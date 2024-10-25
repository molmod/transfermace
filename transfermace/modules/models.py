from typing import Any, Callable, Dict, List, Optional, Type

import torch
import numpy as np

from e3nn.util.jit import compile_mode
from e3nn import o3, nn

from mace.modules import MACE
from mace.tools.scatter import scatter_sum

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    #EquivariantProductBasisBlock,
    #InteractionBlock,
    #LinearDipoleReadoutBlock,
    #LinearNodeEmbeddingBlock,
    #LinearReadoutBlock,
    #NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    #RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

def extract_kwargs(model):
    kwargs = {
            'r_max': model.radial_embedding.bessel_fn.r_max.item(),
            'num_bessel': torch.numel(model.radial_embedding.bessel_fn.bessel_weights),
            'num_polynomial_cutoff': model.radial_embedding.cutoff_fn.p.item(),
            'max_ell': model.spherical_harmonics.irreps_out.lmax,
            'interaction_cls': type(model.interactions[1]),
            'interaction_cls_first': type(model.interactions[0]),
            'num_interactions': model.num_interactions.item(),
            'num_elements': model.node_embedding.linear.irreps_in.dim,
            'hidden_irreps': model.readouts[0].linear.irreps_in,
            'MLP_irreps': model.readouts[1].hidden_irreps,
            'atomic_energies': model.atomic_energies_fn.atomic_energies.numpy(),
            'avg_num_neighbors': model.interactions[0].avg_num_neighbors,
            'atomic_numbers': model.atomic_numbers.numpy(),
            'correlation': model.products[0].symmetric_contractions.contractions[0].correlation,
            'gate': torch.nn.functional.silu, # hardcoded
            'radial_MLP': model.interactions[0].radial_MLP,
            }
    if hasattr(model, 'scale_shift'):
        kwargs['atomic_inter_scale'] = model.scale_shift.scale.item()
        kwargs['atomic_inter_shift'] = model.scale_shift.shift.item()
    return kwargs


@compile_mode("script")
class TransferReadoutBlock(torch.nn.Module):
    def __init__(
            self, irreps_in: o3.Irreps, delta_MLP: list[int], gate: Optional[Callable]
    ):
        super().__init__()
        assert len(delta_MLP) > 0
        self.layers = torch.nn.ModuleList()
        for size in delta_MLP:
            irreps = o3.Irreps('{}x0e'.format(size))
            self.layers.append(o3.Linear(irreps_in=irreps_in, irreps_out=irreps))
            self.layers.append(nn.Activation(irreps_in=irreps, acts=[gate]))
            irreps_in = irreps
        self.layers.append(o3.Linear(irreps_in=irreps, irreps_out=o3.Irreps('0e')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        for layer in self.layers:
            x = layer(x)
        return x


@compile_mode("script")
class TransferMACE(MACE):

    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        delta_MLP: list[int],
        delta_atomic_energies: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        hidden_irreps_out = str(kwargs['hidden_irreps'][0]) # ?
        self.readouts_delta = torch.nn.ModuleList()

        assert kwargs['num_interactions'] == 2
        readout0 = TransferReadoutBlock(kwargs['hidden_irreps'], delta_MLP, kwargs['gate'])
        readout1 = TransferReadoutBlock(hidden_irreps_out, delta_MLP, kwargs['gate'])
        self.readouts_delta.append(readout0)
        self.readouts_delta.append(readout1)
        self.delta_atomic_energies_fn = AtomicEnergiesBlock(delta_atomic_energies)

    @classmethod
    def from_model(cls, model, **kwargs):
        model_kwargs = extract_kwargs(model)
        transfer = cls(**kwargs, **model_kwargs)
        transfer.load_state_dict( # loads existing model parameters
                model.state_dict(),
                strict=False,
                )
        return transfer

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.delta_atomic_energies_fn(data["node_attrs"])
        if not self.training: # add regular
            node_e0 += self.atomic_energies_fn(data['node_attrs'])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        node_es_list_delta = []
        for interaction, product, readout, readout_delta in zip(
            self.interactions, self.products, self.readouts, self.readouts_delta
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
            node_es_list_delta.append(readout_delta(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)
        node_inter_es_delta = torch.sum(
            torch.stack(node_es_list_delta, dim=0), dim=0
        )  # [n_nodes, ]

        # Sum over nodes in graph
        if not self.training:
            node_inter_es = node_inter_es + node_inter_es_delta
        else:
            node_inter_es = node_inter_es_delta

        inter_e = scatter_sum(
            src=node_inter_es,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }

        return output


