# Transfer learning using energy-only datasets using MACE (see [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/668d2d8e5101a2ffa8dd39ca))

Legacy implementation of a simple transfer learning approach using MACE. An more extensive and up-to-date version will be released soon.

![transfer](https://github.com/user-attachments/assets/7dca8dae-2485-44bf-ab4b-6cfbfa2ed825)


# Usage
Transfer learning can be used to augment the accuracy of a given baseline MACE model to a higher level of theory using a small additional readout layer.
To achieve this, you only need a baseline `model.pth` and a dataset of atomic geometries in XYZ format that is labeled with the energy *difference* between the target level of theory and the baseline level of theory used to train `model.pth`.
The train script in `scripts/run_train.py` is similar to the original MACE training script except that it takes two additional arguments:
- `base_model`: path to your baseline model which will be modified with an additional energy readout. In this example:

  `--base_model=model.pth`
- `delta_MLP`: layer sizes for the additional readout. This follows a similar input format as `radial_MLP` and should be a string representation of a list of layer sizes:

  `--delta_MLP="[8, 8]"`

# Installation
Via `pip`: 
`pip install git+https://github.com/molmod/transfermace`
Make sure you are using `mace v0.3.0` and `ase<3.23`.
