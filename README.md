# DLquantification
Repository for researching deep learning approaches for quantification.

## GMNet
GMNet implementation is based on pytorch. It is a neural network that focus on the quantification problem. It uses gaussian mixture models to represent the samples.

## Experiments
All the experimentation done with GMNet (HistNet or other DL approaches for quantification) lives in the folder ˋexperimentsˋ. You can check the corresponding documentation [here](dlquantification/README.md)

- The experiments related to training with GMNet (or other DL approaches for quantification) are in the folder `train_lequa`. 

- The experiments using validation data with the labels provided by the organizers ^[The data for the experiments in `comp_trad_DL_meth` must be requested from the organizers of the LeQua2024 quantification competition. Specifically, we requested the individual labels of dataset T2.] (section **"Impact of data availability on quantification methods"**) are in the folder `comp_trad_DL_meth`.

    