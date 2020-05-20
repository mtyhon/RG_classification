
Network Training
===

Execute run_training.py. The script contains the following arguments:


* '--root': Path to folder containing background-corrected power spectra for training
* '--k_folds': Positive integer number of folds for k-fold validation. Set this above 1 if k-fold validation is desired, otherwise normal training is done 
* '--weight_llrgb': Boolean flag to determine if stars with dnu above 9uHz (no ambiguity in classification) should be weighted differently. Default: False
* '--llrgb_factor': If weight_llrgb is enabled, this float values determines how much more/less high dnu stars should be weighted. Default: 0.5
* '--holdout_frac': Only for normal training. Determines what fraction of the training set to hold out for validation. Default: 0.15
* '--obs_len': Observational length of data in days. Used to scale the delta_nu/numax uncertainties in the generator
* '--init_lr': Initial learning rate for the Adam optimizer. Default: 0.001
* '--num_epochs': Number of training iterations. Default: 250


Example
---

python run_training.py --root npz_examples --k_folds 3 --num_epochs 200


How training works
===

The data generator loads in .npz files containing the background corrected power spectra, delta_nu, sigma_delta_nu, numax, sigma_numax, and the truth labels of each star. To create such files, place ASCII whitespace delimited files of background corrected power spectra into the /psd_file folder. The file NEEDS to have its star identification number in its name. Next, run the convert.py script:


python convert.py


This script currently works only for Kepler red giants, and will cross-reference each star's power spectrum with evolutionary state measurements from the [Elsworth et al. (2019) catalogue](https://arxiv.org/abs/1909.06266), and asteroseismic measurements from the [Yu et al. (2018) catalogue](https://arxiv.org/abs/1802.04455). Thus, ensure that the star you are making belongs to both these catalogues.


During training, the generator will create folded spectra on-the-fly. The delta_nu and numax used to create each folded spectrum will be sampled from a Gaussian with its nominal value as the mean and its sigma as the dispersion. The obs_len parameter scales the uncertainty of delta_nu and numax to match shorter/longer observation lengths. 

The best models during training are saved in the saved_models/ folder.


