# Deep Learning for the Asteroseismic Classification of Red Giants
Based on the papers by [Hon et al. (2017)](https://arxiv.org/abs/1802.07260) and [Hon et al. (2018)](https://arxiv.org/abs/1705.06405). The deep learning classifier learns to classify folded frequency power spectra of red giants as either hydrogen shell-burning (RGB; label 0) or core helium-burning (HeB; label 1).

The classifier requires background-subtracted frequency power spectrum of a red giant, along with measurements of delta_nu (large frequency separation) and numax (frequency at maximum oscillation power). It is trained using the consensus asteroseismic evolutionary states from APOKASC ([Elsworth et al. (2019)](https://arxiv.org/abs/1909.06266)), which has labels for red giants down to a delta_nu of 2.8uHz. 

Required libraries:
---

* numpy
* scipy
* pandas
* torch (>= 0.4.0)

Running the script
===

To perform inference on a star, download the folder and run inference.py. The script accepts the following arguments:

* '--classifier': Path to classifier. Default is 4-year classifier in the /saved_models directory
* '--psd_file': Path to ASCII file of backgrond-corrected power spectrum. Needs to have frequency in first column and power in the second. Can be either whitespace or comma separated. 
* '--numax': Frequency at maximum oscillation power in uHz
* '--dnu': Large frequency separation in uHz
* '--mc_samples': Number of Monte Carlo samples for uncertainty estimation
* '--plot_spectrum': Boolean flag specifying whether the spectrum should be plotted in tandem with prediction


Running inference, example 1:
---

python inference.py --psd_file example_ps/1027337.csv --numax 74.21 --dnu 6.937

![alt text](https://github.com/mtyhon/deep-sub/raw/master/sample/results_RGB.png "RGB Example")



Running inference, example 2:
---

python inference.py --psd_file example_ps/2014377.dat --numax 39 --dnu 4.046

![alt text](https://github.com/mtyhon/deep-sub/raw/master/sample/results_HeB.png "HeB Example")



The delta_nu and numax values in these examples are taken from the [measurements of 16,000 Kepler red giants by the SYD pipeline](https://arxiv.org/abs/1802.04455).

The deep learning algorithm is a Bayesian convolutional neural network that is based on [Bayes by Backprop](https://arxiv.org/abs/1901.02731). It borrows heavily from the implementation in [this repo](https://github.com/kumar-shridhar/PyTorch-BayesianCNN). 

This repo will be updated soon with scripts to train a network from scratch.



