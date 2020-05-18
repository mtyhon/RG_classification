import numpy as np
from scipy.misc import imresize
from scipy import signal

def stack(lower_limit_index, upper_limit_index, freq, dnu, power): #folding function
    modulo = []
    for i in range(lower_limit_index, upper_limit_index):
        modulo.append(np.remainder(freq[i], dnu)/dnu)

    return modulo, power[lower_limit_index: upper_limit_index]

def supergausswindow(interval):
    window = signal.general_gaussian(2000, p=3,sig=650)
    window = window/np.max(window)
    return np.multiply(interval, window)


def create_folded_spectrum(freq, power, numax, dnu):

    lower_limit_index = np.argmin(np.abs(freq - (numax - 2 * dnu)))
    upper_limit_index = np.argmin(np.abs(freq - (numax + 2 * dnu)))

    if not lower_limit_index:
        lower_limit_index = 0

    if not upper_limit_index:
        upper_limit_index = len(freq) - 1

    eps, mod_power = stack(lower_limit_index, upper_limit_index, freq, dnu, power)
    mod_power = mod_power[np.argsort(eps)].reshape((len(mod_power), 1))
    resize = imresize(mod_power, size=(1000,1) , interp='lanczos')
    resize = np.append(resize, resize)
    resize = supergausswindow(resize).flatten()
    row_max = np.max(resize)

    return resize/row_max


