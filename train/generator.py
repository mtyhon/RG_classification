import numpy as np
import os
from torch.utils import data
from scipy import signal, stats
from scipy.misc import imresize

class npz_generator(data.Dataset):
    def __init__(self, root, dim=(2000,), extension='.npz', select_kic=np.array([]), obs_len=82):
        self.root = root  # root folder containing subfolders
        self.extension = extension  # file extension
        self.filenames = []
        self.dim = dim  # image/2D array dimensions
        self.perturb_factor = np.sqrt(365. * 4. / obs_len)
        self.kic = []

        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(extension):
                    self.filenames.append(os.path.join(dirpath, file))
                    self.kic.append(int(file.split('-')[0]))

        self.kic = np.array(self.kic)
        self.filenames = np.array(self.filenames)
        self.indexes = np.arange(len(self.kic))

        self.filenames = self.filenames[np.in1d(self.kic, select_kic)]
        self.indexes = self.indexes[np.in1d(self.kic, select_kic)]
        self.kic = self.kic[np.in1d(self.kic, select_kic)]
        print('Number of files: ', len(self.filenames))
        print('Number of unique IDs: ', len(select_kic))

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        # Get a list of filenames of the batch
        batch_filenames = self.filenames[index]
        batch_kic = self.kic[index]
        # Generate data
        X, y, dnu_vec, flag_vec = self.__data_generation(batch_filenames)

        return X, y, dnu_vec, flag_vec, batch_kic

    def __supergausswindow(self, interval):
        window = signal.general_gaussian(2000, p=3, sig=650)
        window = window / np.max(window)
        return np.multiply(interval, window)

    def __echelle(self, lower_limit_index, upper_limit_index, freq, dnu, power):
        modulo = []
        for i in range(lower_limit_index, upper_limit_index):
            modulo.append((np.remainder(freq[i], dnu) / dnu)[0])

        return modulo, power[lower_limit_index: upper_limit_index]

    def __create_folded_spectra(self, freq, power, dnu, dnu_err, numax, numax_err):
        dnu_perturb = np.random.normal(loc=dnu, scale=dnu_err * self.perturb_factor)
        numax_perturb = np.random.normal(loc=numax, scale=numax_err * self.perturb_factor)
        lower_limit_index = np.argmin(np.abs(freq - (numax_perturb - 2 * dnu_perturb)))
        upper_limit_index = np.argmin(np.abs(freq - (numax_perturb + 2 * dnu_perturb)))
        if not lower_limit_index:
            lower_limit_index = 0
        if not upper_limit_index:
            upper_limit_index = len(freq) - 1

        eps, mod_power = self.__echelle(lower_limit_index, upper_limit_index, freq, dnu_perturb, power)
        mod_power = mod_power[np.argsort(eps)]
        eps = np.sort(eps)
        if mod_power.shape[0] == 0:
            return np.zeros(self.dim), [0.]
        reshape_pow = mod_power.reshape((len(mod_power), 1))

        resize = imresize(reshape_pow, size=(1000, 1), interp='lanczos')
        resize = np.append(resize, resize)
        resize = self.__supergausswindow(resize)
        assert resize.shape == self.dim, 'Folded Spectra is of Wrong Shape!'

        if np.max(resize, axis=0) == 0:
            return np.zeros(self.dim), [0.]
        resize /= np.max(resize, axis=0)
        return resize, dnu_perturb

    def __data_generation(self, batch_filenames):

        data = np.load(batch_filenames)
        freq = data['freq']
        power = data['power']
        numax = data['numax']
        numax_err = data['numax_err']
        dnu = data['dnu']
        dnu_err = data['dnu_err']
        y = data['pop']
        flag_vec = 0

        folded_spectra, dnu_perturb = self.__create_folded_spectra(freq, power, dnu, dnu_err, numax, numax_err)
        if (dnu_perturb == 0):  # focus training on overlapping region
            flag_vec = 1

        X = folded_spectra
        dnu_perturb = np.round(dnu_perturb, 2)
        return X, y, dnu_perturb, flag_vec
