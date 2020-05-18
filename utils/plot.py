import numpy as np
import matplotlib.pyplot as plt


def plot_folded_spectrum(folded_spectrum, filename, dnu, numax, pred, package_dir, freq, power):
    lower_limit_index = np.argmin(np.abs(freq - (numax - 2 * dnu)))
    upper_limit_index = np.argmin(np.abs(freq - (numax + 2 * dnu)))

    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(freq[lower_limit_index:upper_limit_index], power[lower_limit_index:upper_limit_index])
    ax1.set_ylabel('S/N')
    ax1.set_title('Original Spectrum')
    ax1.set_xlabel('Frequency ($\\mu$Hz)')
    ax2.plot(folded_spectrum.flatten(), c='k')
    ax2.set_xlabel('Arbitrary Units')
    ax2.set_title('Folded Spectrum\n%s, $\\Delta\\nu$: %.2f$\\mu$Hz, $\\nu_{\\mathrm{max}}$: %.1f$\\mu$Hz, Pred: %d' %(filename, dnu, numax, pred))
    plt.tight_layout(h_pad=2)
    plt.savefig(package_dir+'/results.png')
