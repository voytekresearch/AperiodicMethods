"""
Figure for Reccomendation 2: Generate high signal-to-noise power spectra.
a, Smoothing / transforming data
b, Windowing and padding
c, Computing spectra for short time windows
"""

# SET-UP #######################################################################

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import get_window


from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
from neurodsp.spectral import compute_spectrum
from neurodsp.filt import filter_signal

# settings - figure
FIGSIZE = [5, 7]
plt.style.use('mplstyle/nature_reviews.mplstyle')

# settings - panel b
FS = 500 # sampling frequency
N_SECONDS = 2 # signal duration
PAD_FRACTION = 0.2 # pad duration / signal duration

# MAIN #########################################################################

def main():

    # create figure and gridspec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=6, 
                             height_ratios=[0.02, 0.8, 0.02, 0.8, 0.02, 1])

    # plot panels a and b
    plot_panel_a(fig, spec[1], fs=1200) 
    plot_panel_b(fig, spec[3], fs=FS, n_seconds=N_SECONDS, 
                 pad_length=int(N_SECONDS*FS*PAD_FRACTION))
    
    # plot panel c: Cohen, 2014
    panel_c_path = "notebooks/images/cohen_2014_multitaper.png"
    ax_c = fig.add_subplot(spec[5])
    ax_c.imshow(plt.imread(panel_c_path), aspect='auto')
    ax_c.axis('off')

    # add subplot titles
    titles = ["Smoothing / transforming data", 
              "Windowing and padding", 
              "Multitaper method"]
    for ii, title in enumerate(titles):
        ax_title = fig.add_subplot(spec[ii*2])
        ax_title.set_title(title, fontsize=12, pad=0)
        ax_title.axis("off")

    # save/show
    fig.savefig(os.path.join('figures', 'figure_2.png'))
    plt.show()


def plot_panel_a(fig, subplot_spec, fs, n_seconds=10):
    # create nested subgridspec
    spec = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, 
                                            width_ratios=[1, 1])
    ax_0 = fig.add_subplot(spec[0, 0])
    ax_1 = fig.add_subplot(spec[0, 1])

    # simulate signal (aperiodic activity + line noise)
    sim_components = {'sim_powerlaw': {'exponent' : -2},
                      'sim_oscillation': [{'freq' : 60}, {'freq' : 120}]}
    signal = sim_combined(n_seconds=n_seconds, fs=fs, 
                          components=sim_components, 
                          component_variances=[1, 0.5, 0.1])

    # apply bandpass filter
    signal_mfilt = signal.copy()
    for center_freq in [60, 120]:
        signal_mfilt = filter_signal(signal_mfilt, fs=fs, pass_type='bandstop',
                                      f_range=[center_freq-3, center_freq+3],
                                      butterworth_order=3)

    # compute power spectra
    freqs, psd = compute_spectrum(signal, fs=fs, method='welch')
    _, psd_mfilt = compute_spectrum(signal_mfilt, fs=fs, method='welch')

    # interpolate spectrum
    psd_interp = interp_spectra(freqs, psd, f_range=[58, 62])
    psd_interp = interp_spectra(freqs, psd_interp, f_range=[118, 122])

    # plot
    ax_0.loglog(freqs, psd, color='k', alpha=0.5, label='raw signal')
    ax_0.loglog(freqs, psd_mfilt, color='b', alpha=0.5, label='filtered signal')
    ax_1.loglog(freqs, psd, color='k', alpha=0.5, label='original psd')
    ax_1.loglog(freqs, psd_interp, color='g', alpha=0.5, label='interpolated psd')

    # label
    ax_0.set_title('Bandpass filter')
    ax_1.set_title('Interpolation')
    ax_0.legend()
    ax_1.legend()

    for ax in [ax_0, ax_1]:
        ax.set(xlabel='frequency (Hz)', ylabel='power (au)')

   # beautify
    for ax in [ax_0, ax_1]:
        remove_spines(ax)


def plot_panel_b(fig, subplot_spec, fs, n_seconds, pad_length):
    """
    a, signal
    b, windowed signal
    c, padded signal
    d, padded windowed signal
    """

    # create nested subgridspec
    spec = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=subplot_spec, 
                                            height_ratios=[1, 1, 1, 1],
                                            width_ratios=[2, 1])
    axes = np.array([fig.add_subplot(spec[i,0]) for i in range(4)])
    for ax in axes[1:-1]:
        ax.sharex(axes[0])

    # simulate signal (oscillation + aperiodic activity)
    sim_components = {'sim_powerlaw': {'exponent' : -2},
                      'sim_oscillation': [{'freq' : 10}, {'freq' : 45}]}
    signal = sim_combined(n_seconds=n_seconds, fs=fs, 
                          components=sim_components, 
                          component_variances=[1, 0.5, 0.2])
    time = create_times(n_seconds=n_seconds, fs=fs)
    axes[0].plot(time, signal, color='k')

    # pad signal (mirror-padding)
    signal_pad = np.concatenate((np.flip(signal[:pad_length]), signal, 
                                 np.flip(signal[-pad_length:])))
    time_pad = create_times(n_seconds=(len(signal) / fs) + (pad_length * 2 / fs), 
                        fs=fs, start_val=-pad_length / fs)
    axes[2].plot(time_pad[pad_length-1:-pad_length-1], 
            signal_pad[pad_length-1:-pad_length-1], color='k')
    axes[2].plot(time_pad[:pad_length], signal_pad[:pad_length], color='r')
    axes[2].plot(time_pad[-pad_length:], signal_pad[-pad_length:], color='r')

    # window signals (orignal and padded)
    window = get_window('hann', len(signal))
    signal_w = signal * window
    signal_pw = signal_pad * get_window('hann', len(signal_pad))
    axes[1].plot(time, signal_w, color='k')
    axes[1].plot(time, window, color='b', alpha=0.5)
    axes[3].plot(time_pad, signal_pw, color='g')

    # label
    axes[0].set_title('Time-series')
    axes[-1].set_xlabel('time (s)')
    axes[2].set_ylabel('             voltage (au)')

    # remove clutter
    for ax in axes:
        ax.set_yticks([])
    for ax in axes[:-1]:
        ax.set_xticks([])

    # compute power spectra
    freqs, psd = compute_spectrum(signal, fs=fs, method='welch')
    # _, psd_w = compute_spectrum(signal_w, fs=fs, method='welch')
    # _, psd_p = compute_spectrum(signal_pad, fs=fs, method='welch')
    freqs_pw, psd_pw = compute_spectrum(signal_pw, fs=fs, method='welch')
    ax_psd = fig.add_subplot(spec[:, 1])
    ax_psd.loglog(freqs, psd, color='k', alpha=0.5, label='original signal')
    # ax_psd.loglog(freqs, psd_w, color='b', alpha=0.5, label='windowed signal')
    # ax_psd.loglog(freqs_pw, psd_p, color='r', alpha=0.5, label='padded signal')
    ax_psd.loglog(freqs_pw, psd_pw, color='g', alpha=0.5, label='pad + window')
    ax_psd.legend()
    ax_psd.set(xlabel='frequency (Hz)', ylabel='power (au)', 
               title='Power spectra')

    # beautify
    for ax in axes:
        remove_spines(ax)
    remove_spines(ax_psd)


def remove_spines(ax):
    """
    Remove the top and left spines from a matplotlib axis.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def interp_spectra(freqs, spectra, f_range):

    # Build mask for the band
    freq_mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    temp_x = freqs[~freq_mask]
    temp_y = spectra[~freq_mask]

    # Prepare data for interpolation
    x_log = np.log10(temp_x)
    y_log = np.log10(temp_y)
    xi = np.log10(freqs[freq_mask])
    f_interp = interp1d(x_log, y_log, kind='linear', bounds_error=True, assume_sorted=True)
    yi = f_interp(xi)
    y_new = 10 ** yi

    spectrum_interp = spectra.copy()
    spectrum_interp[freq_mask] = y_new

    return spectrum_interp


if __name__ == "__main__":
    main()
