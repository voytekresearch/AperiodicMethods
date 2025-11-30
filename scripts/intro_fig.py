"""
Figure 0: Background / Pedgogical Fig

This is a conceptual figure, to introduce the methodology of time-resolved 
parameterization.

Panels:
a, Simulated neural time-series
b, 

"""


# SET-UP #######################################################################

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mne.time_frequency import tfr_array_multitaper
import seaborn as sns

from neurodsp.sim import (
    sim_synaptic_current,
    sim_knee,
    sim_combined,
    sim_oscillation
)
from neurodsp.spectral import compute_spectrum

from neurodsp.utils import create_times
from neurodsp.sim.utils import rotate_timeseries
from specparam import SpectralModel, SpectralTimeModel

import fooof
from fooof.utils.params import compute_knee_frequency

import sys
sys.path.append('code')
from plt_utils import remove_spines, FIGURE_WIDTH, PANEL_FONTSIZE
from tfr_utils import plot_evoked_tfr

# settings - figure
plt.style.use('mplstyle/nature_reviews.mplstyle')
FIGSIZE = [FIGURE_WIDTH+2, 10]
# TIME_POINTS = [-0.35, -0.25, -0.15, 1.35] # which to plot
# COLORS = sns.color_palette("Greens", len(TIME_POINTS))
TITLE_FONTSIZE = PANEL_FONTSIZE - 5
# sns.set_context('talk')

# settings - simulation parameters
N_SECONDS = 2 # signal duration (s)
T_MIN = -0.5 # start time (s)
FS = 1000 # sampling frequency (Hz)
EXPONENT = -2.5 # baseline exponent
DELTA_EXP = -1 # task-evoked change in exponent (negative for flattening)
F_ROTATION = 45 # rotation frequency (Hz)

# settings - fitting parameters
SPECPARAM_SETTINGS = {
    'aperiodic_mode' : 'fixed',
    'max_n_peaks' : 0,
    'verbose' : False,
}

# settings - multitaper
TFR_WINDOW = 0.3 # window length (s)
FREQ_BANDWIDTH = 7 # frequency bandwidth (Hz)

# set random seed
np.random.seed(39)
colors_pal = (sns.color_palette('crest'))
prestim_color = colors_pal[0]
poststim_color = colors_pal[3]

# MAIN #########################################################################

def main():

    # create figure and gridspec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(figure=fig, ncols=1, nrows=3, 
                           height_ratios=[0.75, 0.5, 0.5])

    # # Add variable freq range plots
    # ax_e = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
    #                                         width_ratios=[1, 1, 1, 1])
    # plot_variable_freq_ranges(fig, ax_e)
    # plot_diff_time_wins(fig, plt.subplot(ax_e[3]))

    # Simulate and plot aperiodic + oscillation with events 
    ax_a = fig.add_subplot(gs[0])
    events = [0.75, 1.25, 3.25, 4.5]
    event_win = 0.25
    fs = 1000
    sig, times = generate_modulated_signal(events, event_win, fs)
    for ev in events: 
        ax_a.axvline(ev, color='grey', linewidth=3)
        ax_a.axvspan(xmin = ev-event_win, xmax=ev,  color = prestim_color)
        ax_a.axvspan(xmin = ev, xmax=ev+event_win, color = poststim_color)
    ax_a.plot(times, sig, color='k', alpha=0.85)

    # Compute and plot TFR
    # ax_b = fig.add_subplot(gs[2])
    # tfr, time_tfr, freqs = compute_and_plot_tfr(sig, fig, ax_b)

    # Plot spectral parameterization
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1],
                                            width_ratios=[1, 1, 1, 1, 1])
    ax_b_0 = fig.add_subplot(gs_b[0])
    ax_b_1 = fig.add_subplot(gs_b[1])
    ax_b_2 = fig.add_subplot(gs_b[2])
    ax_b_3 = fig.add_subplot(gs_b[3])

    gs_c = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2],
                                            width_ratios=[1, 1, 1, 1, 1])
    ax_c_0 = fig.add_subplot(gs_c[0])
    ax_c_1 = fig.add_subplot(gs_c[1])
    ax_c_2 = fig.add_subplot(gs_c[2])
    ax_c_3 = fig.add_subplot(gs_c[3])

    axes_c = [(ax_b_0, ax_c_0), (ax_b_1, ax_c_1), (ax_b_2, ax_c_2), (ax_b_3, ax_c_3)]
    for axs, ev in zip(axes_c, events):
        ev_idx = int(ev*fs)
        plot_prestim_poststim_psd(sig, ev_idx, event_win, fs, axs)
    # ax_c_1.set_title("                                Traditional Analysis", fontsize=TITLE_FONTSIZE)
    # for ax in [ax_c_1, ax_c_2, ax_c_3]:
    #     ax.sharey(ax_c_0)
    #     ax.label_outer()
    # for ax, col in zip(axes_c, COLORS):
    #     add_background(ax, col)

    # # add large text elipsis
    # ax_c_x = fig.add_subplot(gs_c[3])
    # ax_c_x.text(0.5, 0.5, r"$\cdots$", fontsize=40, ha="center", va="center")
    # ax_c_x.axis("off")

    # # Compute and plot sliding window parameters
    # ax_d = fig.add_subplot(gs[4])
    # ax_d.set_title("Time-resolved spectral features", fontsize=TITLE_FONTSIZE)
    # # compute_and_plot_sliding_window_params(tfr, time_tfr, freqs, ax=ax_d)

    # add panel labels
    fig.text(0.01, 0.97, 'A', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.75, 0.97, 'B', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.76, 'C', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.60, 'D', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.43, 'E', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.25, 'F', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # remove spines
    # for ax in [ ax_b, *axes_c, ax_d]:
    #     remove_spines(ax)

    # # Save
    fig.savefig('figures\\figure_pedagogical.png')#os.path.join('figures', 'figure_0.png'))

def generate_modulated_signal(events, event_win, fs):

    sim_components = {
        "sim_powerlaw": {"exponent": -1},
        "sim_oscillation": [{"freq": 10}],
    }
    sig = sim_combined(n_seconds=5, fs=fs, components=sim_components)
    osc_sig = (sim_oscillation(n_seconds=5, fs=fs, freq=10)*0.10)

    times = create_times(n_seconds=5, fs=fs)

    for ev in events:
        ev_idx = int(ev*fs)
        ev_idx_end = int(ev_idx + (event_win*fs))
        print(ev_idx, ev_idx_end)

        mod_sig = sig[ev_idx : ev_idx_end]
        osc_sig_add = osc_sig[ev_idx : ev_idx_end]
        rotated = rotate_timeseries(sig=mod_sig, fs=fs, delta_exp=-1, f_rotation=40)
        rotated = rotated + osc_sig_add
        sig[ev_idx : ev_idx_end ] = rotated

    return sig, times

def plot_prestim_poststim_psd(sig, ev_idx, event_win, fs, axs):

    psd_ax, bar_ax = axs

    freqs, pows_pre = compute_spectrum(sig = sig[ev_idx - int(event_win*fs) : ev_idx ], fs=fs)
    freqs, pows_post = compute_spectrum(sig = sig[ev_idx : ev_idx + int(event_win*fs) ], fs=fs)
    pows_pre = pows_pre[(freqs > 0.5) & (freqs < 50)]
    pows_post = pows_post[(freqs > 0.5) & (freqs < 50)]
    freqs = freqs[(freqs > 0.5) & (freqs < 50)]

    alpha_mask = (freqs > 8) & (freqs <=12)
    pre_total_power = np.mean(pows_pre[alpha_mask])
    pst_total_power = np.mean(pows_post[alpha_mask])
    delta_total = (pst_total_power - pre_total_power)

    specpar_pre = fooof.FOOOF(**SPECPARAM_SETTINGS)
    specpar_pre.fit(freqs=freqs, power_spectrum=pows_pre, freq_range=(0.5, 50))    
    specpar_pst = fooof.FOOOF(**SPECPARAM_SETTINGS)
    specpar_pst.fit(freqs=freqs, power_spectrum=pows_post, freq_range=(0.5, 50))

    flat_spec_delta = ((specpar_pst._spectrum_flat) - (specpar_pre._spectrum_flat))
    flat_spec_delta = np.mean(flat_spec_delta[alpha_mask])
    ap_delta = ((specpar_pst._ap_fit) - (specpar_pre._ap_fit))
    ap_delta = np.mean(ap_delta[alpha_mask])

    pre_ap_fit = 10**(specpar_pre._ap_fit)
    pst_ap_fit = 10**(specpar_pst._ap_fit)

    psd_ax.plot(freqs,pows_pre, color = prestim_color, alpha=0.85)
    psd_ax.plot(freqs,pre_ap_fit, color = 'k', linewidth=2)
    psd_ax.plot(freqs,pows_post, color = poststim_color, alpha=0.85)
    psd_ax.plot(freqs,pst_ap_fit, color = 'k', linewidth=2)
    psd_ax.axvspan(xmin=8, xmax=12, color='grey', alpha=0.5)

    psd_ax.fill_between(
        freqs[alpha_mask], pows_post[alpha_mask], pst_ap_fit[alpha_mask], where=(pows_post[alpha_mask] != pst_ap_fit[alpha_mask]), 
        interpolate=True, color="#054907", alpha=0.25
        )
    # psd_ax.set_xscale('log')
    psd_ax.set_yscale('log')

    bar_ax.bar(['delta total', 'delta corr osc', 'delta aper'], height=[delta_total, flat_spec_delta, ap_delta])

if __name__ == "__main__":
    main()
