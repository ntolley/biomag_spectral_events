# %% import modules
from os import listdir
import os.path as op
from glob import glob

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

import mne
from mne.decoding import SSD
from mne.time_frequency import psd_array_multitaper
from mne.viz import plot_topomap

mne.viz.set_browser_backend('qt')

# %% extract subject IDs from data directory


data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
subj_ids = list({fname[:8] for fname in listdir(data_dir)})
print(subj_ids)

# %% define alpha SSD function


def alpha_ssd(data, info):
    """Alpha-band Spatio-Spectral Decomposition (SSD) from raw data"""
    freqs_sig = 9, 14
    freqs_noise = 8, 15

    ssd = SSD(info=info,
              reg='oas',
              sort_by_spectral_ratio=True,
              filt_params_signal=dict(l_freq=freqs_sig[0],
                                      h_freq=freqs_sig[1],
                                      l_trans_bandwidth=1,
                                      h_trans_bandwidth=1),
              filt_params_noise=dict(l_freq=freqs_noise[0],
                                     h_freq=freqs_noise[1],
                                     l_trans_bandwidth=1,
                                     h_trans_bandwidth=1))
    data_filtered = ssd.fit_transform(X=data)
    return data_filtered, ssd

# %% load data for selected subjects and fit SSD transform


subj_ids = ['WWIZIWSJ']  # XXXfor now, only select a single arbitrary subj
for subj_id in subj_ids:
    # read in both sessions belonging to a single subject
    raw_sessions = list()
    session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))
    for session_idx, fname in enumerate(session_fnames):
        raw = mne.io.read_raw_ctf(fname, verbose=False)
        raw.load_data()
        raw.pick_types(meg=True, eeg=False, ref_meg=False)
        raw.filter(l_freq=1., h_freq=50)
        raw._data = zscore(raw.get_data(), axis=1)
        raw.resample(sfreq=500)
        raw_sessions.append(raw)

    # concatenate data and compute a single SSD filter that is applied to both
    # sessions
    raw_data = np.stack([raw.get_data() for raw in raw_sessions], axis=0)
    filtered_data, ssd = alpha_ssd(raw_data, raw_sessions[0].info)

    # store filtered data in new Raw instances
    raw_sessions_filtered = [raw.copy() for raw in raw_sessions]
    for session_idx, raw_filtered in enumerate(raw_sessions_filtered):
        raw_filtered._data = filtered_data[session_idx].copy()
# %% plot and compare the average PSD across all channels, between sessions


fig, ax = plt.subplots(1)
for raw_session in raw_sessions:
    raw_session.plot_psd(average=True, fmax=50, dB=False, ax=ax)
# %% plot SSD-tranformed PSD where the raw time series has been divided into
# mulitiple epochs)


num_filters = 5  # select first num_filters SSD dimensions off the top

fig_ssd, axes = plt.subplots(num_filters, 2, figsize=[10, 10])
colors = ['k', 'grey']
n_epochs = 100
dt = 1/raw_sessions_filtered[0].info['sfreq']
times = raw_sessions_filtered[0].times
dt_epoch = dt * len(times) / n_epochs
time_epochs = np.arange(0., times[-1] + dt_epoch, dt_epoch)

# plot SSD for selected dimensions
for filt_idx in range(num_filters):
    # plot SSD filter topography
    im, _ = plot_topomap(ssd.patterns_[filt_idx], ssd.info,
                         axes=axes[filt_idx, 0], show=False)
    plt.colorbar(im, ax=axes[filt_idx, 0])
    axes[filt_idx, 0].set_title(f'SSD transform dim. #{filt_idx + 1}')

    for sess_idx in range(len(raw_sessions_filtered)):
        raw_filtered = raw_sessions_filtered[sess_idx]
        color = colors[sess_idx]

        data_epochs = np.zeros((n_epochs, int(len(raw_filtered.times)
                                              / n_epochs)))
        for epoch_idx, (tmin, tmax) in enumerate(zip(time_epochs[:-1],
                                                     time_epochs[1:])):
            # define epoch
            times = raw_filtered.times
            time_mask = np.logical_and(times >= tmin, times < tmax)
            # select epoched data from current SSD dimension
            data_epochs[epoch_idx, :] = raw_filtered._data[filt_idx, time_mask]

        raw_filtered_psds, freqs = psd_array_multitaper(
            data_epochs,
            raw_filtered.info['sfreq'],
            fmin=1., fmax=50.
            )

        alpha_mask = np.logical_and(freqs >= 9, freqs <= 14)
        alpha_max = np.max(raw_filtered_psds[:, alpha_mask], axis=1)
        alpha_max_mean = np.mean(alpha_max)
        alpha_max_std = np.std(alpha_max)
        avg_filtered_psd = np.mean(raw_filtered_psds, axis=0)
        label = (r"session {0}" "\n" r"alpha mean: {1:.2e}" "\n"
                 r"alpha std: {2:.2e}").format(sess_idx + 1,
                                               alpha_max_mean,
                                               alpha_max_std)
        axes[filt_idx, 1].plot(freqs, avg_filtered_psd, c=color, label=label)
        axes[filt_idx, 1].set_ylabel('power')
        axes[filt_idx, 1].set_xlabel('freq. (Hz)')
        axes[filt_idx, 1].legend()

# %%
