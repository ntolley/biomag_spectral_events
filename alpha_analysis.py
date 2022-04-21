# %% import modules
from os import listdir
import os.path as op
from glob import glob

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import mne
from mne.decoding import SSD
from mne.time_frequency import psd_array_multitaper
from mne.viz import plot_topomap
from mne.report import Report

mne.viz.set_browser_backend('qt')


def ssd_alpha(raw_sessions_filtered, ssd, num_filters=5):

    fig_ssd, axes = plt.subplots(num_filters, 2, figsize=[10, 10],
                                 sharey='col')
    colors = ['k', 'grey']
    n_epochs = 100
    dt = 1/raw_sessions_filtered[0].info['sfreq']
    times = raw_sessions_filtered[0].times
    dt_epoch = dt * len(times) / n_epochs
    time_epochs = np.arange(0., times[-1] + dt_epoch, dt_epoch)

    sessions_alpha = list()

    # plot SSD for selected dimensions
    for filt_idx in range(num_filters):

        # plot SSD filter topography
        im, _ = plot_topomap(ssd.patterns_[filt_idx], ssd.info,
                             axes=axes[filt_idx, 0], show=False)
        plt.colorbar(im, ax=axes[filt_idx, 0])
        axes[filt_idx, 0].set_title(f'SSD transform dim. #{filt_idx + 1}')

        session_alpha = dict()

        for sess_idx in range(len(raw_sessions_filtered)):
            raw_filtered = raw_sessions_filtered[sess_idx]
            color = colors[sess_idx]

            # segment transformed data into epochs
            data_epochs = np.zeros((n_epochs, int(len(raw_filtered.times)
                                                  / n_epochs)))
            for epoch_idx, (tmin, tmax) in enumerate(zip(time_epochs[:-1],
                                                         time_epochs[1:])):
                # define epoch
                times = raw_filtered.times
                time_mask = np.logical_and(times >= tmin, times < tmax)
                # select epoched data from current SSD dimension
                data_epochs[epoch_idx, :] = raw_filtered.get_data()[filt_idx,
                                                                    time_mask]

            raw_filtered_psds, freqs = psd_array_multitaper(
                data_epochs,
                raw_filtered.info['sfreq'],
                fmin=1., fmax=50.
                )

            alpha_mask = np.logical_and(freqs >= 9, freqs <= 14)
            alpha_mean_pow = raw_filtered_psds[:, alpha_mask].mean(axis=1)
            # alpha_max_mean = np.mean(alpha_max)
            # alpha_max_std = np.std(alpha_max)
            avg_filtered_psd = np.mean(raw_filtered_psds, axis=0)

            session_alpha[sess_idx] = raw_filtered_psds[:, alpha_mask]
            label = f'session {sess_idx}'
            # label = (r"session {0}" "\n" r"alpha mean: {1:.2e}" "\n"
            #          r"alpha std: {2:.2e}").format(sess_idx + 1,
            #                                        alpha_max_mean,
            #                                        alpha_max_std)
            # plot epoch-avg PSD
            axes[filt_idx, 1].plot(freqs, avg_filtered_psd,
                                   c=color, label=label)
            axes[filt_idx, 1].set_ylabel('power')
            axes[filt_idx, 1].set_xlabel('freq. (Hz)')
            axes[filt_idx, 1].legend()
        sessions_alpha.append(session_alpha)
    return fig_ssd, sessions_alpha


def fit_ssd(data, info):
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


def analysis(subj_id):
    # read in both sessions belonging to a single subject
    raw_sessions = list()
    session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))

    fig_alpha_topo, axes = plt.subplots(1, 3, figsize=(9, 3))
    for session_idx, fname in enumerate(session_fnames):
        raw = mne.io.read_raw_ctf(fname, verbose=False)
        raw.load_data()
        raw.pick_types(meg=True, eeg=False, ref_meg=False)
        raw.filter(l_freq=1., h_freq=50)
        # raw._data = zscore(raw._data, axis=1)
        raw._data -= raw._data.mean()
        raw._data /= raw._data.std()
        raw.resample(sfreq=500)

        # compute raw PSD
        raw_psds, freqs = psd_array_multitaper(raw.get_data(),
                                               raw.info['sfreq'],
                                               fmin=1., fmax=50.)
        # plot channel-mean psd
        chan_avg_psd = raw_psds.mean(axis=0)
        axes[0].plot(freqs, chan_avg_psd)
        axes[0].set_ylabel('power')
        axes[0].set_xlabel('freq. (Hz)')
        # plot alpha topography
        alpha_mask = np.logical_and(freqs >= 9, freqs <= 14)
        avg_alpha_pow = raw_psds[:, alpha_mask].mean(axis=1)
        im, _ = plot_topomap(avg_alpha_pow, raw.info,
                             axes=axes[session_idx + 1],
                             show=False)
        plt.colorbar(im, ax=axes[session_idx + 1])

        raw_sessions.append(raw)

    # concatenate data and compute a single SSD filter that is applied to both
    # sessions
    raw_data = np.stack([raw.get_data() for raw in raw_sessions], axis=0)
    filtered_data, ssd = fit_ssd(raw_data, raw_sessions[0].info)

    # store filtered data in new Raw instances
    raw_sessions_filtered = [raw.copy() for raw in raw_sessions]
    for session_idx, raw_filtered in enumerate(raw_sessions_filtered):
        raw_filtered._data = filtered_data[session_idx].copy()

    fig_ssd, sessions_alpha = ssd_alpha(raw_sessions_filtered, ssd)

    return fig_alpha_topo, fig_ssd, sessions_alpha


if __name__ == '__main__':
    n_jobs = 1
    data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
    subj_ids = list({fname[:8] for fname in listdir(data_dir)})
    print(subj_ids)

    subj_ids = subj_ids[:5]
    subj_ids = ['RASMDGZN']

    out = Parallel(n_jobs=n_jobs)(delayed(analysis)(subj_id) for subj_id
                                  in subj_ids)
    fig_alpha_topo_list, fig_ssd_list, sessions_alpha_list = zip(*out)

    report = Report()
    for fig_alpha_topo, fig_ssd, subj_id in zip(fig_alpha_topo_list,
                                                fig_ssd_list,
                                                subj_ids):
        report.add_figure(fig_alpha_topo, title=subj_id, tags=('topo',))
        report.add_figure(fig_ssd, title=subj_id, tags=('ssd_psd',))
    report.save('alpha_analysis.html', overwrite=True, open_browser=True)
    