from os import listdir
import os.path as op
from glob import glob

import numpy as np
from scipy.stats import zscore, norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import mne
from mne.decoding import SSD
from mne.time_frequency import psd_array_multitaper
from mne.viz import plot_topomap
from mne.report import Report

mne.viz.set_browser_backend('qt')

bads = dict(
    DXFTLCJA=[[], ['MRT35-1609', 'MRT15-1609', 'MRP57-1609', 'MRP56-1609',
                   'MRP23-1609', 'MRP22-1609', 'MRO24-1609', 'MLF67-1609',
                   'MLF66-1609', 'MRP34-1609', 'MLT14-1609']],
    BQBBKEBX=[[], ['MRT21-1609', 'MLT31-1609', 'MLT41-1609', 'MLT51-1609',
                   'MLT21-1609', 'MLT16-1609', 'MLT11-1609', 'MLT22-1609',
                   'MLT12-1609', 'MLT32-1609', 'MLT33-1609', 'MLT42-1609',
                   'MLT43-1609', 'MRF14-1609', 'MRT34-1609', 'MRT27-1609']],
    JBGAZIEO=[[], ['MLF25-1609']],
    QGFMDSZY=[['MLP41-1609'], []],
    ZDIAXRUW=[[], []]
)


def conf_int_mean(data, conf=0.95):
    n = data.shape[0]
    mean = data.mean(axis=0)
    std_mean = data.std(axis=0, ddof=1) / n**(1/2)
    lb = norm.ppf((1 - conf) / 2, loc=mean, scale=std_mean)
    ub = norm.ppf((1 + conf) / 2, loc=mean, scale=std_mean)
    return mean, lb, ub


def ssd_alpha(raw_sessions_filtered, ssd, num_filters=5):

    fig_ssd, axes = plt.subplots(num_filters, 2, figsize=[10, 10],
                                 sharey='col')
    colors = ['grey', 'orange']
    n_epochs = 100
    dt = 1/raw_sessions_filtered[0].info['sfreq']
    times = raw_sessions_filtered[0].times
    dt_epoch = dt * len(times) / n_epochs
    time_epochs = np.arange(0., times[-1] + dt_epoch, dt_epoch)

    sessions_alpha = list()

    # plot SSD for selected dimensions
    for filt_idx in range(num_filters):

        # plot SSD filter topography
        # XXX SSD.filters_ or SSD.patterns_???
        # https://mne.tools/stable/generated/mne.decoding.SSD.html#mne.decoding.SSD.get_spectral_ratio  # noqa
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
            # label = (r"session {0}" "\n" r"alpha mean: {1:.2e}" "\n"
            #          r"alpha std: {2:.2e}").format(sess_idx + 1,
            #                                        alpha_max_mean,
            #                                        alpha_max_std)
            # plot epoch-avg PSD
            axes[filt_idx, 1].plot(freqs, avg_filtered_psd,
                                   c=color, label=f'session {sess_idx + 1}')
            axes[filt_idx, 1].set_ylabel('power')
            axes[filt_idx, 1].set_xlabel('freq. (Hz)')
            axes[filt_idx, 1].legend()
        sessions_alpha.append(session_alpha)
    return fig_ssd, sessions_alpha


def ssd_spec_ratios(raw_sessions_filtered, ssd):

    n_ssd_dims = ssd.patterns_.shape[0]
    fig_spec_ratios, axes = plt.subplots(1)
    colors = ['grey', 'orange']

    data_agg = np.stack([raw_filtered.get_data() for raw_filtered
                         in raw_sessions_filtered], axis=0)
    spec_ratio_agg, _ = ssd.get_spectral_ratio(data_agg)

    # store stats of comparison across sessions
    spec_ratio_stats = {'mean': np.zeros([2, n_ssd_dims]),
                        'lb': np.zeros([2, n_ssd_dims]),
                        'ub': np.zeros([2, n_ssd_dims])}

    for sess_idx, raw_filtered in enumerate(raw_sessions_filtered):
        epochs = mne.make_fixed_length_epochs(raw_filtered,
                                              duration=2.5,
                                              reject_by_annotation=False,
                                              proj=False)

        n_epochs = epochs.get_data().shape[0]
        spec_ratio_epochs = np.zeros([n_epochs, n_ssd_dims])
        for idx, data_epoch in enumerate(epochs):
            spec_ratio_epochs[idx, :], _ = ssd.get_spectral_ratio(data_epoch)
            # axes.plot(spec_ratio_epochs, color=colors[sess_idx],
            #           marker=',', linewidth=0., alpha=0.5)

        sess_mean, lb, ub = conf_int_mean(spec_ratio_epochs, conf=0.95)
        spec_ratio_stats['mean'][sess_idx, :] = sess_mean
        spec_ratio_stats['lb'][sess_idx, :] = lb
        spec_ratio_stats['ub'][sess_idx, :] = ub

        axes.plot(sess_mean, color=colors[sess_idx],
                  linewidth=0.5, label=f'session {sess_idx + 1}')
        axes.fill_between(x=range(n_ssd_dims), y1=lb, y2=ub, linewidth=0,
                          color=colors[sess_idx], alpha=0.5)

    axes.plot(spec_ratio_agg, color='black', linewidth=0.5, linestyle='--',
              alpha=0.7, label='aggregate')
    axes.set_xlabel("Eigenvalue Index")
    axes.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")

    # find highest SSD component where the mean spectral ratio is different
    sess_argmin = spec_ratio_stats['mean'].argmin(axis=0)
    for filter_idx in range(n_ssd_dims):
        argmin = sess_argmin[filter_idx]
        min_mean = spec_ratio_stats['mean'][argmin, filter_idx]
        min_ub = spec_ratio_stats['ub'][argmin, filter_idx]
        max_mean = spec_ratio_stats['mean'][argmin - 1, filter_idx]
        max_lb = spec_ratio_stats['lb'][argmin - 1, filter_idx]

        if (min_ub < max_mean and min_mean < max_lb):
            axes.vlines(filter_idx, 0.98, 1.02,
                        label=f'SSD component {filter_idx}')
            break

    axes.legend()
    axes.axhline(1, linestyle='--')

    return fig_spec_ratios


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


def analysis(subj_id, subj_bads):
    # read in both sessions belonging to a single subject
    raw_sessions = list()
    session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))

    fig_alpha_topo, axes = plt.subplots(1, 3, figsize=(9, 3))
    colors = ['grey', 'orange']

    for session_idx, fname in enumerate(session_fnames):
        raw = mne.io.read_raw_ctf(fname, verbose=False)
        raw.load_data()
        # mark bad channels; maintain symmetry of channels across sessions
        for bad_sess_idx in range(2):
            for bad_ch in bads[subj_id][bad_sess_idx]:
                if bad_ch not in raw.info['bads']:
                    raw.info['bads'].append(bad_ch)
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
        axes[0].plot(freqs, chan_avg_psd, color=colors[session_idx],
                     alpha=0.8)
        axes[0].set_ylabel('power')
        axes[0].set_xlabel('freq. (Hz)')
        # plot alpha topography
        alpha_mask = np.logical_and(freqs >= 9, freqs <= 14)
        avg_alpha_pow = raw_psds[:, alpha_mask].mean(axis=1)
        im, _ = plot_topomap(avg_alpha_pow, raw.info, vmax=70,
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

    fig_spec_ratios = ssd_spec_ratios(raw_sessions_filtered, ssd)
    fig_ssd, sessions_alpha = ssd_alpha(raw_sessions_filtered, ssd)

    return fig_alpha_topo, fig_ssd, sessions_alpha, fig_spec_ratios, subj_id


if __name__ == '__main__':
    n_jobs = 6
    n_subjs = 6

    data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
    subj_ids = list({fname[:8] for fname in listdir(data_dir)})
    print(subj_ids)

    subj_ids = subj_ids[:n_subjs]
    # subj_ids = ['RASMDGZN']

    out = Parallel(n_jobs=n_jobs)(delayed(analysis)(subj_id) for subj_id
                                  in subj_ids)
    # fig_alpha_topo_list, fig_ssd_list, sessions_alpha_list = zip(*out)

    report = Report()
    for fig_alpha_topo, fig_ssd, _, fig_spec_ratios, subj_id in out:
        report.add_figure(fig_alpha_topo, title=subj_id, tags=('topo',))
        report.add_figure(fig_ssd, title=subj_id, tags=('ssd_psd',))
        report.add_figure(fig_spec_ratios, title=subj_id,
                          tags=('spec_ratios',))
    report.save('alpha_analysis.html', overwrite=True, open_browser=True)
