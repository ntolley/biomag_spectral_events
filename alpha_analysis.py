from os import listdir
import os.path as op
from glob import glob

import numpy as np
from scipy.stats import zscore, norm, chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
sns.set()

import mne
from mne.decoding import SSD
from mne.time_frequency import psd_array_multitaper
from mne.viz import plot_topomap
from mne.report import Report

# mne.viz.set_browser_backend('qt')


def add_bad_labels(raw, subj_id):
    """Label the the bad channels in a raw object"""

    bads = dict(
        DXFTLCJA=[[], ['MLF67-1609', 'MLF66-1609', 'MLT14-1609', 'MRO24-1609',
                       'MRP23-1609', 'MRP22-1609', 'MRP34-1609', 'MRT35-1609',
                       'MRT15-1609', 'MRP57-1609', 'MRP56-1609', 'MRO14-1609',
                       'MLT44-1609', 'MRO34-1609']],
        BQBBKEBX=[[], ['MLF67-1609', 'MLF66-1609', 'MLT14-1609', 'MRO24-1609',
                       'MRP23-1609', 'MRP22-1609', 'MRP34-1609', 'MRT35-1609',
                       'MRT15-1609', 'MRP57-1609', 'MRP56-1609', 'MRO14-1609',
                       'MLT44-1609']],
        JBGAZIEO=[[], ['MLF25-1609']],
        QGFMDSZY=[['MLP41-1609'], []],
        ZDIAXRUW=[['MRF61-1609', 'MRC62-1609', 'MRC15-1609', 'MRO41-1609',
                   'MRO53-1609', 'MRT41-1609', 'MRT42-1609', 'MRT43-1609'],
                  ['MRC55-1609']]
    )

    # mark bad channels; maintain symmetry of channels across sessions
    if subj_id in bads:
        for bad_sess in bads[subj_id]:
            for bad_ch in bad_sess:
                if bad_ch not in raw.info['bads']:
                    raw.info['bads'].append(bad_ch)


def conf_int_mean(data, conf=0.95):
    n = data.shape[0]
    mean = data.mean(axis=0)
    std_mean = data.std(axis=0, ddof=1) / n**(1/2)
    lb = norm.ppf((1 - conf) / 2, loc=mean, scale=std_mean)
    ub = norm.ppf((1 + conf) / 2, loc=mean, scale=std_mean)
    return mean, lb, ub


def ssd_alpha(raw_sessions_filtered, ssd, num_filters=3):

    fig_ssd, axes = plt.subplots(num_filters, 2, figsize=[10, 10],
                                 constrained_layout=True, sharey='col')
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
        plt.colorbar(im, ax=axes[filt_idx, 0], fraction=0.05)
        axes[filt_idx, 0].set_title(f'SSD transform dim. #{filt_idx}')

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
                fmin=3., fmax=55.
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
        # sessions_alpha.append(session_alpha)  # speed up
    return fig_ssd, sessions_alpha


def ssd_spec_ratios(raw_sessions_filtered, ssd):

    n_ssd_dims = ssd.patterns_.shape[0]
    fig_spec_ratios, axes = plt.subplots(1, 2)
    colors = ['grey', 'orange']

    # store first 100 spec_ratios for each session
    spec_ratios_100 = np.zeros((2, 100))

    data_agg = np.stack([raw_filtered.get_data() for raw_filtered
                         in raw_sessions_filtered], axis=0)
    spec_ratio_agg, _ = ssd.get_spectral_ratio(data_agg)

    for sess_idx, raw_filtered in enumerate(raw_sessions_filtered):
        data = raw_filtered.get_data()
        spec_ratios_sess, _ = ssd.get_spectral_ratio(data)
        spec_ratios_diff = spec_ratios_sess - spec_ratio_agg

        spec_ratios_100[sess_idx, :] = spec_ratios_sess[:100]

        df = len(spec_ratios_diff) - 1
        loc = spec_ratios_diff.mean()
        scale = spec_ratios_diff.std()
        chi_2_stat = np.sum((spec_ratios_diff ** 2) / spec_ratio_agg)
        if chi2.cdf(chi_2_stat, df=df, loc=loc, scale=scale) > 0.95:
            label = f'session {sess_idx + 1}*'
        else:
            label = f'session {sess_idx + 1}'
        axes[0].plot(spec_ratios_sess, color=colors[sess_idx],
                     linewidth=2, alpha=0.8, label=label)
        axes[1].hist(spec_ratios_diff, density=True,
                     histtype='stepfilled', color=colors[sess_idx], alpha=0.8)

        # plot theoretical normal distribution of sess-agg spectral ratios
        norm_dist = norm(loc=loc, scale=scale)
        norm_bounds = norm_dist.ppf([0.001, 0.999])
        norm_x = np.linspace(norm_bounds[0], norm_bounds[1], num=100)
        axes[1].plot(norm_x, norm_dist.pdf(norm_x), color=colors[sess_idx],
                     lw=2, alpha=0.95)

    axes[0].plot(spec_ratio_agg, color='black', linewidth=2, linestyle='--',
                 alpha=0.7, label='aggregate')
    axes[0].set_xlabel("Eigenvalue Index")
    axes[0].set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")

    axes[0].legend()
    axes[0].axhline(1, color='k', linestyle=':')

    spec_ratios_diff = spec_ratios_100[1, :] - spec_ratios_100[0, :]

    return fig_spec_ratios, spec_ratios_diff


def fit_ssd(data, info):
    """Alpha-band Spatio-Spectral Decomposition (SSD) from raw data"""
    freqs_sig = 9, 14  # alpha
    freqs_noise = 8, 15  # alpha
    # freqs_sig = 15, 29  # beta
    # freqs_noise = 14, 30  # beta
    # freqs_sig = 35, 45  # gamma
    # freqs_noise = 30, 50  # gamma

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
    alpha_sess = list()
    session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))

    fig_alpha_topo, axes = plt.subplots(1, 4, constrained_layout=True)
    colors = ['grey', 'orange']

    for session_idx, fname in enumerate(session_fnames):
        raw = mne.io.read_raw_ctf(fname, verbose=False)
        raw.load_data()
        add_bad_labels(raw, subj_id)  # modifies Raw object in-place
        raw.pick_types(meg=True, eeg=False, ref_meg=False)
        raw.filter(l_freq=3., h_freq=55)
        # raw._data = zscore(raw._data, axis=1)
        raw._data -= raw._data.mean()
        raw._data /= raw._data.std()
        raw.resample(sfreq=500)

        # compute raw PSD
        raw_psds, freqs = psd_array_multitaper(raw.get_data(),
                                               raw.info['sfreq'],
                                               fmin=3., fmax=55.)
        # plot channel-mean psd
        chan_avg_psd = raw_psds.mean(axis=0)
        axes[0].plot(freqs, np.log(chan_avg_psd), color=colors[session_idx], alpha=0.8)
        axes[0].set_ylabel('log power')
        axes[0].set_xlabel('freq. (Hz)')
        # plot alpha topography
        alpha_mask = np.logical_and(freqs >= 9, freqs <= 14)
        avg_alpha_pow = raw_psds[:, alpha_mask].mean(axis=1)
        im, _ = plot_topomap(avg_alpha_pow, raw.info, vmax=70,
                             axes=axes[session_idx + 1],
                             show=False)

        alpha_sess.append(avg_alpha_pow)
        raw_sessions.append(raw)
    plt.colorbar(im, ax=axes[1:3], fraction=0.05, shrink=0.5)

    # plot alpha topography diff between sessions
    alpha_sess_diff = alpha_sess[1] - alpha_sess[0]
    im, _ = plot_topomap(alpha_sess_diff, raw.info, axes=axes[3], show=False)
    plt.colorbar(im, ax=axes[3], shrink=0.5)

    # concatenate data and compute a single SSD filter that is applied to both
    # sessions
    raw_data = np.stack([raw.get_data() for raw in raw_sessions], axis=0)
    filtered_data, ssd = fit_ssd(raw_data, raw_sessions[0].info)

    # store filtered data in new Raw instances
    raw_sessions_filtered = [raw.copy() for raw in raw_sessions]
    for session_idx, raw_filtered in enumerate(raw_sessions_filtered):
        raw_filtered._data = filtered_data[session_idx].copy()

    fig_ssd, sessions_alpha = ssd_alpha(raw_sessions_filtered, ssd)
    fig_spec_ratios, spec_ratios_diff = ssd_spec_ratios(raw_sessions_filtered, ssd)

    return (fig_alpha_topo, alpha_sess_diff.mean(), fig_ssd, sessions_alpha,
            fig_spec_ratios, spec_ratios_diff, subj_id)


if __name__ == '__main__':
    n_jobs = 5
    n_subjs = 5

    data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
    # data_dir = '/Volumes/THORPE/MDD_ketamine_2022'
    subj_ids = list({fname[:8] for fname in listdir(data_dir)})
    print(subj_ids)

    subj_ids = subj_ids[:n_subjs]
    # subj_ids = ['RASMDGZN']

    out = Parallel(n_jobs=n_jobs)(delayed(analysis)(subj_id)
                                  for subj_id in subj_ids)

    report = Report()
    for (fig_alpha_topo, _, fig_ssd, _, fig_spec_ratios, _, subj_id) in out:
        report.add_figure(fig_alpha_topo, title=subj_id, tags=('topo',))
        report.add_figure(fig_ssd, title=subj_id, tags=('ssd_psd',))
        report.add_figure(fig_spec_ratios, title=subj_id,
                          tags=('spec_ratios',))

    _, alpha_sess_diffs, _, _, _, spec_ratios_diffs, _ = zip(*out)

    fig_summary, axes = plt.subplots(1, 2, figsize=(7, 3),
                                     constrained_layout=True)
    sns.kdeplot(data=alpha_sess_diffs, bw_adjust=0.5, ax=axes[0])
    sns.rugplot(data=alpha_sess_diffs, ax=axes[0])
    axes[0].axvline(0, color='k', linestyle=':')
    axes[0].set_xlabel('mean alpha power (session 2 - session 1)')

    x = np.stack(spec_ratios_diffs, axis=0)
    pca = PCA(n_components=x.shape[0])
    x_pca = pca.fit_transform(x)
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=alpha_sess_diffs,
                    hue_norm=(-1, 1), palette='Spectral', ax=axes[1])
    # axes[1].scatter(x_pca[:, 0], x_pca[:, 1], marker='.')
    axes[1].set_xlabel('PC_1 eigenvector')
    axes[1].set_ylabel('PC_2 eigenvector')
    # ax.set_zlabel('PC_3 eigenvector')
    report.add_figure(fig_summary, title='summary')

    report.save('alpha_analysis.html', overwrite=True, open_browser=True)
