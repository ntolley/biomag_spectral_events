from os import listdir
import os.path as op
from glob import glob

import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
sns.set()

import mne
from mne.decoding import SSD
from mne.time_frequency import psd_array_multitaper
from mne.viz import plot_topomap
from mne.report import Report


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
    session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))

    fig_alpha_ssd, axes = plt.subplots(1, 2)

    ssd_weights = list()
    for sess_idx, fname in enumerate(session_fnames):
        raw = mne.io.read_raw_ctf(fname, verbose=False)
        raw.load_data()
        add_bad_labels(raw, subj_id)  # modifies Raw object in-place
        raw.pick_types(meg=True, eeg=False, ref_meg=False)
        raw.filter(l_freq=3., h_freq=55)
        # raw._data = zscore(raw._data, axis=1)
        raw._data -= raw._data.mean()
        raw._data /= raw._data.std()
        raw.resample(sfreq=500)
        raw_sessions.append(raw)

        filtered_data, ssd = fit_ssd(raw.get_data(), raw.info)

        im, _ = plot_topomap(ssd.patterns_[0], ssd.info, axes=axes[sess_idx],
                             vmin=-.1, vmax=.1, show=False)
        plt.colorbar(im, ax=axes[sess_idx], fraction=0.05)
        axes[sess_idx].set_title(f'session {sess_idx + 1}')
        ssd_weights.append(ssd.patterns_[0])

    ssd_std_diff = ssd_weights[1].std() - ssd_weights[0].std()

    ssd_weights_diff = ssd_weights[1] - ssd_weights[0]
    ssd_weights_diff_flip = ssd_weights[1] + ssd_weights[0]
    # note that some subjects have sessions where the spatial SSD pattern
    # remains consistent, but with opposite sign
    # given that an opposite sign of an SSD weight reflects a pi-radians phase
    # difference, here I explored the possibility that the sign assigned to a
    # given SSD vector is somewhat arbitrary by taking the SSD weight
    # difference with minimal magnitude
    ssd_norm = min(np.linalg.norm(ssd_weights_diff),
                   np.linalg.norm(ssd_weights_diff_flip))

    fig_alpha_ssd.tight_layout()

    return fig_alpha_ssd, ssd_std_diff, ssd_norm, subj_id


if __name__ == '__main__':
    n_jobs = 5
    n_subjs = 36

    data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
    # data_dir = '/Volumes/THORPE/MDD_ketamine_2022'
    subj_ids = list({fname[:8] for fname in listdir(data_dir)})
    print(subj_ids)

    subj_ids = subj_ids[:n_subjs]
    # subj_ids = ['RASMDGZN']

    out = Parallel(n_jobs=n_jobs)(delayed(analysis)(subj_id)
                                  for subj_id in subj_ids)

    _, ssd_std_diffs, ssd_norms, _ = zip(*out)

    X = np.stack([ssd_std_diffs, ssd_norms], axis=1)
    X = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=0.7, min_samples=10).fit_predict(X)
    clustered_subj_ids = np.array(subj_ids)[labels == 0]
    unclustered_subj_ids = np.array(subj_ids)[labels == -1]
    print(f'clustered subjects: {clustered_subj_ids}')
    print(f'outlier subjects: {unclustered_subj_ids}')

    g = sns.JointGrid()
    sns.kdeplot(x=ssd_std_diffs, bw_adjust=0.85, fill=True, linewidth=0,
                alpha=.5, color='C0', ax=g.ax_marg_x, label='agg')
    sns.kdeplot(y=ssd_norms, bw_adjust=0.85, fill=True, linewidth=0, alpha=.5,
                color='C0', ax=g.ax_marg_y)
    for class_idx, class_ in enumerate(set(labels)):
        if class_ == -1:
            label = 'outliers'
            edgecolor = None
        else:
            label = 'cluster'
            edgecolor = 'k'
        subj_class_ids = np.array(subj_ids)[labels == class_]
        cluster_mask = labels == class_
        x = np.array(ssd_std_diffs)[cluster_mask]
        y = np.array(ssd_norms)[cluster_mask]
        sns.scatterplot(x=x, y=y, alpha=.5, edgecolor=edgecolor,
                        color=f'C{class_idx + 1}', ax=g.ax_joint)
        sns.kdeplot(x=x, bw_adjust=0.85, fill=True, linewidth=0, alpha=.5,
                    color=f'C{class_idx + 1}', ax=g.ax_marg_x, label=label)
        sns.kdeplot(y=y, bw_adjust=0.85, fill=True, linewidth=0, alpha=.5,
                    color=f'C{class_idx + 1}', ax=g.ax_marg_y)
        for x_coord, y_coord, subj_id in zip(x, y, subj_class_ids):
            g.ax_joint.text(x_coord + 0.002, y_coord, subj_id[:3], fontsize=5,
                            va='center_baseline')
    g.ax_joint.set_xlabel(r'$std(W_{sess_2})-std(W_{sess_1})$')
    g.ax_joint.set_ylabel(r'$\parallel W_{sess_2}-W_{sess_1}\parallel$')
    g.ax_marg_x.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    g.fig.tight_layout()

    report = Report()
    report.add_figure(g.fig, title='summary')
    report.add_code(str(clustered_subj_ids), 'clustered subjects (class 0)')
    report.add_code(str(unclustered_subj_ids), 'outlier subjects (class -1)')

    for (fig_alpha_ssd, _, _, subj_id) in out:
        report.add_figure(fig_alpha_ssd, title=subj_id, tags=('ssd_topo',))

    report.save('alpha_analysis_spatial.html',
                overwrite=True, open_browser=True)
