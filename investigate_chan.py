from os import listdir
import os.path as op
from glob import glob

import mne

mne.viz.set_browser_backend('qt')

data_dir = '/home/ryan/Documents/datasets/MDD_ketamine_2022'
subj_ids = list({fname[:8] for fname in listdir(data_dir)})
print(subj_ids)

subj_id, session_idx = 'ZDIAXRUW', 0


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

session_fnames = sorted(glob(op.join(data_dir, subj_id + '*')))
fname = session_fnames[session_idx]

raw = mne.io.read_raw_ctf(fname, verbose=False)
raw.load_data()
for session_idx in range(2):
    for bad_ch in bads[subj_id][session_idx]:
        if bad_ch not in raw.info['bads']:
            raw.info['bads'].append(bad_ch)

raw.plot(scalings='auto')
raw.plot_psd(fmax=50)
