# biomag_spectral_events
## Analysis Pipeline for the Ketamine in Depression Data Analysis Competition
## BIOMAG 2022

In order to run the primary analysis pipeline for our final submission, run
`alpha_analysis_spatial.py` after setting `n_jobs` to the number of cores you
wish to parallelize the computation across on your computer (e.g., 
`n_jobs=8` for an 8 core computer) and `data_dir` to the directory containing 
your MEG session data. The script assumes you have downloaded the entire
Ketamine in Depression dataset containing 36 subjects with 2 sessions/subject.

### Dependencies
* numpy
* scipy
* scikit-learn
* matplotlib
* joblib
* mne
* seaborn


