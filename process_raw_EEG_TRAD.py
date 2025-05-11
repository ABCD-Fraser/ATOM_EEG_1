# %%
import numpy as np
import mne
import pandas as pd
import glob
from mne.stats import linear_regression
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from datetime import datetime
import time


start_time = time.time()

run_ICA = True

# ---- 1. Get All Participants ----
raw_files = sorted(glob.glob("RAW_data/RAW_TRAD/*.vhdr"))  # Load all BrainVision files in the directory

# ---- 2. Define event codes ----
event_id = {'StimA': 31, 'StimB': 32}

for raw_file in raw_files:
    print(f"Processing: {raw_file}")

    # ---- 3. Load raw EEG data ----
    raw = mne.io.read_raw_brainvision(raw_file, preload=True)
    events, event_dict = mne.events_from_annotations(raw)  # Extract events
    sfreq = raw.info['sfreq']  # Sampling frequency
    PID = raw_file.split("/")[-1].split("_")[0]  # Extract participant ID
    print(f"Processing: {PID}")

    alt_events = []
    phase_2 = False
    for i in range(len(events)):
        

        if events[i][2] == 81:
            phase_2 = True

        if phase_2:

            if events[i][2] == 32:
                if events[i+1][2] > 7:
                    print(f'no trial id in {i}')
                else:    
                    alt_events.append(abs(events[i+1][2] - 4))

                if events[i+3][2] not in [38,39]:
                    print(f'no response id in {i}')
                elif events[i+3][2] == 38:
                    
                    alt_events[-1] = 320




    
    # ---- 4. Apply Filtering ----
    raw.filter(l_freq=0.1, h_freq=40, fir_design='firwin')  # High-pass at 0.1 Hz, Low-pass at 40 Hz

    # ---- 5. Run ICA to Remove Artifacts ----
    if run_ICA:
    
        ica = ICA(n_components=20, random_state=97, method='fastica')
        ica.fit(raw)
        raw = ica.apply(raw)  # Apply ICA correction

    # ---- 6. Compute Time Shifts from Raw Data ----
    event_times = events[:, 0] / sfreq  # Convert sample indices to seconds
    stimA_times = event_times[np.isin(events[:, 2], [31, 81])]  # Include StimA (31, 81)
    stimB_times = event_times[events[:, 2] == 32]  # StimB (32) times

    # ---- 7. Adjust for 44ms Delay ----
    stimA_times += 0.044  # Adjust all StimA times forward by 44ms

    # Compute time shifts for StimB relative to the most recent StimA
    trial_time_shifts = np.full(len(stimB_times), np.nan)  # Initialize NaNs
    for i, stimB_time in enumerate(stimB_times):
        preceding_stimA = stimA_times[stimA_times < stimB_time]  # Find all previous StimAs
        if len(preceding_stimA) > 0:
            trial_time_shifts[i] = stimB_time - preceding_stimA[-1]  # Compute time difference

    # Convert time shifts to samples
    trial_time_shifts_samples = np.round(trial_time_shifts * sfreq).astype(int)

    # Loop through events and create alt list for conditions


    # ---- 8. Create epochs AFTER computing time shifts ----
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.8, preload=True)

    # ---- 9. Create metadata for regression ----
    metadata = pd.DataFrame({
        'StimA': (epochs.events[:, 2] == event_id['StimA']).astype(int),
        'StimB': (epochs.events[:, 2] == event_id['StimB']).astype(int),
    })
    epochs.metadata = metadata

    # ---- 10. Define design matrix ----
    design_matrix = epochs.metadata[['StimA', 'StimB']]

    # ---- 11. Run regression ----
    lm_results = linear_regression(epochs, design_matrix, names=['StimA', 'StimB'])

    # ---- 12. Extract betas ----
    betas_stimA_trials = lm_results['StimA'].beta.data  # (n_channels, n_times)
    betas_stimB_trials = lm_results['StimB'].beta.data  # (n_channels, n_times)

    # ---- 13. Compute Grand Average for Non-Deconvolved Data ----
    evoked_original = epochs['StimB'].average()  # Get original evoked response

    # ---- 14. Initialize corrected trial-level EEG data ----
    corrected_trials = np.tile(betas_stimB_trials[:, :, np.newaxis], (1, 1, len(stimB_times)))  # (n_channels, n_times, n_trials)

    # ---- 15. Process each trial separately ----
    for trial_idx, shift_samples in enumerate(trial_time_shifts_samples):
        if not np.isnan(shift_samples) and 0 < shift_samples < betas_stimA_trials.shape[1]:  # Ensure valid shift range
            betas_stimA_shifted = np.zeros_like(betas_stimA_trials)  # Initialize zero array
            betas_stimA_shifted[:, :-shift_samples] = betas_stimA_trials[:, shift_samples:]  # Shift backward

            # Remove StimA's contribution for this trial
            corrected_trials[:, :, trial_idx] -= betas_stimA_shifted

    # ---- 16. Compute Grand Average for Deconvolved Data ----

    

    corrected_epochs = mne.EpochsArray(
    data=corrected_trials.transpose(2, 0, 1),  # Change to (n_trials, n_channels, n_times)
    info=epochs['StimB'].info,
    events=epochs['StimB'].events,  # Use the original event structure
    tmin=epochs['StimB'].tmin
)


    corrected_epochs.events[:, 2] = alt_events
    corrected_epochs = corrected_epochs[corrected_epochs.events[:,2] != 320]
    corrected_epochs.event_id = {'Far': 3, 'Mid': 2, 'Close': 1}
    corrected_epochs.apply_baseline((-0.1, 0))
    

    epochs_test = epochs['StimB'].copy()
    epochs_test.events[:,2] = alt_events
    epochs_test = epochs_test[epochs_test.events[:,2] != 320]

    corrected_epochs.save(f'deconvolved_data/TRAD/{PID}-epo.fif', overwrite=True)
    epochs_test.save(f'original_data/TRAD/{PID}-epo.fif', overwrite=True)


# End timing
end_time = time.time()
total_time = end_time - start_time

print(f"Total processing took {total_time:.2f} seconds.")
# %%
