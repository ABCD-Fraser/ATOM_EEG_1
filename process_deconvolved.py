 
# %%

####  PROCESSING ####

import numpy as np
import mne
import pandas as pd
import glob
from mne.stats import linear_regression
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import time


def get_mean_amplitude(epochs, tmin, tmax):
    """Compute the mean amplitude of the signal within a time window.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object.
    tmin : float
        The start of the time window.
    tmax : float
        The end of the time window.

    Returns
    -------
    mean_amplitude : float
        The mean amplitude within the time window.
    """
    data = epochs.copy().average().crop(tmin=tmin, tmax=tmax).get_data()  # Extract data
    mean_amplitude = np.mean(np.abs(data))  # Compute mean amplitude
    return mean_amplitude

def get_mean_amplitude_single(epochs, tmin, tmax):
    """Compute the mean amplitude of the signal within a time window.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object.
    tmin : float
        The start of the time window.
    tmax : float
        The end of the time window.

    Returns
    -------
    mean_amplitude : float
        The mean amplitude within the time window.
    """

    data = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()  # Extract data
    mean_amplitude = np.mean(np.abs(data))  # Compute mean amplitude
    condition = epochs.events[0][2]
    return mean_amplitude, condition

def get_peak_measures(epochs, PID, tmin, tmax, channels, conditions, cond_list, trial=False):

    trial_avgdf = pd.DataFrame()
    avgdf = pd.DataFrame()
    peak_df = pd.DataFrame()

    for ch in channels:

        if trial:
            amplitudes = []
            conditions = []

            for i in range(len(epochs)):
                amp, cond = get_mean_amplitude_single(epochs[i].copy().pick(picks=ch), tmin=tmin, tmax=tmax)
                amplitudes.append(amp)
                conditions.append(cond)

            trial_avgdf[f'Amplitude_{ch}'] = amplitudes

            trial_avgdf = pd.DataFrame()

        for cond in conditions:

            # Filter epochs by condition
            epochs_cond = epochs[epochs.events[:, 2] == cond]
            amp = get_mean_amplitude(epochs_cond.copy().pick(picks=ch), tmin=tmin, tmax=tmax)
            avgdf[f'{cond_list[cond-1]}_amp_{ch}'] = [amp]

            _, lat, amp = epochs_cond.copy().pick(picks=ch).average().get_peak(tmin=tmin, tmax=tmax, return_amplitude=True)
            peak_df[f'{cond_list[cond-1]}_amp_{ch}'] = [amp]
            peak_df[f'{cond_list[cond-1]}_lat_{ch}'] = [lat]

    trial_avgdf.insert(0, 'Conditions', conditions)
    trial_avgdf.insert(0, 'PID', PID)

    avgdf.insert(0, 'PID', PID)

    peak_df.insert(0, 'PID', PID)

    if trial:
        return trial_avgdf, avgdf, peak_df
    else:
        return avgdf, peak_df

def plot_erp_with_windows(evoked, channels, component_windows, colors=None):
    """
    Plot ERP data for selected channels with highlighted component windows.

    Parameters:
    evoked : mne.Evoked
        The Evoked (ERP) object to plot.
    channels : list
        Channels to plot (e.g., ['P3']).
    component_windows : dict
        Dictionary of components and their time windows.
        e.g., {'N1': (0.1, 0.2), 'P2': (0.2, 0.3)}
    colors : dict or None
        Dictionary specifying colours for each component window.
        e.g., {'N1': 'blue', 'P2': 'green'}
        If None, default colours will be used.
    """

    if colors is None:
        # Default colours if none provided
        colors = ['orange', 'blue', 'green', 'purple', 'red']
        colors = {comp: colors[i % len(colors)] for i, comp in enumerate(component_windows)}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ERP data
    evoked_ch = evoked.copy().pick_channels(channels)
    evoked_ch.plot(picks=channels, spatial_colors=True, show=False, axes=ax)

    # Highlight component windows
    for component, (start, end) in component_windows.items():
        plt.axvspan(start, end, color=colors[component], alpha=0.2, label=f'{component} window')

    # Formatting
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'ERP at {", ".join(channels)} with highlighted components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

start_time = time.time()

peak_extract = True
run_trial = False
overwrite = True


plot_erp_windows = True
plot_condition_ERP = True
get_peaks = True

channels = ['P3', 'P4']
conditions = [1,2,3]
cond_list = ['close', 'mid', 'far']


n2_trial_overall = pd.DataFrame()
p2_trial_overall = pd.DataFrame()
n2b_trial_overall = pd.DataFrame()
p3_trial_overall = pd.DataFrame()

n2_avgoverall = pd.DataFrame()
p2_avgoverall = pd.DataFrame()
n2b_avgoverall = pd.DataFrame()
p3_avgoverall = pd.DataFrame()

n2_peak_overall = pd.DataFrame()
p2_peak_overall = pd.DataFrame()
n2b_peak_overall = pd.DataFrame()
p3_peak_overall = pd.DataFrame()

evoked_list = []
evoked_list_close = []
evoked_list_mid = []
evoked_list_far = []

# Define the directory containing the desired data
f_dir = 'deconvolved'

files = sorted(glob.glob(f"{f_dir}_data/TRAD/*.fif"))  # Load all BrainVision files in the directory


# Define your component time windows (in seconds)
component_windows = {
    'N1': (0.18, 0.22),
    'P2': (0.24, 0.27),
    'N2b': (0.28, 0.32),
    'P3': (0.36, 0.41),
}

for f in files:


    corrected_epochs = mne.read_epochs(f, preload=True)
    PID = f.split("\\")[-1].split("_")[1].split(".")[0] # Extract participant ID
    
    if PID == 'A004':
        continue 

    grand_avg = corrected_epochs.average()
    close_avg = corrected_epochs[corrected_epochs.events[:,2] == 1].average()
    mid_avg = corrected_epochs[corrected_epochs.events[:,2] == 2].average()
    far_avg = corrected_epochs[corrected_epochs.events[:,2] == 3].average()
    
    evoked_list.append(grand_avg)
    evoked_list_close.append(close_avg)
    evoked_list_mid.append(mid_avg)
    evoked_list_far.append(far_avg)

    #### extract amplitudes
    if peak_extract:

        if run_trial:
            n2_trial_avgdf, n2_avgdf, n2_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['N1'][0], tmax=component_windows['N1'][1], channels=channels, conditions=conditions, cond_list=cond_list, trial = True)
            p2_trial_avgdf, p2_avgdf, p2_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['P2'][0], tmax=component_windows['P2'][0], channels=channels, conditions=conditions, cond_list=cond_list, trial = True)
            n2b_trial_avgdf, n2b_avgdf, n2b_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['N2b'][0], tmax=component_windows['N2b'][0], channels=channels, conditions=conditions, cond_list=cond_list, trial = True)
            p3_trial_avgdf, p3_avgdf, p3_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['P3'][0], tmax=component_windows['P3'][0], channels=channels, conditions=conditions, cond_list=cond_list, trial = True)  

            n2_trial_overall = pd.concat([n2_trial_overall, n2_trial_avgdf])
            p2_trial_overall = pd.concat([p2_trial_overall, p2_trial_avgdf])
            n2b_trial_overall = pd.concat([n2b_trial_overall, n2b_trial_avgdf])
            p3_trial_overall = pd.concat([p3_trial_overall, p3_trial_avgdf])
        else:
            n2_avgdf, n2_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['N1'][0], tmax=component_windows['N1'][1], channels=channels, conditions=conditions, cond_list=cond_list)
            p2_avgdf, p2_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['P2'][0], tmax=component_windows['P2'][0], channels=channels, conditions=conditions, cond_list=cond_list)
            n2b_avgdf, n2b_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['N2b'][0], tmax=component_windows['N2b'][0], channels=channels, conditions=conditions, cond_list=cond_list)
            p3_avgdf, p3_peak_df = get_peak_measures(corrected_epochs, PID, tmin=component_windows['P3'][0], tmax=component_windows['P3'][0], channels=channels, conditions=conditions, cond_list=cond_list)

        n2_avgoverall = pd.concat([n2_avgoverall, n2_avgdf])
        p2_avgoverall = pd.concat([p2_avgoverall, p2_avgdf])
        n2b_avgoverall = pd.concat([n2b_avgoverall, n2b_avgdf])
        p3_avgoverall = pd.concat([p3_avgoverall, p3_avgdf])

        n2_peak_overall = pd.concat([n2_peak_overall, n2_peak_df])
        p2_peak_overall = pd.concat([p2_peak_overall, p2_peak_df])
        n2b_peak_overall = pd.concat([n2b_peak_overall, n2b_peak_df])
        p3_peak_overall = pd.concat([p3_peak_overall, p3_peak_df])

        print("Processing complete for all participants.")

# ---- 17. plot grand averages and component windows ----

# ---- 17. Compute Grand Averages ----

grand_avg = mne.grand_average(evoked_list)
close_grand_avg = mne.grand_average(evoked_list_close)
mid_grand_avg = mne.grand_average(evoked_list_mid)
far_grand_avg = mne.grand_average(evoked_list_far)



if plot_erp_windows:
    # Plotting parameters
    for ch in channels:
        plot_erp_with_windows(grand_avg, channels=[ch], component_windows=component_windows)


if plot_condition_ERP:
    # mne.viz.plot_compare_evokeds([close_grand_avg, mid_grand_avg, far_grand_avg], picks='P3')
    # mne.viz.plot_compare_evokeds([close_grand_avg, mid_grand_avg, far_grand_avg], picks='P4')

    mne.viz.plot_compare_evokeds([close_grand_avg, mid_grand_avg], picks='P3')
    mne.viz.plot_compare_evokeds([close_grand_avg, mid_grand_avg], picks='P4')
    grand_avg.drop_channels('BIP2AUX').set_montage('easycap-M1').plot(picks=['P3', 'P4'], spatial_colors=True)

if peak_extract:
    n2_trial_overall['component'] = 'N1'
    p2_trial_overall['component'] = 'P2'
    n2b_trial_overall['component'] = 'N2b'
    p3_trial_overall['component'] = 'P3'
    trial_avgoverall = pd.concat([n2_trial_overall, p2_trial_overall, p3_trial_overall])

    n2_avgoverall['component'] = 'N1'
    p2_avgoverall['component'] = 'P2'
    n2b_avgoverall['component'] = 'N2b'   
    p3_avgoverall['component'] = 'P3'
    avgoverall = pd.concat([n2_avgoverall, n2b_avgoverall, p2_avgoverall, p3_avgoverall])

    n2_peak_overall['component'] = 'N1'
    p2_peak_overall['component'] = 'P2'
    n2b_peak_overall['component'] = 'N2b'
    p3_peak_overall['component'] = 'P3'
    peak_overall = pd.concat([n2_peak_overall, n2b_peak_overall, p2_peak_overall, p3_peak_overall])

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save Data with date in the filename
    if overwrite:
        if run_trial:
            trial_avgoverall.to_csv(f'stats/{f_dir}_trial_avg_overall_TRAD.csv', index=False)
        avgoverall.to_csv(f'stats/{f_dir}_avg_overall_TRAD.csv', index=False)
        peak_overall.to_csv(f'stats/{f_dir}_peak_overall_TRAD.csv', index=False)
    else:
        if run_trial:
            trial_avgoverall.to_csv(f'stats/{f_dir}_trial_avg_overall_{current_date}.csv', index=False)
        avgoverall.to_csv(f'stats/{f_dir}_avg_overall_{current_date}.csv', index=False)
        peak_overall.to_csv(f'stats/{f_dir}_peak_overall_{current_date}.csv', index=False)


end_time = time.time()
total_time = end_time - start_time  # Calculate total time taken   
print(f"Total processing took {total_time:.2f} seconds.")  # Print total time taken

   
# %%
