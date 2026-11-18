import numpy as np

def calculate_alpha_power(raw, channels):
    spectrum = raw.compute_psd(
        method="welch",
        fmin=8,
        fmax=13,
        picks=channels,
        verbose=False
    )

    psds, freqs = spectrum.get_data(return_freqs=True)

    # Integrate power spectral density across alpha frequencies
    alpha_power = np.trapezoid(psds, freqs, axis=-1)

    # Average across posterior channels
    return alpha_power.mean()


import gc
import pandas as pd
import mne

corrected_results = []
corrected_errors = []

for i, subject in enumerate(subjects, start=1):
    print(f"Processing {subject} ({i}/43)")

    for session in ["ses-OA", "ses-OC"]:
        for run in [1, 2]:
            filename = (
                dataset / subject / session / "eeg"
                / f"{subject}_{session}_task-resting_run-{run}_eeg.vhdr"
            )

            try:
                raw = mne.io.read_raw_brainvision(
                    filename, preload=True, verbose=False
                )

                # Remove non-EEG Status channel FIRST
                if "Status" in raw.ch_names:
                    raw.drop_channels(["Status"])

                raw.pick(picks="eeg")
                raw.filter(1, 40, verbose=False)
                raw.set_eeg_reference("average", verbose=False)

                channels = [
                    ch for ch in posterior_channels
                    if ch in raw.ch_names
                ]

                if len(channels) != len(posterior_channels):
                    raise ValueError(
                        "Missing posterior electrodes"
                    )

                power = calculate_alpha_power(raw, channels)

                corrected_results.append({
                    "participant": subject,
                    "condition": session,
                    "run": run,
                    "alpha_power": power
                })

                del raw

            except Exception as e:
                corrected_errors.append({
                    "participant": subject,
                    "condition": session,
                    "run": run,
                    "error": str(e)
                })
                print(f"ERROR: {subject}, {session}, run {run}: {e}")

    gc.collect()

corrected_df = pd.DataFrame(corrected_results)

corrected_df.to_csv(
    dataset / "corrected_alpha_results.csv",
    index=False
)

print("\nFinished!")
print("Successful recordings:", len(corrected_df))
print("Errors:", len(corrected_errors))

display(corrected_df.head())
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path, help="Folder containing the sub-* folders")
dataset = parser.parse_args().dataset

posterior_channels = (
    [f"A{i}" for i in range(4, 33)]
    + [f"B{i}" for i in range(3, 14)]
    + [f"D{i}" for i in range(29, 33)]
)

subjects = []
for folder in sorted(dataset.glob("sub-*")):
    if folder.is_dir() and all(
        (folder / session / "eeg"
         / f"{folder.name}_{session}_task-resting_run-{run}_eeg.vhdr").is_file()
        for session in ("ses-OA", "ses-OC")
        for run in (1, 2)
    ):
        subjects.append(folder.name)

if len(subjects) != 43:
    raise SystemExit(f"Expected 43 participants with four runs; found {len(subjects)}.")
    