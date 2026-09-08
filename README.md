# 4datasets

# Method
1. Execute pretrainedmodel
2. Execute research of best number of electrodes
3. Execute research of ratio with static and dynamic
4. Execute research of comparison with candidates

# Datasets
## SEED-VLA
**preprocessing**
```text
================================================================================
Binary Label Generation from Raw PERCLOS
================================================================================

[Label Definition]
  - Class 0 (Alert)    : PERCLOS <= 0.45
  - Class 1 (Fatigued) : PERCLOS >= 0.50
  - Excluded           : 0.45 < PERCLOS < 0.50 (ambiguous transition interval; 510 samples)

[Subject Exclusion Criteria]
  - Exclude subjects whose labels are exclusively Class 0 throughout all samples.
  - Retain subjects whose labels are exclusively Class 1.
    * Note: As the fatigued class represents the overall minority, they are 
      retained as a trade-off to preserve data volume.

[Input Data]
  - processdData/subject_wise/label_{s}.npy : Raw PERCLOS values
  - processdData/subject_wise/eeg_{s}.npy   : EEG feature array of shape (N_s, 16, 5, 18)
================================================================================
```

**project_files**
```text
project_root/
├── GNN/                          # Graph Neural Network (GNN) models and related scripts
├── LOSO_pretrainedmodel/         # Leave-One-Subject-Out (LOSO) pretrained model weights
├── model/                        # Model architecture definitions (e.g., EEGNet)
├── research_number_of_electrodes/# Evaluation scripts for total electrode count
├── research_ratio_staticdynamic/ # Experiments on static vs. dynamic channel selection ratios
├── research_try_candidate/       # Validation scripts for candidate channel selection
└── select_net/               # Channel selection algorithms and utilities
```

**execute order**
```text
LOSO_pretrainedmodel → research_number_of_electrodes → research_ratio_staticdynamic → research_try_candidate
(Exclude Subject 10 due to completely skewed (single-class) labels)
```

## SEED-VIG
**preprocessing**
```text:
================================================================================
EEG Feature Extraction (DE) and Binary Label Generation for SEED-VIG
================================================================================

[Step 1: Differential Entropy (DE) Feature Extraction]
  - Sampling Rate        : fs = 200 Hz
  - Window Size          : 100 samples (0.5 s per DE segment)
  - Frequency Bands (5)  : Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-14 Hz),
                           Beta (14-31 Hz), Gamma (31-51 Hz) (Butterworth bandpass, order=3)
  - Sequence Length      : 16 segments (8.0 s per sample)
  - EEG Array Shape      : (Samples, TimeSteps=16, Bands=5, Channels=17)
  - Source Inputs        : Raw EEG and PERCLOS .mat files
  - Output Directory     : processedData/subject_wise/
                           - eeg_{session_id}.npy
                           - label_{session_id}.npy

[Step 2: Binary Label Generation & Dataset Formatting]
  - Binarization Threshold : THRESHOLD = 0.35
      * Class 0 (Awake)   : PERCLOS <  0.35
      * Class 1 (Fatigue) : PERCLOS >= 0.35
  - Sample Alignment       : Truncate label array to match the sample count of 
                             EEG features (aligned to 16-segment boundaries).
  - Storage Optimization   : Create symbolic links for eeg_{session_id}.npy 
                             pointing to the source directory to save disk space.
  - Output Directory       : processedData/subject_wise_2class/
                             - label_{session_id}.npy (shape: (Samples,), dtype: int64)
                             - eeg_{session_id}.npy   (symlink to subject_wise/)
================================================================================
```

**project_files**
```text
project_root/
├── GNN/                          # Graph Neural Network (GNN) models and related scripts
├── LOSO_pretrainedmodel/         # Leave-One-Subject-Out (LOSO) pretrained model weights
├── model/                        # Model architecture definitions (e.g., EEGNet)
├── research_number_of_electrodes/# Evaluation scripts for total electrode count
├── research_ratio_staticdynamic/ # Experiments on static vs. dynamic channel selection ratios
├── research_try_candidate/       # Validation scripts for candidate channel selection
└── select_channel/               # Channel selection algorithms and utilities
```

**execute order**
```text
LOSO_pretrainedmodel → research_number_of_electrodes → research_ratio_staticdynamic → research_try_candidate
```

## TUEV
**preprocessing**
```text
================================================================================
EEG Preprocessing & Differential Entropy (DE) Extraction for TUH EEG Event (TUEV)
================================================================================

[Dataset Overview: TUH EEG Event Corpus (TUEV v2.0.0)]
  - Task                 : Multi-class EEG event detection / classification (6 classes)
  - Target Classes       : 1: spsw (spike and slow wave)
                           2: gped (generalized periodic epileptiform discharge)
                           3: pled (periodic lateralized epileptiform discharge)
                           4: eyem (eye movement)
                           5: artf (artifact)
                           6: bckg (background)
  - Channel Montage      : Standard ACNS TCP montage (16 bipolar differential channels)
                           Channels 0–7 & 14–21 (Temporal and Parasagittal chains)

[Step 1: Signal Preprocessing]
  - Filtering            : Bandpass filter 0.1–75.0 Hz; Notch filter at 50.0 Hz
  - Resampling           : Resampled to 256 Hz
  - Event Windowing      : Event start/stop extracted from .rec files with +/- 2.0 s padding

[Step 2: Differential Entropy (DE) Feature Extraction]
  - Frequency Bands (5)  : Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-14 Hz),
                           Beta (14-31 Hz), Gamma (31-50 Hz) (Butterworth bandpass, order=3)
  - Time Segmentation    : 10 temporal windows per event window
  - Feature Array Shape  : (Events, Channels=16, TimeWindows=10, Bands=5)

[Step 3: Subject-wise Splitting & Dataset Export]
  - Split Strategy       : Subject-independent split on training data
                           - Train Set : 80% of unique subjects from train folder
                           - Val Set   : 20% of unique subjects from train folder
                           - Eval Set  : Benchmark evaluation set kept completely disjoint
  - Output Files (.npy)  : Saved to designated root_save directory
                           - {train, val, eval}_data.npy        : Extracted DE features
                           - {train, val, eval}_labels.npy      : Event class labels (1–6)
                           - {train, val, eval}_subject_ids.npy : Subject identifiers
                           - count.txt                          : Summary of subjects & sample counts
================================================================================
```

## TUAB
**preprocessing**
```text
================================================================================
Differential Entropy (DE) Extraction for TUH Abnormal EEG Corpus (TUAB)
================================================================================

[Dataset Overview: TUH Abnormal EEG Corpus (TUAB v3.0.1)]
  - Task                 : Binary EEG classification (Normal vs. Abnormal)
  - Data Configuration   : Standard TCP AR montage (16 channels)
  - Input Format         : Preprocessed .pkl files partitioned into train / val / test
                           * X: Segmented EEG signal [16 channels, 2,560 points (10 s @ 256 Hz)]
                           * y: Clinical label (Normal / Abnormal)
                           * Skip samples that do not match the expected 2,560 points

[Step 1: Differential Entropy (DE) Feature Extraction]
  - Sampling Rate        : fs = 256 Hz
  - Window Size          : 0.5 s window (128 samples) -> 20 temporal windows per 10 s epoch
  - Frequency Bands (5)  : Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-14 Hz),
                           Beta (14-31 Hz), Gamma (31-50 Hz) (Butterworth bandpass, order=3)
  - Filtering Method     : Zero-phase forward-backward digital filtering (filtfilt) along axis=1
  - DE Calculation       : 0.5 * ln(2 * pi * e * variance + 1e-10)
  - Feature Array Shape  : (N_samples, Channels=16, TimeWindows=20, Bands=5)

[Step 2: Parallel Processing & Dataset Export]
  - Execution            : Multi-process extraction via multiprocessing.Pool across splits
  - Splits Processed     : train, val, test
  - Metadata Handling    : Subject IDs parsed from filename prefixes (e.g., "aaaaamye" from file string)
  - Output Files (.npy)  : Saved to designated output directory
                           - {train, val, test}_data.npy        : Extracted DE features [N, 16, 20, 5]
                           - {train, val, test}_labels.npy      : Classification labels [N]
                           - {train, val, test}_subject_ids.npy : Corresponding subject IDs [N]
================================================================================
```