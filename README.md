# 4datasets

# Method
1. Execute pretrainedmodel
2. Execute research of best number of electrodes
3. Execute research of ratio with static and dynamic
4. Execute research of comparison with candidates

# Datasets
## SEED-VLA
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

## SEED-VIG
```text:preprocessing
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


## TUEV



## TUAB
