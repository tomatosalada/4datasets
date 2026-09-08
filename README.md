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
'''preprocessing

'''
## TUEV
'''preprocessing

'''
## TUAB
'''preprocessing

'''