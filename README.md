# 4datasets

# Method
1. Execute pretrainedmodel
2. Execute research of best number of electrodes
3. Execute research of ratio with static and dynamic
4. Execute research of comparison with candidates

# Datasets
## SEED-VLA
'''

preprocessing
    # Binary Label Generation from Raw PERCLOS

## Label Definition
- **Class 0 (Alert):** $\text{PERCLOS} \le 0.45$
- **Class 1 (Fatigued):** $\text{PERCLOS} \ge 0.50$
- **Excluded:** $0.45 < \text{PERCLOS} < 0.50$ (ambiguous transition interval; 510 samples excluded)

## Subject Exclusion Criteria
- **Exclude:** Subjects whose labels are exclusively Class 0 throughout the recording.
- **Retain:** Subjects whose labels are exclusively Class 1. 
  - *Rationale:* Because the fatigued class constitutes the overall minority, these subjects are retained to strike a balance between class representation and preserving available data volume.

## Input Data
- `processdData/subject_wise/label_{s}.npy`: Raw PERCLOS values
- `processdData/subject_wise/eeg_{s}.npy`: EEG data array of shape $(N_s, 16, 5, 18)$

'''

## SEED-VIG
'''preprocessing

'''
## TUEV
'''preprocessing

'''
## TUAB
'''preprocessing

'''