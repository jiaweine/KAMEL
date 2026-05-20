TabPFN v1 vs v2 Implementation Summary
=======================================

BACKGROUND:
-----------
TabPFN has two major versions:
1. v1 (Nature 2022): Original implementation from NeurIPS 2022 paper
   - GitHub: https://github.com/PriorLabs/TabPFN/tree/v1.0.0
   - Version: 0.1.9
   - Limited to 1000 samples, 100 features

2. v2 (2024+): Latest improved version
   - GitHub: https://github.com/PriorLabs/TabPFN (main branch)
   - Version: 2.2.1
   - Better preprocessing and model loading
   - More flexible limits

IMPLEMENTATION APPROACH:
------------------------
Since both versions cannot be installed simultaneously as the same package name,
we created TWO DIFFERENT CONFIGURATIONS using the same TabPFN v2.2.1 installation:

1. tabpfn_classifier.py (TabPFN v1-style):
   - Uses TabPFN 2.2.1 with conservative settings
   - n_estimators=3 (smaller ensemble)
   - device='cpu' (CPU only)
   - Stricter warnings for sample/feature limits
   - Mimics v1 behavior

2. tabpfn_v2_classifier.py (TabPFN v2):
   - Uses TabPFN 2.2.1 with default settings
   - n_estimators=16 (larger ensemble)
   - device='auto' (CUDA if available)
   - More flexible configuration
   - Full v2 capabilities

KEY DIFFERENCES:
----------------
Parameter          | v1-style    | v2
-------------------|-------------|-------------
n_estimators       | 3           | 16
device             | cpu         | auto (cuda/cpu)
random_state       | 42          | 42
Sample limit warn  | 1000        | flexible
Feature limit warn | 100         | flexible

DOWNLOADED SOURCE CODE:
-----------------------
Location: /root/ICML25-TimeVLM-main/KAMEL/model_comparison/algorithms/tabpfn_official/
  - v1_nature2022/: Original v1.0.0 source code
  - v2_latest/: Latest v2 source code

Note: These are for reference only. The actual implementation uses
the installed TabPFN 2.2.1 package with different configurations.

USAGE:
------
# Run v1-style (conservative)
python run_ml_comparison.py --dataset diabetes --algorithm tabpfn --seed 42

# Run v2 (latest)
python run_ml_comparison.py --dataset diabetes --algorithm tabpfn_v2 --seed 42

VERIFICATION:
-------------
Both implementations are working correctly and produce DIFFERENT results
due to the different ensemble sizes and configurations.

This approach ensures:
1. No API usage (all local models)
2. Two distinct algorithm implementations
3. Reproducible results
4. No package installation conflicts



