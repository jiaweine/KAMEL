"""
Dataset-specific configuration for dual-path preprocessing.
Each dataset follows the principle: "First Clean, Then Bifurcate"
"""

import numpy as np
from typing import Dict, List, Any, Callable
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder

DATASET_CONFIGS = {
    "adult": {
        "name": "Adult Census Income Prediction",
        "target_col": "income",
        "missing_values": ["?"],
        "numerical_cols": ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        "categorical_cols": ["workclass", "education", "marital-status", "occupation", 
                           "relationship", "race", "sex", "native-country"],
        "drop_cols": ["fnlwgt"],  # sampling weight column
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median",
                "categorical": "mode"
            },
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "quantile"
        },
        "path_b_text_template": "{feature_name} is {feature_value}",
        "imbalance_ratio": 0.239,
        "positive_class": ">50K"
    },
    
    "creditcard": {
        "name": "Credit Card Fraud Detection", 
        "target_col": "Class",
        "missing_values": [],
        "numerical_cols": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {},
            "duplicate_removal": True,
            "outlier_treatment": "none"  # V1-V28 already PCA transformed
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard",
            "scale_only": ["Time", "Amount"]  # V1-V28 already scaled
        },
        "path_b_text_template": "{feature_name} is {feature_value:.2f}",
        "imbalance_ratio": 0.00172,
        "positive_class": 1
    },
    
    "diabetes": {
        "name": "Diabetes Prediction",
        "target_col": "target", 
        "missing_values": [],
        "numerical_cols": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median"  # Standard median imputation
            }
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"  # Apply standard scaling
        },
        "path_b_text_template": "{feature_name} is {feature_value}",
        "imbalance_ratio": 0.500,
        "positive_class": 1
    },
    
    
    
    # ========== NEW DATASETS ==========
    
    "spambase": {
        "name": "Spambase Email Classification", 
        "target_col": 57,  # Last column index
        "missing_values": [],
        "numerical_cols": list(range(57)),  # All feature columns are numerical
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median"
            },
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "quantile"
        },
        "path_b_text_template": "Email feature {feature_name} has value {feature_value}",
        "imbalance_ratio": 0.394,
        "positive_class": 1
    },
    
    "magic_telescope": {
        "name": "MAGIC Gamma Telescope",
        "target_col": 10,  # Last column index (0-based)
        "missing_values": [],
        "numerical_cols": list(range(10)),  # Columns 0-9 are numerical features
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median"
            },
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.005, 0.995]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "target_mapping": {"g": 1, "h": 0},  # gamma=1 (signal), hadron=0 (background)
        "path_b_text_template": "Telescope measurement {feature_name} equals {feature_value}",
        "imbalance_ratio": 0.648,
        "positive_class": 1  # gamma signal
    },
    
    

    "mushroom": {
        "name": "Mushroom Classification",
        "target_col": 0,  # First column index (0-based)
        "missing_values": ["?"],
        "numerical_cols": [],
        "categorical_cols": list(range(1, 23)),  # Columns 1-22 are categorical features
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "categorical": "mode"
            }
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "none"
        },
        "target_mapping": {"e": 1, "p": 0},  # edible=1 (safe), poisonous=0 (danger)
        "path_b_text_template": "Mushroom {feature_name} is {feature_value}",
        "imbalance_ratio": 0.518,
        "positive_class": 1,  # edible
        "note": "Binary dataset: edible (e) vs poisonous (p)"
    },
    
    "nursery": {
        "name": "Nursery School Decision", 
        "target_col": 8,
        "missing_values": [],
        "numerical_cols": [],
        "categorical_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "categorical": "mode"
            }
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "none"
        },
        "path_b_text_template": "Nursery {feature_name} is {feature_value}",
        "num_classes": 4,
        "task_type": "multiclass",
        "class_names": ["not_recom", "priority", "spec_prior", "very_recom"]
    },
    
    "car_evaluation": {
        "name": "Car Evaluation",
        "target_col": 6,
        "missing_values": [],
        "numerical_cols": [],
        "categorical_cols": [0, 1, 2, 3, 4, 5],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "categorical": "mode"
            }
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "none"
        },
        "path_b_text_template": "Car {feature_name} is {feature_value}",
        "num_classes": 4,
        "task_type": "multiclass",
        "class_names": ["acc", "good", "unacc", "vgood"]
    },




    # ========== FAULT DETECTION DATASETS ==========
    
    "electrical_grid_stability": {
        "name": "Electrical Grid Stability Prediction",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"],
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median"
            },
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Grid {feature_name} is {feature_value}",
        "imbalance_ratio": 0.362,  # 36.2% stable (minority class)
        "positive_class": 1  # Unstable (fault)
    },
    
    "steel_plates_fault_detection": {
        "name": "Steel Plates Fault Detection",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [f"V{i}" for i in range(1, 34)],
        "categorical_cols": [],
        "drop_cols": [],
        "cleaning_rules": {
            "missing_strategy": {
                "numerical": "median"
            },
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.005, 0.995]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Steel plate {feature_name} measurement is {feature_value}",
        "num_classes": 7,
        "task_type": "multiclass",
        "class_names": ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    },

    'breast_cancer_wisconsin': {
        "name": "Breast Cancer Wisconsin",
        "target_col": "diagnosis", 
        "missing_values": None,
        "numerical_cols": [
            "radius1", "texture1", "perimeter1", "area1", "smoothness1",
            "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
            "radius2", "texture2", "perimeter2", "area2", "smoothness2",
            "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
            "radius3", "texture3", "perimeter3", "area3", "smoothness3",
            "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
        ],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "drop_columns": ["id"],
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Patient with {feature_name} measurement of {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Benign", "Malignant"]
    },

    'heart_disease': {
        "name": "Heart Disease Prediction",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_cols": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "missing_strategy": {"numerical": "median", "categorical": "mode"},
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Patient with {feature_name} value of {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["No Disease", "Heart Disease"]
    },

    'ionosphere': {
        "name": "Ionosphere Radar Signal Classification",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [f"a{i:02d}" for i in range(1, 35)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Radar signal {feature_name} has intensity {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Bad", "Good"]
    },

    'sonar': {
        "name": "Sonar Rock vs Mine Classification",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [f"attribute_{i}" for i in range(1, 61)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Sonar {feature_name} reflection strength is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Rock", "Mine"]
    },

    'australian_credit': {
        "name": "Australian Credit Approval",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [f"A{i}" for i in range(14)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Credit application {feature_name} is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Rejected", "Approved"]
    },

    'haberman': {
        "name": "Haberman's Survival Data",
        "target_col": "survival",
        "missing_values": [],
        "numerical_cols": ["age", "operation_year", "positive_auxillary_nodes"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Patient {feature_name} value is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Died", "Survived"]
    },

    'liver_disorders': {
        "name": "Liver Disorders Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["mcv", "alkphos", "sgpt", "sgot", "gammagt"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Liver function {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Group 1", "Group 2"]
    },

    'monks1': {
        "name": "Monk's Problems Dataset 1",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [],
        "categorical_cols": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "none"
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "none"
        },
        "path_b_text_template": "Robot attribute {feature_name} has value {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Class 0", "Class 1"]
    },

    'tictactoe': {
        "name": "Tic-Tac-Toe Endgame Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [],
        "categorical_cols": ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "none"
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "none"
        },
        "path_b_text_template": "Tic-tac-toe {feature_name} contains {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Negative", "Positive"]
    },

    'iris': {
        "name": "Iris Flower Classification",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Flower {feature_name} measurement is {feature_value}",
        "num_classes": 3,
        "task_type": "multiclass",
        "class_names": ["Setosa", "Versicolor", "Virginica"]
    },

    'wine': {
        "name": "Wine Recognition Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Wine {feature_name} content is {feature_value}",
        "num_classes": 3,
        "task_type": "multiclass",
        "class_names": ["Class 1", "Class 2", "Class 3"]
    },

    'glass': {
        "name": "Glass Identification Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Glass {feature_name} composition is {feature_value}",
        "num_classes": 6,
        "task_type": "multiclass",
        "class_names": ["Building Windows Float", "Building Windows Non-Float", "Vehicle Windows Float", "Containers", "Tableware", "Headlamps"]
    },

    'ecoli': {
        "name": "E.coli Protein Localization",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Protein {feature_name} score is {feature_value}",
        "num_classes": 8,
        "task_type": "multiclass",
        "class_names": ["CP", "IM", "IMU", "IML", "IMS", "OM", "OML", "PP"]
    },

    'seeds': {
        "name": "Seeds Classification Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["V1", "V2", "V3", "V4", "V5", "V6", "V7"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.01, 0.99]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Seed {feature_name} measurement is {feature_value}",
        "num_classes": 3,
        "task_type": "multiclass",
        "class_names": ["Kama", "Rosa", "Canadian"]
    },

    'banknote': {
        "name": "Banknote Authentication Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["variance", "skewness", "curtosis", "entropy"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Banknote {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Genuine", "Forged"]
    },

    'german_credit': {
        "name": "German Credit Risk Assessment",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"],
        "categorical_cols": ["checking_status", "credit_history", "purpose", "savings_status", "employment", "personal_status", "other_parties", "property_magnitude", "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Credit applicant {feature_name} score is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Good Credit", "Bad Credit"]
    },

    'hepatitis': {
        "name": "Hepatitis Diagnosis Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders", "Ascites", "Varices", "Bilirubin", "Alk Phosphate", "Sgot", "Albumin", "Protime", "Histology"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "missing_strategy": {"numerical": "median"},
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Patient hepatitis {feature_name} level is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Live", "Die"]
    },

    'parkinsons': {
        "name": "Parkinson's Disease Detection Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["MDVP:Fo", "MDVP:Fhi", "MDVP:Flo", "MDVP:Jitter", "MDVP:Jitter.1", 
                          "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer.1", "Shimmer:APQ3", 
                          "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Voice {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Healthy", "Parkinson's"]
    },

    'blood_transfusion': {
        "name": "Blood Transfusion Service Center Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["recency", "frequency", "monetary", "time"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Donor {feature_name} value is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["No Donation", "Donated"]
    },

    'fertility': {
        "name": "Fertility Diagnosis Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Fertility factor {feature_name} is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Normal", "Altered"]
    },

    'mammographic': {
        "name": "Mammographic Mass Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["age", "shape", "margin", "density"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "missing_strategy": {"numerical": "median"},
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Mammography {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Benign", "Malignant"]
    },

    'qsar_biodegradation': {
        "name": "QSAR Biodegradation Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [f"V{i}" for i in range(1, 42)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "handle_missing": "drop",
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Molecular descriptor {feature_name} value is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Non-biodegradable", "Biodegradable"]
    },

    'seismic_bumps': {
        "name": "Seismic Bumps Prediction Dataset",
        "target_col": "class",
        "missing_values": [],
        "numerical_cols": ["genergy", "gpuls", "gdenergy", "gdpuls", "nbumps", "nbumps2",
                          "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
                          "energy", "maxenergy"],
        "categorical_cols": ["seismic", "seismoacoustic", "shift", "ghazard"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "label",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Seismic {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["No Hazard", "Hazardous"]
    },

    'vertebral_column': {
        "name": "Vertebral Column Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Spine {feature_name} measurement is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["Normal", "Abnormal"]
    },

    'vehicle': {
        "name": "Vehicle Silhouettes Dataset",
        "target_col": "target",
        "missing_values": [],
        "numerical_cols": [
            "compactness", "circularity", "distance_circularity", "radius_ratio",
            "pr_axis_aspect_ratio", "max_length_aspect_ratio", "scatter_ratio",
            "elongatedness", "pr_axis_rectangularity", "max_length_rectangularity",
            "scaled_variance_along_major_axis", "scaled_variance_along_minor_axis",
            "scaled_radius_of_gyration", "skewness_about_major_axis",
            "skewness_about_minor_axis", "kurtosis_about_minor_axis",
            "kurtosis_about_major_axis", "hollows_ratio"
        ],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Vehicle silhouette {feature_name} is {feature_value}",
        "num_classes": 4,
        "task_type": "multiclass",
        "class_names": ["bus", "opel", "saab", "van"]
    },

    'authorship': {
        "name": "Authorship Attribution Dataset",
        "target_col": "L4",
        "missing_values": [],
        "numerical_cols": [f"A{i}" for i in range(1, 71)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Authorship feature {feature_name} is {feature_value}",
        "num_classes": 4,
        "task_type": "multiclass",
        "class_names": ["1", "2", "3", "4"]
    },

    'mfeat_morphological': {
        "name": "Multiple Features Morphological Dataset",
        "target_col": "class",
        "missing_values": [],
        "numerical_cols": ["att1", "att2", "att3", "att4", "att5", "att6"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Morphological feature {feature_name} is {feature_value}",
        "num_classes": 10,
        "task_type": "multiclass",
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    },

    'cmc': {
        "name": "Contraceptive Method Choice Dataset",
        "target_col": "contraceptive_method",
        "missing_values": [],
        "numerical_cols": ["wife_age", "num_children"],
        "categorical_cols": ["wife_edu", "husband_edu", "wife_religion", "wife_working",
                           "husband_occupation", "standard_of_living_index", "media_exposure"],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {},
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Contraceptive survey {feature_name} is {feature_value}",
        "num_classes": 3,
        "task_type": "multiclass",
        "class_names": ["1", "2", "3"]
    },

    'solar_flare': {
        "name": "Solar Flare Dataset (UCI) - Binary C-class Prediction",
        "target_col": "target",
        "drop_cols": ["common flares", "moderate flares", "severe flares"],
        "missing_values": [],
        "numerical_cols": [],
        "categorical_cols": [
            "modified Zurich class", "largest spot size", "spot distribution",
            "activity", "evolution", "previous 24 hour flare activity",
            "historically-complex", "became complex on this pass",
            "area", "area of largest spot"
        ],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {},
        "path_a_transforms": {
            "categorical_encoding": "onehot",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Solar flare feature {feature_name} is {feature_value}",
        "num_classes": 2,
        "task_type": "binary",
        "class_names": ["no_flare", "flare"]
    },

    'cardiotocography': {
        "name": "Cardiotocography Dataset (UCI) - 10-class Pattern Classification",
        "target_col": "CLASS",
        "drop_cols": ["NSP"],
        "missing_values": [],
        "numerical_cols": [
            "LB", "AC", "FM", "UC", "DL", "DS", "DP",
            "ASTV", "MSTV", "ALTV", "MLTV",
            "Width", "Min", "Max", "Nmax", "Nzeros",
            "Mode", "Mean", "Median", "Variance", "Tendency"
        ],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Cardiotocography feature {feature_name} is {feature_value}",
        "num_classes": 10,
        "task_type": "multiclass",
        "class_names": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    },

    'mfeat_fourier': {
        "name": "Multiple Features Fourier Coefficients Dataset",
        "target_col": "class",
        "missing_values": [],
        "numerical_cols": [f"att{i}" for i in range(1, 77)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Fourier feature {feature_name} is {feature_value}",
        "num_classes": 10,
        "task_type": "multiclass",
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    },

    'mfeat_zernike': {
        "name": "Multiple Features Zernike Moments Dataset",
        "target_col": "class",
        "missing_values": [],
        "numerical_cols": [f"att{i}" for i in range(1, 48)],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Zernike feature {feature_name} is {feature_value}",
        "num_classes": 10,
        "task_type": "multiclass",
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    },

    'wine_quality': {
        "name": "Wine Quality Dataset (UCI) - Multiclass Quality Prediction",
        "target_col": "quality",
        "missing_values": [],
        "numerical_cols": [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol"
        ],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {
            "outlier_treatment": "quantile_clip",
            "outlier_quantiles": [0.02, 0.98]
        },
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Wine chemistry {feature_name} is {feature_value}",
        "num_classes": 7,
        "task_type": "multiclass",
        "class_names": ["3", "4", "5", "6", "7", "8", "9"]
    },

    'yeast': {
        "name": "Yeast Protein Localization Dataset (UCI)",
        "target_col": "localization_site",
        "missing_values": [],
        "numerical_cols": ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"],
        "categorical_cols": [],
        "text_cols": [],
        "image_cols": [],
        "cleaning_rules": {},
        "path_a_transforms": {
            "categorical_encoding": "none",
            "numerical_scaling": "standard"
        },
        "path_b_text_template": "Yeast feature {feature_name} is {feature_value}",
        "num_classes": 10,
        "task_type": "multiclass",
        "class_names": ["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL"]
    },

}

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for specified dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name].copy()

