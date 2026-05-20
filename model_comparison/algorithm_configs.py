"""
Algorithm parameter configurations for ML comparison experiments.
Dataset-specific parameters for optimal performance on each dataset.
"""

# Dataset-specific algorithm configurations
ALGORITHM_CONFIGS = {
    'adult': {
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6, 'reg_lambda': 3.0, 'random_state': 42},
        # SOTA Deep Learning Models (2023-2025)
        'tabnet': {'n_d': 64, 'n_a': 64, 'n_steps': 7, 'gamma': 1.5, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 5e-5, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 15, 'batch_size': 512, 'virtual_batch_size': 128, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 64, 'depth': 4, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 256, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 0.85, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 10, 'max_features': 50, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 128, 'attention_dropout': 0.1, 'ffn_d_hidden': 256, 'ffn_dropout': 0.1, 'learning_rate': 5e-5, 'batch_size': 512, 'max_epochs': 20, 'patience': 4, 'min_epochs': 10, 'early_stopping_min_delta': 5e-4, 'random_state': 42},
        'tabicl': {'n_estimators': 64, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 4, 'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 20, 'patience': 10, 'outlier_threshold': 4.0, 'softmax_temperature': 0.9, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 128, 'dropout': 0.1, 'k': 32, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 
            'd_main': 128, 'd_multiplier': 2.0, 
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.05, 'dropout0': 0.1, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models (LAMDA-Tabular/TALENT) - excluding TabR which has official implementation
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'dim': 128, 'n_blocks': 2, 'dropout': 0.1, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_main': 128, 'd_hidden': 256, 'dropout_first': 0.25, 'dropout_second': 0.0, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_block': 128, 'dropout': 0.1, 'random_state': 42},

        'node': {'num_trees': 1024, 'depth': 6, 'num_layers': 2, 'tree_dim': 3, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 512, 'max_epochs': 50, 'patience': 8, 'random_state': 42},
        'gandalf': {'gflu_stages': 6, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 512, 'max_epochs': 50, 'patience': 8, 'random_state': 42}
    },
    
    
    'car_evaluation': {
        'logistic_regression': {
            'random_state': 42, 'max_iter': 1000, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 5.0, 'multi_class': 'multinomial'
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False, 'objective': 'multi:softprob'
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'force_col_wise': True, 'objective': 'multiclass'
        },
        'catboost': {'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 6, 'reg_lambda': 3.0, 'random_state': 42, 'loss_function': 'MultiClass'},
        'tabnet': {'n_d': 64, 'n_a': 64, 'n_steps': 5, 'gamma': 1.3, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 5e-5, 'learning_rate': 8e-3, 'weight_decay': 5e-6, 'max_epochs': 100, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 48, 'depth': 3, 'heads': 6, 'dim_head': 12, 'attn_dropout': 0.15, 'ff_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 64, 'device': 'cuda', 'softmax_temperature': 0.65, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 48, 'n_epochs': 40, 'learning_rate': 5e-4, 'batch_size': 32, 'patience': 8, 'max_features': 30, 'random_state': 42},
        'ft_transformer': {'n_blocks': 4, 'd_token': 128, 'attention_dropout': 0.2, 'ffn_d_hidden': 256, 'ffn_dropout': 0.15, 'learning_rate': 1e-4, 'batch_size': 128, 'max_epochs': 60, 'patience': 8, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 64, 'n_heads': 8, 'row_layers': 3, 'icl_layers': 4, 'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 80, 'patience': 12, 'outlier_threshold': 3.5, 'softmax_temperature': 0.85, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 60, 'patience': 10, 'n_blocks': 3, 'd_block': 128, 'dropout': 0.15, 'k': 32, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 12, 
            'd_main': 128, 'd_multiplier': 2.5, 
            'encoder_n_blocks': 3, 'predictor_n_blocks': 3,
            'context_dropout': 0.15, 'dropout0': 0.2, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42
        },
        'talent_modernnca': {'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 20, 'patience': 7, 'dim': 128, 'n_blocks': 4, 'dropout': 0.15, 'temperature': 0.9, 'sample_rate': 0.85, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 20, 'patience': 10, 'n_blocks': 4, 'd_main': 128, 'd_hidden': 256, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 10, 'patience': 7, 'n_blocks': 6, 'd_block': 128, 'dropout': 0.15, 'random_state': 42},

        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 10, 'random_state': 42}
    },
    
    'creditcard': {
        'logistic_regression': {
            'random_state': 42, 'max_iter': 5000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1
        },
        'random_forest': {
            'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 20, 'min_samples_leaf': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 12, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.7, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 500, 'max_depth': 12, 'learning_rate': 0.01, 'subsample': 0.7, 'colsample_bytree': 0.7, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 500, 'max_depth': 12, 'learning_rate': 0.01, 'num_leaves': 100, 'subsample': 0.7, 'colsample_bytree': 0.7, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42},
        # SOTA Deep Learning Models (2023-2025)
        'tabnet': {'n_d': 96, 'n_a': 96, 'n_steps': 8, 'gamma': 1.3, 'n_independent': 4, 'n_shared': 4, 'lambda_sparse': 1e-5, 'learning_rate': 3e-3, 'weight_decay': 5e-6, 'max_epochs': 60, 'patience': 10, 'batch_size': 2048, 'virtual_batch_size': 256, 'momentum': 0.005, 'random_state': 42},
        'saint': {'dim': 64, 'depth': 6, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 5e-5, 'batch_size': 512, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 64, 'device': 'cuda', 'softmax_temperature': 0.8, 'balance_probabilities': True, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 128, 'attention_dropout': 0.3, 'ffn_d_hidden': 384, 'ffn_dropout': 0.2,
            'learning_rate': 1e-5, 'batch_size': 256, 'weight_decay': 1e-4,
            'max_epochs': 20, 'patience': 5, 'random_state': 42
        },
        'tabicl': {'n_estimators': 64, 'embed_dim': 128, 'n_heads': 16, 'row_layers': 3, 'icl_layers': 6, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 150, 'patience': 15, 'outlier_threshold': 5.0, 'softmax_temperature': 0.7, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 512, 'max_epochs': 30, 'patience': 10, 'n_blocks': 3, 'd_block': 128, 'dropout': 0.2, 'k': 64, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation  
        'tabr': {
            'learning_rate': 3e-4, 'batch_size': 256, 'max_epochs': 40, 'patience': 10,
            'd_main': 128, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 512, 'max_epochs': 40, 'patience': 10, 'dim': 128, 'n_blocks': 3, 'dropout': 0.2, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 512, 'max_epochs': 40, 'patience': 10, 'n_blocks': 3, 'd_main': 128, 'd_hidden': 256, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 512, 'max_epochs': 40, 'patience': 10, 'n_blocks': 3, 'd_block': 128, 'dropout': 0.2, 'random_state': 42},

        'node': {'num_trees': 1024, 'depth': 6, 'num_layers': 2, 'tree_dim': 3, 'input_dropout': 0.0, 'learning_rate': 5e-4, 'batch_size': 1024, 'max_epochs': 30, 'patience': 8, 'random_state': 42},
        'gandalf': {'gflu_stages': 6, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 1024, 'max_epochs': 30, 'patience': 8, 'random_state': 42}
    },
    
    'diabetes': {
        # Traditional ML - Optimized
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'saga', 'class_weight': 'balanced', 'C': 0.5, 'penalty': 'l2'},
        'random_forest': {'n_estimators': 8, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.85, 'max_features': 'sqrt', 'random_state': 42},
        
        # Gradient Boosting - Enhanced
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.85, 'colsample_bytree': 0.85, 'gamma': 0.1, 'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31, 'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'l2_leaf_reg': 3.0, 'random_strength': 0.5, 'bagging_temperature': 0.2, 'random_state': 42, 'verbose': 0},
        
        # Deep Learning Models - Optimized for Small Dataset (442 samples)
        # Strategy: Reduce complexity + Increase regularization + Smaller batches
        'tabnet': {
            'n_d': 32, 'n_a': 32,  # Reduced from 64→32 (less overfitting)
            'n_steps': 4,  # Reduced from 7→4
            'gamma': 1.6,  # Increased from 1.4→1.6 (more regularization)
            'n_independent': 2, 'n_shared': 2,  # Reduced from 3→2
            'lambda_sparse': 1e-4,  # Increased from 8e-5→1e-4 (more sparsity)
            'learning_rate': 5e-3,  # Reduced from 9e-3→5e-3 (more stable)
            'weight_decay': 5e-5,  # Increased from 2e-5→5e-5
            'max_epochs': 200,  # Increased from 150→200
            'patience': 40,  # Increased from 30→40 (more tolerance)
            'batch_size': 32,  # Reduced from 96→32 (better for small data)
            'virtual_batch_size': 16,  # Reduced from 48→16
            'momentum': 0.02,  # Increased from 0.015→0.02
            'mask_type': 'sparsemax',
            'random_state': 42
        },
        'saint': {
            'dim': 32,  # Reduced from 64→32 (small dataset needs small model)
            'depth': 4,  # Reduced from 8→4
            'heads': 4,  # Reduced from 8→4
            'dim_head': 8,  # Reduced from 16→8
            'attn_dropout': 0.2,  # Increased from 0.05→0.2 (prevent overfitting)
            'ff_dropout': 0.2,  # Increased from 0.05→0.2
            'learning_rate': 5e-4,  # Reduced from 3e-4→1e-4 (more conservative)
            'batch_size': 64,  # Reduced from 64→32
            'max_epochs': 30,  # Increased from 100→150
            'patience': 10,  # Increased from 15→25
            'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 12, 'device': 'cuda', 'softmax_temperature': 1, 'balance_probabilities': True, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 3,  # Reduced from 3→2
            'd_token': 8,  # Reduced from 96→64
            'attention_dropout': 0.3,  # Increased from 0.2→0.3
            'ffn_d_hidden': 64,  # Reduced from 192→128
            'learning_rate': 1e-4,  # Increased from 5e-5→1e-4
            'batch_size': 16,  # Reduced from 128→32
            'max_epochs': 50,  # Increased from 30→50
            'patience': 10,  # Increased from 5→10
            'random_state': 42
        },
        'tabicl': {
            'n_estimators': 8,  # Reduced from 32→16
            'embed_dim': 24,  # Reduced from 32→24
            'n_heads': 2,  # Reduced from 8→4
            'row_layers': 2, 
            'icl_layers': 3,  # Reduced from 4→3
            'learning_rate': 3e-4,  # Reduced from 5e-4→3e-4
            'batch_size': 32,  # Reduced from 64→32
            'max_epochs': 80,  # Reduced from 100→80
            'patience': 15,  # Increased from 10→15
            'outlier_threshold': 3.0,  # Reduced from 3.5→3.0
            'softmax_temperature': 0.85,  # Increased from 0.8→0.85
            'class_shift': True, 
            'feat_shuffle_method': 'latin', 
            'norm_methods': ['none', 'standard'],  # Changed from power→standard
            'random_state': 42, 
            'verbose': False
        },
        'tabm': {
            'learning_rate': 5e-4, 
            'batch_size': 32,  
            'max_epochs': 80, 
            'patience': 20,  
            'n_blocks': 2, 
            'd_block': 32, 
            'dropout': 0.35,  
            'k': 16,  
            'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation (OPTIMIZED v2 for diabetes)
        'tabr': {           
            # Training hyperparameters
            'learning_rate': 3e-4, 
            'batch_size': 32,  
            'max_epochs': 20,  
            'patience': 10,  
            'weight_decay': 1e-5,  
            
            # Architecture parameters (TabR-specific)
            'd_main': 64, 
            'd_multiplier': 2.5, 
            'encoder_n_blocks': 1,  
            'predictor_n_blocks': 3, 
            

            'context_dropout': 0.05,  
            'dropout0': 0.15,  
            'dropout1': 0.20,  
            
            # Retrieval settings (CRITICAL for TabR performance)
            'context_size': 64,  
            
            # Advanced settings
            'normalization': 'LayerNorm',  
            'activation': 'GELU', 
            'random_state': 42
        },
             
        'talent_modernnca': {
            'learning_rate': 5e-4,  
            'batch_size': 64, 
            'max_epochs': 20,  
            'patience': 10,  
            'dim': 64,  
            'n_blocks': 3,  
            'dropout': 0.25,  
            'temperature': 0.8, 
            'sample_rate': 0.85, 
            'random_state': 42
        },
        
        'talent_resnet': {
            'learning_rate': 5e-4, 
            'batch_size': 64, 
            'max_epochs': 50, 
            'patience': 10, 
            'n_blocks': 3,  
            'd_main': 96,  
            'd_hidden': 192,  
            'dropout_first': 0.25, 
            'dropout_second': 0.15, 
            'random_state': 42
        },
        
        'talent_realmlp': {
            'learning_rate': 5e-4,  
            'batch_size': 32,  
            'max_epochs': 20,  
            'patience': 12, 
            'n_blocks': 2,  
            'd_block': 64, 
            'dropout': 0.3,  
            'random_state': 42
        },

        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.2, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },
    
    
    'magic_telescope': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 1.0},
        'random_forest': {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 3, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.08, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 250, 'max_depth': 10, 'learning_rate': 0.08, 'num_leaves': 40, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42},
        # SOTA Deep Learning Models (2023-2025)
        'tabnet': {'n_d': 80, 'n_a': 80, 'n_steps': 6, 'gamma': 1.4, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 5e-5, 'learning_rate': 5e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 12, 'batch_size': 1024, 'virtual_batch_size': 256, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 48, 'depth': 5, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 8e-5, 'batch_size': 256, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 48, 'device': 'cuda', 'softmax_temperature': 0.85, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 48, 'n_epochs': 30, 'learning_rate': 8e-4, 'batch_size': 64, 'patience': 8, 'max_features': 50, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 160, 'attention_dropout': 0.15, 'ffn_d_hidden': 320, 'learning_rate': 8e-5, 'batch_size': 256, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 32, 'n_heads': 8, 'row_layers': 3, 'icl_layers': 4, 'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 20, 'patience': 12, 'outlier_threshold': 4.5, 'softmax_temperature': 0.8, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 40, 'patience': 8, 'n_blocks': 3, 'd_block': 160, 'dropout': 0.15, 'k': 48, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        

        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 256, 'max_epochs': 50, 'patience': 10,
            'd_main': 160, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.05, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42
        },
        
        'talent_modernnca': {'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'dim': 128, 'n_blocks': 3, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_main': 128, 'd_hidden': 256, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_block': 160, 'dropout': 0.15, 'random_state': 42},

        'node': {'num_trees': 512, 'depth': 6, 'num_layers': 2, 'tree_dim': 3, 'input_dropout': 0.0, 'learning_rate': 8e-4, 'batch_size': 512, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 6, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 512, 'max_epochs': 50, 'patience': 10, 'random_state': 42}
    },
    
    'mushroom': {
        'logistic_regression': {'random_state': 42, 'max_iter': 10, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 1.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42},

        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 5e-4, 'learning_rate': 1.5e-2, 'weight_decay': 1e-5, 'max_epochs': 60, 'patience': 10, 'batch_size': 256, 'virtual_batch_size': 128, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 3, 'heads': 4, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 2e-4, 'batch_size': 64, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.95, 'balance_probabilities': False, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 20, 'learning_rate': 2e-3, 'batch_size': 64, 'patience': 5, 'max_features': 22, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'learning_rate': 2e-4, 'batch_size': 64, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabicl': {'n_estimators': 8, 'embed_dim': 16, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 2, 'learning_rate': 2e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 5, 'outlier_threshold': 3.0, 'softmax_temperature': 0.9, 'class_shift': False, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 2e-3, 'batch_size': 64, 'max_epochs': 30, 'patience': 5, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.2, 'k': 8, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {
            'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8,
            'd_main': 64, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42
        },

        'talent_modernnca': {'learning_rate': 2e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'dim': 64, 'n_blocks': 2, 'dropout': 0.2, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42},
        'talent_resnet': {'learning_rate': 2e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 2e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.2, 'random_state': 42},

        'node': {'num_trees': 256, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'random_state': 42}
    },
    
    'nursery': {
        'logistic_regression': {'random_state': 42, 'max_iter': 500, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 0.5, 'multi_class': 'multinomial'},
        'random_forest': {'n_estimators': 60, 'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 30, 'learning_rate': 0.5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 60, 'max_depth': 5, 'learning_rate': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False, 'objective': 'multi:softprob'},
        'lightgbm': {'n_estimators': 60, 'max_depth': 5, 'learning_rate': 0.2, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'force_col_wise': True, 'objective': 'multiclass'},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42, 'loss_function': 'MultiClass'},
   
        'tabnet': {'n_d': 24, 'n_a': 24, 'n_steps': 3, 'gamma': 1.6, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 5e-3, 'weight_decay': 1e-5, 'max_epochs': 20, 'patience': 10, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 12, 'depth': 2, 'heads': 3, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': False, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 25, 'learning_rate': 3e-3, 'batch_size': 32, 'patience': 5, 'max_features': 20, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 48, 'attention_dropout': 0.2, 'ffn_d_hidden': 96, 'learning_rate': 3e-4, 'batch_size': 32, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabicl': {'n_estimators': 4, 'embed_dim': 12, 'n_heads': 3, 'row_layers': 1, 'icl_layers': 2, 'learning_rate': 3e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 3, 'outlier_threshold': 2.5, 'softmax_temperature': 1.0, 'class_shift': False, 'feat_shuffle_method': 'random', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-3, 'batch_size': 128, 'max_epochs': 10, 'patience': 7, 'n_blocks': 4, 'd_block': 48, 'dropout': 0.2, 'k': 4, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        

        'tabr': {
            'learning_rate': 2e-3, 'batch_size': 64, 'max_epochs': 35, 'patience': 7,
            'd_main': 64, 'd_multiplier': 2.0,
            'encoder_n_blocks': 1, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42
        },
        

        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 256, 'max_epochs': 10, 'patience': 7, 'dim': 64, 'n_blocks': 4, 'dropout': 0.02, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 256, 'max_epochs': 10, 'patience': 7, 'n_blocks': 2, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 3e-3, 'batch_size': 256, 'max_epochs': 35, 'patience': 12, 'n_blocks': 4, 'd_block': 64, 'dropout': 0.02, 'random_state': 42},

        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'random_state': 42}
    },
    
    'spambase': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0},
        'random_forest': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42},

        'tabnet': {'n_d': 64, 'n_a': 64, 'n_steps': 5, 'gamma': 1.4, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 1e-4, 'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 12, 'batch_size': 1024, 'virtual_batch_size': 256, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 4, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 256, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 128, 'attention_dropout': 0.2, 'ffn_d_hidden': 256, 'learning_rate': 1e-4, 'batch_size': 256, 'max_epochs': 20, 'patience': 3, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 32, 'n_heads': 8, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 100, 'patience': 10, 'outlier_threshold': 4.0, 'softmax_temperature': 0.9, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 128, 'dropout': 0.2, 'k': 32, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},        
        'tabr': {
            'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 50, 'patience': 10,
            'd_main': 128, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.05, 'dropout0': 0.1, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42
        },       
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'dim': 128, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_main': 128, 'd_hidden': 256, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 128, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 512, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 6, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 50, 'patience': 10, 'random_state': 42}
    },
    
    'electrical_grid_stability': {
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 2.0
        },
        'random_forest': {
            'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 24, 'learning_rate': 0.08, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 24, 'max_depth': 10, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 24, 'max_depth': 10, 'learning_rate': 0.08, 'num_leaves': 24, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 24, 'learning_rate': 0.2, 'max_depth': 4, 'reg_lambda': 2.0, 'random_state': 42},

        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 4, 'lambda_sparse': 1e-3, 'learning_rate': 8e-3, 'max_epochs': 25, 'patience': 5, 'batch_size': 256, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 4, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 8e-5, 'batch_size': 128, 'max_epochs': 25, 'patience': 5, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 4, 'device': 'cuda', 'softmax_temperature': 0.85, 'balance_probabilities': True, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.15, 'ffn_d_hidden': 64, 'learning_rate': 8e-5, 'batch_size': 128, 'max_epochs': 25, 'patience': 5, 'random_state': 42},
        'tabicl': {'n_estimators': 6, 'embed_dim': 4, 'n_heads': 2, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 80, 'patience': 8, 'outlier_threshold': 4.0, 'softmax_temperature': 0.8, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 20, 'patience': 7, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'k': 20, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        

        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 40, 'patience': 8,
            'd_main': 96, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 80, 'random_state': 42
        },
        

        'talent_modernnca': {'learning_rate': 5e-3, 'batch_size': 64, 'max_epochs': 10, 'patience': 8, 'dim': 96, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 0.85, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-3, 'batch_size': 32, 'max_epochs': 10, 'patience': 4, 'n_blocks': 1, 'd_main': 32, 'd_hidden': 32, 'dropout_first': 0.1, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 512, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 40, 'patience': 8, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 96, 'head_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 256, 'max_epochs': 40, 'patience': 8, 'random_state': 42}
    },
    
    'steel_plates_fault_detection': {
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'class_weight': 'balanced', 'C': 1.5, 'multi_class': 'multinomial'
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 18, 'min_samples_split': 4, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False, 'objective': 'multi:softprob'
        },
        'lightgbm': {
            'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 35, 'subsample': 0.8, 'colsample_bytree': 0.8, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'force_col_wise': True, 'objective': 'multiclass'
        },
        'catboost': {'n_estimators': 120, 'learning_rate': 0.18, 'max_depth': 4, 'reg_lambda': 2.2, 'random_state': 42, 'loss_function': 'MultiClass'},

        'tabnet': {'n_d': 40, 'n_a': 40, 'n_steps': 4, 'lambda_sparse': 1e-3, 'learning_rate': 1e-2, 'max_epochs': 30, 'patience': 5, 'batch_size': 256, 'random_state': 42},
        'saint': {'dim': 40, 'depth': 4, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 128, 'max_epochs': 30, 'patience': 5, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 96, 'attention_dropout': 0.2, 'ffn_d_hidden': 256, 'learning_rate': 1e-4, 'batch_size': 128, 'max_epochs': 30, 'patience': 5, 'random_state': 42},
        'tabicl': {'n_estimators': 20, 'embed_dim': 40, 'n_heads': 8, 'row_layers': 3, 'icl_layers': 4, 'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 100, 'patience': 10, 'outlier_threshold': 4.5, 'softmax_temperature': 0.9, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'power'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 40, 'patience': 8, 'n_blocks': 3, 'd_block': 96, 'dropout': 0.2, 'k': 20, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 45, 'patience': 10,
            'd_main': 96, 'd_multiplier': 2.0,
            'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 80, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 45, 'patience': 10, 'dim': 96, 'n_blocks': 3, 'dropout': 0.2, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 45, 'patience': 10, 'n_blocks': 3, 'd_main': 96, 'd_hidden': 192, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 45, 'patience': 10, 'n_blocks': 3, 'd_block': 96, 'dropout': 0.2, 'random_state': 42},
        'node': {'num_trees': 512, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 96, 'head_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 50, 'patience': 10, 'random_state': 42}
    },
    
    'breast_cancer_wisconsin': {
        # Traditional ML Models - Optimized for medical diagnostic task (569 samples, 30 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'num_leaves': 15, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for medium-sized medical dataset
        'tabnet': {
            'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 64, 
            'virtual_batch_size': 32, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 40, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 8, 'max_features': 30, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.2, 
            'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 40, 'patience': 8, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 
            'batch_size': 32, 'max_epochs': 60, 'patience': 10, 'outlier_threshold': 3.5, 'softmax_temperature': 0.85, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 64, 
            'dropout': 0.2, 'k': 16, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 40, 'patience': 10, 
            'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 40, 'patience': 10, 'dim': 64, 'n_blocks': 2, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 40, 'patience': 10, 'n_blocks': 2, 'd_main': 64, 
            'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 40, 'patience': 10, 'n_blocks': 2, 'd_block': 64, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'heart_disease': {
        # Traditional ML Models - Optimized for medical diagnostic task (303 samples, 13 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for small medical dataset
        'tabnet': {
            'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 20, 'batch_size': 32, 
            'virtual_batch_size': 16, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 10, 'max_features': 13, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.2, 
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 1e-3, 
            'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 0.85, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.2, 'k': 8, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 50, 'patience': 15, 
            'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 50, 'patience': 15, 'dim': 32, 'n_blocks': 2, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 50, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 
            'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 50, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'ionosphere': {
        # Traditional ML Models - Optimized for signal processing (351 samples, 34 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'reg_lambda': 1.5, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for signal processing
        'tabnet': {
            'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 64, 
            'virtual_batch_size': 32, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 4, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 40, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 8, 'max_features': 34, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 3, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.2, 
            'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 3, 'icl_layers': 3, 'learning_rate': 5e-4, 
            'batch_size': 32, 'max_epochs': 70, 'patience': 12, 'outlier_threshold': 3.0, 'softmax_temperature': 0.85, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 3, 'd_block': 64, 
            'dropout': 0.2, 'k': 16, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 12, 
            'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 3, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 12, 'dim': 64, 'n_blocks': 3, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 12, 'n_blocks': 3, 'd_main': 64, 
            'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 12, 'n_blocks': 3, 'd_block': 64, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'sonar': {
        # Traditional ML Models - Optimized for sonar classification (208 samples, 60 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 3000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.01
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 8, 'reg_lambda': 1.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for small high-dimensional dataset
        'tabnet': {
            'n_d': 24, 'n_a': 24, 'n_steps': 5, 'gamma': 1.5, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 1e-3, 
            'learning_rate': 5e-3, 'weight_decay': 1e-4, 'max_epochs': 120, 'patience': 25, 'batch_size': 32, 
            'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42
        },
        'saint': {
            'dim': 64, 'depth': 4, 'heads': 8, 'dim_head': 8, 'attn_dropout': 0.2, 'ff_dropout': 0.2, 
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 0.8, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 128, 'n_epochs': 60, 'learning_rate': 5e-4, 'batch_size': 16, 'patience': 12, 'max_features': 60, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 4, 'd_token': 128, 'attention_dropout': 0.3, 'ffn_d_hidden': 256, 'ffn_dropout': 0.3, 
            'learning_rate': 5e-5, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 32, 'embed_dim': 128, 'n_heads': 8, 'row_layers': 4, 'icl_layers': 4, 'learning_rate': 1e-4, 
            'batch_size': 16, 'max_epochs': 100, 'patience': 25, 'outlier_threshold': 2.5, 'softmax_temperature': 0.8, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'n_blocks': 4, 'd_block': 128, 
            'dropout': 0.3, 'k': 32, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 
            'd_main': 128, 'd_multiplier': 2.0, 'encoder_n_blocks': 4, 'predictor_n_blocks': 3,
            'context_dropout': 0.2, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 128, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'dim': 128, 'n_blocks': 4, 
            'dropout': 0.2, 'temperature': 0.8, 'sample_rate': 0.9, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'n_blocks': 4, 'd_main': 128, 
            'd_hidden': 256, 'dropout_first': 0.3, 'dropout_second': 0.2, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'n_blocks': 4, 'd_block': 128, 
            'dropout': 0.2, 'random_state': 42
        },
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.2, 'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.2, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.3, 'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 20, 'random_state': 42}
    },

    'australian_credit': {
        # Traditional ML Models - Optimized for credit approval (690 samples, 14 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for credit scoring
        'tabnet': {
            'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 64, 
            'virtual_batch_size': 32, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 40, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 8, 'max_features': 14, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.2, 
            'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 
            'batch_size': 32, 'max_epochs': 70, 'patience': 12, 'outlier_threshold': 3.5, 'softmax_temperature': 0.85, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 2, 'd_block': 64, 
            'dropout': 0.2, 'k': 16, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 
            'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'dim': 64, 'n_blocks': 2, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_main': 64, 
            'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 64, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'haberman': {
        # Traditional ML Models - Optimized for small medical survival dataset (306 samples, 3 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 5, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'num_leaves': 15, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'reg_lambda': 3.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for very small dataset
        'tabnet': {
            'n_d': 8, 'n_a': 8, 'n_steps': 2, 'gamma': 1.2, 'n_independent': 1, 'n_shared': 1, 'lambda_sparse': 1e-3, 
            'learning_rate': 5e-3, 'weight_decay': 1e-4, 'max_epochs': 80, 'patience': 20, 'batch_size': 32, 
            'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42
        },
        'saint': {
            'dim': 16, 'depth': 2, 'heads': 2, 'dim_head': 8, 'attn_dropout': 0.2, 'ff_dropout': 0.2, 
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 8, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 16, 'n_epochs': 40, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 10, 'max_features': 3, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 1, 'd_token': 16, 'attention_dropout': 0.3, 'ffn_d_hidden': 32, 'ffn_dropout': 0.3, 
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 8, 'embed_dim': 16, 'n_heads': 2, 'row_layers': 1, 'icl_layers': 2, 'learning_rate': 1e-3, 
            'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'outlier_threshold': 4.0, 'softmax_temperature': 1.0, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 50, 'patience': 15, 'n_blocks': 1, 'd_block': 16, 
            'dropout': 0.3, 'k': 4, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 10, 
            'd_main': 16, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 1,
            'context_dropout': 0.2, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 16, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 10, 'dim': 16, 'n_blocks': 1, 
            'dropout': 0.2, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 10, 'n_blocks': 1, 'd_main': 16, 
            'd_hidden': 32, 'dropout_first': 0.3, 'dropout_second': 0.2, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 40, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 
            'dropout': 0.2, 'random_state': 42
        },
        'node': {'num_trees': 64, 'depth': 3, 'num_layers': 1, 'tree_dim': 1, 'input_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 2, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 16, 'head_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 60, 'patience': 15, 'random_state': 42}
    },

    'liver_disorders': {
        # Traditional ML Models - Optimized for medical dataset (345 samples, 6 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.5
        },
        'random_forest': {
            'n_estimators': 150, 'max_depth': 8, 'min_samples_split': 8, 'min_samples_leaf': 3, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.08, 'max_depth': 5, 'min_samples_split': 8, 'min_samples_leaf': 4, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.08, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.08, 'max_depth': 5, 'reg_lambda': 2.5, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for medical data
        'tabnet': {
            'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 20, 'batch_size': 64, 
            'virtual_batch_size': 32, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 50, 'learning_rate': 8e-4, 'batch_size': 32, 'patience': 10, 'max_features': 6, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.2, 
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 8e-4, 
            'batch_size': 32, 'max_epochs': 70, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 0.9, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.2, 'k': 12, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 
            'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'dim': 32, 'n_blocks': 2, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_main': 32, 
            'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.2, 'learning_rate': 8e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'monks1': {
        # Traditional ML Models - Optimized for categorical logic problem (124 samples, 6 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.9, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'num_leaves': 15, 'subsample': 0.9, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for small categorical dataset
        'tabnet': {
            'n_d': 8, 'n_a': 8, 'n_steps': 2, 'gamma': 1.5, 'n_independent': 1, 'n_shared': 1, 'lambda_sparse': 1e-3, 
            'learning_rate': 1e-2, 'weight_decay': 1e-4, 'max_epochs': 60, 'patience': 15, 'batch_size': 16, 
            'virtual_batch_size': 8, 'momentum': 0.02, 'random_state': 42
        },
        'saint': {
            'dim': 16, 'depth': 2, 'heads': 2, 'dim_head': 8, 'attn_dropout': 0.2, 'ff_dropout': 0.2, 
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 8, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 16, 'n_epochs': 40, 'learning_rate': 1e-3, 'batch_size': 16, 'patience': 8, 'max_features': 6, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 1, 'd_token': 16, 'attention_dropout': 0.3, 'ffn_d_hidden': 32, 'ffn_dropout': 0.3, 
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 8, 'embed_dim': 16, 'n_heads': 2, 'row_layers': 1, 'icl_layers': 2, 'learning_rate': 1e-3, 
            'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'outlier_threshold': 4.0, 'softmax_temperature': 1.0, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 
            'dropout': 0.3, 'k': 6, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 
            'd_main': 16, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 1,
            'context_dropout': 0.2, 'dropout0': 0.2, 'dropout1': 0.2,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 16, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'dim': 16, 'n_blocks': 1, 
            'dropout': 0.2, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'n_blocks': 1, 'd_main': 16, 
            'd_hidden': 32, 'dropout_first': 0.3, 'dropout_second': 0.2, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 
            'dropout': 0.2, 'random_state': 42
        },
        'node': {'num_trees': 64, 'depth': 3, 'num_layers': 1, 'tree_dim': 1, 'input_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 2, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 16, 'head_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42}
    },

    'tictactoe': {
        # Traditional ML Models - Optimized for categorical game dataset (958 samples, 9 features)
        'logistic_regression': {
            'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0
        },
        'random_forest': {
            'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.8, 'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        },
        'lightgbm': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 'auto', 'random_state': 42, 'verbose': -1, 'force_col_wise': True
        },
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models - Optimized for categorical features
        'tabnet': {
            'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 
            'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 64, 
            'virtual_batch_size': 32, 'momentum': 0.01, 'random_state': 42
        },
        'saint': {
            'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42
        },
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 50, 'learning_rate': 5e-4, 'batch_size': 32, 'patience': 10, 'max_features': 9, 'random_state': 42},
        'ft_transformer': {
            'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.2, 
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'random_state': 42
        },
        'tabicl': {
            'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 
            'batch_size': 32, 'max_epochs': 70, 'patience': 12, 'outlier_threshold': 3.5, 'softmax_temperature': 0.9, 
            'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False
        },
        'tabm': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.2, 'k': 16, 'random_state': 42
        },
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        
        # TabR - Official Yandex Research Implementation
        'tabr': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 
            'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2,
            'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15,
            'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42
        },
        
        # TALENT Deep Tabular Models
        'talent_modernnca': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'dim': 32, 'n_blocks': 2, 
            'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.85, 'random_state': 42
        },
        'talent_resnet': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_main': 32, 
            'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42
        },
        'talent_realmlp': {
            'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 50, 'patience': 10, 'n_blocks': 2, 'd_block': 32, 
            'dropout': 0.15, 'random_state': 42
        },
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'iris': {
        # Traditional ML Models - Small dataset (150 samples, 4 features, 3 classes)
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0},
        'random_forest': {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'num_leaves': 7, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 1.0, 'random_state': 42},
        
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 2, 'gamma': 1.2, 'n_independent': 1, 'n_shared': 1, 'lambda_sparse': 1e-3, 'learning_rate': 1e-2, 'weight_decay': 1e-4, 'max_epochs': 50, 'patience': 10, 'batch_size': 16, 'virtual_batch_size': 8, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 2, 'heads': 2, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 8, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 16, 'n_epochs': 30, 'learning_rate': 1e-3, 'batch_size': 16, 'patience': 8, 'max_features': 4, 'random_state': 42},
        'ft_transformer': {'n_blocks': 1, 'd_token': 16, 'attention_dropout': 0.2, 'ffn_d_hidden': 32, 'ffn_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'random_state': 42},
        'tabicl': {'n_estimators': 8, 'embed_dim': 16, 'n_heads': 2, 'row_layers': 1, 'icl_layers': 2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'outlier_threshold': 4.0, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.2, 'k': 4, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'd_main': 16, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 1, 'context_dropout': 0.1, 'dropout0': 0.1, 'dropout1': 0.1, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 16, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'dim': 16, 'n_blocks': 1, 'dropout': 0.1, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'n_blocks': 1, 'd_main': 16, 'd_hidden': 32, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 30, 'patience': 8, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.1, 'random_state': 42},
        'node': {'num_trees': 64, 'depth': 3, 'num_layers': 1, 'tree_dim': 1, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 2, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 16, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 40, 'patience': 10, 'random_state': 42}
    },

    'wine': {
        # Traditional ML Models - Wine dataset (178 samples, 13 features, 3 classes)
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 1.5, 'random_state': 42},
        
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 60, 'patience': 12, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 40, 'learning_rate': 8e-4, 'batch_size': 16, 'patience': 8, 'max_features': 13, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.2, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 8e-4, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'outlier_threshold': 3.5, 'softmax_temperature': 0.9, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 12, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.15, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42}
    },

    'glass': {
        # Traditional ML Models - Glass dataset (214 samples, 9 features, 6 classes)
        'logistic_regression': {'random_state': 42, 'max_iter': 3000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1},
        'random_forest': {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 24, 'n_a': 24, 'n_steps': 4, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 5e-3, 'weight_decay': 1e-4, 'max_epochs': 80, 'patience': 15, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.2, 'ff_dropout': 0.2, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.8, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 50, 'learning_rate': 5e-4, 'batch_size': 16, 'patience': 10, 'max_features': 9, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 64, 'attention_dropout': 0.3, 'ffn_d_hidden': 128, 'ffn_dropout': 0.3, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 3, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 60, 'patience': 12, 'outlier_threshold': 3.0, 'softmax_temperature': 0.8, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.3, 'k': 18, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 3, 'predictor_n_blocks': 2, 'context_dropout': 0.2, 'dropout0': 0.2, 'dropout1': 0.2, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'dim': 64, 'n_blocks': 3, 'dropout': 0.2, 'temperature': 0.8, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.3, 'dropout_second': 0.2, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.2, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.2, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'ecoli': {
        # Traditional ML Models - Ecoli dataset (336 samples, 7 features, 8 classes)
        'logistic_regression': {'random_state': 42, 'max_iter': 3000, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 0.1},
        'random_forest': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8, 'reg_lambda': 2.0, 'random_state': 42},
        
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.5, 'n_independent': 3, 'n_shared': 3, 'lambda_sparse': 1e-4, 'learning_rate': 3e-3, 'weight_decay': 1e-4, 'max_epochs': 100, 'patience': 20, 'batch_size': 64, 'virtual_batch_size': 32, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 64, 'depth': 4, 'heads': 8, 'dim_head': 8, 'attn_dropout': 0.2, 'ff_dropout': 0.2, 'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 24, 'device': 'cuda', 'softmax_temperature': 0.7, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 96, 'n_epochs': 60, 'learning_rate': 3e-4, 'batch_size': 32, 'patience': 12, 'max_features': 7, 'random_state': 42},
        'ft_transformer': {'n_blocks': 4, 'd_token': 96, 'attention_dropout': 0.3, 'ffn_d_hidden': 192, 'ffn_dropout': 0.3, 'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'tabicl': {'n_estimators': 24, 'embed_dim': 96, 'n_heads': 8, 'row_layers': 4, 'icl_layers': 4, 'learning_rate': 3e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 2.5, 'softmax_temperature': 0.7, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 4, 'd_block': 96, 'dropout': 0.3, 'k': 28, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'd_main': 96, 'd_multiplier': 2.0, 'encoder_n_blocks': 4, 'predictor_n_blocks': 3, 'context_dropout': 0.2, 'dropout0': 0.25, 'dropout1': 0.25, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'dim': 96, 'n_blocks': 4, 'dropout': 0.25, 'temperature': 0.7, 'sample_rate': 0.95, 'random_state': 42},
        'talent_resnet': {'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 4, 'd_main': 96, 'd_hidden': 192, 'dropout_first': 0.3, 'dropout_second': 0.25, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 3e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'n_blocks': 4, 'd_block': 96, 'dropout': 0.25, 'random_state': 42},
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 96, 'head_dropout': 0.2, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'seeds': {
        # Traditional ML Models - Seeds dataset (210 samples, 7 features, 3 classes)
        'logistic_regression': {'random_state': 42, 'max_iter': 1500, 'solver': 'liblinear', 'class_weight': 'balanced', 'C': 1.0},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 20, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 1.5, 'random_state': 42},
        
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 8e-3, 'weight_decay': 1e-5, 'max_epochs': 60, 'patience': 12, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.01, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 0.9, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 40, 'learning_rate': 8e-4, 'batch_size': 16, 'patience': 8, 'max_features': 7, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.2, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 8e-4, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'outlier_threshold': 3.5, 'softmax_temperature': 0.9, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 14, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 40, 'patience': 8, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.15, 'learning_rate': 8e-4, 'batch_size': 32, 'max_epochs': 50, 'patience': 10, 'random_state': 42}
    },

    # 10 New UCI Binary Classification Datasets
    'banknote': {
        # Traditional ML Models - Banknote dataset (1372 samples, 4 features)
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 1.0},
        'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 31, 'subsample': 1.0, 'colsample_bytree': 1.0, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
        # SOTA Deep Learning Models
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-3, 'learning_rate': 2e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 20, 'batch_size': 256, 'virtual_batch_size': 128, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 6, 'heads': 8, 'dim_head': 16, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 100, 'patience': 20, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 20, 'max_features': 4, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 192, 'attention_dropout': 0.2, 'ffn_d_hidden': 512, 'ffn_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 64, 'n_heads': 8, 'row_layers': 12, 'icl_layers': 6, 'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 100, 'patience': 20, 'outlier_threshold': 50.0, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'n_blocks': 8, 'd_block': 192, 'dropout': 0.0, 'k': 32, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'd_main': 265, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 1, 'context_dropout': 0.2, 'dropout0': 0.38, 'dropout1': 0.0, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 96, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'dim': 265, 'n_blocks': 8, 'dropout': 0.38, 'temperature': 1.0, 'sample_rate': 0.95, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'n_blocks': 8, 'd_main': 265, 'd_hidden': 1056, 'dropout_first': 0.38, 'dropout_second': 0.0, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-4, 'batch_size': 512, 'max_epochs': 100, 'patience': 16, 'n_blocks': 8, 'd_block': 265, 'dropout': 0.38, 'random_state': 42},
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 256, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'german_credit': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 5e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 80, 'learning_rate': 5e-4, 'batch_size': 16, 'patience': 15, 'max_features': 20, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 16, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'hepatitis': {
        'logistic_regression': {'random_state': 42, 'max_iter': 500, 'solver': 'liblinear', 'C': 1.0},
        'random_forest': {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 15, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'max_epochs': 50, 'patience': 10, 'batch_size': 16, 'virtual_batch_size': 8, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 8, 'patience': 10, 'max_features': 19, 'random_state': 42},
        'ft_transformer': {'n_blocks': 1, 'd_token': 16, 'attention_dropout': 0.2, 'ffn_d_hidden': 32, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 16, 'n_heads': 4, 'row_layers': 1, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 8, 'max_epochs': 50, 'patience': 10, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.2, 'k': 9, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'd_main': 16, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 16, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'dim': 16, 'n_blocks': 1, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_main': 16, 'd_hidden': 32, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 64, 'depth': 3, 'num_layers': 1, 'tree_dim': 1, 'input_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 2, 'gflu_dropout': 0.2, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 16, 'head_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42}
    },

    'parkinsons': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 5e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 80, 'learning_rate': 5e-4, 'batch_size': 16, 'patience': 15, 'max_features': 22, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 16, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 256, 'depth': 4, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'blood_transfusion': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 24, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 4, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 24, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 4, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'fertility': {
        'logistic_regression': {'random_state': 42, 'max_iter': 500, 'solver': 'liblinear', 'C': 1.0},
        'random_forest': {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 15, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'max_epochs': 50, 'patience': 10, 'batch_size': 16, 'virtual_batch_size': 8, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 8, 'patience': 10, 'max_features': 9, 'random_state': 42},
        'ft_transformer': {'n_blocks': 1, 'd_token': 16, 'attention_dropout': 0.2, 'ffn_d_hidden': 32, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 16, 'n_heads': 4, 'row_layers': 1, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 8, 'max_epochs': 50, 'patience': 10, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.2, 'k': 9, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'd_main': 16, 'd_multiplier': 2.0, 'encoder_n_blocks': 1, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 16, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'dim': 16, 'n_blocks': 1, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_main': 16, 'd_hidden': 32, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 10, 'n_blocks': 1, 'd_block': 16, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 64, 'depth': 3, 'num_layers': 1, 'tree_dim': 1, 'input_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 2, 'gflu_dropout': 0.2, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 16, 'head_dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 16, 'max_epochs': 50, 'patience': 12, 'random_state': 42}
    },

    'mammographic': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 24, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 4, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 24, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 4, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'qsar_biodegradation': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1500, 'solver': 'liblinear', 'C': 0.1},
        'random_forest': {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 10, 'random_state': 42},
        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 20, 'batch_size': 64, 'virtual_batch_size': 32, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 64, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 128, 'n_epochs': 100, 'learning_rate': 1e-4, 'batch_size': 32, 'patience': 20, 'max_features': 41, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 3, 'icl_layers': 3, 'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 100, 'patience': 20, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.2, 'k': 32, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 3, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'dim': 64, 'n_blocks': 3, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 512, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 6, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 128, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'seismic_bumps': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1500, 'solver': 'liblinear', 'C': 0.1},
        'random_forest': {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 10, 'random_state': 42},
        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 20, 'batch_size': 64, 'virtual_batch_size': 32, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 64, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 128, 'n_epochs': 100, 'learning_rate': 1e-4, 'batch_size': 32, 'patience': 20, 'max_features': 18, 'random_state': 42},
        'ft_transformer': {'n_blocks': 3, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.1, 'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 3, 'icl_layers': 3, 'learning_rate': 1e-4, 'batch_size': 32, 'max_epochs': 100, 'patience': 20, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.2, 'k': 18, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 3, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'dim': 64, 'n_blocks': 3, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs': 100, 'patience': 20, 'n_blocks': 3, 'd_block': 64, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 512, 'depth': 5, 'num_layers': 2, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 5, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 96, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'vertebral_column': {
        'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear', 'C': 0.5},
        'random_forest': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 25, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 5e-3, 'weight_decay': 1e-5, 'max_epochs': 80, 'patience': 15, 'batch_size': 32, 'virtual_batch_size': 16, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 16, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 80, 'learning_rate': 5e-4, 'batch_size': 16, 'patience': 15, 'max_features': 6, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 6, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'vehicle': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'C': 1.0, 'multi_class': 'multinomial'},
        'random_forest': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 18, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 32, 'embed_dim': 64, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 1e-3, 'batch_size': 32, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'k': 16, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'dim': 64, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.8, 'random_state': 42},
        'talent_resnet': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 256, 'depth': 5, 'num_layers': 2, 'tree_dim': 3, 'input_dropout': 0.0, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42},
        'gandalf': {'gflu_stages': 4, 'gflu_dropout': 0.0, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 60, 'patience': 12, 'random_state': 42}
    },

    'authorship': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'C': 1.0, 'multi_class': 'multinomial'},
        'random_forest': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42},
        'tabnet': {'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 15, 'batch_size': 64, 'virtual_batch_size': 32, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 64, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 70, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 64, 'attention_dropout': 0.2, 'ffn_d_hidden': 128, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'k': 8, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'd_main': 64, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 64, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'dim': 64, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 64, 'd_hidden': 128, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 64, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 64, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 64, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'cmc': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'C': 1.0, 'multi_class': 'multinomial'},
        'random_forest': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42},
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 32, 'depth': 3, 'heads': 4, 'dim_head': 8, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 9, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 8, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    },

    'mfeat_morphological': {
        'logistic_regression': {'random_state': 42, 'max_iter': 2000, 'solver': 'lbfgs', 'C': 1.0, 'multi_class': 'multinomial'},
        'random_forest': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8, 'random_state': 42},
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'mlogloss', 'use_label_encoder': False},
        'lightgbm': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1, 'force_col_wise': True},
        'catboost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8, 'random_state': 42},
        'tabnet': {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'gamma': 1.5, 'n_independent': 2, 'n_shared': 2, 'lambda_sparse': 1e-4, 'learning_rate': 1e-2, 'weight_decay': 1e-5, 'max_epochs': 100, 'patience': 15, 'batch_size': 128, 'virtual_batch_size': 64, 'momentum': 0.02, 'random_state': 42},
        'saint': {'dim': 16, 'depth': 2, 'heads': 4, 'dim_head': 4, 'attn_dropout': 0.1, 'ff_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabpfn_v2': {'n_estimators': 32, 'device': 'cuda', 'softmax_temperature': 1.0, 'balance_probabilities': True, 'random_state': 42},
        'tabpfn_offline': {'hidden_dim': 32, 'n_epochs': 80, 'learning_rate': 1e-3, 'batch_size': 32, 'patience': 15, 'max_features': 6, 'random_state': 42},
        'ft_transformer': {'n_blocks': 2, 'd_token': 32, 'attention_dropout': 0.2, 'ffn_d_hidden': 64, 'ffn_dropout': 0.1, 'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'tabicl': {'n_estimators': 16, 'embed_dim': 32, 'n_heads': 4, 'row_layers': 2, 'icl_layers': 3, 'learning_rate': 5e-4, 'batch_size': 16, 'max_epochs': 80, 'patience': 15, 'outlier_threshold': 3.5, 'softmax_temperature': 1.0, 'class_shift': True, 'feat_shuffle_method': 'latin', 'norm_methods': ['none', 'standard'], 'random_state': 42, 'verbose': False},
        'tabm': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.2, 'k': 8, 'random_state': 42},
        'mothernet': {'device': 'cuda', 'scale': True, 'random_state': 42},
        'tabr': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'd_main': 32, 'd_multiplier': 2.0, 'encoder_n_blocks': 2, 'predictor_n_blocks': 2, 'context_dropout': 0.1, 'dropout0': 0.15, 'dropout1': 0.15, 'normalization': 'LayerNorm', 'activation': 'ReLU', 'context_size': 32, 'random_state': 42},
        'talent_modernnca': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'dim': 32, 'n_blocks': 2, 'dropout': 0.15, 'temperature': 1.0, 'sample_rate': 0.9, 'random_state': 42},
        'talent_resnet': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_main': 32, 'd_hidden': 64, 'dropout_first': 0.2, 'dropout_second': 0.1, 'random_state': 42},
        'talent_realmlp': {'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'n_blocks': 2, 'd_block': 32, 'dropout': 0.15, 'random_state': 42},
        'node': {'num_trees': 128, 'depth': 4, 'num_layers': 1, 'tree_dim': 2, 'input_dropout': 0.1, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42},
        'gandalf': {'gflu_stages': 3, 'gflu_dropout': 0.1, 'learnable_sparsity': True, 'feature_init_sparsity': 0.3, 'head_hidden': 32, 'head_dropout': 0.15, 'learning_rate': 5e-4, 'batch_size': 128, 'max_epochs': 80, 'patience': 15, 'random_state': 42}
    }

}

# SMOTE configuration
SMOTE_CONFIG = {
    'random_state': 42,
    'sampling_strategy': 'auto',
    'k_neighbors': 5
}

def get_algorithm_config(algorithm_name: str, dataset_name: str) -> dict:
    """Get configuration for specified algorithm and dataset."""
    if dataset_name not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if algorithm_name not in ALGORITHM_CONFIGS[dataset_name]:
        raise ValueError(f"Unknown algorithm: {algorithm_name} for dataset: {dataset_name}")
    return ALGORITHM_CONFIGS[dataset_name][algorithm_name].copy()


def get_all_algorithms_for_dataset(dataset_name: str) -> list:
    """Get all available algorithms (including TALENT models) for a dataset."""
    if dataset_name not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return list(ALGORITHM_CONFIGS[dataset_name].keys())


def get_smote_config() -> dict:
    """Get SMOTE configuration."""
    return SMOTE_CONFIG.copy()
