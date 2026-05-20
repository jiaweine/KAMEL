"""
Dual-Path Tabular Data Preprocessor for KAMEL Framework.

Implements the core principle: "First Clean, Then Bifurcate"
- Step 1: Universal Cleaning (handle missing values, data types, outliers)  
- Step 2: Dual-Path Bifurcation:
  * Path A (Numerical): Convert all features to pure numerical format
  * Path B (Sequential): Generate descriptive text using cleaned raw data
"""

import os
import pandas as pd
import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, QuantileTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from .dataset_configs import get_dataset_config


class TabularDataProcessor:
    """
    Unified tabular data processor implementing dual-path preprocessing.
    """
    
    def __init__(self, config: Optional[Dict] = None, cache_dir: Optional[str] = None):
        self.config = config
        self.dataset_name = None
        self.feature_names = None
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.label_encoder = None
        self.is_fitted = False
        self.cache_dir = cache_dir or "./cache/data_splits"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_split_cache_key(self, dataset_name: str, test_size: float, val_size: float, random_state: int, data_shape: Optional[Tuple] = None) -> str:
        """Generate a cache key for data split configuration."""
        if data_shape:
            key_str = f"{dataset_name}_{test_size}_{val_size}_{random_state}_{data_shape[0]}_{data_shape[1]}"
        else:
            key_str = f"{dataset_name}_{test_size}_{val_size}_{random_state}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _save_split_indices(self, cache_key: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        """Save data split indices to cache."""
        cache_file = os.path.join(self.cache_dir, f"split_{cache_key}.json")
        split_data = {
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(), 
            'test_indices': test_idx.tolist()
        }
        with open(cache_file, 'w') as f:
            json.dump(split_data, f)
    
    def _load_split_indices(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load data split indices from cache if available."""
        cache_file = os.path.join(self.cache_dir, f"split_{cache_key}.json")
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                split_data = json.load(f)
            
            train_idx = np.array(split_data['train_indices'])
            val_idx = np.array(split_data['val_indices'])
            test_idx = np.array(split_data['test_indices'])
            
            return train_idx, val_idx, test_idx
        except Exception as e:
            print(f"Warning: Failed to load cached split indices: {e}")
            return None
    
    def load_dataset(
        self, 
        dataset_name: str,
        data_path: Optional[str] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        use_cache: bool = True
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Load and preprocess dataset with dual-path processing.
        
        Args:
            dataset_name: Name of dataset from configs
            data_path: Path to data file (optional, uses default if None)
            test_size: Test set proportion  
            val_size: Validation set proportion
            random_state: Random seed
            use_cache: Whether to use cached data splits (default True)
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        self.dataset_name = dataset_name
        self.config = get_dataset_config(dataset_name)
        
        # Load raw data
        if data_path is None:
            data_path = self._get_default_data_path(dataset_name)
        
        df = self._load_data_file(data_path)
        print(f"Loaded {dataset_name} dataset: {df.shape}")
        
        # Special preprocessing for specific datasets
        df = self._handle_special_datasets(df, dataset_name)
        
        # Step 1: Universal Cleaning
        df_cleaned = self._universal_cleaning(df)
        print(f"After cleaning: {df_cleaned.shape}")
        
        # Split data using cached indices for consistency
        X = df_cleaned.drop(columns=[self.config['target_col']])
        y = df_cleaned[self.config['target_col']]
        
        # Check for cached split indices with data shape validation
        cache_key = self._get_split_cache_key(dataset_name, test_size, val_size, random_state, X.shape)
        cached_indices = self._load_split_indices(cache_key) if use_cache else None
        
        if cached_indices is not None and use_cache:
            train_idx, val_idx, test_idx = cached_indices
            
            # Validate cached indices against current data size
            max_idx = max(np.max(train_idx) if len(train_idx) > 0 else -1,
                         np.max(val_idx) if len(val_idx) > 0 else -1, 
                         np.max(test_idx) if len(test_idx) > 0 else -1)
            
            if max_idx >= len(X):
                print(f"Warning: Cached indices are out-of-bounds (max_idx={max_idx}, data_len={len(X)})")
                print(f"Regenerating data split...")
                cached_indices = None
            else:
                print(f"Using cached data split for reproducibility (seed={random_state})")
                # Use cached indices to split data
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]  
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        
        if cached_indices is None:
            print(f"Creating new data split and caching indices")
            # Stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
            
            val_size_adj = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adj, random_state=random_state,
                stratify=y_temp
            )
            
            # Cache the indices for future reproducibility
            train_idx = X_train.index.values
            val_idx = X_val.index.values
            test_idx = X_test.index.values
            self._save_split_indices(cache_key, train_idx, val_idx, test_idx)
        
        print(f"Split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Fit preprocessing on training data
        self._fit_preprocessing(X_train, y_train)
        
        # Step 2: Dual-Path Bifurcation
        train_data = self._dual_path_processing(X_train, y_train, "train")
        val_data = self._dual_path_processing(X_val, y_val, "val") 
        test_data = self._dual_path_processing(X_test, y_test, "test")
        
        # Add dataset name to all data splits for cache file naming
        train_data['dataset_name'] = dataset_name
        val_data['dataset_name'] = dataset_name
        test_data['dataset_name'] = dataset_name
        
        self.is_fitted = True
        
        return train_data, val_data, test_data
    
    def _get_default_data_path(self, dataset_name: str) -> str:
        """Get default data file path."""
        # Get the current directory and find the data directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(current_dir, "..", "data")
        if not os.path.exists(data_dir):
            # Alternative path for different project structures
            data_dir = os.path.join(current_dir, "data")
        if not os.path.exists(data_dir):
            # Try relative path from KAMEL directory
            data_dir = "../data"
        
        file_mapping = {
            # Original datasets
            "adult": "adult.csv",
            "creditcard": "creditcard.csv", 
            "creditdefault": "default_of_credit_card_clients.csv",
            "diabetes": "diabetes.csv",
            # New datasets
            "spambase": "spambase.csv",
            "magic_telescope": "magic_telescope.csv",
            "mushroom": "mushroom.csv",
            "nursery": "nursery_filtered.csv",
            "car_evaluation": "car_evaluation_filtered.csv",
            # New real-world datasets
            # Fault detection datasets
            "electrical_grid_stability": "electrical_grid_stability.csv",
            "steel_plates_fault_detection": "steel_plates_fault_detection.csv",
            # Medical datasets
            "breast_cancer_wisconsin": "breast_cancer_wisconsin.csv",
            "heart_disease": "heart_disease.csv",
            # Signal processing datasets  
            "ionosphere": "ionosphere.csv",
            "sonar": "sonar.csv",
            # Financial datasets
            "australian_credit": "australian_credit.csv",
            # Additional medical/survival datasets
            "haberman": "haberman.csv",
            "liver_disorders": "liver_disorders.csv",
            # Logic/game datasets
            "monks1": "monks1.csv",
            "tictactoe": "tictactoe.csv",
            # Multi-class classification datasets
            "iris": "iris.csv",
            "wine": "wine.csv",
            "glass": "glass.csv",
            "ecoli": "ecoli.csv",
            "seeds": "seeds.csv",
            # 10 new UCI binary classification datasets
            "banknote": "banknote.csv",
            "german_credit": "german_credit.csv",
            "hepatitis": "hepatitis.csv",
            "parkinsons": "parkinsons.csv",
            "blood_transfusion": "blood_transfusion.csv",
            "fertility": "fertility.csv",
            "mammographic": "mammographic.csv",
            "qsar_biodegradation": "qsar_biodegradation.csv",
            "seismic_bumps": "seismic_bumps.csv",
            "vertebral_column": "vertebral_column.csv",
            "vehicle": "vehicle.csv",
            "authorship": "authorship.csv",
            "mfeat_morphological": "mfeat_morphological.csv",
            "cmc": "cmc.csv",
            "solar_flare": "solar_flare.csv",
            "cardiotocography": "cardiotocography.csv",
            "mfeat_fourier": "mfeat_fourier.csv",
            "mfeat_zernike": "mfeat_zernike.csv",
            "wine_quality": "wine_quality.csv",
            "yeast": "yeast.csv"
        }
        
        if dataset_name not in file_mapping:
            raise ValueError(f"No default path for dataset: {dataset_name}")
            
        return os.path.join(data_dir, file_mapping[dataset_name])
    
    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load data file with appropriate encoding."""
        try:
            if file_path.endswith('.csv'):
                # Determine separator and header based on dataset
                sep = ','
                header = 'infer'
                
                # Special handling for bank marketing dataset
                if 'bank-additional' in file_path:
                    sep = ';'
                    header = 0  # Has headers
                # For binary datasets without headers
                elif any(dataset in file_path for dataset in ['spambase', 'magic_telescope', 'mushroom', 
                                                             'nursery', 'car_evaluation']):
                    header = None  # No headers
                
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, header=header)
                        
                        # For datasets without headers, ensure proper column naming
                        if header is None:
                            df.columns = range(len(df.columns))
                            
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"Could not read file with any encoding: {file_path}")
            elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                # Handle Excel files
                try:
                    # For credit clients data, skip first row which is metadata
                    if 'credit card clients' in file_path.lower():
                        df = pd.read_excel(file_path, header=1)
                    else:
                        df = pd.read_excel(file_path)
                    return df
                except Exception as e:
                    raise ValueError(f"Error reading Excel file {file_path}: {str(e)}")
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")
    
    def _handle_special_datasets(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Handle special formatting for specific datasets."""
        df = df.copy()

        if dataset_name == 'solar_flare':
            df['target'] = (df['common flares'].astype(int) > 0).astype(int)

        return df
    
    def _universal_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Universal cleaning operations."""
        df = df.copy()
        
        # Drop specified columns
        if self.config and 'drop_cols' in self.config and self.config['drop_cols']:
            # Handle both named and numeric column indices
            cols_to_drop = []
            for col in self.config['drop_cols']:
                if isinstance(col, int) and col < len(df.columns):
                    cols_to_drop.append(df.columns[col])
                elif col in df.columns:
                    cols_to_drop.append(col)
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Handle missing value indicators
        if 'missing_values' in self.config and self.config['missing_values']:
            for missing_val in self.config['missing_values']:
                df = df.replace(missing_val, np.nan)
        
        # Data type conversions
        cleaning_rules = self.config.get('cleaning_rules', {})
        
        if 'data_type_conversion' in cleaning_rules:
            for col, dtype in cleaning_rules['data_type_conversion'].items():
                # Handle both named and numeric column indices
                actual_col = col
                if isinstance(col, int) and col < len(df.columns):
                    actual_col = df.columns[col]
                    
                if actual_col in df.columns:
                    if dtype == 'numeric':
                        df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
        
        # Handle zeros as missing values (e.g., diabetes dataset)
        if 'zero_as_missing' in cleaning_rules:
            for col in cleaning_rules['zero_as_missing']:
                # Handle both named and numeric column indices
                actual_col = col
                if isinstance(col, int) and col < len(df.columns):
                    actual_col = df.columns[col]
                    
                if actual_col in df.columns:
                    df[actual_col] = df[actual_col].replace(0, np.nan)
        
        # Remove duplicates
        if cleaning_rules.get('duplicate_removal', False):
            df = df.drop_duplicates()
        
        # Handle categorical value mappings
        if 'categorical_mapping' in cleaning_rules:
            for col, mapping in cleaning_rules['categorical_mapping'].items():
                if col in df.columns:
                    df[col] = df[col].replace(mapping)
        
        # Payment status merging (credit default dataset) - before string conversion
        if 'payment_status_merge' in cleaning_rules:
            pmt_config = cleaning_rules['payment_status_merge']
            for col in pmt_config['columns']:
                if col in df.columns:
                    df[col] = df[col].replace(pmt_config['merge_values'])
        
        # Convert columns to string for categorical encoding - MUST be after all numeric operations
        if 'convert_to_string' in cleaning_rules:
            for col in cleaning_rules['convert_to_string']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        # Handle target missing values (if specified)
        if 'handle_target_missing' in cleaning_rules and self.config and 'target_col' in self.config:
            target_col = self.config['target_col']
            if target_col in df.columns:
                if cleaning_rules['handle_target_missing'] == 'impute':
                    # For categorical targets, use mode imputation
                    if df[target_col].isnull().sum() > 0:
                        mode_value = df[target_col].mode()[0]
                        df[target_col] = df[target_col].fillna(mode_value)
                        print(f"Imputed {df[target_col].isnull().sum()} target missing values with mode: {mode_value}")
        
        # Apply target mapping if specified (for multi-class to binary conversion)
        if 'target_mapping' in cleaning_rules and self.config and 'target_col' in self.config:
            target_col = self.config['target_col']
            if target_col in df.columns:
                target_mapping = cleaning_rules['target_mapping']
                original_len = len(df)
                # Apply mapping
                df[target_col] = df[target_col].map(target_mapping)
                # Remove rows with unmapped target values (NaN)
                df = df.dropna(subset=[target_col])
                if len(df) < original_len:
                    print(f"Applied target mapping, dropped {original_len - len(df)} unmapped samples")
                print(f"Target mapping applied: {len(set(df[target_col]))} unique classes")
        
        # Filter categorical values
        if 'categorical_filtering' in cleaning_rules:
            for col, filters in cleaning_rules['categorical_filtering'].items():
                if col in df.columns and 'remove' in filters:
                    df = df[~df[col].isin(filters['remove'])]
        
        return df
    
    def _fit_preprocessing(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit preprocessing transformers on training data."""
        cleaning_rules = self.config.get('cleaning_rules', {})
        
        # Handle missing values
        if 'missing_strategy' in cleaning_rules:
            strategy = cleaning_rules['missing_strategy']
            
            # Numerical missing values
            if 'numerical' in strategy:
                num_cols = [col for col in self.config.get('numerical_cols', []) 
                          if col in X_train.columns]
                
                if strategy['numerical'] == 'median':
                    self.numerical_imputer = SimpleImputer(strategy='median')
                    self.numerical_imputer.fit(X_train[num_cols])
                elif strategy['numerical'] == 'zero_fill':
                    # Simple zero filling for problematic datasets
                    pass  # Handle during transform
                elif strategy['numerical'] == 'knn':
                    # Use specified features for KNN
                    knn_features = cleaning_rules.get('knn_features', num_cols)
                    knn_features = [f for f in knn_features if f in X_train.columns]
                    self.knn_imputer = KNNImputer(n_neighbors=5)
                    # Create numeric version of categorical features for KNN
                    X_knn = X_train[knn_features].copy()
                    for col in X_knn.columns:
                        if X_knn[col].dtype == 'object':
                            le = LabelEncoder()
                            X_knn[col] = le.fit_transform(X_knn[col].astype(str))
                    self.knn_imputer.fit(X_knn)
                elif strategy['numerical'] == 'class_median':
                    # Group by target class for median imputation
                    self.class_medians = {}
                    num_cols = [col for col in self.config.get('numerical_cols', [])
                              if col in X_train.columns]
                    for col in num_cols:
                        self.class_medians[col] = X_train.groupby(y_train)[col].median().to_dict()
        
        # Fit label encoder for target
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train)
        
        # Fit Path A transformers
        self._fit_path_a_transformers(X_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
    
    def _fit_path_a_transformers(self, X_train: pd.DataFrame):
        """Fit Path A (numerical) transformers."""
        path_a_config = self.config.get('path_a_transforms', {})
        
        # Categorical encoding
        if path_a_config.get('categorical_encoding') == 'onehot':
            cat_cols = [col for col in self.config.get('categorical_cols', [])
                       if col in X_train.columns]
            if cat_cols:
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.onehot_encoder.fit(X_train[cat_cols])
        
        # Numerical scaling - Handle both named and numeric column indices
        num_cols = []
        for col in self.config.get('numerical_cols', []):
            if isinstance(col, int) and col < len(X_train.columns):
                actual_col = X_train.columns[col]
                if actual_col in X_train.columns:
                    num_cols.append(actual_col)
            elif col in X_train.columns:
                num_cols.append(col)
        
        if num_cols:
            scaling_method = path_a_config.get('numerical_scaling', 'standard')
            
            if scaling_method == 'standard':
                self.numerical_scaler = StandardScaler()
                self.numerical_scaler.fit(X_train[num_cols])
            elif scaling_method == 'quantile':
                self.numerical_scaler = QuantileTransformer(
                    output_distribution='uniform',
                    n_quantiles=min(1000, len(X_train))
                )
                self.numerical_scaler.fit(X_train[num_cols])
    
    def _dual_path_processing(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        split: str
    ) -> Dict[str, Any]:
        """Apply dual-path processing to generate multimodal data."""
        
        # Handle missing values first
        X_clean = self._handle_missing_values(X, y)
        
        # Path A: Numerical transformation
        X_numerical = self._path_a_numerical(X_clean)
        
        # Path B: Text sequentialization  
        X_text = self._path_b_sequential(X_clean)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Calculate class distribution
        unique_labels, counts = np.unique(y_encoded, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        
        return {
            'tabular': X_numerical,
            'text': X_text,
            'labels': y_encoded,
            'raw_features': X_clean,
            'feature_names': self.feature_names,
            'class_distribution': class_distribution,
            'split': split,
            'dataset': self.dataset_name,
            'num_samples': len(X),
            'num_features': X_numerical.shape[1] if len(X_numerical.shape) > 1 else 1,
            'num_classes': len(unique_labels)
        }
    
    def _handle_missing_values(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Handle missing values according to dataset configuration."""
        X = X.copy()
        cleaning_rules = self.config.get('cleaning_rules', {})
        
        if 'missing_strategy' not in cleaning_rules:
            return X
        
        strategy = cleaning_rules['missing_strategy']
        
        # Handle numerical missing values
        if 'numerical' in strategy:
            num_cols = [col for col in self.config.get('numerical_cols', [])
                       if col in X.columns]
            
            if strategy['numerical'] == 'median':
                X[num_cols] = self.numerical_imputer.transform(X[num_cols])
            elif strategy['numerical'] == 'class_median':
                # Use pre-computed class medians
                for col in num_cols:
                    if col in self.class_medians:
                        for class_label in self.class_medians[col]:
                            mask = (y == class_label) & X[col].isna()
                            X.loc[mask, col] = self.class_medians[col][class_label]
                        # Fill remaining with overall median
                        X[col].fillna(X[col].median(), inplace=True)
            elif strategy['numerical'] == 'knn':
                # KNN imputation (simplified)
                for col in num_cols:
                    if X[col].isna().any():
                        X[col].fillna(X[col].median(), inplace=True)
        
        # Handle categorical missing values
        if 'categorical' in strategy:
            cat_cols = [col for col in self.config.get('categorical_cols', [])
                       if col in X.columns]
            
            if strategy['categorical'] == 'mode':
                for col in cat_cols:
                    mode_val = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
                    X[col].fillna(mode_val, inplace=True)
        
        return X
    
    def _path_a_numerical(self, X: pd.DataFrame) -> np.ndarray:
        """Path A: Convert to pure numerical representation."""
        path_a_config = self.config.get('path_a_transforms', {})
        features = []
        
        # Process categorical features
        cat_cols = [col for col in self.config.get('categorical_cols', [])
                   if col in X.columns]
        
        if cat_cols and path_a_config.get('categorical_encoding') == 'onehot':
            cat_encoded = self.onehot_encoder.transform(X[cat_cols])
            features.append(cat_encoded)
        
        # Process numerical features
        num_cols = [col for col in self.config.get('numerical_cols', [])
                   if col in X.columns]
        
        if num_cols:
            num_scaled = self.numerical_scaler.transform(X[num_cols])
            features.append(num_scaled)
        
        # Combine all features
        if features:
            return np.concatenate(features, axis=1)
        else:
            return X.values
    
    def _path_b_sequential(self, X: pd.DataFrame) -> List[str]:
        """Path B: Generate descriptive text sequences with robust formatting."""
        template = self.config.get('path_b_text_template', "{feature_name} is {feature_value}")
        
        # Use a safe template without format specifiers to avoid errors
        safe_template = "{feature_name} is {feature_value}"
        
        text_sequences = []
        # Convert to numpy array to avoid pandas iterrows() issues
        X_values = X.values
        columns = X.columns.tolist()
        
        for row_idx in range(len(X_values)):
            features_text = []
            for col_idx, col in enumerate(columns):
                value = X_values[row_idx, col_idx]
                # Skip NaN/None values
                if pd.isna(value) or value is None:
                    continue
                
                # Safe formatting for all value types
                try:
                    if pd.isna(value) or value is None:
                        formatted_value = "missing"
                    elif isinstance(value, (int, np.integer)):
                        formatted_value = str(int(value))
                    elif isinstance(value, (float, np.floating)):
                        if np.isinf(value):
                            formatted_value = "infinity" if value > 0 else "negative_infinity"
                        elif np.isnan(value):
                            formatted_value = "missing"
                        else:
                            # Use 2 decimal places for float values
                            formatted_value = f"{float(value):.2f}"
                    else:
                        # For any other type, convert to string
                        formatted_value = str(value)
                    
                    # Always use safe template to avoid format errors
                    # Handle both string and integer column names
                    col_name = str(col).replace('_', ' ') if isinstance(col, str) else f"feature_{col}"
                    feature_text = safe_template.format(
                        feature_name=col_name,
                        feature_value=formatted_value
                    )
                    features_text.append(feature_text)
                
                except Exception as e:
                    # Ultimate fallback for any formatting errors
                    try:
                        safe_value = str(value) if value is not None else "missing"
                        # Handle both string and integer column names
                        col_name = str(col).replace('_', ' ') if isinstance(col, str) else f"feature_{col}"
                        feature_text = f"{col_name} is {safe_value}"
                        features_text.append(feature_text)
                    except:
                        # Final fallback
                        # Handle both string and integer column names
                        col_name = str(col).replace('_', ' ') if isinstance(col, str) else f"feature_{col}"
                        feature_text = f"{col_name} is unknown"
                        features_text.append(feature_text)
            
            text_sequences.append("; ".join(features_text))
        
        return text_sequences
