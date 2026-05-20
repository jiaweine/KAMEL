"""
Configuration management for KAMEL framework.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for KAMEL experiments."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or self._get_default_config()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (.yaml or .json)
            
        Returns:
            Config instance
        """
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def save(self, save_path: str):
        """
        Save configuration to file.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        elif save_path.suffix.lower() == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported save file format: {save_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]):
        """Update configuration with another dictionary."""
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default KAMEL configuration."""
        return {
            # Model configuration
            'model': {
                'embed_dim': 64,
                'kan_type': 'fast_kan',
                'kan_hidden_dims': [128],
                'num_unimodal_experts': 3,
                'num_crossmodal_experts': 3,
                'num_synergy_experts': 1,
                'use_hierarchical_moe': False,
                'use_sparse_routing': True,
                'moe_top_k': 2,
                'gate_temperature': 1.0,
                'classifier_hidden_dims': [32],
                'classifier_dropout': 0.3,
                'encoder_dropout': 0.3,
                'moe_dropout': 0.3,
            },
            
            # Data configuration
            'data': {
                'image_size': [32, 32],
                'max_text_length': 512,
                'tokenizer_name': 'distilbert-base-uncased',
                'train_val_test_split': [0.7, 0.15, 0.15]
            },
            
            # Training configuration
            'training': {
                'total_epochs': 10,
                'phase_1_epochs': 40,
                'batch_size': 64,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'num_workers': 0,
                'patience': 8,
                'evaluation_frequency': 1,
                'save_best_only': True,
                'gradient_clip_norm': 0.5,
                'min_delta': 1e-4,
                'warmup_epochs': 2,
            },
            
            # Imbalance handling
            'imbalance': {
                'sampling_strategy': 'curriculum',
                'loss_strategy': 'focal',
                'oversampling_ratio': 1.0,
                'loss_kwargs': {
                    'gamma': 2.0,
                    'alpha': 0.25
                }
            },
            
            # Experiment configuration
            'experiment': {
                'name': 'kamel_experiment',
                'output_dir': './kamel_outputs',
                'random_seed': 42,
                'device': 'cuda'
            },
            
            # Logging configuration
            'logging': {
                'use_tensorboard': True,
                'log_frequency': 100,
                'save_visualizations': True
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
    
    def __getitem__(self, key: str):
        """Dictionary-like access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-like setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def load_bayesian_optimization_results(self, dataset_name: str, results_dir: str = './bayesian_optimization_results'):
        """
        Load and apply best parameters from Bayesian optimization results.
        
        Args:
            dataset_name: Name of dataset
            results_dir: Directory containing optimization results
        """
        results_path = os.path.join(results_dir, f'{dataset_name}_best_params.json')
        
        if not os.path.exists(results_path):
            print(f"Warning: No Bayesian optimization results found for {dataset_name} at {results_path}")
            print("Using default configuration parameters.")
            return
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            best_params = results.get('best_params', {})
            best_val_f1 = results.get('best_val_f1', 0.0)
            
            print(f"Loading Bayesian optimization results for {dataset_name}")
            print(f"Best validation F1: {best_val_f1:.4f}")
            
            # Apply best parameters to configuration
            self._apply_optimization_params(best_params)
            
            print("Bayesian optimization parameters applied successfully!")
            
        except Exception as e:
            print(f"Error loading Bayesian optimization results: {e}")
            print("Using default configuration parameters.")
    
    def _apply_optimization_params(self, best_params: Dict[str, Any]):
        """Apply best parameters from optimization to configuration."""
        
        # Update model parameters
        if 'embed_dim' in best_params:
            self.set('model.embed_dim', best_params['embed_dim'])
        if 'kan_hidden_dims' in best_params:
            self.set('model.kan_hidden_dims', best_params['kan_hidden_dims'])
        if 'classifier_hidden_dims' in best_params:
            self.set('model.classifier_hidden_dims', best_params['classifier_hidden_dims'])
        if 'classifier_dropout' in best_params:
            self.set('model.classifier_dropout', best_params['classifier_dropout'])
        if 'encoder_dropout' in best_params:
            self.set('model.encoder_dropout', best_params['encoder_dropout'])
        if 'moe_dropout' in best_params:
            self.set('model.moe_dropout', best_params['moe_dropout'])
        
        # Update training parameters
        if 'batch_size' in best_params:
            self.set('training.batch_size', best_params['batch_size'])
        if 'learning_rate' in best_params:
            self.set('training.learning_rate', best_params['learning_rate'])
        if 'weight_decay' in best_params:
            self.set('training.weight_decay', best_params['weight_decay'])
        
        # Update imbalance handling parameters
        if 'oversampling_ratio' in best_params:
            self.set('imbalance.oversampling_ratio', best_params['oversampling_ratio'])
        if 'focal_gamma' in best_params and 'focal_alpha' in best_params:
            self.set('imbalance.loss_kwargs.gamma', best_params['focal_gamma'])
            self.set('imbalance.loss_kwargs.alpha', best_params['focal_alpha'])
    
    def save_with_optimization_results(self, save_path: str, dataset_name: str, results_dir: str = './bayesian_optimization_results'):
        """
        Save configuration with Bayesian optimization results applied.
        
        Args:
            save_path: Path to save optimized configuration
            dataset_name: Name of dataset
            results_dir: Directory containing optimization results
        """
        # Load and apply optimization results
        self.load_bayesian_optimization_results(dataset_name, results_dir)
        
        # Save the updated configuration
        self.save(save_path)
        
        print(f"Optimized configuration saved to: {save_path}")


def create_experiment_config(
    dataset_name: str,
    model_size: str = 'base',
    experiment_name: Optional[str] = None,
    use_bayesian_optimization: bool = False,
    bayesian_results_dir: str = './bayesian_optimization_results'
) -> Config:
    """
    Create experiment configuration for specific dataset.
    
    Args:
        dataset_name: Name of dataset
        model_size: Model size ('small', 'base', 'large')
        experiment_name: Name of experiment
        use_bayesian_optimization: Whether to load and apply Bayesian optimization results
        bayesian_results_dir: Directory containing Bayesian optimization results
        
    Returns:
        Config instance
    """
    config = Config()
    
    # Dataset-specific configurations
    dataset_configs = {
        'adult': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],    
            'model.classifier_hidden_dims': [32],
            'model.classifier_dropout': 0.10387525324236296,
            'model.encoder_dropout': 0.3,
            'model.moe_dropout': 0.26089104674315544,
            'training.batch_size': 64,
            'training.learning_rate': 0.002552088694524925,
            'training.weight_decay': 6.54180018194517e-05,
            'training.phase_1_epochs': 80,
            'training.total_epochs': 50,
            'training.patience': 8,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 1.8996391471230458,
            'imbalance.loss_kwargs': {
                'gamma': 2.4947619961986782,
                'alpha': 0.7562580965200475
            }
        },
        'creditcard': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],
            'model.classifier_hidden_dims': [32],
            'model.classifier_dropout': 0.45,  
            'model.encoder_dropout': 0.3,
            'model.moe_dropout': 0.18,
            'training.batch_size': 128,
            'training.learning_rate': 0.00155,
            'training.weight_decay': 0.00015,  
            'training.phase_1_epochs': 85,
            'training.patience': 12,  
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 2.5,  
            'imbalance.loss_kwargs': {
                'gamma': 2.0,  
                'alpha': 0.82  
            }
        },
        'diabetes': {
            'model.embed_dim': 96,
            'model.kan_hidden_dims': [256],
            'model.classifier_hidden_dims': [32],
            'model.classifier_dropout': 0.03576074924700969,
            'model.encoder_dropout': 0.14430665407439408,
            'model.moe_dropout': 0.1473268979320396,
            'model.use_sparse_routing': True,
            'model.moe_top_k': 2,
            'model.gate_temperature': 1.0,
            'training.batch_size': 64,
            'training.learning_rate': 0.0004871313247363778,
            'training.weight_decay': 0.0007785403481707472,
            'training.total_epochs': 10,
            'training.phase_1_epochs': 10,
            'training.patience': 8,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 0.6774769651535074,
            'imbalance.loss_kwargs': {
                'gamma': 1.6475473395969793,
                'alpha': 0.6964065250351908
            }
        },
        'spambase': {
            # Bayesian optimization best test-based parameters (Test F1: 0.9503)
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],
            'model.classifier_hidden_dims': [32], 
            'model.classifier_dropout': 0.1548470851823389,     
            'model.encoder_dropout': 0.141502582160994,        
            'model.moe_dropout': 0.15300058854932355,            
            'training.batch_size': 64,            
            'training.learning_rate': 0.0013420791745248036,     
            'training.weight_decay': 0.0002209551659526853,     
            'training.phase_1_epochs': 70,
            'training.patience': 6,             
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 2.9101527371831075,   
            'imbalance.loss_kwargs': {
                'gamma': 1.5264608232943773,     
                'alpha': 0.37655535428845394      
            }
        },
        'magic_telescope': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],
            'model.classifier_hidden_dims': [32],
            'model.classifier_dropout': 0.19990452966831734,
            'model.encoder_dropout': 0.1070555985120384,
            'model.moe_dropout': 0.3750859208569214,
            'training.batch_size': 256,
            'training.learning_rate': 0.0005419191554140214,
            'training.weight_decay': 0.0007213434320493333,
            'training.phase_1_epochs': 70,
            'training.patience': 12,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 1.2069988755815175,
            'imbalance.loss_kwargs': {
                'gamma': 1.436061935909855,
                'alpha': 0.6016096415569046
            }
        },
        'mushroom': {
            # Bayesian optimization - Test F1: 1.0 (Perfect!) - Trial 4 (best_test_based)
            'model.embed_dim': 48,
            'model.kan_hidden_dims': [128, 64],
            'model.classifier_hidden_dims': [48],
            'model.classifier_dropout': 0.06286679922671917,
            'model.encoder_dropout': 0.10942875570602029,
            'model.moe_dropout': 0.25820006039287435,
            'training.batch_size': 128,
            'training.learning_rate': 0.001412826646609442,
            'training.weight_decay': 0.00012811830949395153,
            'training.phase_1_epochs': 80,
            'training.patience': 15,
            'imbalance.sampling_strategy': 'curriculum', 
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 1.7555511385430487,
            'imbalance.loss_kwargs': {
                'gamma': 1.6863944964748674,
                'alpha': 0.1615839278630344
            }
        },
        'nursery': {
            # Bayesian optimization - Test F1: 1.0 (Perfect!) - Trial 61
            'model.embed_dim': 96,
            'model.kan_hidden_dims': [64],
            'model.classifier_hidden_dims': [48],
            'model.classifier_dropout': 0.3114196785893291,
            'model.encoder_dropout': 0.10913614039444763,
            'model.moe_dropout': 0.3315990207920251,
            'model.use_sparse_routing': True,
            'model.moe_top_k': 2,
            'model.gate_temperature': 1.0,
            'training.batch_size': 128,
            'training.learning_rate': 0.00042474647358484207,
            'training.weight_decay': 0.00011171922301528475,
            'training.total_epochs': 10,
            'training.phase_1_epochs': 10,
            'training.patience': 8,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 2.005266615865987,
            'imbalance.loss_kwargs': {
                'gamma': 3.037049053057849,
                'alpha': 0.6926562288953855
            }
        },
        'car_evaluation': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [256],
            'model.classifier_hidden_dims': [64],
            'model.classifier_dropout': 0.07493575101699558,
            'model.encoder_dropout': 0.13180794764613904,
            'model.moe_dropout': 0.046423396260332116,
            'training.batch_size': 64,
            'training.learning_rate': 0.00025695034049104534,
            'training.weight_decay': 7.779473813659261e-05,
            'training.phase_1_epochs': 80,
            'training.patience': 15,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 1.9877598256316897,
            'imbalance.loss_kwargs': {
                'gamma': 1.5802954612945985,
                'alpha': 0.4651482247195954
            }
        },
        'electrical_grid_stability': {
            'model.embed_dim': 96,
            'model.kan_hidden_dims': [256],
            'model.classifier_hidden_dims': [128],
            'model.classifier_dropout': 0.021832319115328867,
            'model.encoder_dropout': 0.10860097080343298,
            'model.moe_dropout': 0.13101039933919012,
            'training.batch_size': 128,
            'training.learning_rate': 0.0002605528213509483,
            'training.weight_decay': 0.00015996685388853247,
            'training.phase_1_epochs': 100,
            'training.patience': 35,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 1.5158067342634736,
            'imbalance.loss_kwargs': {
                'gamma': 3.09001192734027,
                'alpha': 0.34915741501007513
            }
        },
        'steel_plates_fault_detection': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],
            'model.classifier_hidden_dims': [48],
            'model.classifier_dropout': 0.272,
            'model.encoder_dropout': 0.3,
            'model.moe_dropout': 0.13,
            'training.batch_size': 64,
            'training.learning_rate': 0.00145,
            'training.weight_decay': 0.000231,
            'training.phase_1_epochs': 30,
            'training.patience': 10,
            'imbalance.sampling_strategy': 'curriculum',
            'imbalance.loss_strategy': 'focal',
            'imbalance.oversampling_ratio': 6.83,
            'imbalance.loss_kwargs': {
                'gamma': 3.25,
                'alpha': 0.219
            }
        },
    }
    
    # Model size configurations - expanded to support wider parameter ranges
    size_configs = {
        'small': {
            'model.embed_dim': 32,
            'model.kan_hidden_dims': [64],
            'model.num_unimodal_experts': 2,
            'model.num_crossmodal_experts': 2,
            'model.classifier_hidden_dims': [16]
        },
        'base': {
            'model.embed_dim': 64,
            'model.kan_hidden_dims': [128],
            'model.num_unimodal_experts': 3,
            'model.num_crossmodal_experts': 3,
            'model.classifier_hidden_dims': [32]
        },
        'large': {
            'model.embed_dim': 128,
            'model.kan_hidden_dims': [256, 128],
            'model.num_unimodal_experts': 4,
            'model.num_crossmodal_experts': 4,
            'model.classifier_hidden_dims': [64, 32]
        },
        'xlarge': {
            'model.embed_dim': 192,
            'model.kan_hidden_dims': [256, 128],
            'model.num_unimodal_experts': 4,
            'model.num_crossmodal_experts': 4,
            'model.classifier_hidden_dims': [96, 48]
        }
    }
    
    # Apply model size configuration first (lower priority)
    if model_size in size_configs:
        for key, value in size_configs[model_size].items():
            config.set(key, value)
    
    # Apply dataset-specific configuration last (higher priority)
    if dataset_name in dataset_configs:
        for key, value in dataset_configs[dataset_name].items():
            config.set(key, value)
    
    # Set experiment name
    if experiment_name is None:
        experiment_name = f'kamel_{dataset_name}_{model_size}'
    
    config.set('experiment.name', experiment_name)
    config.set('experiment.output_dir', f'./outputs/{experiment_name}')
    
    # Load and apply Bayesian optimization results if requested
    if use_bayesian_optimization:
        config.load_bayesian_optimization_results(dataset_name, bayesian_results_dir)
    
    return config
