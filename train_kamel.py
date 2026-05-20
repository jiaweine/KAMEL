#!/usr/bin/env python3
"""
Main training script for KAMEL model.
Usage: python train_kamel.py --dataset adult --model_size base --epochs 200
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from typing import Dict, Any

# Add KAMEL to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kamel.models.kamel_model import KAMELModel as KAMELModelClass
from kamel.training.trainer import KAMELTrainer
from kamel.utils.config import create_experiment_config, Config
from kamel.data.preprocessor import TabularDataProcessor
from kamel.data.transforms import MultiModalTransforms
from kamel.training.train_utils import set_random_seeds, train_evaluate_model



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train KAMEL model on tabular data')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['adult', 'creditcard', 'diabetes', 
                               'magic_telescope', 'spambase', 'mushroom', 
                               'nursery', 'car_evaluation',
                               'electrical_grid_stability', 'steel_plates_fault_detection',
                               'blood_transfusion', 'banknote', 'breast_cancer_wisconsin',
                               'haberman', 'heart_disease', 'hepatitis', 'liver_disorders',
                               'mammographic', 'monks1', 'parkinsons', 'vertebral_column',
                               'authorship', 'cardiotocography', 'cmc',
                               'mfeat_fourier', 'mfeat_morphological', 'mfeat_zernike',
                               'solar_flare', 'vehicle', 'wine_quality', 'yeast'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset file (optional)')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base', 'large', 'xlarge'],
                       help='Model size configuration')
    parser.add_argument('--embed_dim', type=int, default=None,
                       help='Embedding dimension (overrides model_size)')
    parser.add_argument('--kan_type', type=str, default='fast_kan',
                       choices=['kan', 'fast_kan', 'cheby_kan', 'wave_kan'],
                       help='Type of KAN layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--phase1_epochs', type=int, default=None,
                       help='Number of epochs for phase 1 (representation learning)')
    
    # Imbalance handling
    parser.add_argument('--sampling_strategy', type=str, default='curriculum',
                       choices=['curriculum', 'balanced', 'smote'],
                       help='Sampling strategy for imbalanced data')
    parser.add_argument('--loss_strategy', type=str, default='ldam',
                       choices=['ldam', 'focal', 'adaptive', 'cross_entropy'],
                       help='Loss function for imbalanced data')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of experiment')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--save_config', action='store_true',
                       help='Save configuration to output directory')
    parser.add_argument('--use_bayesian_optimization', action='store_true',
                       help='Load and apply Bayesian optimization results')
    parser.add_argument('--bayesian_results_dir', type=str, default='./bayesian_optimization_results',
                       help='Directory containing Bayesian optimization results')
    
    return parser.parse_args()


def create_model_from_config(
    config: Config,
    tabular_dim: int,
    num_classes: int,
    vocab_size: int = 10000
) -> KAMELModelClass:
    """
    Create KAMEL model from configuration.
    
    Args:
        config: Configuration object
        tabular_dim: Tabular feature dimension
        num_classes: Number of classes
        vocab_size: Vocabulary size for text
        
    Returns:
        KAMEL model instance
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model = KAMELModelClass(
        tabular_dim=tabular_dim,
        num_classes=num_classes,
        vocab_size=vocab_size,
        embed_dim=model_config.get('embed_dim', 64),
        image_size=tuple(data_config.get('image_size', [32, 32])),
        max_text_length=data_config.get('max_text_length', 512),
        kan_type=model_config.get('kan_type', 'fast_kan'),
        kan_hidden_dims=model_config.get('kan_hidden_dims', [128]),
        num_unimodal_experts=model_config.get('num_unimodal_experts', 3),
        num_crossmodal_experts=model_config.get('num_crossmodal_experts', 3),
        num_synergy_experts=model_config.get('num_synergy_experts', 1),
        use_hierarchical_moe=model_config.get('use_hierarchical_moe', False),
        classifier_hidden_dims=model_config.get('classifier_hidden_dims', [32]),
        classifier_dropout=model_config.get('classifier_dropout', 0.3),
        encoder_dropout=model_config.get('encoder_dropout', 0.1),  # 添加encoder_dropout
        moe_dropout=model_config.get('moe_dropout', 0.1),          # 添加moe_dropout
        device=config.get('experiment.device', 'cuda')
    )
    
    return model


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("KAMEL: KAN-fused Adaptive Mixture-of-Experts for Tabular Learning")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Create or load configuration
    if args.config_file:
        print(f"Loading configuration from {args.config_file}")
        config = Config.from_file(args.config_file)
    else:
        config = create_experiment_config(
            dataset_name=args.dataset,
            model_size=args.model_size,
            experiment_name=args.experiment_name,
            use_bayesian_optimization=args.use_bayesian_optimization,
            bayesian_results_dir=args.bayesian_results_dir
        )
    
    # Override configuration with command line arguments
    if args.embed_dim is not None:
        config.set('model.embed_dim', args.embed_dim)
    if args.batch_size is not None:
        config.set('training.batch_size', args.batch_size)
    if args.learning_rate is not None:
        config.set('training.learning_rate', args.learning_rate)
    
    config.set('training.total_epochs', args.epochs)
    if args.phase1_epochs is not None:
        config.set('training.phase_1_epochs', args.phase1_epochs)
    if args.num_workers is not None:
        config.set('training.num_workers', args.num_workers)
    else:
        # Force single worker for deterministic results
        config.set('training.num_workers', 0)
    config.set('imbalance.sampling_strategy', args.sampling_strategy)
    # Only set loss_strategy if not already configured in dataset-specific config
    if config.get('imbalance.loss_strategy') is None:
        config.set('imbalance.loss_strategy', args.loss_strategy)
    config.set('experiment.device', args.device)
    config.set('experiment.random_seed', args.seed)
    config.set('model.kan_type', args.kan_type)
    
    if args.output_dir:
        config.set('experiment.output_dir', args.output_dir)
    
    # Create output directory
    output_dir = config.get('experiment.output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    if args.save_config or args.config_file is None:
        config_path = os.path.join(output_dir, 'config.yaml')
        config.save(config_path)
        print(f"Configuration saved to {config_path}")
    
    print("\nConfiguration:")
    print("-" * 40)
    for key, value in config.to_dict().items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("-" * 40)
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing {args.dataset} dataset...")
    processor = TabularDataProcessor()
    
    try:
        train_data, val_data, test_data = processor.load_dataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            test_size=0.15,  
            val_size=0.15,  
            random_state=args.seed
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please check that the dataset file exists in the expected location.")
        return 1
    
    # Use fixed vocab_size to match Bayesian optimization (avoids double IGTD fitting)
    # This matches the tokenizer vocab size for distilbert-base-uncased
    actual_vocab_size = 30522
    print(f"Using fixed vocab_size: {actual_vocab_size} (distilbert-base-uncased)")
    
    # Use unified training function for consistency with Bayesian optimization
    print(f"\nPreparing unified training configuration...")
    
    # Prepare unified configuration matching Bayesian optimization format
    unified_config = {
        'model': {
            'embed_dim': config.get('model.embed_dim'),
            'kan_type': config.get('model.kan_type'),
            'kan_hidden_dims': config.get('model.kan_hidden_dims'),
            'num_unimodal_experts': config.get('model.num_unimodal_experts'),
            'num_crossmodal_experts': config.get('model.num_crossmodal_experts'),
            'num_synergy_experts': config.get('model.num_synergy_experts'),
            'use_hierarchical_moe': config.get('model.use_hierarchical_moe'),
            'classifier_hidden_dims': config.get('model.classifier_hidden_dims'),
            'classifier_dropout': config.get('model.classifier_dropout'),
            'encoder_dropout': config.get('model.encoder_dropout'),
            'moe_dropout': config.get('model.moe_dropout'),
            'use_sparse_routing': config.get('model.use_sparse_routing'),
            'moe_top_k': config.get('model.moe_top_k'),
            'gate_temperature': config.get('model.gate_temperature')
        },
        'training': {
            'total_epochs': config.get('training.total_epochs'),
            'phase_1_epochs': config.get('training.phase_1_epochs'),
            'batch_size': config.get('training.batch_size'),
            'learning_rate': config.get('training.learning_rate'),
            'weight_decay': config.get('training.weight_decay'),
            'patience': config.get('training.patience'),
            'save_best_only': config.get('training.save_best_only'),
            'evaluation_frequency': config.get('training.evaluation_frequency'),
            'enable_visualization': config.get('visualization.enable_visualization', True),
            'output_dir': output_dir
        },
        'imbalance': {
            'sampling_strategy': config.get('imbalance.sampling_strategy'),
            'loss_strategy': config.get('imbalance.loss_strategy'),
            'oversampling_ratio': config.get('imbalance.oversampling_ratio', 1.0),
            'loss_kwargs': config.get('imbalance.loss_kwargs', {})
        },
        'vocab_size': actual_vocab_size,
        'num_classes': train_data['num_classes'],
        'image_size': config.get('data.image_size', [32, 32]),
        'max_text_length': config.get('data.max_text_length', 512)
    }
    
    # Resume functionality note - unified function doesn't support resume yet
    if args.resume:
        print(f"Warning: Resume functionality not yet supported with unified training")
    
    # Start training using unified function
    print(f"\nStarting unified training...")
    try:
        results = train_evaluate_model(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=unified_config,
            device=args.device,
            random_seed=args.seed
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Handle different result formats
        if isinstance(results, dict):
            if 'best_val_metrics' in results and 'test_metrics' in results:
                # Full results available
                val_auc = results['best_val_metrics'].get('auc', 0.0)
                test_auc = results['test_metrics'].get('auc', 0.0)
                test_au_prc = results['test_metrics'].get('au_prc', 0.0)
                test_f1 = results['test_metrics'].get('f1', 0.0)
                test_acc = results['test_metrics'].get('accuracy', results['test_metrics'].get('acc', 0.0))
                test_prec = results['test_metrics'].get('precision', results['test_metrics'].get('prec', 0.0))
                test_rec = results['test_metrics'].get('recall', results['test_metrics'].get('rec', 0.0))
                
                print(f"Best validation AUC-ROC: {val_auc:.4f}")
                print(f"\nTest Set Performance:")
                print(f"  AUC-ROC:   {test_auc:.4f}")
                print(f"  AU-PRC:    {test_au_prc:.4f}")
                print(f"  F1-Score:  {test_f1:.4f}")
                print(f"  Accuracy:  {test_acc:.4f}")
                print(f"  Precision: {test_prec:.4f}")
                print(f"  Recall:    {test_rec:.4f}")
            elif 'message' in results:
                # Simplified results
                print(f"Training status: {results['message']}")
            else:
                # Unknown format
                print("Training completed with unknown result format")
        else:
            print("Training completed")
        
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

