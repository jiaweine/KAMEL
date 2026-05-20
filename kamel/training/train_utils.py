#!/usr/bin/env python3
"""
Core training utilities for KAMEL.
"""

import os
import random
import logging
import numpy as np
import torch
from typing import Dict, Any

from kamel.models.kamel_model import KAMELModel as KAMELModelClass
from kamel.training.trainer import KAMELTrainer


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_evaluate_model(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    config: Dict[str, Any],
    device: str = 'cuda',
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Unified training and evaluation function.

    Args:
        train_data: Training dataset dictionary
        val_data: Validation dataset dictionary
        test_data: Test dataset dictionary
        config: Unified configuration dictionary
        device: Training device
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing validation and test metrics
    """

    set_random_seeds(random_seed)

    model_config = config['model']
    training_config = config['training']
    imbalance_config = config['imbalance']

    model = KAMELModelClass(
        tabular_dim=train_data['tabular'].shape[1],
        num_classes=config['num_classes'],
        vocab_size=config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        image_size=tuple(config['image_size']),
        max_text_length=config['max_text_length'],
        kan_type=model_config['kan_type'],
        kan_hidden_dims=model_config['kan_hidden_dims'],
        num_unimodal_experts=model_config['num_unimodal_experts'],
        num_crossmodal_experts=model_config['num_crossmodal_experts'],
        num_synergy_experts=model_config['num_synergy_experts'],
        use_hierarchical_moe=model_config['use_hierarchical_moe'],
        classifier_hidden_dims=model_config['classifier_hidden_dims'],
        classifier_dropout=model_config['classifier_dropout'],
        encoder_dropout=model_config['encoder_dropout'],
        moe_dropout=model_config['moe_dropout'],
        use_sparse_routing=model_config.get('use_sparse_routing', True),
        moe_top_k=model_config.get('moe_top_k', 2),
        gate_temperature=model_config.get('gate_temperature', 1.0),
        device=device
    )

    model = model.to(device)

    trainer_config = {
        'total_epochs': training_config['total_epochs'],
        'phase_1_epochs': training_config['phase_1_epochs'],
        'batch_size': training_config['batch_size'],
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config['weight_decay'],
        'evaluation_frequency': training_config.get('evaluation_frequency', 1),
        'gradient_clip_norm': training_config.get('gradient_clip_norm', 1.0),
        'sampling_strategy': imbalance_config['sampling_strategy'],
        'loss_strategy': imbalance_config['loss_strategy'],
        'oversampling_ratio': imbalance_config.get('oversampling_ratio', 1.0),
        'loss_kwargs': imbalance_config.get('loss_kwargs', {}),
        'patience': training_config.get('patience', 10),
        'save_best_only': training_config.get('save_best_only', True)
    }

    trainer = KAMELTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        config=trainer_config
    )

    output_dir = training_config.get('output_dir', './temp_training')
    os.makedirs(output_dir, exist_ok=True)
    trainer.output_dir = output_dir

    try:
        trainer.train()

        best_model_path = os.path.join(trainer.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))

        model.eval()
        val_metrics = trainer.evaluate(trainer.val_loader, 'val')
        test_metrics = trainer.evaluate(trainer.test_loader, 'test')

        val_metrics_clean = {k.replace('val_', ''): v for k, v in val_metrics.items()}
        test_metrics_clean = {k.replace('test_', ''): v for k, v in test_metrics.items()}

        return {
            'best_val_metrics': val_metrics_clean,
            'test_metrics': test_metrics_clean,
            'model_path': best_model_path,
            'training_completed': True
        }

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return {
            'error': str(e),
            'training_completed': False,
            'best_val_metrics': {'f1': 0.0, 'auc': 0.0},
            'test_metrics': {'f1': 0.0, 'auc': 0.0}
        }
