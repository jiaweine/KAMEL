#!/usr/bin/env python3
"""
Complete training framework for KAMEL model.
Handles multimodal data loading, imbalanced training, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from ..data.transforms import MultiModalTransforms
from .losses import get_loss_function
from .imbalance_handler import ImbalanceAwareTrainer, DynamicCurriculumSampler
from ..utils.metrics import MetricsCalculator
from ..utils.visualization import VisualizationUtils


class MultiModalDataset(data.Dataset):
    """Dataset class for multimodal tabular data."""
    
    def __init__(self, data_dict: Dict[str, Any], transforms: Optional[MultiModalTransforms] = None, mode: str = 'train'):
        self.data_dict = data_dict
        self.transforms = transforms
        self.mode = mode
        
        # Extract data components
        self.tabular_data = torch.tensor(data_dict['tabular'], dtype=torch.float32)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.long)
        self.text_sequences = data_dict['text']
        self.feature_names = data_dict.get('feature_names', [])
        
        # Apply transforms if provided
        if transforms and transforms.is_fitted:
            transformed = transforms.transform(
                X_numerical=data_dict['tabular'],
                text_sequences=self.text_sequences
            )
            
            self.image_data = transformed['image']
            self.text_input_ids = transformed['text_input_ids']
            self.text_attention_mask = transformed['text_attention_mask']
        else:
            # Dummy data if transforms not available
            batch_size = len(self.labels)
            self.image_data = torch.zeros((batch_size, 1, 32, 32), dtype=torch.float32)
            self.text_input_ids = torch.zeros((batch_size, 512), dtype=torch.long)
            self.text_attention_mask = torch.zeros((batch_size, 512), dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_data[idx],
            'image': self.image_data[idx],
            'text_input_ids': self.text_input_ids[idx],
            'text_attention_mask': self.text_attention_mask[idx],
            'label': self.labels[idx],
            'text_sequence': self.text_sequences[idx] if idx < len(self.text_sequences) else ""
        }


class KAMELTrainer:
    """Complete trainer for KAMEL model with imbalanced tabular learning."""
    
    def __init__(self, model, train_data: Dict[str, Any], val_data: Dict[str, Any], test_data: Dict[str, Any], 
                 config: Optional[Dict[str, Any]] = None, device: str = 'cuda', enable_caching: bool = True):
        self.model = model
        self.device = device
        self.config = config or self._get_default_config()
        self.enable_caching = False  # 禁用缓存，全参数优化
        
        # Store dataset name for cache file naming
        self.dataset_name = train_data.get('dataset_name', 'unknown')
        
        # Store class distribution for imbalance handling
        self.class_counts = list(train_data['class_distribution'].values())
        self.num_classes = len(self.class_counts)
        
        print(f"Class distribution: {dict(train_data['class_distribution'])}")
        
        # Initialize multimodal transforms
        self.transforms = MultiModalTransforms(
            image_size=(32, 32),
            max_text_length=512,
            tokenizer_name='distilbert-base-uncased'
        )
        
        # Fit transforms on training data
        print("Fitting multimodal transforms...")
        self.transforms.fit(
            X_numerical=train_data['tabular'],
            feature_names=train_data['feature_names']
        )
        
        # Create datasets
        self.train_dataset = MultiModalDataset(train_data, self.transforms, 'train')
        self.val_dataset = MultiModalDataset(val_data, self.transforms, 'val')
        self.test_dataset = MultiModalDataset(test_data, self.transforms, 'test')
        
        # Initialize imbalance handler
        self.imbalance_trainer = ImbalanceAwareTrainer(
            sampling_strategy=self.config['sampling_strategy'],
            loss_strategy=self.config['loss_strategy'],
            phase_1_epochs=self.config['phase_1_epochs'],
            total_epochs=self.config['total_epochs']
        )
        
        # Create samplers
        # For SMOTE, we need to pass dataset features
        sampler_kwargs = {
            'dataset_labels': train_data['labels'],
            'batch_size': self.config['batch_size']
        }
        
        # Add features for SMOTE sampling
        if self.config.get('sampling_strategy') == 'smote':
            # Try to get features from different possible keys
            if 'features' in train_data:
                sampler_kwargs['dataset_features'] = train_data['features']
            elif 'raw_features' in train_data:
                # Convert DataFrame to numpy array if needed
                raw_features = train_data['raw_features']
                if hasattr(raw_features, 'values'):
                    sampler_kwargs['dataset_features'] = raw_features.values
                else:
                    sampler_kwargs['dataset_features'] = raw_features
            elif 'tabular' in train_data:
                sampler_kwargs['dataset_features'] = train_data['tabular']
        
        self.train_sampler = self.imbalance_trainer.create_sampler(**sampler_kwargs)
        
        # Create data loaders
        # drop_last is disabled when sampler length <= batch_size to avoid an empty loader
        try:
            sampler_len = len(self.train_sampler)
        except TypeError:
            sampler_len = len(self.train_dataset)
        drop_last = sampler_len > self.config['batch_size']
        self.train_loader = data.DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], sampler=self.train_sampler,
            num_workers=0, pin_memory=True, drop_last=drop_last
        )
        
        self.val_loader = data.DataLoader(
            self.val_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )
        
        self.test_loader = data.DataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )
        
        # Initialize components
        self.loss_fn = get_loss_function(
            loss_type=self.config['loss_strategy'],
            cls_num_list=self.class_counts,
            **self.config.get('loss_kwargs', {})
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['total_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Initialize metrics and logging
        self.metrics_calculator = MetricsCalculator(optimize_threshold=True)
        self.visualization_utils = VisualizationUtils()
        
        # Create output directory
        self.output_dir = self.config.get('output_dir', './kamel_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create unified cache directory for embeddings (dataset-specific)
        if self.enable_caching:
            # 统一缓存位置：./cache/{dataset_name}/
            self.cache_dir = os.path.join('./cache', self.dataset_name)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Embedding cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None
        
        # Training history
        self.train_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
            'val_f1': [], 'val_auc': [], 'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_val_auc = 0.0  # 兼容旧代码
        self.best_epoch = 0
        
        # Two-stage training flags
        self.encoding_cached = False
        self.cached_embeddings = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'total_epochs': 50,
            'phase_1_epochs': 40,
            'batch_size': 128,
            'learning_rate': 5e-4,  # 降低学习率减少过拟合
            'weight_decay': 1e-3,   # 增强L2正则化
            'sampling_strategy': 'curriculum',
            'loss_strategy': 'ldam',
            'num_workers': 0,
            'patience': 8,          # 更早停止，从20降到8
            'save_best_only': True,
            'evaluation_frequency': 1,
            'gradient_clip_norm': 0.5,  # 添加梯度裁剪
            'dropout_rate': 0.5     # 增加dropout
        }
    

    
    def evaluate(self, data_loader, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on given data loader.
        Works with full data (not cached embeddings) for end-to-end evaluation.
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                tabular_data = batch['tabular'].to(self.device)
                image_data = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass through complete model
                logits = self.model(
                    tabular_data=tabular_data,
                    image_data=image_data,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask
                )
                
                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                # Get probabilities
                if self.num_classes == 2:
                    # Binary classification
                    probabilities = torch.softmax(logits, dim=1)[:, 1]
                else:
                    # Multi-class classification  
                    probabilities = torch.softmax(logits, dim=1)
                
                # Store results
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
                
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        
        # Concatenate results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        
        # Compute metrics  
        is_validation = (split == 'validation' or split == 'val')
        raw_metrics = self.metrics_calculator.compute_metrics(
            all_labels.numpy(),
            all_probabilities.numpy(),
            num_classes=self.num_classes,
            is_validation=is_validation,
            verbose=is_validation
        )
        
        # Add prefix to metrics
        metrics = {}
        for key, value in raw_metrics.items():
            metrics[f'{split}_{key}'] = value
        
        # Add loss
        metrics[f'{split}_loss'] = total_loss / total_samples
        
        return metrics
    
    def cache_embeddings(self) -> bool:
        """Stage 2: Cache embeddings from all modalities."""
        if not self.enable_caching:
            print("Caching disabled, skipping embedding caching stage")
            return True
            
        # 简化缓存文件命名：仅按数据集区分，参数变化时直接覆盖
        cache_files = {
            'train': os.path.join(self.cache_dir, 'train_embeddings.pt'),
            'val': os.path.join(self.cache_dir, 'val_embeddings.pt'),
            'test': os.path.join(self.cache_dir, 'test_embeddings.pt')
        }
        
        # Check if embeddings already cached
        if all(os.path.exists(f) for f in cache_files.values()):
            print("[OK] Found cached embeddings, loading...")
            try:
                # Load embeddings to CPU to avoid pin_memory issues
                self.cached_embeddings = {
                    split: torch.load(cache_file, map_location='cpu')
                    for split, cache_file in cache_files.items()
                }
                self.encoding_cached = True
                print("[OK] Cached embeddings loaded successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Error loading cached embeddings: {e}")
                print("Regenerating embeddings...")
        
        print("Stage 2: Caching embeddings from encoders...")
        self.model.eval()
        
        loaders = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}
        
        for split_name, data_loader in loaders.items():
            print(f"Caching {split_name} embeddings...")
            
            all_tabular_embeds = []
            all_image_embeds = []
            all_text_embeds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f'Caching {split_name} embeddings'):
                    # Move to device
                    tabular_data = batch['tabular'].to(self.device)
                    image_data = batch['image'].to(self.device)
                    text_input_ids = batch['text_input_ids'].to(self.device)
                    text_attention_mask = batch['text_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Get embeddings from encoders only
                    tabular_embed, image_embed, text_embed = self.model.encoders(
                        tabular_data=tabular_data,
                        image_data=image_data,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask
                    )
                    
                    # Store embeddings
                    all_tabular_embeds.append(tabular_embed.cpu())
                    all_image_embeds.append(image_embed.cpu())
                    all_text_embeds.append(text_embed.cpu())
                    all_labels.append(labels.cpu())
            
            # Concatenate all embeddings
            embeddings_data = {
                'tabular_embeds': torch.cat(all_tabular_embeds, dim=0),
                'image_embeds': torch.cat(all_image_embeds, dim=0),
                'text_embeds': torch.cat(all_text_embeds, dim=0),
                'labels': torch.cat(all_labels, dim=0)
            }
            
            # Save to cache (keep on CPU for better compatibility)
            cache_file = cache_files[split_name]
            torch.save(embeddings_data, cache_file)
            self.cached_embeddings[split_name] = embeddings_data
            
            print(f"[OK] {split_name} embeddings cached: {embeddings_data['labels'].shape[0]} samples")
        
        self.encoding_cached = True
        print("[OK] Stage 2 completed: All embeddings cached successfully")
        return True
    
    def create_cached_dataloaders(self):
        """Create data loaders from cached embeddings for stage 2."""
        if not self.encoding_cached:
            raise ValueError("Embeddings not cached yet. Run cache_embeddings() first.")
        
        from torch.utils.data import TensorDataset, DataLoader
        
        cached_loaders = {}
        
        for split_name, embeddings_data in self.cached_embeddings.items():
            # Create tensor dataset
            dataset = TensorDataset(
                embeddings_data['tabular_embeds'],
                embeddings_data['image_embeds'], 
                embeddings_data['text_embeds'],
                embeddings_data['labels']
            )
            
            # Create data loader
            shuffle = (split_name == 'train')
            batch_size = self.config['batch_size']
            
            # Don't use pin_memory with cached embeddings (they're already optimized)
            pin_memory = False  # Cached embeddings don't benefit from pinning
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)
            cached_loaders[split_name] = loader
            print(f"[OK] Created cached data loader for {split_name}: {len(dataset)} samples")
        
        return cached_loaders
    
    def train_fusion_stage(self, cached_loaders: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Train fusion module with cached embeddings."""
        print("Stage 3: Training MoE fusion with cached embeddings...")
        
        # Freeze encoders for fusion-only training
        self.model.freeze_encoders()
        print("[OK] Encoders frozen, training fusion module only")
        
        # Use cached loaders
        train_loader = cached_loaders['train']
        val_loader = cached_loaders['val']
        
        best_val_metrics = {}
        patience_counter = 0
        
        for epoch in range(self.config['total_epochs']):
            # Training with cached embeddings
            train_metrics = self._train_fusion_epoch(train_loader, epoch)
            
            # Validation
            if epoch % self.config['evaluation_frequency'] == 0:
                val_metrics = self._evaluate_fusion(val_loader, 'val')
                
                # Update history
                self.train_history['epoch'].append(epoch)
                self.train_history['train_loss'].append(train_metrics['train_loss'])
                self.train_history['val_loss'].append(val_metrics['val_loss'])
                self.train_history['val_accuracy'].append(val_metrics['accuracy'])
                self.train_history['val_f1'].append(val_metrics['f1'])
                self.train_history['val_auc'].append(val_metrics['auc'])
                self.train_history['learning_rate'].append(train_metrics['learning_rate'])
                
                # Check for best model
                current_val_f1 = val_metrics['f1']
                if current_val_f1 > self.best_val_f1:
                    self.best_val_f1 = current_val_f1
                    self.best_epoch = epoch
                    best_val_metrics = val_metrics.copy()
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.get('save_best_only', True):
                        self.save_model(epoch, 'best_fusion')
                else:
                    patience_counter += 1
                
                # Print progress
                phase = self.imbalance_trainer.get_current_phase()
                print(f"Fusion Epoch {epoch+1}/{self.config['total_epochs']} [{phase}] - "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val F1: {current_val_f1:.4f}")
                
                # Early stopping with validation loss monitoring
                patience = self.config.get('patience', 8)
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    print(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                    break
            
            # Update learning rate
            self.scheduler.step()
        
        # Re-optimize threshold on validation set for the best model
        print("\nRe-optimizing threshold on validation set for the best model...")
        _ = self._evaluate_fusion(cached_loaders['val'], 'validation')
        # Final evaluation on test set
        print("\nEvaluating fusion model on test set...")
        test_metrics = self._evaluate_fusion(cached_loaders['test'], 'test')
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_metrics': best_val_metrics,
            'test_metrics': test_metrics,
            'train_history': self.train_history,
            'model_info': self.model.get_model_info()
        }
    
    def _train_fusion_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train fusion module for one epoch with cached embeddings."""
        self.model.train()
        
        # Only fusion module is trainable (encoders frozen)
        self.imbalance_trainer.update_epoch(epoch, [])
        
        if hasattr(self.loss_fn, 'update_epoch'):
            self.loss_fn.update_epoch(epoch)
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Fusion Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Cached embeddings (already computed)
            tabular_embed = batch[0].to(self.device)
            image_embed = batch[1].to(self.device)
            text_embed = batch[2].to(self.device)
            labels = batch[3].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Fusion and classification (encoders are frozen)
            fused_features, moe_auxiliary = self.model.fusion(
                tabular_embed=tabular_embed,
                image_embed=image_embed,
                text_embed=text_embed,
                training=self.model.training
            )
            
            # Classification
            logits = self.model.classifier(fused_features)
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                logits=logits,
                targets=labels,
                loss_fn=self.loss_fn,
                auxiliary={'moe_auxiliary': moe_auxiliary}
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (更保守的裁剪)
            max_norm = self.config.get('gradient_clip_norm', 0.5)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            # Update progress bar
            avg_loss = total_loss / total_samples
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        return {
            'train_loss': total_loss / total_samples,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _evaluate_fusion(self, data_loader, split: str = 'val') -> Dict[str, float]:
        """Evaluate fusion model with cached embeddings."""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Evaluating {split}'):
                # Cached embeddings
                tabular_embed = batch[0].to(self.device)
                image_embed = batch[1].to(self.device)
                text_embed = batch[2].to(self.device)
                labels = batch[3].to(self.device)
                
                # Fusion and classification
                fused_features, _ = self.model.fusion(
                    tabular_embed=tabular_embed,
                    image_embed=image_embed,
                    text_embed=text_embed,
                    training=False
                )
                
                logits = self.model.classifier(fused_features)
                
                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                # Get probabilities
                if self.num_classes == 2:
                    # For binary classification with 2 logits, use softmax and take positive class probability
                    probabilities = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probabilities = torch.softmax(logits, dim=1)
                
                # Store results
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
                
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        
        # Concatenate results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        
        # Compute metrics
        is_validation = (split == 'validation' or split == 'val')
        metrics = self.metrics_calculator.compute_metrics(
            y_true=all_labels.numpy(),
            y_pred_proba=all_probabilities.numpy(),
            num_classes=self.num_classes,
            is_validation=is_validation,
            verbose=is_validation
        )
        
        metrics[f'{split}_loss'] = total_loss / total_samples
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        End-to-end training with intelligent caching strategy:
        - Check if cached embeddings exist and encoder config matches
        - If yes: Use cached embeddings for efficiency (fusion-only training)
        - If no: Full end-to-end training while caching embeddings for future use
        """
        print(f"Starting KAMEL end-to-end training...")
        print(f"Model info: {self.model.get_model_info()}")
        
        # 全参数端到端训练 - 不使用缓存，追求最优性能
        print("[E2E] Running full end-to-end training with complete parameter optimization...")
        results = self._train_end_to_end()
        
        # Save training history
        try:
            self.save_training_history()
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")
        
        # Generate visualizations (if enabled)
        if self.config.get('enable_visualization', True):
            try:
                if 'test_metrics' in results:
                    self.generate_visualizations(results['test_metrics'])
            except:
                print("Warning: Could not generate visualizations")
        
        # Save results (convert numpy arrays to lists for JSON serialization)
        try:
            def convert_numpy_to_list(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_list(v) for v in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj
            
            results_serializable = convert_numpy_to_list(results)
            with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
                json.dump(results_serializable, f, indent=2)
        except:
            print("Warning: Could not save results to JSON")
        
        print(f"\nTraining completed!")
        if 'best_val_metrics' in results and 'test_metrics' in results:
            print(f"Best Val F1-Score: {results['best_val_metrics']['f1']:.4f} (Epoch {results['best_epoch']})")
            print(f"\nFinal Test Performance:")
            print(f"  F1-Score:  {results['test_metrics']['f1']:.4f}")
            print(f"  AUC-ROC:   {results['test_metrics']['auc']:.4f}")
            print(f"  AU-PRC:    {results['test_metrics'].get('au_prc', 0.0):.4f}")
            print(f"  Accuracy:  {results['test_metrics']['accuracy']:.4f}")
            print(f"  Precision: {results['test_metrics']['precision']:.4f}")
            print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
        
        return results
    
    def _can_use_cached_embeddings(self) -> bool:
        """Check if cached embeddings exist for current dataset."""
        if not self.cache_dir:
            return False
            
        cache_files = {
            'train': os.path.join(self.cache_dir, 'train_embeddings.pt'),
            'val': os.path.join(self.cache_dir, 'val_embeddings.pt'),
            'test': os.path.join(self.cache_dir, 'test_embeddings.pt')
        }
        
        # Check if all cache files exist
        return all(os.path.exists(f) for f in cache_files.values())
    
    def _can_use_cached_encoder_weights(self) -> bool:
        """Check if cached encoder weights exist."""
        encoder_weights_path = os.path.join(self.cache_dir, 'encoder_weights.pt')
        return os.path.exists(encoder_weights_path)
    
    def _cache_encoder_weights(self):
        """Cache encoder weights for consistent initialization."""
        os.makedirs(self.cache_dir, exist_ok=True)
        encoder_weights_path = os.path.join(self.cache_dir, 'encoder_weights.pt')
        
        # Extract encoder state dict
        encoder_state = {}
        for name, param in self.model.named_parameters():
            if any(encoder_name in name for encoder_name in ['tabular_encoder', 'image_encoder', 'text_encoder']):
                encoder_state[name] = param.data.clone()
        
        torch.save(encoder_state, encoder_weights_path)
        print(f"[CACHE] Encoder weights cached to {encoder_weights_path}")
    
    def _load_cached_encoder_weights(self):
        """Load cached encoder weights for consistent initialization."""
        encoder_weights_path = os.path.join(self.cache_dir, 'encoder_weights.pt')
        
        if os.path.exists(encoder_weights_path):
            encoder_state = torch.load(encoder_weights_path, map_location=self.device)
            
            # Load encoder weights
            current_state = self.model.state_dict()
            for name, cached_weight in encoder_state.items():
                if name in current_state:
                    current_state[name] = cached_weight
            
            self.model.load_state_dict(current_state, strict=False)
            print(f"[CACHE] Loaded cached encoder weights from {encoder_weights_path}")
            return True
        return False
    
    def _train_with_cached_embeddings(self) -> Dict[str, Any]:
        """Train fusion module using existing cached embeddings."""
        # Load cached embeddings  
        cache_files = {
            'train': os.path.join(self.cache_dir, 'train_embeddings.pt'),
            'val': os.path.join(self.cache_dir, 'val_embeddings.pt'),
            'test': os.path.join(self.cache_dir, 'test_embeddings.pt')
        }
        
        try:
            self.cached_embeddings = {
                split: torch.load(cache_file, map_location='cpu')
                for split, cache_file in cache_files.items()
            }
            self.encoding_cached = True
            print("[OK] Cached embeddings loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading cached embeddings: {e}")
            return {'message': 'Training failed - could not load cached embeddings'}
        
        # Create cached data loaders
        cached_loaders = self.create_cached_dataloaders()
        
        # Train fusion module only (encoders frozen)
        return self.train_fusion_stage(cached_loaders)
    
    def _train_end_to_end(self) -> Dict[str, Any]:
        """Full end-to-end training of the complete KAMEL model."""
        print("Starting full end-to-end training...")
        
        # Initialize loss function epoch tracking
        if hasattr(self.loss_fn, 'update_epoch'):
            self.loss_fn.update_epoch(0)
        
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = self.config.get('patience', 8)
        
        print(f"Training for {self.config['total_epochs']} epochs...")
        
        for epoch in range(self.config['total_epochs']):
            # Update imbalance trainer epoch for dynamic sampling
            self.imbalance_trainer.update_epoch(epoch, [self.train_sampler])
            
            # Update loss function epoch if applicable
            if hasattr(self.loss_fn, 'update_epoch'):
                self.loss_fn.update_epoch(epoch)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["total_epochs"]}')
            for batch in pbar:
                # Move to device
                tabular_data = batch['tabular'].to(self.device)
                image_data = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass through complete model
                outputs = self.model(
                    tabular_data=tabular_data,
                    image_data=image_data,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask
                )
                
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_norm']
                    )
                
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader, 'validation')
            val_f1 = val_metrics['validation_f1']
            val_acc = val_metrics['validation_accuracy'] 
            val_loss = val_metrics['validation_loss']
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config['total_epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Update training history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['val_auc'].append(val_metrics.get('validation_auc', 0.0))
            self.train_history['learning_rate'].append(current_lr)
            
            # Model checkpointing based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_auc = val_metrics.get('validation_auc', 0.0)  # 同步更新AUC
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                if self.config.get('save_best_only', True):
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_val_f1': best_val_f1,
                        'config': self.config
                    }
                    
                    model_path = os.path.join(self.output_dir, 'best_fusion_model.pth')
                    torch.save(checkpoint, model_path)
                    print(f"  [SAVE] Best model saved (F1: {best_val_f1:.4f})")
                
                # 不缓存embeddings，每次都进行完整训练
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            print("-" * 60)
        
        # Load best model for final evaluation
        if self.config.get('save_best_only', True):
            try:
                model_path = os.path.join(self.output_dir, 'best_fusion_model.pth')
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"[OK] Loaded best model from epoch {best_epoch} (F1: {best_val_f1:.4f})")
            except Exception as e:
                print(f"Warning: Could not load best model: {e}")
                print("Using current model state for final evaluation")
        
        # Re-optimize threshold on validation set for the best model
        print("\nRe-optimizing threshold on validation set for the best model...")
        _ = self.evaluate(self.val_loader, 'validation')
        # Final test evaluation
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        print(f"Final Test Metrics:")
        print(f"  Test Accuracy:  {test_metrics['test_accuracy']:.4f}")
        print(f"  Test F1:        {test_metrics['test_f1']:.4f}")
        print(f"  Test Precision: {test_metrics['test_precision']:.4f}")
        print(f"  Test Recall:    {test_metrics['test_recall']:.4f}")
        print(f"  Test AUC:       {test_metrics.get('test_auc', 0.0):.4f}")
        print(f"  Test AU-PRC:    {test_metrics.get('test_au_prc', 0.0):.4f}")
        print("="*60)
        
        return {
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'test_metrics': test_metrics,
            'train_history': self.train_history
        }
    
    def _cache_embeddings_from_trained_model(self):
        """Cache embeddings from the currently trained model for future use."""
        if not self.enable_caching:
            return
            
        print("\n[INFO] Caching embeddings from trained model...")
        
        cache_files = {
            'train': os.path.join(self.cache_dir, 'train_embeddings.pt'),
            'val': os.path.join(self.cache_dir, 'val_embeddings.pt'),
            'test': os.path.join(self.cache_dir, 'test_embeddings.pt')
        }
        
        self.model.eval()
        loaders = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}
        
        for split_name, data_loader in loaders.items():
            all_tabular_embeds = []
            all_image_embeds = []
            all_text_embeds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in data_loader:
                    tabular_data = batch['tabular'].to(self.device)
                    image_data = batch['image'].to(self.device)
                    text_input_ids = batch['text_input_ids'].to(self.device)
                    text_attention_mask = batch['text_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Get embeddings from trained encoders
                    tabular_embed, image_embed, text_embed = self.model.encoders(
                        tabular_data=tabular_data,
                        image_data=image_data,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask
                    )
                    
                    all_tabular_embeds.append(tabular_embed.cpu())
                    all_image_embeds.append(image_embed.cpu())
                    all_text_embeds.append(text_embed.cpu())
                    all_labels.append(labels.cpu())
            
            # Save embeddings
            embeddings_data = {
                'tabular_embeds': torch.cat(all_tabular_embeds, dim=0),
                'image_embeds': torch.cat(all_image_embeds, dim=0),
                'text_embeds': torch.cat(all_text_embeds, dim=0),
                'labels': torch.cat(all_labels, dim=0)
            }
            
            torch.save(embeddings_data, cache_files[split_name])
            print(f"[OK] Cached {split_name} embeddings: {embeddings_data['labels'].shape[0]} samples")
        
        print("[OK] Embeddings cached for future use")
    
    def save_model(self, epoch: int, name: str = 'checkpoint'):
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_f1': self.best_val_f1,
                'best_val_auc': self.best_val_auc,
                'config': self.config,
                'class_counts': self.class_counts
            }
            
            checkpoint_path = os.path.join(self.output_dir, f'{name}_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def save_training_history(self):
        """Save training history to CSV."""
        try:
            history_df = pd.DataFrame(self.train_history)
            history_path = os.path.join(self.output_dir, 'training_history.csv')
            history_df.to_csv(history_path, index=False)
            print(f"Training history saved to {history_path}")
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")
    
    def generate_visualizations(self, test_metrics: Dict[str, float]):
        """Generate training and evaluation visualizations."""
        try:
            # Training curves
            self.visualization_utils.plot_training_curves(
                self.train_history,
                save_path=os.path.join(self.output_dir, 'training_curves.png')
            )
            
            # Confusion matrix
            if 'confusion_matrix' in test_metrics:
                self.visualization_utils.plot_confusion_matrix(
                    test_metrics['confusion_matrix'],
                    save_path=os.path.join(self.output_dir, 'confusion_matrix.png')
                )
            
            print(f"Visualizations saved to {self.output_dir}")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
