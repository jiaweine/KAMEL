"""
Machine Learning Comparison Experiment Framework.
Compares 6 ML algorithms against KAMEL on all supported datasets.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add KAMEL to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add algorithms path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from kamel.data.preprocessor import TabularDataProcessor
from kamel.utils.metrics import MetricsCalculator
from .algorithms import ALGORITHM_REGISTRY
from .algorithm_configs import get_algorithm_config, get_smote_config, get_all_algorithms_for_dataset


class MLComparisonExperiment:
    """
    Main experiment class for ML algorithm comparison.
    """
    
    def __init__(self, output_dir: str = "./outputs/ml_comparison", random_state: int = 42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.processor = TabularDataProcessor()
        self.metrics_calculator = MetricsCalculator(optimize_threshold=True)

        # Global seeding for reproducibility between single and batched runs
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        try:
            import torch
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
        except Exception:
            pass
        
        # Supported datasets (filtered per user request)
        self.datasets = [
            'adult', 'car_evaluation', 'creditcard',
            'diabetes', 'magic_telescope',
            'mushroom', 'nursery', 'spambase',
            'electrical_grid_stability', 'steel_plates_fault_detection',
            'breast_cancer_wisconsin', 'heart_disease', 'ionosphere',
            'sonar', 'australian_credit', 'haberman', 'liver_disorders',
            'monks1', 'tictactoe', 'iris', 'wine', 'glass', 'ecoli', 'seeds',
            # 10 new UCI binary classification datasets
            'banknote', 'german_credit', 'hepatitis', 'parkinsons', 'blood_transfusion',
            'fertility', 'mammographic', 'qsar_biodegradation', 'seismic_bumps', 'vertebral_column',
            'vehicle',
            'authorship', 'cmc', 'mfeat_morphological'
        ]
        
        # Algorithm names - get from a reference dataset to ensure all algorithms are included
        # This ensures we get all 17 algorithms including TALENT models
        self.algorithms = get_all_algorithms_for_dataset('adult')  # Use 'adult' as reference since all datasets have same algorithms
        
        # Results storage
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _normalize_proba(self, proba: np.ndarray, expected_samples: int) -> np.ndarray:
        """
        Normalize probability predictions to consistent format.
        For binary classification: returns 1D array of positive class probabilities.
        For multi-class: returns full probability matrix.
        
        Args:
            proba: Probability predictions from model
            expected_samples: Expected number of samples
            
        Returns:
            Normalized probability array
        """
        # Handle various probability array shapes
        if proba is None:
            return np.full(expected_samples, 0.5)  # Return neutral probabilities
        
        # Flatten if 3D (e.g., (67, 1, 2) -> (67, 2))
        if len(proba.shape) == 3:
            if proba.shape[1] == 1:
                proba = proba.squeeze(1)  # (67, 1, 2) -> (67, 2)
            else:
                # Take first slice if multiple
                proba = proba[:, 0, :]
        
        # Ensure correct number of samples
        if len(proba) != expected_samples:
            # Truncate or pad
            if len(proba) > expected_samples:
                proba = proba[:expected_samples]
            else:
                # Pad with neutral probabilities
                if len(proba.shape) == 1:
                    # 1D array
                    padding = np.full(expected_samples - len(proba), 0.5)
                    proba = np.concatenate([proba, padding])
                else:
                    # 2D array
                    padding_shape = (expected_samples - len(proba), proba.shape[1])
                    padding = np.full(padding_shape, 0.5)
                    proba = np.vstack([proba, padding])
        
        # Convert to appropriate format based on shape
        if len(proba.shape) == 1:
            # Already 1D, assume it's positive class probability for binary
            return proba
        elif proba.shape[1] == 2:
            # Binary classification, take positive class (index 1)
            return proba[:, 1]
        elif proba.shape[1] == 1:
            # Single column, assume it's positive class probability
            return proba.ravel()
        else:
            # Multi-class (>2 classes), return full probability matrix
            return proba
        
    def run_single_experiment(self, dataset_name: str, algorithm_name: str) -> Dict[str, Any]:
        """
        Run single experiment for one dataset and one algorithm.
        """
        print(f"Running {algorithm_name} on {dataset_name}...")
        
        try:
            # Load dataset
            train_data, val_data, test_data = self.processor.load_dataset(
                dataset_name=dataset_name,
                test_size=0.15,
                val_size=0.15,
                random_state=self.random_state
            )
            
            # Extract data
            X_train = train_data['tabular']
            y_train = train_data['labels']
            X_val = val_data['tabular']
            y_val = val_data['labels']
            X_test = test_data['tabular']
            y_test = test_data['labels']
            
            # Get algorithm configuration for this dataset
            config = get_algorithm_config(algorithm_name, dataset_name)
            
            # Set fixed seeds inside config if supported by algorithms
            config['random_state'] = self.random_state
            if 'seed' in config:
                config['seed'] = self.random_state
            
            # Initialize algorithm with dataset-specific configuration
            # Handle TALENT models specially (they use TALENTClassifier with model_name parameter)
            is_talent_model = algorithm_name.startswith('talent_')
            
            if is_talent_model:
                from .algorithms.talent_models import TALENTClassifier
                # Map algorithm name to TALENT model name
                # e.g., 'talent_modernnca' -> 'modernNCA', 'talent_resnet' -> 'resnet'
                # Note: TabR now uses official Yandex Research implementation (tabr_classifier.py)
                talent_model_map = {
                    'talent_modernnca': 'modernNCA',
                    'talent_resnet': 'resnet',
                    'talent_realmlp': 'realmlp'
                }
                model_name = talent_model_map.get(algorithm_name, algorithm_name.replace('talent_', ''))
                config['model_name'] = model_name
                algorithm = TALENTClassifier(**config)
            else:
                algorithm = ALGORITHM_REGISTRY[algorithm_name](**config)
            
            # Train model
            start_time = time.time()
            
            if is_talent_model:
                # TALENT uses sklearn interface (fit/predict)
                # Combine train and val for fitting (TALENT does internal validation)
                X_train_val = np.vstack([X_train, X_val])
                y_train_val = np.hstack([y_train, y_val])
                algorithm.fit(X_train_val, y_train_val)
                
                # Get validation predictions with robust shape handling
                val_proba = algorithm.predict_proba(X_val)
                val_proba = self._normalize_proba(val_proba, len(X_val))
                    
                train_results = {
                    'val_proba': val_proba,
                    'training_samples': len(y_train_val)
                }
            else:
                # Standard training interface
                train_results = algorithm.train(X_train, y_train, X_val, y_val)
                
            training_time = time.time() - start_time
            
            # Create a fresh metrics calculator per algorithm to avoid state leakage
            local_metrics = MetricsCalculator(optimize_threshold=True)
            
            # Validation metrics FIRST to set optimal threshold
            val_metrics = local_metrics.compute_metrics(
                y_true=y_val,
                y_pred_proba=train_results['val_proba'],
                num_classes=train_data['num_classes'],
                verbose=False,
                is_validation=True
            )
            
            # Make predictions on test
            start_time = time.time()
            
            if is_talent_model:
                # TALENT uses sklearn interface
                y_pred = algorithm.predict(X_test)
                y_proba_raw = algorithm.predict_proba(X_test)
                y_proba = self._normalize_proba(y_proba_raw, len(X_test))
            else:
                # Standard prediction interface
                y_pred, y_proba = algorithm.predict(X_test)
                
            inference_time = time.time() - start_time
            
            # Test metrics using the optimized threshold determined on validation
            test_metrics = local_metrics.compute_metrics(
                y_true=y_test,
                y_pred_proba=y_proba,
                num_classes=train_data['num_classes'],
                verbose=False
            )
            
            # Compile results
            # Get model info (TALENT models don't have this method)
            if is_talent_model:
                model_info = {
                    'algorithm': f'TALENT-{model_name}',
                    'model_name': model_name,
                    'hyperparameters': config
                }
            else:
                model_info = algorithm.get_model_info()
            
            experiment_result = {
                'dataset': dataset_name,
                'algorithm': algorithm_name,
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'num_samples': len(X_train) + len(X_val) + len(X_test),
                    'num_features': train_data['num_features'],
                    'num_classes': train_data['num_classes'],
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                },
                'model_info': model_info,
                'performance': {
                    'training_time': training_time,
                    'inference_time': inference_time,
                    'training_samples_used': train_results['training_samples']
                },
                'test_metrics': {
                    'accuracy': float(test_metrics['accuracy']),
                    'precision': float(test_metrics['precision']),
                    'recall': float(test_metrics['recall']),
                    'f1': float(test_metrics['f1']),
                    'auc': float(test_metrics['auc']),
                    'au_prc': float(test_metrics['au_prc']),
                    'mcc': float(test_metrics['mcc']),
                    'kappa': float(test_metrics['kappa'])
                },
                'validation_metrics': {
                    'accuracy': float(val_metrics['accuracy']),
                    'precision': float(val_metrics['precision']),
                    'recall': float(val_metrics['recall']),
                    'f1': float(val_metrics['f1']),
                    'auc': float(val_metrics['auc']),
                    'au_prc': float(val_metrics['au_prc'])
                },
                'optimal_threshold': float(local_metrics.optimal_threshold),
                'status': 'completed'
            }
            
            # Save single experiment result to file
            single_result_file = os.path.join(
                self.output_dir, 
                f"{dataset_name}_{algorithm_name}_result.json"
            )
            with open(single_result_file, 'w') as f:
                json.dump(experiment_result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {single_result_file}")
            
            return experiment_result
            
        except Exception as e:
            print(f"Error in {algorithm_name} on {dataset_name}: {str(e)}")
            return {
                'dataset': dataset_name,
                'algorithm': algorithm_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    def run_dataset_experiments(self, dataset_name: str, include_talent: bool = True) -> Dict[str, Any]:
        """
        Run all algorithms on a single dataset.
        
        Args:
            dataset_name: Name of the dataset
            include_talent: If True, include TALENT models (default: True)
        """
        print(f"\n{'='*60}")
        print(f"Running ML Comparison Experiments on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        dataset_results = {
            'dataset': dataset_name,
            'experiment_start': datetime.now().isoformat(),
            'algorithms': {}
        }
        
        # Get all algorithms for this dataset (including TALENT models if requested)
        if include_talent:
            algorithms_to_run = get_all_algorithms_for_dataset(dataset_name)
            print(f"Running {len(algorithms_to_run)} algorithms (including TALENT models)\n")
        else:
            algorithms_to_run = self.algorithms
            print(f"Running {len(algorithms_to_run)} core algorithms\n")
        
        for algorithm_name in algorithms_to_run:
            # Per-algorithm seeding to ensure identical results in batch vs single runs
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            try:
                import torch
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
            except Exception:
                pass
            try:
                print(f"[Start] {algorithm_name} on {dataset_name}")
                result = self.run_single_experiment(dataset_name, algorithm_name)
                dataset_results['algorithms'][algorithm_name] = result
                
                if result['status'] == 'completed':
                    metrics = result['test_metrics']
                    print(f"  {algorithm_name:20} | "
                          f"F1: {metrics['f1']:.4f} | "
                          f"ACC: {metrics['accuracy']:.4f} | "
                          f"PREC: {metrics['precision']:.4f} | "
                          f"REC: {metrics['recall']:.4f} | "
                          f"AUC: {metrics['auc']:.4f} | "
                          f"AUPRC: {metrics['au_prc']:.4f}")
                elif result['status'] == 'failed':
                    print(f"  {algorithm_name:20} | FAILED: {result.get('error', 'unknown')}")
                else:
                    print(f"  {algorithm_name:20} | ERROR: unexpected status {result.get('status')}")
                    
            except Exception as e:
                print(f"  {algorithm_name:20} | ERROR: {str(e)}")
                dataset_results['algorithms'][algorithm_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        dataset_results['experiment_end'] = datetime.now().isoformat()
        
        # Save dataset results
        dataset_file = os.path.join(self.output_dir, f"{dataset_name}_ml_comparison.json")
        with open(dataset_file, 'w') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {dataset_file}")
        return dataset_results
    
    def run_all_experiments(self, datasets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comparison experiments on all datasets.
        """
        if datasets is None:
            datasets = self.datasets
            
        print("KAMEL vs Traditional ML Algorithms Comparison")
        print(f"Datasets: {len(datasets)}")
        print(f"Algorithms: {len(self.algorithms)}")
        print(f"Total experiments: {len(datasets) * len(self.algorithms)}")
        print("="*80)
        
        all_results = {
            'experiment_info': {
                'start_time': datetime.now().isoformat(),
                'datasets': datasets,
                'algorithms': self.algorithms,
                'random_state': self.random_state
            },
            'dataset_results': {}
        }
        
        for dataset_name in datasets:
            try:
                dataset_result = self.run_dataset_experiments(dataset_name)
                all_results['dataset_results'][dataset_name] = dataset_result
            except Exception as e:
                print(f"Failed to process dataset {dataset_name}: {str(e)}")
                all_results['dataset_results'][dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        all_results['experiment_info']['end_time'] = datetime.now().isoformat()
        
        # Save comprehensive results
        summary_file = os.path.join(self.output_dir, "ml_comparison_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Generate summary table
        self.generate_summary_table(all_results)
        
        print(f"\nAll experiments completed!")
        print(f"Summary saved to: {summary_file}")
        
        return all_results
    
    def generate_summary_table(self, results: Dict[str, Any]) -> None:
        """
        Generate summary comparison table.
        """
        summary_data = []
        
        for dataset_name, dataset_result in results['dataset_results'].items():
            if dataset_result.get('status') == 'failed':
                continue
                
            for algo_name, algo_result in dataset_result.get('algorithms', {}).items():
                if algo_result.get('status') == 'completed':
                    metrics = algo_result['test_metrics']
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Algorithm': algo_name,
                        'F1': metrics['f1'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'AUC': metrics['auc'],
                        'AU_PRC': metrics['au_prc'],
                        'MCC': metrics['mcc'],
                        'Kappa': metrics['kappa'],
                        'Training_Time': algo_result['performance']['training_time']
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save CSV
            csv_file = os.path.join(self.output_dir, "ml_comparison_summary.csv")
            df.to_csv(csv_file, index=False)
            print(f"Summary table saved to: {csv_file}")
            
            # Print top performers by F1 score
            print(f"\nTop 5 performers by F1 score:")
            print("-" * 80)
            top_performers = df.nlargest(5, 'F1')[['Dataset', 'Algorithm', 'F1', 'Accuracy', 'Precision', 'Recall', 'AUC', 'AU_PRC']]
            print(top_performers.to_string(index=False))


def main():
    """
    Main function to run ML comparison experiments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Algorithm Comparison Experiments')
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset to run experiments on (or "all" for all datasets)')
    parser.add_argument('--algorithm', type=str, default='all',
                       help='Algorithm to run (or "all" for all algorithms)')
    parser.add_argument('--output_dir', type=str, default='./outputs/ml_comparison',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MLComparisonExperiment(
        output_dir=args.output_dir,
        random_state=args.seed
    )
    
    if args.dataset == 'all':
        # Run all datasets
        experiment.run_all_experiments()
    else:
        # Run single dataset
        if args.algorithm == 'all':
            experiment.run_dataset_experiments(args.dataset)
        else:
            # Run single algorithm on single dataset
            result = experiment.run_single_experiment(args.dataset, args.algorithm)
            if result['status'] == 'completed':
                metrics = result['test_metrics']
                print(f"Algorithm: {result['algorithm']} | Dataset: {result['dataset']}")
                print(f"F1: {metrics['f1']:.4f} | ACC: {metrics['accuracy']:.4f} | PREC: {metrics['precision']:.4f} | REC: {metrics['recall']:.4f} | AUC: {metrics['auc']:.4f} | AUPRC: {metrics['au_prc']:.4f}")
                print(f"Status: {result['status']}")
            else:
                print(f"Failed: {result['status']}")
                if 'error' in result:
                    print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
