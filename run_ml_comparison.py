#!/usr/bin/env python3
"""
Simple script to run ML comparison experiments.
Usage: python run_ml_comparison.py --dataset adult --algorithm all
"""

import sys
import os
from model_comparison.ml_experiment import MLComparisonExperiment

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML Comparison Experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (adult, creditcard, etc.) or "all"')
    parser.add_argument('--algorithm', type=str, default='all',
                       help='Algorithm name or "all"')
    parser.add_argument('--output_dir', type=str, default='./outputs/ml_comparison',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MLComparisonExperiment(
        output_dir=args.output_dir,
        random_state=args.seed
    )
    
    print("Available algorithms:")
    for algo in experiment.algorithms:
        print(f"  - {algo}")
    print()
    
    if args.dataset == 'all':
        print("Running experiments on all datasets...")
        experiment.run_all_experiments()
    else:
        if args.algorithm == 'all':
            print(f"Running all algorithms on {args.dataset}...")
            experiment.run_dataset_experiments(args.dataset)
        else:
            print(f"Running {args.algorithm} on {args.dataset}...")
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