'''
Author: Chao Li
Contact: lucianolee@zju.edu.cn
Affiliation: Zhejiang University
Created: Wednesday September 20th 2023 3:19:17 pm
Last Modified: Wednesday March 26th 2025 2:56:47 pm

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import argparse
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from func import *

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantized Naive Bayes classifier experiment')
    parser.add_argument('--dataset', type=str, default='iris', choices=['iris', 'wine', 'breast_cancer'],
                        help='Dataset to use (default: iris)')
    parser.add_argument('--trials', type=int, default=100, 
                        help='Number of trials to run (default: 100)')
    parser.add_argument('--feature-level', type=int, default=4,
                        help='Power of 2 for feature quantization level (default: 4)')
    parser.add_argument('--device-level', type=int, default=2,
                        help='Power of 2 for device quantization level (default: 2)')
    parser.add_argument('--test-size', type=float, default=0.5,
                        help='Test size for train-test split (default: 0.5)')
    parser.add_argument('--truncate-threshold', type=float, default=1e-3,
                        help='Truncation threshold for probability values (default: 1e-3)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    return parser.parse_args()

def load_dataset(dataset_name):
    """Load the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load ('iris', 'wine', or 'breast_cancer')
        
    Returns:
        data_all: Loaded dataset
    """
    if dataset_name == 'iris':
        return load_iris()
    elif dataset_name == 'wine':
        return load_wine()
    elif dataset_name == 'breast_cancer':
        return load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def run_experiment(data_all, trial_num, test_size, feature_level, device_level, truncate_threshold):
    """Run the quantization experiment.
    
    Args:
        data_all: Dataset to use
        trial_num: Number of trials to run
        test_size: Test size for train-test split
        feature_level: Feature quantization level (power of 2)
        device_level: Device quantization level (power of 2)
        truncate_threshold: Threshold for truncating probability values
        
    Returns:
        acc_quant: List of accuracy results for quantized model
        acc_orig: List of accuracy results for original model
    """
    acc_quant = []
    acc_orig = []
    
    feature_level_pow = np.power(2, feature_level)
    device_level_pow = np.power(2, device_level)

    for trial_index in tqdm(range(trial_num)):
        # Split data for this trial
        x_train, x_test, y_train, y_test = self_split(
            data_all.data, data_all.target, self_ts=test_size, self_rs=trial_index)
        
        # Train and evaluate original model
        model = GaussianNB()
        model.fit(x_train, y_train)
        acc_original = accuracy_score(y_test, model.predict(x_test))
        acc_orig.append(acc_original)

        # Feature quantization
        prob_array, prob_array_log, quant_config = parray_fquanz(
            model, data_all.data, feature_level_pow)
        x_test_quant = input_quantize(x_test, quant_config, feature_level_pow, model.n_features_in_)
        
        # Apply device quantization with the single threshold
        prob_array_quant = parray_dquanz_simp(
            prob_array_log, feature_level_pow, device_level_pow, 
            model.classes_.size, model.n_features_in_, 
            col_norm=True, log_style=True, discard=True, truncate_theta=truncate_threshold)
        
        # Make predictions with quantized model
        predictions = approx_predict(
            prob_array_quant, x_test_quant, 
            model.classes_.size, model.n_features_in_, log_style=True)
        
        # Calculate accuracy
        acc_quantized = accuracy_score(y_test, predictions)
        acc_quant.append(acc_quantized)

    return acc_quant, acc_orig

def save_results(acc_quant, acc_orig, output_dir):
    """Save experimental results to files.
    
    Args:
        acc_quant: Quantized model accuracy results
        acc_orig: Original model accuracy results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'acc_original.npy'), 'wb') as f:
        np.save(f, acc_orig)

    with open(os.path.join(output_dir, 'acc_quantized.npy'), 'wb') as f:
        np.save(f, acc_quant)
    
    # Print summary statistics
    print(f'Original model accuracy: {np.mean(acc_orig) * 100:.2f}% ± {np.std(acc_orig) * 100:.2f}%')
    print(f'Quantized model accuracy: {np.mean(acc_quant) * 100:.2f}% ± {np.std(acc_quant) * 100:.2f}%')

def plot_results(acc_quant, acc_orig, output_dir):
    """Plot distribution of classification accuracy before and after quantization.
    
    Args:
        acc_quant: Quantized model accuracy results
        acc_orig: Original model accuracy results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms of accuracy distributions
    bins = np.linspace(min(min(acc_orig), min(acc_quant)), max(max(acc_orig), max(acc_quant)), 20)
    
    plt.hist(acc_orig, bins=bins, alpha=0.5, label='Original', color='blue')
    plt.hist(acc_quant, bins=bins, alpha=0.5, label='Quantized', color='red')
    
    # Add vertical lines for means
    plt.axvline(x=np.mean(acc_orig), color='blue', linestyle='--', 
                label=f'Original mean: {np.mean(acc_orig):.4f}')
    plt.axvline(x=np.mean(acc_quant), color='red', linestyle='--', 
                label=f'Quantized mean: {np.mean(acc_quant):.4f}')
    
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classification Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_distribution.png'))
    
    # Create a second plot for boxplot comparison
    plt.figure(figsize=(8, 6))
    
    data = [acc_orig, acc_quant]
    plt.boxplot(data, labels=['Original', 'Quantized'])
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Original vs. Quantized')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_boxplot.png'))
    
    plt.close('all')

def main():
    """Main function to run the experiment."""
    args = parse_arguments()
    
    # Load dataset
    data_all = load_dataset(args.dataset)
    
    # Run the experiment
    acc_quant, acc_orig = run_experiment(
        data_all, args.trials, args.test_size, 
        args.feature_level, args.device_level,
        args.truncate_threshold)
    
    # Save and visualize results
    save_results(acc_quant, acc_orig, args.output_dir)
    plot_results(acc_quant, acc_orig, args.output_dir)

if __name__ == "__main__":
    main() 