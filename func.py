'''
Author: Chao Li
Contact: lucianolee@zju.edu.cn
Affiliation: Zhejiang University
Created: Wednesday September 20th 2023 3:53:03 pm
Last Modified: Friday March 28th 2025 1:47:41 pm

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

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


def self_split(data, target, self_ts=0.5, self_rs=42):
    """Split data with balanced class distribution.
    
    Args:
        data: Input feature data
        target: Target labels
        self_ts: Test size proportion (default: 0.5)
        self_rs: Random state for reproducibility (default: 42)
    
    Returns:
        x_train: Training features
        x_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    class_num = np.unique(target).shape[0]
    sample_num_pc = [np.sum(target == class_index) for class_index in range(class_num)]
    sample_num_min = np.min(sample_num_pc)

    x_train_all = []
    x_test_all = []
    y_train_all = []
    y_test_all = []

    for class_index in range(class_num):
        
        cur_data = data[target == class_index]
        cur_target = target[target == class_index]

        # Balance classes by taking only up to the minimum class size
        if sample_num_pc[class_index] > sample_num_min:
            cur_data = cur_data[:sample_num_min]
            cur_target = cur_target[:sample_num_min]
                
        cur_x_train, cur_x_test, cur_y_train, cur_y_test = train_test_split(
            cur_data, cur_target, test_size=self_ts, random_state=self_rs)
        
        if class_index == 0:
            x_train_all = cur_x_train
            x_test_all = cur_x_test 
            y_train_all = cur_y_train
            y_test_all = cur_y_test 
        else:
            x_train_all = np.append(x_train_all, cur_x_train, axis=0)
            x_test_all = np.append(x_test_all, cur_x_test, axis=0)
            y_train_all = np.append(y_train_all, cur_y_train, axis=0)
            y_test_all = np.append(y_test_all, cur_y_test, axis=0)

    return x_train_all, x_test_all, y_train_all, y_test_all


def parray_fquanz(model, data, feature_level):
    """Quantize feature level probabilities.
    
    Args:
        model: Trained Gaussian Naive Bayes model
        data: Input feature data
        feature_level: Number of quantization levels for features
    
    Returns:
        prob_array: Quantized probability array
        prob_array_log: Log-space quantized probability array
        quant_config: Quantization configuration parameters
    """
    feature_num = model.n_features_in_
    class_num = model.classes_.size
    prob_array = np.zeros((class_num, feature_num, feature_level))
    prob_array_log = np.zeros((class_num, feature_num, feature_level))
    quant_config = np.zeros((feature_num, 3))

    for feature_index in range(feature_num):
        # Determine quantization range and step size
        cur_min = np.min(data[:, feature_index])
        cur_max = np.max(data[:, feature_index])
        cur_gap = (cur_max - cur_min) / feature_level
        cur_series = np.linspace(cur_min, cur_max, num=feature_level, endpoint=False) + cur_gap / 2
        quant_config[feature_index, :] = [cur_min, cur_max, cur_gap]

        for class_index in range(class_num):
            # Calculate probability for each quantization level
            theta_cur = model.theta_[class_index, feature_index]
            var_cur = model.var_[class_index, feature_index]
            
            # Calculate probabilities and log-probabilities
            prob_fquanz = np.exp(-1 * ((cur_series - theta_cur) ** 2) / (2 * var_cur)) / np.sqrt(2 * np.pi * var_cur)
            prob_fquanz_log = -1 * ((cur_series - theta_cur) ** 2) / (2 * var_cur) - np.log(2 * np.pi * var_cur) / 2
            
            prob_array[class_index, feature_index, :] = prob_fquanz
            prob_array_log[class_index, feature_index, :] = prob_fquanz_log

    return prob_array, prob_array_log, quant_config


def input_quantize(x_test, quant_config, feature_level, feature_num):
    """Quantize input test data.
    
    Args:
        x_test: Test data to be quantized
        quant_config: Quantization configuration from parray_fquanz
        feature_level: Number of quantization levels
        feature_num: Number of features
    
    Returns:
        x_test_quant: Quantized test data
    """
    x_test_quant = []

    for sample_cur in x_test:
        sample_quant = []

        for feature_index in range(feature_num):
            feature_cur = sample_cur[feature_index]
            [cur_min, cur_max, cur_gap] = quant_config[feature_index, :]
            
            # Create quantization levels
            cur_series = np.linspace(cur_min, cur_max, num=feature_level, endpoint=False) + cur_gap / 2
            
            # Find closest quantization level
            quant_idx = np.argmin(np.abs(cur_series - feature_cur))
            sample_quant.append(quant_idx)

        x_test_quant.append(sample_quant)

    return np.array(x_test_quant)


def approx_predict(prob_array, x_test_quant, class_num, feature_num, log_style=False):
    """Make predictions using quantized model.
    
    Args:
        prob_array: Quantized probability array
        x_test_quant: Quantized test data
        class_num: Number of classes
        feature_num: Number of features
        log_style: Whether to use log-space probabilities (default: False)
    
    Returns:
        predictions: Class predictions
    """
    predictions = []

    for sample_cur in x_test_quant:
        class_probs = []

        for class_index in range(class_num):
            feature_probs = []

            for feature_index in range(feature_num):
                quant_idx = sample_cur[feature_index]
                prob = prob_array[class_index, feature_index, quant_idx]
                feature_probs.append(prob)

            # Combine probabilities across features
            if log_style:
                class_probs.append(np.sum(feature_probs))
            else:
                class_probs.append(np.prod(feature_probs))

        # Predict class with highest probability
        predictions.append(np.argmax(class_probs))

    return predictions
