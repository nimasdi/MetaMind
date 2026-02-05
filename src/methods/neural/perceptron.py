import sys
sys.path.insert(0, 'e:\\Taha\\Term 7\\Computational Intelligence\\Project\\metamind')
from src.core.base_method import BaseMethod

import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Perceptron(BaseMethod):

    PARAM_SPECS = {
        'learning_rate': {
            'default' : 0.01,
            'type': float,
            'range': (0.001, 0.1),
            'description': 'Learning Rate'
        },
        'max_epochs': {
            'default': 100,
            'type': int,
            'range': (50, 1000),
            'description': 'Maximum Training Epochs'
        },
        'bias': {
            'default': True,
            'type': bool,
            'options': [True, False],
            'description': 'Bias is available or not'
        }
    }

    def __init__(self, **parameters):
        full_params = self.get_default_parameters()
        full_params.update(parameters)
        super().__init__(**full_params)

        self.weights = None
        self.bias_term = None
        self.classes_ = None
        self.n_features_ = None


    def fit(self, problem_data, **kwargs):

        self.start_time = time.time()
        self.log("Starting Perceptron training with Hebbian Learning")

        X_train = problem_data['X_train']
        y_train = problem_data['y_train']

        if isinstance(X_train, list):
            X_train = np.array(X_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        X_test = problem_data.get('X_test', None)
        y_test = problem_data.get('y_test', None)

        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)

        if n_classes > 2:
            raise ValueError(
                f"Perceptron is for binary classification. "
                f"Found {n_classes} classes: {self.classes_}"
            )
        
        label_map = {self.classes_[0]: -1, self.classes_[1]: 1}
        y_train_hebbian = np.array([label_map[label] for label in y_train])

        n_samples, self.n_features_= X_train.shape

        np.random.seed(404)
        self.weights = np.random.randn(self.n_features_) * 0.01

        if self.parameters['bias']:
            self.bias_term = np.random.randn() * 0.01
        else:
            self.bias_term = 0.0

        self.log(f"Initialized weights: shape={self.weights.shape}, " f"bias={self.bias_term if self.parameters['bias'] else 'None'}")

        learning_rate = self.parameters['learning_rate']
        max_epochs = self.parameters['max_epochs']

        train_accuracies = []
        test_accuracies = []
        weight_magnitudes = []

        best_accuracy = 0.0
        best_weights= self.weights.copy()
        best_bias = self.bias_term

        for epoch in range(max_epochs):
            epoch_correct = 0

            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_hebbian[indices]

            for i in range(n_samples):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                linear_output = np.dot(self.weights, x_i) + self.bias_term
                y_pred = 1 if linear_output >= 0 else -1

                if y_pred != y_i:
                    self.weights += learning_rate * x_i * (y_i - y_pred)

                    if self.parameters['bias']:
                        self.bias_term += learning_rate * (y_i - y_pred)
                
                linear_output = np.dot(self.weights, x_i) + self.bias_term
                y_pred = 1 if linear_output >= 0 else -1
                if y_pred == y_i:
                    epoch_correct += 1

            epoch_accuracy = epoch_correct / n_samples
            train_accuracies.append(epoch_accuracy)

            weight_norm = np.linalg.norm(self.weights)
            weight_magnitudes.append(weight_norm)

            self.convergence_history.append(epoch_accuracy)

            if X_test is not None and y_test is not None:
                test_acc = self._calculate_accuarcy(X_test, y_test)
                test_accuracies.append(test_acc)

                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_weights = self.weights.copy()
                    best_bias = self.bias_term

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = (f"Epoch {epoch+1}/{max_epochs}: " f"Train Acc={epoch_accuracy:.4f}")
                if test_accuracies:
                    log_msg += f", Test Acc={test_accuracies[-1]:.4f}"
                self.log(log_msg)
            
            if epoch_accuracy >= 1.0:
                self.log(f"Perfect training accuracy reached at epoch {epoch+1}")
                break
        
        if X_test is not None and y_test is not None:
            self.weights = best_weights
            self.bias_term = best_bias

        final_train_accuracy = self._calculate_accuracy(X_train, y_train)

        final_test_accuracy = None
        if X_test is not None and y_test is not None:
            final_test_accuracy = self._calculate_accuracy(X_test, y_test)

        self.end_time = time.time()
        computation_time = self.end_time - self.start_time

        self.results = {
            'method_used': 'Perceptron (Hebbian)',
            'weights': self.weights.copy(),
            'bias': self.bias_term if self.parameters['bias'] else None,
            'n_features': self.n_features_,
            'classes': self.classes_,
            'final_train_accuracy': final_train_accuracy,
            'final_test_accuracy': final_test_accuracy,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'weight_magnitudes': weight_magnitudes,
            'computation_time': computation_time,
            'epochs_completed': len(train_accuracies),
            'convergence_history': self.convergence_history.copy(),
            'parameters': self.parameters.copy()
        }

        self.log(f"Training completed in {computation_time:.2f}s")
        self.log(f"Final train accuracy: {final_train_accuracy:.4f}")
        if final_test_accuracy is not None:
            self.log(f"Final test accuracy: {final_test_accuracy:.4f}")
        
        return self.results

    def _calculate_accuarcy(self, X, y):

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        if isinstance(X, list):
            X = np.array(X)

        linear_output = np.dot(X, self.weights) + self.bias_term

        binary_predictions = (linear_output >= 0).astype(int)

        predictions = np.where(
            binary_predictions == 0,
            self.classes_[0],
            self.classes_[1]
        )
        
        return predictions
    
    def decision_function(self, X):

        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        if isinstance(X, list):
            X = np.array(X)
        
        return np.dot(X, self.weights) + self.bias_term
    
    def get_weights(self):

        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias_term if self.parameters['bias'] else None
        }
