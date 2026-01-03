import time
import numpy as np
import sys
import copy
from pathlib import Path

try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(BaseMethod):    
    PARAM_SPECS = {
        'hidden_layers': {
            'default': [64, 32],
            'type': list,
            'description': 'List of hidden layer sizes'
        },
        'activation': {
            'default': 'relu',
            'type': str,
            'options': ['relu', 'sigmoid', 'tanh'],
            'description': 'Activation function'
        },
        'learning_rate': {
            'default': 0.001,
            'type': float,
            'range': (0.0001, 0.01),
            'description': 'Learning rate'
        },
        'max_epochs': {
            'default': 500,
            'type': int,
            'range': (100, 2000),
            'description': 'Maximum training epochs'
        },
        'batch_size': {
            'default': 32,
            'type': int,
            'range': (16, 128),
            'description': 'Batch size for training'
        },
        'optimizer': {
            'default': 'adam',
            'type': str,
            'options': ['adam', 'sgd', 'rmsprop'],
            'description': 'Optimizer type'
        },
        'validation_split': {
            'default': 0.15,
            'type': float,
            'range': (0.1, 0.3),
            'description': 'Validation set proportion'
        },
        'early_stopping_patience': {
            'default': 50,
            'type': int,
            'range': (10, 200),
            'description': 'Early stopping patience'
        },
        'random_state': {
            'default': 404,
            'type': int,
            'description': 'Random seed for reproducibility'
        }
    }
    
    def __init__(self, **parameters):
        full_params = self.get_default_parameters()
        full_params.update(parameters)
        super().__init__(**full_params)
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_state = None
        
    def _build_model(self, input_size: int, output_size: int) -> nn.Module:
        layers = []
        hidden_layers = self.parameters['hidden_layers']
        activation = self.parameters['activation']
        
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        act_fn = activation_map[activation]
        
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(0.2)) 
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        lr = self.parameters['learning_rate']
        optimizer_type = self.parameters['optimizer']
        
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def fit(self, problem_data, **kwargs) :

        self.start_time = time.time()
        self.log("Starting MLP training")
        
        X_train = problem_data['X_train']
        y_train = problem_data['y_train']
        X_test = problem_data.get('X_test', None)
        y_test = problem_data.get('y_test', None)
        
        if isinstance(X_train, list):
            X_train = np.array(X_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)
        
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
        
        val_split = self.parameters['validation_split']
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_split,
            random_state=self.parameters['random_state'],
            stratify=y_train
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_train))
        self.model = self._build_model(input_size, output_size).to(self.device)
        
        self.log(f"Model architecture: input={input_size}, "
                f"hidden={self.parameters['hidden_layers']}, output={output_size}")
        self.log(f"Device: {self.device}")
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.parameters['batch_size'],
            shuffle=True
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self._get_optimizer(self.model)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.parameters['early_stopping_patience']
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.parameters['max_epochs']):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)
                
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = (predicted == y_val_t).float().mean().item()
                val_accuracies.append(val_accuracy)
            
            self.convergence_history.append(-val_loss)
            
            if (epoch + 1) % 50 == 0:
                self.log(f"Epoch {epoch+1}/{self.parameters['max_epochs']}: "
                        f"Train Loss={avg_train_loss:.4f}, "
                        f"Val Loss={val_loss:.4f}, "
                        f"Val Acc={val_accuracy:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.log(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_t)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val_t).float().mean().item()
        
        self.end_time = time.time()
        computation_time = self.end_time - self.start_time
        
        test_accuracy = None
        if X_test is not None and y_test is not None:
            X_test = self.scaler.transform(X_test)
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            y_test_t = torch.LongTensor(y_test).to(self.device)
            
            with torch.no_grad():
                test_outputs = self.model(X_test_t)
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy = (test_predicted == y_test_t).float().mean().item()
            
            self.log(f"Test accuracy: {test_accuracy:.4f}")
        
        self.results = {
            'method_used': 'MLP',
            'best_val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'test_accuracy': test_accuracy,
            'computation_time': computation_time,
            'epochs_completed': len(train_losses),
            'convergence_history': self.convergence_history,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'parameters': self.parameters.copy()
        }
        
        self.log(f"Training completed in {computation_time:.2f}s")
        self.log(f"Best validation accuracy: {val_accuracy:.4f}")
        
        return self.results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        
        X = self.scaler.transform(X)
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        
        X = self.scaler.transform(X)
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

