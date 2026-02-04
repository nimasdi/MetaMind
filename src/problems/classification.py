import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from ..core.base_problem import BaseProblem
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.base_problem import BaseProblem

class ClassificationProblem(BaseProblem):
    
    def __init__(self, problem_name: str = "Classification Problem"):
        super().__init__(problem_name)
        self.problem_type = 'classification'
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.raw_data = None
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = 'Survived'
        
        self.n_classes = 2
        self.n_features = None
        self.class_balance = {}
        
    def load_data(
        self,
        filepath = None,
        data = None,
        target_column = 'Survived',
        test_size = 0.15,
        val_size = 0.15,
        random_state = 40000,
        **kwargs
    ):
        if filepath is not None:
            self.raw_data = pd.read_csv(filepath)
        elif data is not None:
            self.raw_data = data.copy()
        else:
            raise ValueError("Either filepath or data must be provided")
        
        self.target_name = target_column
        
        self.metadata['original_samples'] = len(self.raw_data)
        self.metadata['original_features'] = len(self.raw_data.columns)
        
        X, y = self.preprocess_data(self.raw_data, target_column)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
        
        val_relative_size = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_relative_size), 
            random_state=random_state, stratify=y_temp
        )
        
        self.scaler.fit(X_train)
        
        self.X_train = self.scaler.transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.n_features = self.X_train.shape[1]
        self.metadata.update({
            'n_samples_train': len(self.X_train),
            'n_samples_val': len(self.X_val),
            'n_samples_test': len(self.X_test),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        })
        
        unique, counts = np.unique(y_train, return_counts=True)
        self.class_balance = {
            int(cls): {
                'count': int(count),
                'percentage': float(count / len(y_train) * 100)
            }
            for cls, count in zip(unique, counts)
        }
        self.metadata['class_balance'] = self.class_balance
        
    def preprocess_data(self, data, target_column):
        df = data.copy()
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = df[target_column].values
        df = df.drop(columns=[target_column])
        
        # drop columns with too many missing values or low utility
        columns_to_drop = []
        if 'PassengerId' in df.columns:
            columns_to_drop.append('PassengerId')
        if 'Name' in df.columns:
            columns_to_drop.append('Name')
        if 'Ticket' in df.columns:
            columns_to_drop.append('Ticket')
        if 'Cabin' in df.columns and df['Cabin'].isna().sum() > len(df) * 0.7:
            columns_to_drop.append('Cabin')
        
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # age: fill with median
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # embarked: fill with mode
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        # fare: fill with median
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        self.feature_names = df.columns.tolist()
        
        X = df.values
        
        return X, y
    
    def evaluate(self, solution):
        if hasattr(solution, 'predict'):
            predictions = solution.predict(self.X_val)
        elif isinstance(solution, tuple) and len(solution) == 2:
            predictions = solution[0]
        else:
            predictions = solution
        
        return accuracy_score(self.y_val, predictions)
    
    def compute_metrics(
        self,
        solution,
        dataset = 'validation'
    ):
        if dataset == 'train':
            X, y_true = self.X_train, self.y_train
        elif dataset == 'test':
            X, y_true = self.X_test, self.y_test
        else:  # validation
            X, y_true = self.X_val, self.y_val
        
        if hasattr(solution, 'predict'):
            # solution is a model
            predictions = solution.predict(X)
            if hasattr(solution, 'predict_proba'):
                probabilities = solution.predict_proba(X)
            else:
                probabilities = None
        elif isinstance(solution, tuple) and len(solution) == 2:
            # solution is (predictions, probabilities)
            predictions, probabilities = solution
        else:
            # solution is predictions array
            predictions = solution
            probabilities = None
        
        predictions = np.array(predictions).astype(int)
        
        metrics = {
            'dataset': dataset,
            'accuracy': float(accuracy_score(y_true, predictions)),
            'precision': float(precision_score(y_true, predictions, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, predictions, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_true, predictions, average='binary', zero_division=0)),
        }
        
        if probabilities is not None:
            try:
                if probabilities.ndim > 1 and probabilities.shape[1] == 2:
                    metrics['auc_roc'] = float(roc_auc_score(y_true, probabilities[:, 1]))
                else:
                    metrics['auc_roc'] = float(roc_auc_score(y_true, probabilities))
            except (ValueError, IndexError):
                metrics['auc_roc'] = None
        else:
            metrics['auc_roc'] = None
        
        cm = confusion_matrix(y_true, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        metrics['classification_report'] = classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        )
        
        metrics['n_samples'] = len(y_true)
        metrics['n_features'] = X.shape[1]
        
        return metrics
    
    def validate_solution(self, solution):
        try:
            if hasattr(solution, 'predict'):
                predictions = solution.predict(self.X_val)
            elif isinstance(solution, tuple):
                predictions = solution[0]
            else:
                predictions = solution
            
            predictions = np.array(predictions)
            
            if len(predictions) != len(self.y_val):
                return False
            
            unique_values = np.unique(predictions)
            if not all(val in [0, 1] for val in unique_values):
                return False
            
            return True
        except Exception:
            return False
    
    def get_dimension(self):
        return self.n_features
    
    def get_bounds(self):
        if self.X_train is not None:
            lower = np.min(self.X_train, axis=0)
            upper = np.max(self.X_train, axis=0)
            return (lower, upper)
        return None
    
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        return self.X_val, self.y_val
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_all_data(self):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.concatenate([self.y_train, self.y_val, self.y_test])
        return X, y
    
    def __repr__(self) -> str:
        return (
            f"ClassificationProblem(name='{self.problem_name}', "
            f"n_samples={self.metadata.get('n_samples_train', 'N/A')}, "
            f"n_features={self.n_features}, "
            f"n_classes={self.n_classes})"
        )


class TitanicProblem(ClassificationProblem):
    def __init__(self):
        super().__init__(problem_name="Titanic Survival Prediction")
        
        self.n_classes = 2
        self.optimal_value = None  # no known optimal (depends on test set)
        
        self.metadata.update({
            'description': 'Predict passenger survival on the Titanic',
            'source': 'Kaggle Titanic Dataset',
            'classes': ['Did not survive', 'Survived'],
            'typical_balance': '~38% survived, ~62% died'
        })
    
    def load_data(
        self,
        filepath=None,
        data=None,
        **kwargs
    ):
        kwargs.setdefault('target_column', 'Survived')
        kwargs.setdefault('test_size', 0.15)
        kwargs.setdefault('val_size', 0.15)
        kwargs.setdefault('random_state', 40000)
        
        super().load_data(filepath=filepath, data=data, **kwargs)
        
        self.metadata['problem_specific'] = {
            'features_used': self.feature_names,
            'preprocessing': [
                'Dropped: PassengerId, Name, Ticket, Cabin (high missing rate)',
                'Age: filled missing with median',
                'Embarked: filled missing with mode',
                'Fare: filled missing with median',
                'Categorical encoding: Sex, Embarked (LabelEncoder)',
                'Normalization: StandardScaler on all features'
            ]
        }
