import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    adjusted_rand_score, calinski_harabasz_score,
    normalized_mutual_info_score
)
from sklearn.datasets import make_blobs


try:
    from ..core.base_problem import BaseProblem
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.base_problem import BaseProblem



class ClusteringProblem(BaseProblem):
    def __init__(self, problem_name = "Clustering Problem"):
        super().__init__(problem_name)
        self.problem_type = 'clustering'
        
        self.X = None
        self.true_labels = None  
        self.raw_data = None  
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        
        self.n_samples = None
        self.n_features = None
        self.n_true_clusters = None
        self.has_true_labels = False
        
    def load_data(
        self,
        filepath = None,
        data = None,
        dataset_type = 'custom',
        features = None,
        target_column = None,
        normalize = True,
        **kwargs
    ):
        if dataset_type == 'iris':
            self.load_iris_data()
        elif dataset_type == 'synthetic':
            self.load_synthetic_data(**kwargs)
        elif dataset_type == 'mall_customers':
            if filepath is None and data is None:
                raise ValueError("Mall Customers dataset requires filepath or data")
            self._load_mall_customers_data(filepath, data)
        else:  # custom
            if filepath is not None:
                self.raw_data = pd.read_csv(filepath)
            elif data is not None:
                self.raw_data = data.copy()
            else:
                raise ValueError("Either filepath or data must be provided")
            
            self.preprocess_custom_data(features, target_column)
        
        if normalize and self.X is not None:
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
        
        self.n_samples, self.n_features = self.X.shape
        self.metadata.update({
            'dataset_type': dataset_type,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'normalized': normalize,
            'has_true_labels': self.has_true_labels
        })
        
        if self.has_true_labels:
            self.metadata['n_true_clusters'] = self.n_true_clusters
    
    def load_iris_data(self):
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        self.X = iris.data
        self.true_labels = iris.target
        self.feature_names = iris.feature_names
        self.has_true_labels = True
        self.n_true_clusters = len(np.unique(self.true_labels))
        
        self.raw_data = pd.DataFrame(self.X, columns=self.feature_names)
        self.raw_data['species'] = iris.target_names[self.true_labels]
        
        self.metadata['description'] = 'Iris flower dataset - 3 species classification'
        self.metadata['source'] = 'sklearn.datasets'
        self.metadata['species'] = iris.target_names.tolist()
    
    def load_synthetic_data(
        self,
        n_samples = 500,
        n_features = 2,
        n_clusters = 5,
        cluster_std = 1.0,
        random_state = 40000,
        **kwargs
    ):
        X, labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state,
            **kwargs
        )
        
        self.X = X
        self.true_labels = labels
        self.has_true_labels = True
        self.n_true_clusters = n_clusters
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.raw_data = pd.DataFrame(X, columns=self.feature_names)
        self.raw_data['true_cluster'] = labels
        
        self.metadata['description'] = f'Synthetic data with {n_clusters} well-separated clusters'
        self.metadata['source'] = 'sklearn.datasets.make_blobs'
        self.metadata['generation_params'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'cluster_std': cluster_std,
            'random_state': random_state
        }
    
    def _load_mall_customers_data(
        self,
        filepath = None,
        data = None
    ):
        if filepath is not None:
            df = pd.read_csv(filepath)
        else:
            df = data.copy()
        
        self.raw_data = df.copy()
        
        feature_cols = []
        
        age_cols = ['Age', 'age']
        income_cols = ['Annual Income (k$)', 'Annual Income', 'AnnualIncome', 'Income']
        spending_cols = ['Spending Score (1-100)', 'Spending Score', 'SpendingScore', 'Score']
        
        for col in age_cols:
            if col in df.columns:
                feature_cols.append(col)
                break
        
        for col in income_cols:
            if col in df.columns:
                feature_cols.append(col)
                break
        
        for col in spending_cols:
            if col in df.columns:
                feature_cols.append(col)
                break
        
        if len(feature_cols) < 3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if 'id' not in col.lower() and 'customer' not in col.lower()]
        
        self.feature_names = feature_cols
        self.X = df[feature_cols].values
        self.true_labels = None
        self.has_true_labels = False
        
        self.metadata['description'] = 'Mall Customer Segmentation - demographic and spending behavior'
        self.metadata['source'] = 'Kaggle Mall Customer Segmentation Dataset'
        self.metadata['expected_clusters'] = '4-6 customer segments'
    
    def preprocess_custom_data(
        self,
        features = None,
        target_column = None
    ):
        df = self.raw_data.copy()
        
        if target_column is not None and target_column in df.columns:
            self.true_labels = df[target_column].values
            
            if not np.issubdtype(self.true_labels.dtype, np.number):
                self.true_labels = self.label_encoder.fit_transform(self.true_labels)
            
            self.has_true_labels = True
            self.n_true_clusters = len(np.unique(self.true_labels))
            df = df.drop(columns=[target_column])
        
        if features is not None:
            feature_cols = features
        else:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = feature_cols
        self.X = df[feature_cols].values
    
    def evaluate(self, solution) -> float:
        labels = self.extract_labels(solution)
        
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= len(labels):
            return -1.0
        
        try:
            score = silhouette_score(self.X, labels)
            return float(score)
        except Exception:
            return -1.0
    
    def compute_metrics(
        self,
        solution,
    ):
        labels = self.extract_labels(solution)
        
        n_clusters = len(np.unique(labels))
        n_samples = len(labels)
        
        metrics = {
            'n_clusters': int(n_clusters),
            'n_samples': n_samples,
            'cluster_sizes': {}
        }
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            metrics['cluster_sizes'][int(label)] = int(count)
        
        if n_clusters >= 2 and n_clusters < n_samples:
            try:
                metrics['silhouette_score'] = float(silhouette_score(self.X, labels))
            except Exception as e:
                metrics['silhouette_score'] = None
                metrics['silhouette_error'] = str(e)
        else:
            metrics['silhouette_score'] = None
        
        if n_clusters >= 2:
            try:
                metrics['davies_bouldin_index'] = float(davies_bouldin_score(self.X, labels))
            except Exception as e:
                metrics['davies_bouldin_index'] = None
                metrics['davies_bouldin_error'] = str(e)
        else:
            metrics['davies_bouldin_index'] = None
        
        if n_clusters >= 2 and n_clusters < n_samples:
            try:
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(self.X, labels))
            except Exception as e:
                metrics['calinski_harabasz_score'] = None
        else:
            metrics['calinski_harabasz_score'] = None
        
        if self.has_true_labels and self.true_labels is not None:
            try:
                # Adjusted Rand Index (range: -1 to 1, 1 is perfect)
                metrics['adjusted_rand_index'] = float(adjusted_rand_score(self.true_labels, labels))
                
                # Normalized Mutual Information (range: 0 to 1)
                metrics['normalized_mutual_info'] = float(
                    normalized_mutual_info_score(self.true_labels, labels)
                )
            except Exception as e:
                metrics['adjusted_rand_index'] = None
                metrics['normalized_mutual_info'] = None
                metrics['supervised_metrics_error'] = str(e)
        else:
            metrics['adjusted_rand_index'] = None
            metrics['normalized_mutual_info'] = None
        
        metrics['cluster_statistics'] = self.compute_cluster_statistics(labels)
        
        return metrics
    
    def compute_cluster_statistics(self, labels):
        stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_points = self.X[cluster_mask]
            
            stats[int(cluster_id)] = {
                'size': int(np.sum(cluster_mask)),
                'centroid': cluster_points.mean(axis=0).tolist(),
                'std': cluster_points.std(axis=0).tolist(),
                'min': cluster_points.min(axis=0).tolist(),
                'max': cluster_points.max(axis=0).tolist()
            }
        
        return stats
    
    def extract_labels(self, solution):
        if hasattr(solution, 'labels_'):
            # Solution is a clustering model
            return solution.labels_
        elif hasattr(solution, 'predict'):
            # Solution has predict method
            return solution.predict(self.X)
        else:
            # Solution is array of labels
            return np.array(solution).astype(int)
    
    def validate_solution(self, solution):
        try:
            labels = self.extract_labels(solution)
            
            if len(labels) != self.n_samples:
                return False
            
            # Check that there are clusters
            n_clusters = len(np.unique(labels))
            if n_clusters < 1 or n_clusters > self.n_samples:
                return False
            
            if not np.all(labels >= 0):
                return False
            
            return True
        except Exception:
            return False
    
    def get_dimension(self):
        return self.n_features
    
    def get_bounds(self):
        if self.X is not None:
            lower = np.min(self.X, axis=0)
            upper = np.max(self.X, axis=0)
            return (lower, upper)
        return None
    
    def get_data(self):
        return self.X
    
    def get_true_labels(self):
        return self.true_labels if self.has_true_labels else None
    
    def __repr__(self):
        return (
            f"ClusteringProblem(name='{self.problem_name}', "
            f"n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"has_labels={self.has_true_labels})"
        )


class IrisProblem(ClusteringProblem):
    def __init__(self):
        super().__init__(problem_name="Iris Clustering")
        self.metadata['description'] = 'Cluster iris flowers into species groups'
    
    def load_data(self, **kwargs):
        super().load_data(dataset_type='iris', **kwargs)


class MallCustomersProblem(ClusteringProblem):
    def __init__(self):
        super().__init__(problem_name="Mall Customer Segmentation")
        self.metadata['description'] = 'Segment customers by demographics and spending behavior'
    
    def load_data(self, filepath = None, data = None, **kwargs):
        super().load_data(filepath=filepath, data=data, dataset_type='mall_customers', **kwargs)


class SyntheticClusteringProblem(ClusteringProblem):
    
    def __init__(self, n_clusters = 5):
        super().__init__(problem_name=f"Synthetic Clustering ({n_clusters} clusters)")
        self.n_clusters_param = n_clusters
    
    def load_data(
        self,
        n_samples = 500,
        n_features = 2,
        cluster_std = 1.0,
        random_state = 40000,
        **kwargs
    ):
        super().load_data(
            dataset_type='synthetic',
            n_samples=n_samples,
            n_features=n_features,
            n_clusters=self.n_clusters_param,
            cluster_std=cluster_std,
            random_state=random_state,
            **kwargs
        )
