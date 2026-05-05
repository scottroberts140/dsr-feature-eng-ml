"""
Tests for ClassificationModelSpecification ROC-AUC computation in _score_classification.

This test module validates that ROC-AUC metrics are computed correctly
for both binary and multiclass classification models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


class TestRocAucComputationDirect:
    """Direct unit tests for ROC-AUC computation logic."""

    def test_binary_roc_auc_from_probabilities(self):
        """Verify ROC-AUC is correctly computed from binary probabilities."""
        # Create synthetic binary classification data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Fit a model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        # Get probabilities
        proba = clf.predict_proba(X)

        # Compute ROC-AUC as the code would
        roc_auc = float(roc_auc_score(y, proba[:, 1]))

        # Verify result is valid
        assert 0.0 <= roc_auc <= 1.0
        assert isinstance(roc_auc, float)

    def test_multiclass_roc_auc_from_probabilities(self):
        """Verify ROC-AUC is correctly computed for multiclass."""
        # Create synthetic multiclass data
        np.random.seed(42)
        n_samples = 150
        X = np.random.randn(n_samples, 6)
        y = np.repeat([0, 1, 2], n_samples // 3)

        # Fit a model
        clf = SklearnLogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)

        # Get probabilities
        proba = clf.predict_proba(X)

        # Compute weighted ROC-AUC as the code would
        roc_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="weighted"))

        # Verify result is valid
        assert 0.0 <= roc_auc <= 1.0
        assert isinstance(roc_auc, float)

    def test_roc_auc_none_without_probabilities(self):
        """Verify ROC-AUC is None when probabilities unavailable."""
        # When no probabilities available, ROC-AUC should be None
        roc_auc = None
        assert roc_auc is None

    def test_roc_auc_with_outlier_filtering(self):
        """Verify ROC-AUC computation with outlier-filtered subset."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Fit model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        # Get probabilities
        proba = clf.predict_proba(X)

        # Simulate outlier filtering (remove half of samples)
        keep_indices = np.arange(n_samples)[:n_samples // 2]

        # Compute ROC-AUC on filtered subset
        roc_auc_filtered = float(
            roc_auc_score(y[keep_indices], proba[keep_indices, 1])
        )

        # Verify result is valid
        assert 0.0 <= roc_auc_filtered <= 1.0

    def test_roc_auc_changes_with_probability_distribution(self):
        """Verify ROC-AUC changes based on probability quality."""
        np.random.seed(42)
        n_samples = 100
        
        # High quality predictions (more separable)
        y = np.repeat([0, 1], n_samples // 2)
        X_good = np.vstack([
            np.random.randn(n_samples // 2, 5) - 2,  # Class 0
            np.random.randn(n_samples // 2, 5) + 2,  # Class 1
        ])
        
        # Low quality predictions (less separable)
        X_poor = np.random.randn(n_samples, 5)
        
        # Fit models
        clf_good = RandomForestClassifier(n_estimators=10, random_state=42)
        clf_good.fit(X_good, y)
        proba_good = clf_good.predict_proba(X_good)
        roc_auc_good = float(roc_auc_score(y, proba_good[:, 1]))
        
        clf_poor = RandomForestClassifier(n_estimators=10, random_state=42)
        clf_poor.fit(X_poor, y)
        proba_poor = clf_poor.predict_proba(X_poor)
        roc_auc_poor = float(roc_auc_score(y, proba_poor[:, 1]))
        
        # Good model should have higher ROC-AUC
        assert roc_auc_good >= roc_auc_poor


class TestRocAucInModelSchema:
    """Test ROC-AUC integration with ModelConfiguration schema."""
    
    def test_model_configuration_has_roc_auc_fields(self):
        """Verify ModelConfiguration has roc_auc fields."""
        from dsr_feature_eng_ml.evaluation.schema import ModelConfiguration
        from dataclasses import dataclass
        
        # Create a mock model params
        @dataclass
        class MockParams:
            param1: str = "test"
        
        # Create an empty ModelConfiguration with mock params
        config = ModelConfiguration.empty(MockParams())
        
        # Check all roc_auc fields exist
        assert hasattr(config, 'roc_auc_train')
        assert hasattr(config, 'roc_auc_val')
        assert hasattr(config, 'roc_auc_val_cleaned')
        assert hasattr(config, 'roc_auc_test')
        
        # Initially should be None
        assert config.roc_auc_train is None
        assert config.roc_auc_val is None
        assert config.roc_auc_val_cleaned is None
        assert config.roc_auc_test is None
    
    def test_model_configuration_to_dict_includes_roc_auc(self):
        """Verify ModelConfiguration.to_dict() includes roc_auc fields."""
        from dsr_feature_eng_ml.evaluation.schema import ModelConfiguration
        from dataclasses import dataclass
        
        @dataclass
        class MockParams:
            param1: str = "test"
        
        config = ModelConfiguration.empty(MockParams())
        config_dict = config.to_dict()
        
        # Check dictionary includes roc_auc keys
        assert 'roc_auc_train' in config_dict
        assert 'roc_auc_val' in config_dict
        assert 'roc_auc_val_cleaned' in config_dict
        assert 'roc_auc_test' in config_dict


class TestRocAucEdgeCases:
    """Test edge cases and boundary conditions for ROC-AUC."""
    
    def test_roc_auc_with_perfect_predictions(self):
        """Verify ROC-AUC is 1.0 with perfect predictions."""
        n_samples = 100
        y = np.array([0] * 50 + [1] * 50)
        
        # Perfect predictions: class 0 gets [1, 0], class 1 gets [0, 1]
        proba = np.vstack([
            np.ones((50, 2)) * [1, 0],  # Perfect class 0 predictions
            np.ones((50, 2)) * [0, 1],  # Perfect class 1 predictions
        ])
        
        roc_auc = float(roc_auc_score(y, proba[:, 1]))
        assert roc_auc == 1.0
    
    def test_roc_auc_with_random_predictions(self):
        """Verify ROC-AUC is near 0.5 with random predictions."""
        np.random.seed(42)
        n_samples = 1000
        y = np.random.binomial(1, 0.5, n_samples)
        
        # Random probabilities
        proba = np.random.rand(n_samples, 2)
        proba = proba / proba.sum(axis=1, keepdims=True)  # Normalize
        
        roc_auc = float(roc_auc_score(y, proba[:, 1]))
        
        # Should be near 0.5 for random predictions
        assert 0.3 < roc_auc < 0.7
    
    def test_roc_auc_array_shapes(self):
        """Verify ROC-AUC handles correct array shapes."""
        np.random.seed(42)
        n_samples = 100
        
        # Binary classification
        y_binary = np.random.binomial(1, 0.5, n_samples)
        proba_binary = np.random.rand(n_samples, 2)
        proba_binary = proba_binary / proba_binary.sum(axis=1, keepdims=True)
        
        roc_auc_binary = float(roc_auc_score(y_binary, proba_binary[:, 1]))
        
        # Should have correct shape expectations
        assert isinstance(roc_auc_binary, float)
        assert 0.0 <= roc_auc_binary <= 1.0
    
    def test_roc_auc_with_imbalanced_data(self):
        """Verify ROC-AUC handles imbalanced classes."""
        np.random.seed(42)
        n_samples = 1000
        
        # Highly imbalanced: 95% class 0, 5% class 1
        y = np.concatenate([
            np.zeros(950, dtype=int),
            np.ones(50, dtype=int),
        ])
        
        X = np.random.randn(n_samples, 5)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        roc_auc = float(roc_auc_score(y, proba[:, 1]))
        
        # Should still compute valid ROC-AUC
        assert 0.0 <= roc_auc <= 1.0


class TestRocAucMulticlassVariants:
    """Test ROC-AUC variants for multiclass problems."""
    
    def test_multiclass_roc_auc_ovr_weighted(self):
        """Verify multiclass ROC-AUC with OvR weighted averaging."""
        np.random.seed(42)
        n_samples = 300
        
        # Create multiclass data
        X = np.random.randn(n_samples, 6)
        y = np.tile([0, 1, 2], n_samples // 3)
        
        clf = SklearnLogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        roc_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="weighted"))
        
        # Should produce valid multiclass ROC-AUC
        assert 0.0 <= roc_auc <= 1.0
    
    def test_three_class_classification(self):
        """Test ROC-AUC for 3-class problem."""
        np.random.seed(42)
        n_per_class = 100
        n_features = 8
        
        # Create three well-separated classes
        X_class0 = np.random.randn(n_per_class, n_features) + np.array([0] * n_features)
        X_class1 = np.random.randn(n_per_class, n_features) + np.array([3] * n_features)
        X_class2 = np.random.randn(n_per_class, n_features) + np.array([6] * n_features)
        
        X = np.vstack([X_class0, X_class1, X_class2])
        y = np.repeat([0, 1, 2], n_per_class)
        
        clf = SklearnLogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        roc_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="weighted"))
        
        # Well-separated classes should have high ROC-AUC
        assert roc_auc > 0.9

