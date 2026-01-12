"""
Modeling Engine Module untuk NLP
Hyperparameter Tuning dan Model Optimization
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        make_scorer, accuracy_score, precision_score, recall_score, f1_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Install with: pip install scikit-learn")


class ModelingEngine:
    """
    Kelas untuk model optimization dan hyperparameter tuning
    """
    
    def __init__(self, verbose=1):
        """
        Inisialisasi modeling engine
        
        Args:
            verbose: Verbosity level (0, 1, or 2)
        """
        self.verbose = verbose
        self.best_models = {}
        self.cv_results = {}
    
    def tune_naive_bayes(self, X_train, y_train, cv=5):
        """
        Tune Naive Bayes dengan GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            
        Returns:
            dict: Best parameters dan model
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Tuning] Naive Bayes...")
        
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'fit_prior': [True, False]
        }
        
        model = MultinomialNB()
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            verbose=self.verbose,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': round(grid_search.best_score_, 4),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        self.best_models['naive_bayes'] = results
        
        print(f"  ✓ Best params: {results['best_params']}")
        print(f"  ✓ Best F1 score (CV): {results['best_score']}")
        
        return results
    
    def tune_svm(self, X_train, y_train, cv=5):
        """
        Tune SVM dengan GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            
        Returns:
            dict: Best parameters dan model
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Tuning] SVM...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'max_iter': [1000, 2000]
        }
        
        model = LinearSVC(random_state=42, loss='squared_hinge')
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            verbose=self.verbose,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': round(grid_search.best_score_, 4),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        self.best_models['svm'] = results
        
        print(f"  ✓ Best params: {results['best_params']}")
        print(f"  ✓ Best F1 score (CV): {results['best_score']}")
        
        return results
    
    def tune_logistic_regression(self, X_train, y_train, cv=5):
        """
        Tune Logistic Regression dengan GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            
        Returns:
            dict: Best parameters dan model
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Tuning] Logistic Regression...")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [500, 1000, 2000],
            'solver': ['lbfgs', 'liblinear']
        }
        
        model = LogisticRegression(random_state=42)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            verbose=self.verbose,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': round(grid_search.best_score_, 4),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        self.best_models['logistic_regression'] = results
        
        print(f"  ✓ Best params: {results['best_params']}")
        print(f"  ✓ Best F1 score (CV): {results['best_score']}")
        
        return results
    
    def tune_random_forest(self, X_train, y_train, cv=5):
        """
        Tune Random Forest dengan GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            
        Returns:
            dict: Best parameters dan model
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Tuning] Random Forest...")
        
        # Convert sparse to dense
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            verbose=self.verbose,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': round(grid_search.best_score_, 4),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        self.best_models['random_forest'] = results
        
        print(f"  ✓ Best params: {results['best_params']}")
        print(f"  ✓ Best F1 score (CV): {results['best_score']}")
        
        return results
    
    def tune_all_models(self, X_train, y_train, cv=5):
        """
        Tune semua models
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation folds
            
        Returns:
            dict: Results dari semua tuning
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - ALL MODELS")
        print("="*80)
        
        self.tune_naive_bayes(X_train, y_train, cv)
        self.tune_svm(X_train, y_train, cv)
        self.tune_logistic_regression(X_train, y_train, cv)
        self.tune_random_forest(X_train, y_train, cv)
        
        return self.best_models
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        cv_results = cross_validate(
            model,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        
        # Calculate mean and std
        summary = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            summary[f'{metric}_test'] = {
                'mean': round(test_scores.mean(), 4),
                'std': round(test_scores.std(), 4)
            }
            summary[f'{metric}_train'] = {
                'mean': round(train_scores.mean(), 4),
                'std': round(train_scores.std(), 4)
            }
        
        return summary
    
    def get_best_model(self, model_name):
        """
        Get best tuned model
        
        Args:
            model_name: Name of the model
            
        Returns:
            sklearn model: Best tuned model
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in tuned models")
        
        return self.best_models[model_name]['best_model']
    
    def get_tuning_results_summary(self):
        """
        Get summary of tuning results
        
        Returns:
            DataFrame: Summary of all tuned models
        """
        summary_data = []
        
        for model_name, results in self.best_models.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Best Params': str(results['best_params']),
                'Best F1 Score': results['best_score']
            })
        
        return pd.DataFrame(summary_data)


# Testing
if __name__ == "__main__":
    if SKLEARN_AVAILABLE:
        print("="*80)
        print("MODELING ENGINE TEST")
        print("="*80)
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        
        # Sample data
        texts = [
            "Produk sangat bagus dan pengiriman cepat!",
            "Barang rusak dan tidak sesuai dengan deskripsi.",
            "Produknya biasa saja, tidak spesial.",
            "Tokopedia bagus tapi promo terbatas.",
            "Pengiriman lambat dan customer service tidak responsif.",
            "Sangat merekomendasikan produk ini!",
            "Jelek banget, tidak worth it.",
            "Baik-baik saja, acceptable quality.",
            "Sempurna, akan membeli lagi!",
            "Mengecewakan, penipuan!"
        ]
        
        labels = ['positive', 'negative', 'neutral', 'positive', 'negative',
                  'positive', 'negative', 'neutral', 'positive', 'negative']
        
        # Vectorize
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, random_state=42
        )
        
        # Tune models
        engine = ModelingEngine(verbose=0)
        engine.tune_all_models(X_train, y_train, cv=3)
        
        # Results summary
        print("\n[Tuning Results Summary]")
        print(engine.get_tuning_results_summary())
    else:
        print("⚠️  scikit-learn is required for testing")
