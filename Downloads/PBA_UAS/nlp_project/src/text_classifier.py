"""
Text Classification Module untuk NLP
Mendukung Naive Bayes, SVM, Logistic Regression, Random Forest
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Install with: pip install scikit-learn")


class TextClassifier:
    """
    Kelas untuk klasifikasi teks dengan multiple algorithms
    Mendukung: Naive Bayes, SVM, Logistic Regression, Random Forest
    """
    
    def __init__(self):
        """Inisialisasi text classifier"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def train_naive_bayes(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train Multinomial Naive Bayes classifier
        
        Args:
            X_train: Training features (BoW/TF-IDF matrix)
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            dict: Model and evaluation results
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Training] Naive Bayes Classifier...")
        
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        
        results = {
            'model': model,
            'algorithm': 'Multinomial Naive Bayes',
            'training_size': X_train.shape[0]
        }
        
        # Evaluation
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            results.update(self._evaluate_model(y_test, y_pred, 'Naive Bayes'))
        
        self.models['naive_bayes'] = model
        self.results['naive_bayes'] = results
        
        return results
    
    def train_svm(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train Support Vector Machine (SVM) classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            dict: Model and evaluation results
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Training] SVM Classifier...")
        
        model = LinearSVC(max_iter=1000, random_state=42, loss='squared_hinge')
        model.fit(X_train, y_train)
        
        results = {
            'model': model,
            'algorithm': 'Linear SVM',
            'training_size': X_train.shape[0]
        }
        
        # Evaluation
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            results.update(self._evaluate_model(y_test, y_pred, 'SVM'))
        
        self.models['svm'] = model
        self.results['svm'] = results
        
        return results
    
    def train_logistic_regression(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train Logistic Regression classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            dict: Model and evaluation results
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Training] Logistic Regression Classifier...")
        
        model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X_train, y_train)
        
        results = {
            'model': model,
            'algorithm': 'Logistic Regression',
            'training_size': X_train.shape[0]
        }
        
        # Evaluation
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            results.update(self._evaluate_model(y_test, y_pred, 'Logistic Regression'))
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return results
    
    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None, n_estimators=100):
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            n_estimators: Number of trees
            
        Returns:
            dict: Model and evaluation results
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required")
            return None
        
        print("\n[Training] Random Forest Classifier...")
        
        # Convert sparse matrix to dense for Random Forest
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if X_test is not None and hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        )
        model.fit(X_train, y_train)
        
        results = {
            'model': model,
            'algorithm': 'Random Forest',
            'training_size': X_train.shape[0],
            'n_estimators': n_estimators
        }
        
        # Evaluation
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            results.update(self._evaluate_model(y_test, y_pred, 'Random Forest'))
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train semua models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            dict: Results dari semua models
        """
        print("\n" + "="*80)
        print("TRAINING ALL CLASSIFIERS")
        print("="*80)
        
        # Train Naive Bayes
        self.train_naive_bayes(X_train, y_train, X_test, y_test)
        
        # Train SVM
        self.train_svm(X_train, y_train, X_test, y_test)
        
        # Train Logistic Regression
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Find best model
        self._find_best_model()
        
        return self.results
    
    def _evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluasi model dengan berbagai metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Nama model untuk logging
            
        Returns:
            dict: Evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        evaluation = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'test_size': len(y_true)
        }
        
        # Classification report
        try:
            report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            evaluation['classification_report'] = report
        except:
            pass
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            evaluation['confusion_matrix'] = cm.tolist()
        except:
            pass
        
        print(f"  ✓ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return evaluation
    
    def _find_best_model(self):
        """Find best model based on F1 score"""
        best_score = 0
        best_model_name = None
        
        for name, results in self.results.items():
            if 'f1_score' in results:
                score = results['f1_score']
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        if best_model_name:
            self.best_model = best_model_name
            self.best_score = best_score
            print(f"\n✓ Best Model: {best_model_name.replace('_', ' ').title()} (F1: {best_score:.4f})")
    
    def predict(self, X, model_name=None):
        """
        Predict menggunakan trained model
        
        Args:
            X: Features to predict
            model_name: Model name (default: best model)
            
        Returns:
            array: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model trained. Train a model first.")
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        return model.predict(X)
    
    def predict_proba(self, X, model_name=None):
        """
        Predict probabilities
        
        Args:
            X: Features to predict
            model_name: Model name (default: best model)
            
        Returns:
            array: Probability predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model trained. Train a model first.")
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # For models without predict_proba
            return self.predict(X, model_name)
    
    def get_feature_importance(self, model_name='random_forest', feature_names=None, top_n=20):
        """
        Get feature importance dari model
        
        Args:
            model_name: Model name (usually random_forest)
            feature_names: List of feature names
            top_n: Top N features to return
            
        Returns:
            DataFrame: Feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  Model {model_name} does not support feature importance")
            return None
        
        importance = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def get_results_summary(self):
        """
        Get summary of all models
        
        Returns:
            DataFrame: Summary of all models
        """
        summary_data = []
        
        for model_name, results in self.results.items():
            summary_data.append({
                'Algorithm': results.get('algorithm', model_name),
                'Accuracy': results.get('accuracy', 'N/A'),
                'Precision': results.get('precision', 'N/A'),
                'Recall': results.get('recall', 'N/A'),
                'F1-Score': results.get('f1_score', 'N/A'),
                'Training Size': results.get('training_size', 'N/A'),
                'Test Size': results.get('test_size', 'N/A')
            })
        
        return pd.DataFrame(summary_data)


# Testing
if __name__ == "__main__":
    if SKLEARN_AVAILABLE:
        print("="*80)
        print("TEXT CLASSIFIER TEST")
        print("="*80)
        
        # Sample data
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [
            "Produk sangat bagus dan pengiriman cepat!",
            "Barang rusak dan tidak sesuai dengan deskripsi.",
            "Produknya biasa saja, tidak spesial.",
            "Tokopedia bagus tapi promo terbatas.",
            "Pengiriman lambat dan customer service tidak responsif.",
            "Sangat merekomendasikan produk ini!",
            "Jelek banget, tidak worth it.",
            "Baik-baik saja, acceptable quality."
        ]
        
        labels = ['positive', 'negative', 'neutral', 'positive', 'negative',
                  'positive', 'negative', 'neutral']
        
        # Vectorize
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        
        # Train-test split (80-20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train classifiers
        classifier = TextClassifier()
        classifier.train_all_models(X_train, y_train, X_test, y_test)
        
        # Results summary
        print("\n[Results Summary]")
        print(classifier.get_results_summary())
    else:
        print("⚠️  scikit-learn is required for testing")
