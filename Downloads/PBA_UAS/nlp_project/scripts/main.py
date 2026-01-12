"""
Main NLP Pipeline untuk Tokopedia Dataset Analysis
Mengintegrasikan semua komponen: Sentiment, Features, Classification, NER, Modeling
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import NLP modules
from sentiment_analyzer import SentimentAnalyzer
from feature_extraction import TextFeatureExtractor
from text_classifier import TextClassifier
from modeling_engine import ModelingEngine
from named_entity_recognition import NamedEntityRecognizer, POS_Tagger
from visualization import VisualizationHelper, AnalysisReporter
from data_processor import DataProcessor


class NLPPipeline:
    """
    Main NLP Pipeline untuk analisis komprehensif
    """
    
    def __init__(self, output_dir='output'):
        """
        Inisialisasi NLP Pipeline
        
        Args:
            output_dir: Output directory untuk results
        """
        self.output_dir = output_dir
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_extractor = TextFeatureExtractor()
        self.text_classifier = TextClassifier()
        self.modeling_engine = ModelingEngine(verbose=0)
        self.ner = NamedEntityRecognizer()
        self.pos_tagger = POS_Tagger()
        self.visualizer = VisualizationHelper()
        self.reporter = AnalysisReporter(output_dir)
        self.data_processor = DataProcessor()
        
        # Results storage
        self.results = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self, csv_file):
        """
        Load dataset
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            DataFrame: Loaded data
        """
        print("\n" + "="*80)
        print("[STEP 1] LOADING DATA")
        print("="*80)
        
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            return None
        
        df = pd.read_csv(csv_file)
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"✓ Columns: {', '.join(df.columns)}")
        
        self.results['data'] = df
        self.results['text_column'] = 'text_clean' if 'text_clean' in df.columns else 'text_original'
        
        return df
    
    def analyze_sentiments(self, df, text_column):
        """
        Analyze sentiments
        
        Args:
            df: DataFrame
            text_column: Text column name
            
        Returns:
            DataFrame: DataFrame with sentiment results
        """
        print("\n" + "="*80)
        print("[STEP 2] SENTIMENT ANALYSIS")
        print("="*80)
        
        texts = df[text_column].astype(str).tolist()
        
        print(f"\nAnalyzing {len(texts)} texts...")
        
        # Analyze with lexicon method
        sentiment_results = []
        for i, text in enumerate(texts):
            result = self.sentiment_analyzer.analyze_sentiment_combined(text)
            sentiment_results.append({
                'positive_words': result['lexicon']['positive_count'],
                'negative_words': result['lexicon']['negative_count'],
                'sentiment': result['final_sentiment'],
                'confidence': round(result.get('confidence', 0), 4)
            })
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(texts)} texts...")
        
        # Create results DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        results_df = pd.concat([df, sentiment_df], axis=1)
        
        # Distribution
        sentiment_dist = self.sentiment_analyzer.get_sentiment_distribution(
            sentiment_df['sentiment'].tolist()
        )
        
        print("\nSentiment Distribution:")
        for sentiment, stats in sentiment_dist.items():
            print(f"  {sentiment}: {stats['count']} ({stats['percentage']}%)")
        
        self.results['sentiment'] = sentiment_df
        self.results['sentiment_distribution'] = sentiment_dist
        
        return results_df
    
    def extract_features(self, df, text_column):
        """
        Extract text features
        
        Args:
            df: DataFrame
            text_column: Text column name
            
        Returns:
            dict: Feature extraction results
        """
        print("\n" + "="*80)
        print("[STEP 3] FEATURE EXTRACTION")
        print("="*80)
        
        texts = df[text_column].astype(str).tolist()
        
        # Word frequency
        print("\nExtracting word frequency...")
        word_freq = self.feature_extractor.extract_word_frequency(texts)
        print(f"✓ Found {len(word_freq)} unique words")
        print(f"  Top 5: {', '.join(word_freq.head()['word'].tolist())}")
        
        # Bigrams
        print("\nExtracting bigrams...")
        bigrams = self.feature_extractor.extract_bigrams(texts, top_n=200)
        print(f"✓ Found {len(bigrams)} unique bigrams")
        print(f"  Top 5: {', '.join(bigrams.head()['bigram'].tolist())}")
        
        # Text statistics
        print("\nCalculating text statistics...")
        text_stats = self.feature_extractor.extract_text_statistics(texts)
        print(f"✓ Average text length: {text_stats['avg_words_per_document']} words")
        
        self.results['word_frequency'] = word_freq
        self.results['bigrams'] = bigrams
        self.results['text_stats'] = text_stats
        
        return {
            'word_frequency': word_freq,
            'bigrams': bigrams,
            'text_stats': text_stats
        }
    
    def extract_named_entities(self, df, text_column, max_texts=None):
        """
        Extract named entities
        
        Args:
            df: DataFrame
            text_column: Text column name
            max_texts: Maximum number of texts to process (for performance)
            
        Returns:
            dict: NER results
        """
        print("\n" + "="*80)
        print("[STEP 4] NAMED ENTITY RECOGNITION (NER)")
        print("="*80)
        
        if self.ner.nlp is None:
            print("⚠️  spaCy model not available. Skipping NER...")
            return None
        
        texts = df[text_column].astype(str).tolist()
        
        if max_texts:
            texts = texts[:max_texts]
        
        print(f"\nExtracting entities from {len(texts)} texts...")
        
        # Extract entities
        batch_entities = self.ner.extract_entities_batch(texts)
        
        # Get statistics
        entity_dist = self.ner.get_entity_distribution()
        top_entities = self.ner.get_top_entities(top_n=20)
        entity_stats = self.ner.get_entity_statistics()
        
        print(f"✓ Found {entity_stats['total_entities']} total entities")
        print(f"✓ Found {entity_stats['unique_entities']} unique entities")
        print(f"✓ Found {entity_stats['entity_types']} entity types")
        
        if entity_dist:
            print("\nEntity Type Distribution:")
            for entity_type, stats in entity_dist.items():
                print(f"  {entity_type}: {stats['count']} ({stats['percentage']}%)")
        
        self.results['entities'] = batch_entities
        self.results['entity_distribution'] = entity_dist
        self.results['top_entities'] = top_entities
        
        return {
            'entities': batch_entities,
            'entity_distribution': entity_dist,
            'top_entities': top_entities,
            'statistics': entity_stats
        }
    
    def classify_texts(self, df, text_column, sentiment_column=None, test_size=0.2):
        """
        Train text classifiers
        
        Args:
            df: DataFrame
            text_column: Text column name
            sentiment_column: Sentiment column name (for training)
            test_size: Test set size
            
        Returns:
            dict: Classification results
        """
        print("\n" + "="*80)
        print("[STEP 5] TEXT CLASSIFICATION & MODELING")
        print("="*80)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("⚠️  scikit-learn is required for classification")
            return None
        
        texts = df[text_column].astype(str).tolist()
        
        # Get labels
        if sentiment_column and sentiment_column in df.columns:
            labels = df[sentiment_column].tolist()
        else:
            # Use predicted sentiment as label
            labels = self.results.get('sentiment', {}).get('sentiment', []).tolist()
        
        print(f"\nPreparing data for classification...")
        print(f"  Total samples: {len(texts)}")
        print(f"  Unique classes: {len(set(labels))}")
        
        # Vectorize with TF-IDF
        print("\nVectorizing texts with TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        print(f"✓ TF-IDF matrix shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\nTrain/Test split:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        # Train classifiers
        print("\nTraining classifiers...")
        classification_results = self.text_classifier.train_all_models(
            X_train, y_train, X_test, y_test
        )
        
        # Get results summary
        results_summary = self.text_classifier.get_results_summary()
        print("\nClassification Results:")
        print(results_summary.to_string())
        
        # Hyperparameter tuning
        print("\n" + "-"*80)
        print("Performing hyperparameter tuning...")
        print("-"*80)
        
        self.modeling_engine.tune_all_models(X_train, y_train, cv=3)
        tuning_summary = self.modeling_engine.get_tuning_results_summary()
        print("\nTuning Results:")
        print(tuning_summary.to_string())
        
        self.results['classifier'] = self.text_classifier
        self.results['vectorizer'] = vectorizer
        self.results['modeling_engine'] = self.modeling_engine
        self.results['classification_summary'] = results_summary
        self.results['tuning_summary'] = tuning_summary
        
        return {
            'classifier': self.text_classifier,
            'vectorizer': vectorizer,
            'results_summary': results_summary,
            'tuning_summary': tuning_summary
        }
    
    def generate_reports(self, df, text_column):
        """
        Generate analysis reports
        
        Args:
            df: DataFrame
            text_column: Text column name
            
        Returns:
            dict: Generated report paths
        """
        print("\n" + "="*80)
        print("[STEP 6] GENERATING REPORTS")
        print("="*80)
        
        report_paths = {}
        
        # 1. Text Report
        print("\nGenerating text report...")
        sections = {
            'Summary': f"Tokopedia Dataset Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'Dataset Info': pd.DataFrame({
                'Metric': ['Total Records', 'Text Column', 'Missing Values'],
                'Value': [len(df), text_column, df[text_column].isnull().sum()]
            }),
            'Text Statistics': pd.DataFrame(list(self.results.get('text_stats', {}).items()), 
                                          columns=['Metric', 'Value']),
            'Sentiment Distribution': pd.DataFrame(
                list(self.results.get('sentiment_distribution', {}).items()),
                columns=['Sentiment', 'Stats']
            ),
            'Word Frequency': self.results.get('word_frequency', pd.DataFrame()).head(20),
            'Top Bigrams': self.results.get('bigrams', pd.DataFrame()).head(20)
        }
        
        if 'classification_summary' in self.results:
            sections['Classification Results'] = self.results['classification_summary']
        
        if 'tuning_summary' in self.results:
            sections['Tuning Results'] = self.results['tuning_summary']
        
        text_report = self.reporter.generate_text_report(
            "TOKOPEDIA DATASET NLP ANALYSIS REPORT",
            sections,
            'NLP_Analysis_Report.txt'
        )
        report_paths['text'] = text_report
        print(f"✓ Text report: {text_report}")
        
        # 2. Excel Report
        print("\nGenerating Excel report...")
        excel_data = {
            'Word_Frequency': self.results.get('word_frequency', pd.DataFrame()),
            'Bigrams': self.results.get('bigrams', pd.DataFrame()),
            'Sentiment_Analysis': self.results.get('sentiment', pd.DataFrame())
        }
        
        if 'top_entities' in self.results:
            excel_data['Top_Entities'] = self.results['top_entities']
        
        if 'classification_summary' in self.results:
            excel_data['Classification_Results'] = self.results['classification_summary']
        
        excel_report = self.reporter.generate_excel_report(
            excel_data,
            'NLP_Analysis_Report.xlsx'
        )
        report_paths['excel'] = excel_report
        print(f"✓ Excel report: {excel_report}")
        
        return report_paths
    
    def run_full_pipeline(self, csv_file, sentiment_column=None):
        """
        Run complete NLP pipeline
        
        Args:
            csv_file: Path to CSV file
            sentiment_column: Sentiment column name (optional)
            
        Returns:
            dict: All results
        """
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "TOKOPEDIA DATASET - COMPLETE NLP PIPELINE ANALYSIS".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        # Load data
        df = self.load_data(csv_file)
        if df is None:
            return None
        
        text_column = self.results['text_column']
        
        # Run pipeline steps
        df = self.analyze_sentiments(df, text_column)
        self.extract_features(df, text_column)
        self.extract_named_entities(df, text_column, max_texts=500)  # Limit for performance
        self.classify_texts(df, text_column, sentiment_column, test_size=0.2)
        
        # Generate reports
        self.generate_reports(df, text_column)
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*80)
        print(f"\n✓ Analysis completed successfully!")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Results saved for review")
        
        return self.results


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = NLPPipeline(output_dir='../output')
    
    # Run pipeline
    csv_file = "../data/Dataset pengguna tokopedia.csv"
    
    if os.path.exists(csv_file):
        pipeline.run_full_pipeline(csv_file, sentiment_column='sentiment_label')
    else:
        print(f"❌ File not found: {csv_file}")
