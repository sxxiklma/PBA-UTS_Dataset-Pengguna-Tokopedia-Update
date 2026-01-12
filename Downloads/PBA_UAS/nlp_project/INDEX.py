"""
Quick Index & Navigation Guide
Tokopedia NLP Analysis Project
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_STRUCTURE = """
📁 nlp_project/
│
├── 📂 src/                           [NLP MODULES - Core Components]
│   ├── sentiment_analyzer.py        → Sentiment analysis (TextBlob + Lexicon)
│   ├── feature_extraction.py        → BoW & TF-IDF extraction
│   ├── text_classifier.py           → Classification (4 algorithms)
│   ├── modeling_engine.py           → Hyperparameter tuning
│   ├── named_entity_recognition.py  → NER & POS tagging
│   ├── visualization.py             → Charts & reports
│   └── data_processor.py            → Excel/CSV I/O
│
├── 📂 scripts/                       [EXECUTABLE SCRIPTS]
│   ├── main.py                      → Full NLP pipeline (run this!)
│   ├── dashboard.py                 → Interactive Streamlit dashboard
│   ├── analyze_tokopedia.py         → Full analysis with all features
│   └── analyze_tokopedia_simple.py  → Lightweight analysis
│
├── 📂 data/                          [INPUT DATA]
│   └── Dataset pengguna tokopedia.csv → 1,999 user reviews
│
├── 📂 output/                        [RESULTS & OUTPUTS]
│   ├── *.txt                        → Text reports
│   └── *.xlsx                       → Excel results
│
├── 📄 README.md                      → Full documentation
├── 📄 config.json                   → Configuration file
└── 📄 TOKOPEDIA_ANALYSIS_SUMMARY.txt → Analysis summary
"""

# ============================================================================
# QUICK START COMMANDS
# ============================================================================

QUICK_START = """
🚀 QUICK START

1. Run Full Pipeline Analysis:
   $ cd scripts
   $ python main.py

2. View Interactive Dashboard:
   $ cd scripts
   $ streamlit run dashboard.py
   
3. Simple Analysis (No Heavy Dependencies):
   $ cd scripts
   $ python analyze_tokopedia_simple.py

4. Read Documentation:
   $ Open: README.md
"""

# ============================================================================
# MODULE DESCRIPTIONS
# ============================================================================

MODULES = {
    'sentiment_analyzer.py': {
        'description': 'Sentiment Analysis Module',
        'key_classes': ['SentimentAnalyzer'],
        'key_methods': [
            'analyze_sentiment_textblob()',
            'analyze_sentiment_lexicon()',
            'analyze_sentiment_combined()',
            'analyze_batch()',
            'get_sentiment_distribution()'
        ],
        'features': [
            'TextBlob polarity/subjectivity',
            'Lexicon-based classification',
            'Positive/negative word detection',
            'Intensifier & negator handling',
            'Combined sentiment prediction'
        ]
    },
    
    'feature_extraction.py': {
        'description': 'Feature Extraction Module',
        'key_classes': ['TextFeatureExtractor'],
        'key_methods': [
            'extract_bow()',
            'extract_tfidf()',
            'extract_word_frequency()',
            'extract_bigrams()',
            'extract_text_statistics()',
            'get_top_features()'
        ],
        'features': [
            'Bag of Words (BoW)',
            'TF-IDF vectorization',
            'Word frequency analysis',
            'Bigram extraction',
            'Text length statistics',
            'Vocabulary analysis'
        ]
    },
    
    'text_classifier.py': {
        'description': 'Text Classification Module',
        'key_classes': ['TextClassifier'],
        'algorithms': [
            'Multinomial Naive Bayes',
            'Linear SVM',
            'Logistic Regression',
            'Random Forest'
        ],
        'key_methods': [
            'train_naive_bayes()',
            'train_svm()',
            'train_logistic_regression()',
            'train_random_forest()',
            'train_all_models()',
            'predict()',
            'get_results_summary()'
        ]
    },
    
    'modeling_engine.py': {
        'description': 'Model Optimization Module',
        'key_classes': ['ModelingEngine'],
        'key_methods': [
            'tune_naive_bayes()',
            'tune_svm()',
            'tune_logistic_regression()',
            'tune_random_forest()',
            'tune_all_models()',
            'cross_validate()',
            'get_tuning_results_summary()'
        ],
        'features': [
            'GridSearchCV hyperparameter tuning',
            'Cross-validation (k-fold)',
            'Multiple scoring metrics',
            'Best model selection'
        ]
    },
    
    'named_entity_recognition.py': {
        'description': 'NER & POS Tagging Module',
        'key_classes': [
            'NamedEntityRecognizer',
            'POS_Tagger',
            'EntitySentimentAnalyzer'
        ],
        'entities': [
            'PERSON', 'ORG', 'GPE', 'DATE', 'LOCATION', 'etc'
        ],
        'key_methods': [
            'extract_entities()',
            'extract_entities_batch()',
            'tag_pos()',
            'extract_by_pos()',
            'get_entity_distribution()',
            'analyze_entity_sentiment()'
        ]
    },
    
    'visualization.py': {
        'description': 'Visualization & Reporting Module',
        'key_classes': [
            'VisualizationHelper',
            'AnalysisReporter'
        ],
        'visualization_types': [
            'Sentiment distribution (pie, bar)',
            'Word frequency plots',
            'Confusion matrices',
            'Text length distributions',
            'Model metrics comparison'
        ],
        'report_types': [
            'Text reports (.txt)',
            'Excel reports (.xlsx)',
            'Summary statistics'
        ]
    },
    
    'data_processor.py': {
        'description': 'Data Processing Module',
        'key_classes': [
            'ExcelDataHandler',
            'DataProcessor'
        ],
        'key_methods': [
            'load_excel()',
            'save_to_excel()',
            'load_csv()',
            'apply_styling()',
            'clean_data()'
        ]
    }
}

# ============================================================================
# PIPELINE FLOW
# ============================================================================

PIPELINE_FLOW = """
📊 NLP PIPELINE - 6 STAGES

[1] LOAD DATA
    └─ Load CSV dataset (1,999 reviews)

[2] SENTIMENT ANALYSIS
    ├─ TextBlob analysis
    ├─ Lexicon-based sentiment
    └─ Distribution calculation

[3] FEATURE EXTRACTION
    ├─ Word frequency
    ├─ Bigram extraction
    ├─ TF-IDF vectorization
    └─ Text statistics

[4] NAMED ENTITY RECOGNITION (Optional)
    ├─ Entity extraction
    ├─ POS tagging
    └─ Entity sentiment analysis

[5] TEXT CLASSIFICATION & MODELING
    ├─ TF-IDF vectorization
    ├─ Train 4 classifiers
    ├─ Evaluate metrics
    └─ Hyperparameter tuning

[6] GENERATE REPORTS
    ├─ Text reports (.txt)
    ├─ Excel files (.xlsx)
    └─ Dashboard visualization
"""

# ============================================================================
# KEY FILES & WHAT THEY DO
# ============================================================================

KEY_FILES = {
    'main.py': 'Main NLP pipeline - orchestrates all components',
    'dashboard.py': 'Interactive Streamlit dashboard - visualize results',
    'analyze_tokopedia.py': 'Full analysis using all modules',
    'analyze_tokopedia_simple.py': 'Lightweight analysis (minimal deps)',
    'README.md': 'Complete documentation',
    'config.json': 'Project configuration',
    'TOKOPEDIA_ANALYSIS_SUMMARY.txt': 'Analysis summary & findings'
}

# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_FILES = {
    '00_Tokopedia_Analysis_Report.txt': 'Initial analysis report',
    '01_Tokopedia_Sentiment_Analysis.xlsx': 'Per-review sentiment results',
    '02_Tokopedia_Word_Frequency.xlsx': 'Top 200 most frequent words',
    '03_Tokopedia_Bigram_Frequency.xlsx': 'Top 200 most frequent bigrams',
    'NLP_Analysis_Report.txt': 'Complete pipeline analysis report',
    'NLP_Analysis_Report.xlsx': 'Multi-sheet Excel report'
}

# ============================================================================
# ANALYSIS RESULTS SUMMARY
# ============================================================================

RESULTS_SUMMARY = """
📈 ANALYSIS RESULTS

Dataset: 1,999 Tokopedia user reviews

Sentiment Distribution:
├─ POSITIF (53.38%): 1,067 reviews ✅
├─ NETRAL (28.26%): 565 reviews 😐
└─ NEGATIF (18.36%): 367 reviews ❌

Text Statistics:
├─ Unique words: 3,810
├─ Unique bigrams: 200+
├─ Average text length: 12.09 words
└─ Vocabulary richness: High

Top Keywords:
1. tokopedia (288x)
2. bisa (274x)
3. saya (263x)
4. tidak (224x)
5. promo (210x)

Top Phrases:
1. pengguna baru (51x)
2. tidak bisa (50x)
3. kurir rekomendasi (42x)
4. gak bisa (41x)
5. gratis ongkir (40x)
"""

# ============================================================================
# IMPORT PATHS
# ============================================================================

IMPORT_GUIDE = """
🔗 HOW TO IMPORT MODULES

From scripts/:
  from sys import path
  path.insert(0, '../src')
  
  from sentiment_analyzer import SentimentAnalyzer
  from feature_extraction import TextFeatureExtractor
  from text_classifier import TextClassifier
  # ... etc

From main.py (already configured):
  # Just use: from sentiment_analyzer import SentimentAnalyzer
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
🔧 TROUBLESHOOTING

1. ModuleNotFoundError (import errors)
   → Make sure you're in correct directory
   → Check sys.path.insert(0, '../src') is present

2. FileNotFoundError (data/output not found)
   → Check paths use: ../data/ and ../output/
   → Run scripts from scripts/ folder only

3. Missing dependencies
   → pip install pandas numpy scikit-learn textblob spacy matplotlib seaborn openpyxl streamlit
   → python -m spacy download en_core_web_sm

4. Dashboard not loading
   → streamlit run dashboard.py
   → Check http://localhost:8501

5. Excel files empty
   → Ensure openpyxl is installed
   → Check output/ folder has write permissions
"""

# ============================================================================
# USEFUL COMMANDS
# ============================================================================

COMMANDS = """
📋 USEFUL COMMANDS

# Navigate to project
cd nlp_project

# Run main analysis
cd scripts
python main.py

# Start dashboard
cd scripts
streamlit run dashboard.py

# Run simple analysis
cd scripts
python analyze_tokopedia_simple.py

# Check dependencies
pip list | grep -E "pandas|scikit-learn|streamlit"

# View file structure
dir /s  (Windows)
ls -R   (macOS/Linux)

# View analysis results
cat ../output/NLP_Analysis_Report.txt

# Clear output cache
rm -rf __pycache__
rm -rf .streamlit
"""

# ============================================================================
# PRINT GUIDE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TOKOPEDIA NLP ANALYSIS - INDEX & NAVIGATION GUIDE")
    print("=" * 80)
    print()
    print(PROJECT_STRUCTURE)
    print("\n" + "=" * 80)
    print("QUICK START")
    print("=" * 80)
    print(QUICK_START)
    print("\n" + "=" * 80)
    print("PIPELINE FLOW")
    print("=" * 80)
    print(PIPELINE_FLOW)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(RESULTS_SUMMARY)
    print("\n" + "=" * 80)
    print("For detailed information, see: README.md")
    print("=" * 80)
