# 📊 Tokopedia NLP Analysis Project

## 📁 Project Structure

```
nlp_project/
│
├── 📂 src/                           # NLP Modules (Core Components)
│   ├── sentiment_analyzer.py         # Sentiment analysis (TextBlob + Lexicon)
│   ├── feature_extraction.py         # BoW & TF-IDF extraction
│   ├── text_classifier.py            # 4 ML algorithms (NB, SVM, LR, RF)
│   ├── modeling_engine.py            # Hyperparameter tuning & GridSearch
│   ├── named_entity_recognition.py   # NER & POS tagging with spaCy
│   ├── visualization.py              # Charts & report generation
│   └── data_processor.py             # Excel/CSV I/O & data handling
│
├── 📂 scripts/                       # Executable Scripts
│   ├── main.py                       # Full NLP Pipeline (6-stage)
│   ├── dashboard.py                  # Streamlit interactive dashboard
│   ├── analyze_tokopedia.py          # Full analysis with all modules
│   └── analyze_tokopedia_simple.py   # Simple analysis (no heavy deps)
│
├── 📂 data/                          # Input Data
│   └── Dataset pengguna tokopedia.csv  # 1,999 Tokopedia user reviews
│
├── 📂 output/                        # Output Results
│   ├── 00_Tokopedia_Analysis_Report.txt
│   ├── 01_Tokopedia_Sentiment_Analysis.xlsx
│   ├── 02_Tokopedia_Word_Frequency.xlsx
│   ├── 03_Tokopedia_Bigram_Frequency.xlsx
│   ├── NLP_Analysis_Report.txt
│   └── NLP_Analysis_Report.xlsx
│
├── 📄 config.json                    # Project configuration
├── 📄 TOKOPEDIA_ANALYSIS_SUMMARY.txt # Analysis summary
├── 📄 .gitignore                     # Git ignore file
└── 📄 README.md                      # This file
```

---

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. **Install Dependencies**
```bash
pip install pandas numpy scikit-learn textblob spacy matplotlib seaborn openpyxl streamlit
python -m spacy download en_core_web_sm
```

### 3. **Run Full Analysis**
```bash
python scripts/main.py
```

### 4. **View Interactive Dashboard**
```bash
streamlit run scripts/dashboard.py
```

---

## 📋 File Descriptions

### **Modules (src/)**

| File | Purpose |
|------|---------|
| `sentiment_analyzer.py` | Sentiment classification (TextBlob + Lexicon-based) |
| `feature_extraction.py` | BoW, TF-IDF, word/bigram frequency extraction |
| `text_classifier.py` | Naive Bayes, SVM, Logistic Regression, Random Forest |
| `modeling_engine.py` | GridSearchCV, hyperparameter tuning, cross-validation |
| `named_entity_recognition.py` | NER extraction, POS tagging with spaCy |
| `visualization.py` | Matplotlib charts, report generation |
| `data_processor.py` | DataFrame handling, Excel/CSV I/O |

### **Scripts (scripts/)**

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestrating all components |
| `dashboard.py` | Interactive web dashboard with 6 analysis sections |
| `analyze_tokopedia.py` | Full analysis with all NLP features |
| `analyze_tokopedia_simple.py` | Lightweight analysis (minimal dependencies) |

---

## 📊 Analysis Results

### **Dataset Overview**
- **Total Reviews**: 1,999
- **Unique Words**: 3,810
- **Average Text Length**: 12.09 words

### **Sentiment Distribution**
- **POSITIF**: 53.38% (1,067 reviews) ✅
- **NETRAL**: 28.26% (565 reviews) 😐
- **NEGATIF**: 18.36% (367 reviews) ❌

### **Top Keywords**
1. tokopedia (288x)
2. bisa (274x)
3. saya (263x)
4. tidak (224x)
5. promo (210x)

### **Top Phrases**
1. pengguna baru (51x)
2. tidak bisa (50x)
3. kurir rekomendasi (42x)
4. gak bisa (41x)
5. gratis ongkir (40x)

---

## 🔧 Core Features

### ✅ **Sentiment Analysis**
- TextBlob polarity/subjectivity analysis
- Lexicon-based sentiment classification
- Combined sentiment prediction

### ✅ **Feature Extraction**
- Bag of Words (BoW)
- TF-IDF vectorization
- Word frequency analysis
- Bigram extraction

### ✅ **Text Classification**
- Multinomial Naive Bayes
- Linear SVM
- Logistic Regression
- Random Forest

### ✅ **Model Optimization**
- GridSearchCV hyperparameter tuning
- 5-fold cross-validation
- Multiple scoring metrics

### ✅ **Named Entity Recognition**
- spaCy NER (PERSON, ORG, GPE, DATE, etc)
- Part-of-speech tagging
- Dependency parsing

### ✅ **Visualization & Reports**
- Sentiment distribution charts
- Word frequency plots
- Confusion matrices
- Text & Excel reports
- Interactive dashboard

---

## 📈 Dashboard Sections

1. **📊 Overview** - Dataset statistics & quality
2. **😊 Sentiment Analysis** - Sentiment distribution & predictions
3. **📝 Text Statistics** - Text length distribution & statistics
4. **🔤 Word Frequency** - Top words with interactive filtering
5. **🔗 Bigram Analysis** - Top phrases with frequency
6. **💡 Key Insights** - Findings, problems, & recommendations

---

## 💡 Key Insights

### ✅ **Strengths**
- High positive sentiment (53%)
- Good user experience reported
- Strong promotion reception

### ⚠️ **Issues to Address**
- Payment processing problems ("tidak bisa" issues)
- System cancellations ("dibatalkan sistem")
- Delivery concerns
- Seller responsiveness

### 🚀 **Recommendations**
- Fix payment gateway issues (highest priority)
- Improve system stability
- Better courier partnerships
- Enhanced product descriptions
- Strengthen seller training

---

## 📦 Output Files

All analysis results are saved in `output/` folder:

| File | Format | Content |
|------|--------|---------|
| `00_Tokopedia_Analysis_Report.txt` | TXT | Detailed text report |
| `01_Tokopedia_Sentiment_Analysis.xlsx` | XLSX | Per-review sentiment analysis |
| `02_Tokopedia_Word_Frequency.xlsx` | XLSX | Top 200 words |
| `03_Tokopedia_Bigram_Frequency.xlsx` | XLSX | Top 200 bigrams |
| `NLP_Analysis_Report.txt` | TXT | Full pipeline report |
| `NLP_Analysis_Report.xlsx` | XLSX | Multi-sheet results |

---

## 🛠️ Technology Stack

- **Python 3.11**
- **Data**: pandas, numpy
- **NLP**: scikit-learn, NLTK, TextBlob, spaCy
- **Visualization**: matplotlib, seaborn
- **Dashboard**: Streamlit
- **Excel**: openpyxl

---

## 📝 Notes

- All analysis runs on Indonesian language text
- Lexicon-based sentiment uses custom word lists optimized for Tokopedia domain
- NER requires spaCy model download
- Classification requires scikit-learn (optional for simple analysis)

---

## 👨‍💻 Project by

GitHub Copilot | January 12, 2026

---

## 📞 Support

For questions or issues, refer to:
- Source code docstrings
- Output reports
- Dashboard help sections

Happy analyzing! 🎉
