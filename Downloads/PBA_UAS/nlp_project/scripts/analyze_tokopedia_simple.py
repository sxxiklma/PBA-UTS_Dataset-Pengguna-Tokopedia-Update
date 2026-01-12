"""
Script Analisis Tokopedia Dataset - Versi Sederhana (Tanpa External Dependencies)
Menggunakan library standar: pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from datetime import datetime

# Cek apakah file ada
if not os.path.exists('Dataset pengguna tokopedia.csv'):
    print("❌ File Dataset pengguna tokopedia.csv tidak ditemukan!")
    exit(1)

print("\n" + "="*80)
print("ANALISIS DATASET TOKOPEDIA - SIMPLE NLP ANALYSIS")
print("="*80)

# Step 1: Load Data
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('Dataset pengguna tokopedia.csv')
print(f"✓ Dataset loaded: {len(df)} rows")
print(f"✓ Columns: {list(df.columns)}")

# Step 2: Data Overview
print("\n" + "="*80)
print("[STEP 2] DATA OVERVIEW")
print("="*80)

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {', '.join(df.columns)}")

print(f"\nMissing Values:")
print(df.isnull().sum())

# Get text column
text_column = 'text_clean' if 'text_clean' in df.columns else 'text_original'
print(f"\n✓ Using text column: {text_column}")

# Step 3: Sentiment Analysis
print("\n" + "="*80)
print("[STEP 3] SENTIMENT ANALYSIS")
print("="*80)

if 'sentiment_label' in df.columns:
    print(f"\nExisting Sentiment Distribution:")
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  • {sentiment:10s} : {count:5d} ({percentage:5.2f}%)")
    
    print(f"\nExisting Sentiment Score Distribution:")
    if 'sentiment_score' in df.columns:
        print(f"  • Score 0 (Netral/Baik) : {len(df[df['sentiment_score']==0]):5d}")
        print(f"  • Score 1 (Buruk)       : {len(df[df['sentiment_score']==1]):5d}")

# Step 4: Text Statistics
print("\n" + "="*80)
print("[STEP 4] TEXT STATISTICS")
print("="*80)

texts = df[text_column].astype(str).tolist()

# Text length analysis
text_lengths = [len(text.split()) for text in texts]
print(f"\nText Length Statistics (words):")
print(f"  • Mean length:       {np.mean(text_lengths):.2f} words")
print(f"  • Median length:     {np.median(text_lengths):.2f} words")
print(f"  • Min length:        {np.min(text_lengths):.2f} words")
print(f"  • Max length:        {np.max(text_lengths):.2f} words")
print(f"  • Std deviation:     {np.std(text_lengths):.2f} words")

# Step 5: Feature Extraction (Simple)
print("\n" + "="*80)
print("[STEP 5] FEATURE EXTRACTION (Simple BoW)")
print("="*80)

print("\nExtracting words from texts...")

# Tokenize and count words
word_freq = Counter()
bigram_freq = Counter()

for text in texts:
    # Simple preprocessing
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    
    # Count unigrams
    for word in words:
        if len(word) > 2:  # Skip short words
            word_freq[word] += 1
    
    # Count bigrams
    for i in range(len(words)-1):
        bigram = f"{words[i]} {words[i+1]}"
        if len(words[i]) > 2 and len(words[i+1]) > 2:
            bigram_freq[bigram] += 1

print(f"✓ Unique words: {len(word_freq):,}")
print(f"✓ Unique bigrams: {len(bigram_freq):,}")

print(f"\nTop 20 Most Frequent Words:")
for i, (word, freq) in enumerate(word_freq.most_common(20), 1):
    print(f"  {i:2d}. {word:20s} : {freq:6d} occurrences")

print(f"\nTop 15 Most Frequent Bigrams:")
for i, (bigram, freq) in enumerate(bigram_freq.most_common(15), 1):
    print(f"  {i:2d}. {bigram:35s} : {freq:5d}")

# Step 6: Simple Sentiment Lexicon Analysis
print("\n" + "="*80)
print("[STEP 6] SENTIMENT LEXICON ANALYSIS")
print("="*80)

# Simple Indonesian sentiment words
positive_words = {
    'bagus', 'baik', 'sempurna', 'excellent', 'great', 'good', 'amazing',
    'wonderful', 'mantap', 'mantaf', 'manteb', 'keren', 'luar', 'biasa',
    'memuaskan', 'terbaik', 'best', 'awesome', 'puas', 'sangat puas'
}

negative_words = {
    'buruk', 'jelek', 'terrible', 'awful', 'bad', 'worst', 'payah',
    'parah', 'mengecewakan', 'kecewa', 'ribet', 'lemot', 'gokil',
    'complaint', 'komplain', 'susah', 'ribet'
}

sentiment_analysis = []
for i, text in enumerate(texts):
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        predicted = 'POSITIF'
    elif neg_count > pos_count:
        predicted = 'NEGATIF'
    else:
        predicted = 'NETRAL'
    
    sentiment_analysis.append({
        'text': text,
        'positive_words': pos_count,
        'negative_words': neg_count,
        'predicted_sentiment': predicted
    })

sentiment_results_df = pd.DataFrame(sentiment_analysis)

print(f"\nPredicted Sentiment Distribution (Lexicon-based):")
predicted_counts = sentiment_results_df['predicted_sentiment'].value_counts()
for sentiment, count in predicted_counts.items():
    percentage = (count / len(sentiment_results_df)) * 100
    print(f"  • {sentiment:10s} : {count:5d} ({percentage:5.2f}%)")

# Step 7: Comparison
print("\n" + "="*80)
print("[STEP 7] COMPARISON WITH EXISTING LABELS")
print("="*80)

if 'sentiment_label' in df.columns:
    # Map labels
    sentiment_map = {
        'Baik': 'POSITIF',
        'Buruk': 'NEGATIF',
        'Netral': 'NETRAL'
    }
    
    df['predicted'] = sentiment_results_df['predicted_sentiment']
    df['existing_mapped'] = df['sentiment_label'].map(sentiment_map)
    
    # Calculate agreement
    agreement = (df['existing_mapped'] == df['predicted']).sum()
    agreement_rate = (agreement / len(df)) * 100
    
    print(f"\nLexicon-based vs Existing Labels Agreement: {agreement_rate:.2f}%")
    print(f"Matching: {agreement}/{len(df)}")
    
    print(f"\nDetailed Comparison:")
    comparison = pd.crosstab(
        df['existing_mapped'],
        df['predicted'],
        margins=True,
        margins_name='Total'
    )
    print(comparison)

# Step 8: Export Results
print("\n" + "="*80)
print("[STEP 8] EXPORTING RESULTS")
print("="*80)

os.makedirs('output', exist_ok=True)

# Export sentiment analysis
export_df = pd.DataFrame({
    'text': texts,
    'text_length_words': text_lengths,
    'positive_word_count': sentiment_results_df['positive_words'],
    'negative_word_count': sentiment_results_df['negative_words'],
    'predicted_sentiment': sentiment_results_df['predicted_sentiment']
})

if 'sentiment_label' in df.columns:
    export_df['existing_label'] = df['sentiment_label']
    export_df['existing_score'] = df['sentiment_score'] if 'sentiment_score' in df.columns else 0

export_df.to_excel('output/01_Tokopedia_Sentiment_Analysis.xlsx', index=False)
print("✓ Exported: 01_Tokopedia_Sentiment_Analysis.xlsx")

# Export word frequency
word_freq_df = pd.DataFrame([
    {'rank': i+1, 'word': word, 'frequency': freq}
    for i, (word, freq) in enumerate(word_freq.most_common(200))
])
word_freq_df.to_excel('output/02_Tokopedia_Word_Frequency.xlsx', index=False)
print("✓ Exported: 02_Tokopedia_Word_Frequency.xlsx")

# Export bigram frequency
bigram_freq_df = pd.DataFrame([
    {'rank': i+1, 'bigram': bigram, 'frequency': freq}
    for i, (bigram, freq) in enumerate(bigram_freq.most_common(200))
])
bigram_freq_df.to_excel('output/03_Tokopedia_Bigram_Frequency.xlsx', index=False)
print("✓ Exported: 03_Tokopedia_Bigram_Frequency.xlsx")

# Step 9: Generate Report
print("\n" + "="*80)
print("[STEP 9] GENERATING REPORT")
print("="*80)

report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  TOKOPEDIA DATASET - NLP ANALYSIS REPORT                  ║
║                   Natural Language Processing Analysis                     ║
╚════════════════════════════════════════════════════════════════════════════╝

DATASET OVERVIEW
════════════════════════════════════════════════════════════════════════════

Date Analyzed:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset File:                 Dataset pengguna tokopedia.csv
Total Records:                {len(df):,}
Text Column Used:             {text_column}

SENTIMENT DISTRIBUTION (EXISTING LABELS)
════════════════════════════════════════════════════════════════════════════
"""

if 'sentiment_label' in df.columns:
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        report += f"  • {sentiment:15s} : {count:6,d} ({percentage:6.2f}%)\n"

report += f"""
TEXT STATISTICS
════════════════════════════════════════════════════════════════════════════

Average Text Length:          {np.mean(text_lengths):.2f} words
Median Text Length:           {np.median(text_lengths):.2f} words
Shortest Text:                {np.min(text_lengths):.0f} words
Longest Text:                 {np.max(text_lengths):.0f} words
Standard Deviation:           {np.std(text_lengths):.2f} words

Unique Words (Vocabulary):    {len(word_freq):,}
Unique Bigrams:               {len(bigram_freq):,}

TOP 20 MOST FREQUENT WORDS
════════════════════════════════════════════════════════════════════════════

"""

for i, (word, freq) in enumerate(word_freq.most_common(20), 1):
    percentage = (freq / sum(word_freq.values())) * 100
    report += f"  {i:2d}. {word:20s} : {freq:6,d} ({percentage:5.2f}%)\n"

report += f"""
TOP 15 MOST FREQUENT BIGRAMS (2-word phrases)
════════════════════════════════════════════════════════════════════════════

"""

for i, (bigram, freq) in enumerate(bigram_freq.most_common(15), 1):
    percentage = (freq / sum(bigram_freq.values())) * 100
    report += f"  {i:2d}. {bigram:40s} : {freq:5,d} ({percentage:5.2f}%)\n"

report += f"""
LEXICON-BASED SENTIMENT ANALYSIS
════════════════════════════════════════════════════════════════════════════

Predicted Sentiment Distribution:
"""

predicted_counts = sentiment_results_df['predicted_sentiment'].value_counts()
for sentiment in ['POSITIF', 'NEGATIF', 'NETRAL']:
    count = predicted_counts.get(sentiment, 0)
    percentage = (count / len(sentiment_results_df)) * 100
    report += f"  • {sentiment:10s} : {count:6,d} ({percentage:6.2f}%)\n"

if 'sentiment_label' in df.columns:
    report += f"""
AGREEMENT RATE WITH EXISTING LABELS
════════════════════════════════════════════════════════════════════════════

Agreement Rate:               {agreement_rate:.2f}%
Correct Predictions:          {agreement:,} out of {len(df):,}

Note: Lexicon-based sentiment might differ from manual labels due to:
  - Context and sarcasm not detected by lexicon
  - Mixed sentiments in single text
  - Different labeling criteria

CONFUSION MATRIX (Predicted vs Existing)
════════════════════════════════════════════════════════════════════════════

"""
    report += str(comparison) + "\n"

report += f"""
KEY INSIGHTS & FINDINGS
════════════════════════════════════════════════════════════════════════════

1. DOMINANT SENTIMENTS:
   - Most texts in dataset are labeled as: {sentiment_counts.idxmax() if 'sentiment_label' in df.columns else 'N/A'}
   - Lexicon-based prediction shows: {predicted_counts.idxmax()}

2. MOST DISCUSSED TOPICS (by frequency):
   - Top word: '{word_freq.most_common(1)[0][0]}'
   - Common phrase: '{bigram_freq.most_common(1)[0][0]}'

3. TEXT CHARACTERISTICS:
   - Average text length: {np.mean(text_lengths):.0f} words
   - Texts range from {np.min(text_lengths):.0f} to {np.max(text_lengths):.0f} words
   - Shows diversity in user review length

4. SENTIMENT LEXICON FINDINGS:
   - Texts containing positive words: {len(sentiment_results_df[sentiment_results_df['positive_words']>0]):,}
   - Texts containing negative words: {len(sentiment_results_df[sentiment_results_df['negative_words']>0]):,}
   - Mixed sentiment texts: {len(sentiment_results_df[(sentiment_results_df['positive_words']>0) & (sentiment_results_df['negative_words']>0)]):,}

5. LANGUAGE CHARACTERISTICS:
   - Texts are in Indonesian language
   - Contains mix of formal and informal (slang) language
   - Many abbreviations and colloquialisms

RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════

For Tokopedia:
  1. Focus on improving areas mentioned in negative reviews
  2. Top issues appear to be: {', '.join([word for word, _ in word_freq.most_common(5)])}
  3. Implement feedback loop based on sentiment analysis

For Further NLP Analysis:
  1. Build custom Indonesian sentiment classifier
  2. Perform topic modeling (LDA, NMF)
  3. Analyze temporal trends in sentiment
  4. Detect key complaints and issues
  5. Monitor sentiment over time periods

For Feature Engineering:
  1. Use top words as important features
  2. Consider n-grams (bigrams, trigrams) in ML models
  3. Apply TF-IDF weighting for better results
  4. Create domain-specific features from detected entities

EXPORTED FILES
════════════════════════════════════════════════════════════════════════════

Results saved to output/ directory:

1. 01_Tokopedia_Sentiment_Analysis.xlsx
   └─ Contains: Text, text length, sentiment counts, predicted sentiment

2. 02_Tokopedia_Word_Frequency.xlsx
   └─ Contains: Top 200 words with frequencies

3. 03_Tokopedia_Bigram_Frequency.xlsx
   └─ Contains: Top 200 bigrams with frequencies

Additional Analysis Available:
   • Detailed sentiment comparison can be done with machine learning
   • Topic modeling for key themes
   • Temporal analysis for trend detection

TECHNICAL NOTES
════════════════════════════════════════════════════════════════════════════

Analysis Method:           Simple Lexicon-Based + Frequency Analysis
Language:                 Indonesian (with English mix)
Text Preprocessing:       Lowercase + special character removal
Tokenization:            Space-based word splitting
Vocabulary Size:         {len(word_freq):,} unique words
N-gram Coverage:         Bigrams (2-word phrases)

Limitations:
  - Does not handle context/sarcasm
  - Single sentiment per text assumption
  - Limited lexicon for Indonesian
  - No deep semantic analysis

For better results, consider:
  - Training custom sentiment classifier
  - Using transformer models (BERT, etc.)
  - Incorporating domain-specific lexicons
  - Ensemble methods combining multiple approaches

════════════════════════════════════════════════════════════════════════════
ANALYSIS COMPLETE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tool: NLP Dataset Analysis Script
Status: Successfully processed {len(df):,} records
════════════════════════════════════════════════════════════════════════════
"""

# Save report
with open('output/00_Tokopedia_Analysis_Report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ Exported: 00_Tokopedia_Analysis_Report.txt")

# Print summary
print("\n" + report)

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\n📊 All results exported to output/ folder:")
print("   • 00_Tokopedia_Analysis_Report.txt")
print("   • 01_Tokopedia_Sentiment_Analysis.xlsx")
print("   • 02_Tokopedia_Word_Frequency.xlsx")
print("   • 03_Tokopedia_Bigram_Frequency.xlsx")
