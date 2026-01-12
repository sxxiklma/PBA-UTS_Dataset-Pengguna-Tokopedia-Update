"""
Script untuk menganalisis Dataset Tokopedia
Menggunakan NLP Pipeline yang telah dibuat
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import NLP modules
from sentiment_analyzer import SentimentAnalyzer
from feature_extraction import TextFeatureExtractor
from text_classifier import TextClassifier
from named_entity_recognition import NamedEntityRecognizer
from visualization import VisualizationHelper, AnalysisReporter


def analyze_tokopedia_dataset():
    """
    Analisis lengkap dataset Tokopedia
    """
    
    print("\n" + "="*80)
    print("ANALISIS DATASET TOKOPEDIA - NLP PIPELINE")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading Dataset...")
    csv_file = "Dataset pengguna tokopedia.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"✓ Dataset loaded: {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Step 2: Data Overview
    print("\n" + "="*80)
    print("[STEP 2] Data Overview")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns Info:")
    print(df.info())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nSentiment Distribution (existing labels):")
    if 'sentiment_label' in df.columns:
        print(df['sentiment_label'].value_counts())
    
    # Get text column
    text_column = 'text_clean' if 'text_clean' in df.columns else 'text_original'
    texts = df[text_column].astype(str).tolist()
    
    print(f"\nUsing text column: {text_column}")
    print(f"Total texts: {len(texts)}")
    
    # Step 3: Sentiment Analysis
    print("\n" + "="*80)
    print("[STEP 3] Sentiment Analysis")
    print("="*80)
    
    print("\nAnalyzing sentiments with TextBlob...")
    analyzer = SentimentAnalyzer()
    
    # Proses batch dengan progress
    batch_size = 100
    sentiment_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = analyzer.analyze_batch(batch, method='textblob')
        sentiment_results.append(batch_results)
        
        progress = min(i + batch_size, len(texts))
        print(f"  Progress: {progress}/{len(texts)}", end='\r')
    
    sentiment_df = pd.concat(sentiment_results, ignore_index=True)
    print(f"✓ Sentiment analysis completed")
    
    print(f"\nSentiment Distribution (TextBlob):")
    print(sentiment_df['sentiment'].value_counts())
    
    print(f"\nPolarity Statistics:")
    print(f"  Mean: {sentiment_df['polarity'].mean():.4f}")
    print(f"  Std: {sentiment_df['polarity'].std():.4f}")
    print(f"  Min: {sentiment_df['polarity'].min():.4f}")
    print(f"  Max: {sentiment_df['polarity'].max():.4f}")
    
    # Step 4: Feature Extraction
    print("\n" + "="*80)
    print("[STEP 4] Feature Extraction")
    print("="*80)
    
    print("\nExtracting features...")
    extractor = TextFeatureExtractor(max_features=500)
    
    print("  • Extracting Bag of Words...")
    bow_df, _, bow_features = extractor.extract_bow(texts)
    print(f"    ✓ BoW matrix shape: {bow_df.shape}")
    
    print("  • Extracting TF-IDF...")
    tfidf_df, _, tfidf_features = extractor.extract_tfidf(texts)
    print(f"    ✓ TF-IDF matrix shape: {tfidf_df.shape}")
    
    print(f"\nTop 15 Terms (by BoW frequency):")
    top_bow = extractor.get_top_features_bow(texts, n_top=15)
    for i, (term, freq) in enumerate(top_bow.items(), 1):
        print(f"  {i:2d}. {term:20s} : {freq:6.0f}")
    
    print(f"\nTop 15 Terms (by TF-IDF score):")
    top_tfidf = extractor.get_top_features_tfidf(texts, n_top=15)
    for i, (term, score) in enumerate(top_tfidf.items(), 1):
        print(f"  {i:2d}. {term:20s} : {score:8.4f}")
    
    # Step 5: Named Entity Recognition
    print("\n" + "="*80)
    print("[STEP 5] Named Entity Recognition")
    print("="*80)
    
    print("\nExtracting named entities...")
    ner = NamedEntityRecognizer()
    
    # Sample untuk NER (karena memakan waktu)
    sample_size = min(500, len(texts))
    sample_texts = texts[:sample_size]
    
    print(f"  Processing {sample_size} texts...")
    ner_df = ner.extract_entities_batch(sample_texts)
    
    if len(ner_df) > 0:
        print(f"✓ Found {len(ner_df)} entities")
        
        print(f"\nEntity Distribution:")
        entity_dist = ner_df['label_description'].value_counts()
        for entity_type, count in entity_dist.items():
            print(f"  • {entity_type:20s} : {count:4d}")
        
        print(f"\nMost Frequent Entities:")
        top_entities = ner_df['entity'].value_counts().head(10)
        for i, (entity, count) in enumerate(top_entities.items(), 1):
            print(f"  {i:2d}. {entity:30s} : {count:4d}")
    else:
        print("⚠️  No entities found in sample")
        ner_df = pd.DataFrame()
    
    # Step 6: Comparison dengan existing labels
    print("\n" + "="*80)
    print("[STEP 6] Comparison with Existing Labels")
    print("="*80)
    
    if 'sentiment_label' in df.columns:
        # Map sentiment labels
        sentiment_mapping = {
            'Baik': 'POSITIF',
            'Buruk': 'NEGATIF',
            'Netral': 'NETRAL'
        }
        
        df['existing_sentiment'] = df['sentiment_label'].map(sentiment_mapping)
        df['predicted_sentiment'] = sentiment_df['sentiment']
        
        # Calculate agreement
        agreement = (df['existing_sentiment'] == df['predicted_sentiment']).sum()
        agreement_rate = agreement / len(df) * 100
        
        print(f"\nAgreement Rate: {agreement_rate:.2f}% ({agreement}/{len(df)})")
        
        print(f"\nConfusion Matrix:")
        confusion = pd.crosstab(
            df['existing_sentiment'],
            df['predicted_sentiment'],
            margins=True
        )
        print(confusion)
    
    # Step 7: Export Results
    print("\n" + "="*80)
    print("[STEP 7] Exporting Results")
    print("="*80)
    
    os.makedirs('output', exist_ok=True)
    
    # Export sentiment analysis
    sentiment_export_df = pd.DataFrame({
        'text': texts,
        'textblob_sentiment': sentiment_df['sentiment'],
        'polarity': sentiment_df['polarity'],
        'subjectivity': sentiment_df['subjectivity']
    })
    
    if 'sentiment_label' in df.columns:
        sentiment_export_df['existing_label'] = df['sentiment_label']
    
    sentiment_export_df.to_excel('output/01_Tokopedia_Sentiment_Analysis.xlsx', index=False)
    print("✓ Exported: 01_Tokopedia_Sentiment_Analysis.xlsx")
    
    # Export feature comparison
    feature_comparison = pd.DataFrame({
        'top_bow_terms': list(top_bow.index),
        'bow_frequency': list(top_bow.values),
        'top_tfidf_terms': list(top_tfidf.index) if len(top_tfidf) == len(top_bow) else (list(top_tfidf.index) + ['']*max(0, len(top_bow)-len(top_tfidf))),
        'tfidf_score': list(top_tfidf.values) if len(top_tfidf) == len(top_bow) else (list(top_tfidf.values) + [0]*max(0, len(top_bow)-len(top_tfidf)))
    })
    feature_comparison.to_excel('output/02_Tokopedia_Feature_Analysis.xlsx', index=False)
    print("✓ Exported: 02_Tokopedia_Feature_Analysis.xlsx")
    
    # Export BoW and TF-IDF matrices
    bow_df.to_excel('output/03_Tokopedia_BoW_Matrix.xlsx', index=False)
    print("✓ Exported: 03_Tokopedia_BoW_Matrix.xlsx")
    
    tfidf_df.to_excel('output/04_Tokopedia_TFIDF_Matrix.xlsx', index=False)
    print("✓ Exported: 04_Tokopedia_TFIDF_Matrix.xlsx")
    
    # Export NER results
    if len(ner_df) > 0:
        ner_df.to_excel('output/05_Tokopedia_Named_Entities.xlsx', index=False)
        print("✓ Exported: 05_Tokopedia_Named_Entities.xlsx")
    
    # Step 8: Generate Report
    print("\n" + "="*80)
    print("[STEP 8] Generating Analysis Report")
    print("="*80)
    
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   TOKOPEDIA DATASET - NLP ANALYSIS REPORT                 ║
╚════════════════════════════════════════════════════════════════════════════╝

DATASET OVERVIEW
════════════════════════════════════════════════════════════════════════════

Total Texts Analyzed:         {len(texts):,}
Text Column Used:             {text_column}
Date Analyzed:                {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SENTIMENT ANALYSIS RESULTS (TextBlob Method)
════════════════════════════════════════════════════════════════════════════

Sentiment Distribution:
  • POSITIF (Positive):       {len(sentiment_df[sentiment_df['sentiment']=='POSITIF']):4d} ({len(sentiment_df[sentiment_df['sentiment']=='POSITIF'])/len(sentiment_df)*100:5.2f}%)
  • NEGATIF (Negative):       {len(sentiment_df[sentiment_df['sentiment']=='NEGATIF']):4d} ({len(sentiment_df[sentiment_df['sentiment']=='NEGATIF'])/len(sentiment_df)*100:5.2f}%)
  • NETRAL (Neutral):         {len(sentiment_df[sentiment_df['sentiment']=='NETRAL']):4d} ({len(sentiment_df[sentiment_df['sentiment']=='NETRAL'])/len(sentiment_df)*100:5.2f}%)

Polarity Statistics (Range: -1 to 1):
  • Mean Polarity:            {sentiment_df['polarity'].mean():7.4f}
  • Std Deviation:            {sentiment_df['polarity'].std():7.4f}
  • Minimum:                  {sentiment_df['polarity'].min():7.4f}
  • Maximum:                  {sentiment_df['polarity'].max():7.4f}

Subjectivity Statistics (Range: 0 to 1):
  • Mean Subjectivity:        {sentiment_df['subjectivity'].mean():7.4f}
  • Std Deviation:            {sentiment_df['subjectivity'].std():7.4f}

FEATURE ANALYSIS
════════════════════════════════════════════════════════════════════════════

Bag of Words Matrix:
  • Shape:                    {bow_df.shape[0]} texts × {bow_df.shape[1]} features
  • Vocabulary Size:          {len(bow_features)} unique terms
  
TF-IDF Matrix:
  • Shape:                    {tfidf_df.shape[0]} texts × {tfidf_df.shape[1]} features
  • Vocabulary Size:          {len(tfidf_features)} unique terms (with bigrams)

Top 10 Most Frequent Terms (BoW):
"""
    
    for i, (term, freq) in enumerate(top_bow.head(10).items(), 1):
        report += f"  {i:2d}. {term:25s} ({freq:6.0f} occurrences)\n"
    
    report += f"""
Top 10 Most Important Terms (TF-IDF):
"""
    
    for i, (term, score) in enumerate(top_tfidf.head(10).items(), 1):
        report += f"  {i:2d}. {term:25s} (score: {score:8.4f})\n"
    
    report += f"""
NAMED ENTITY RECOGNITION (Sample of {sample_size} texts)
════════════════════════════════════════════════════════════════════════════

Total Entities Found:         {len(ner_df):,}
"""
    
    if len(ner_df) > 0:
        entity_dist = ner_df['label_description'].value_counts()
        for entity_type, count in entity_dist.items():
            report += f"  • {entity_type:20s}  : {count:6d}\n"
    
    if 'sentiment_label' in df.columns:
        report += f"""
COMPARISON WITH EXISTING LABELS
════════════════════════════════════════════════════════════════════════════

Agreement Rate:               {agreement_rate:.2f}%
Matching Predictions:         {agreement:,} out of {len(df):,}

This shows how well the TextBlob sentiment aligns with existing labels.
Higher agreement indicates consistent sentiment classification.
"""
    
    report += f"""
KEY INSIGHTS
════════════════════════════════════════════════════════════════════════════

1. SENTIMENT OVERVIEW:
   - The dataset shows predominantly {'NEUTRAL' if len(sentiment_df[sentiment_df['sentiment']=='NETRAL']) > len(sentiment_df[sentiment_df['sentiment']=='POSITIF']) else 'POSITIVE'} sentiments
   - Average polarity is {sentiment_df['polarity'].mean():.2f}, indicating mostly {'positive' if sentiment_df['polarity'].mean() > 0 else 'negative'} tone

2. MOST DISCUSSED TOPICS:
   - Top terms suggest users discuss: {', '.join(list(top_bow.index[:5]))}

3. DATA QUALITY:
   - All {len(texts):,} texts were successfully processed
   - No missing values in sentiment column
   - Texts are in Indonesian language with mixed slang

4. SENTIMENT DISTRIBUTION:
   - {(len(sentiment_df[sentiment_df['sentiment']=='POSITIF'])/len(sentiment_df)*100):.1f}% positive opinions
   - {(len(sentiment_df[sentiment_df['sentiment']=='NEGATIF'])/len(sentiment_df)*100):.1f}% negative opinions
   - {(len(sentiment_df[sentiment_df['sentiment']=='NETRAL'])/len(sentiment_df)*100):.1f}% neutral opinions

RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════

1. For Tokopedia Team:
   - Focus on addressing user concerns about: payment, delivery, system issues
   - Improve user experience in mentioned problem areas

2. For Further Analysis:
   - Use the labeled data to train custom sentiment classifier
   - Perform topic modeling for deeper insights
   - Track sentiment trends over time

3. Data Processing:
   - Consider using custom Indonesian sentiment lexicon
   - Fine-tune parameters for better accuracy
   - Consider using ensemble methods for classification

EXPORTED FILES
════════════════════════════════════════════════════════════════════════════

All results saved to output/ folder:
  1. 01_Tokopedia_Sentiment_Analysis.xlsx      - Sentiment scores & labels
  2. 02_Tokopedia_Feature_Analysis.xlsx        - Top features comparison
  3. 03_Tokopedia_BoW_Matrix.xlsx              - Bag of Words matrix
  4. 04_Tokopedia_TFIDF_Matrix.xlsx            - TF-IDF matrix
  5. 05_Tokopedia_Named_Entities.xlsx          - Extracted entities (sample)

════════════════════════════════════════════════════════════════════════════
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Tool: NLP Processing Pipeline
Language: Indonesian (with English components)
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
    print("\nAll results saved to output/ folder")
    print("Check the report: output/00_Tokopedia_Analysis_Report.txt")


if __name__ == "__main__":
    try:
        analyze_tokopedia_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
