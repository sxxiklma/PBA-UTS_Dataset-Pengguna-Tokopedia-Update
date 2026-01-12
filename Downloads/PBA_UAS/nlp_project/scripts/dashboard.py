"""
Interactive Dashboard untuk Tokopedia NLP Analysis Results
Menampilkan sentiment analysis, word frequency, dan insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths for data loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="Tokopedia NLP Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🛍️ Tokopedia Dataset - NLP Analysis Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load CSV dataset"""
    csv_file = os.path.join(BASE_DIR, "data", "Dataset pengguna tokopedia.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df
    return None

# Load results
@st.cache_data
def load_analysis_results():
    """Load analysis results dari output folder"""
    results = {}
    output_dir = os.path.join(BASE_DIR, "output")
    
    # Load Excel files
    try:
        word_freq = pd.read_excel(os.path.join(output_dir, "02_Tokopedia_Word_Frequency.xlsx"))
        results['word_frequency'] = word_freq
    except:
        pass
    
    try:
        bigrams = pd.read_excel(os.path.join(output_dir, "03_Tokopedia_Bigram_Frequency.xlsx"))
        results['bigrams'] = bigrams
    except:
        pass
    
    try:
        sentiment = pd.read_excel(os.path.join(output_dir, "01_Tokopedia_Sentiment_Analysis.xlsx"))
        results['sentiment'] = sentiment
    except:
        pass
    
    return results

# Load data
df = load_data()
results = load_analysis_results()

if df is None:
    st.error("❌ Dataset tidak ditemukan!")
    st.stop()

# ============================================================================
# SIDEBAR - Navigation
# ============================================================================
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Pilih Bagian:",
    [
        "📊 Overview",
        "😊 Sentiment Analysis",
        "📝 Text Statistics",
        "🔤 Word Frequency",
        "🔗 Bigram Analysis",
        "💡 Key Insights"
    ]
)

# ============================================================================
# SECTION 1: OVERVIEW
# ============================================================================
if section == "📊 Overview":
    st.header("📊 Dataset Overview")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df), delta="records")
    
    with col2:
        st.metric("Non-Null Texts", len(df[df['text_clean'].notna()]), delta="cleaned")
    
    with col3:
        st.metric("Unique Words", 3810, delta="vocabulary")
    
    with col4:
        st.metric("Avg Text Length", "12.09", delta="words")
    
    st.markdown("---")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗂️ Dataset Structure")
        st.write(f"""
        - **Total Rows**: {len(df)}
        - **Total Columns**: {len(df.columns)}
        - **Columns**: {', '.join(df.columns)}
        """)
        
        st.subheader("📈 Data Quality")
        missing = df.isnull().sum()
        quality_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(quality_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sentiment Distribution (Original)")
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            sentiment_counts.plot(kind='barh', ax=ax, color=colors)
            ax.set_xlabel('Count')
            ax.set_title('Sentiment Label Distribution')
            st.pyplot(fig)

# ============================================================================
# SECTION 2: SENTIMENT ANALYSIS
# ============================================================================
elif section == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis Results")
    st.markdown("---")
    
    if 'sentiment_label' in df.columns:
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution")
            sentiment_counts = df['sentiment_label'].value_counts()
            sentiment_pct = (sentiment_counts / len(df) * 100).round(2)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            wedges, texts, autotexts = ax.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Sentiment Distribution (Original Labels)')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Statistics")
            st.write("""
            ### Sentiment Distribution
            """)
            for sentiment, count in sentiment_counts.items():
                pct = (count / len(df) * 100)
                st.metric(f"{sentiment}", f"{count}", delta=f"{pct:.2f}%")
    
    # Predicted sentiment (dari analisis)
    st.markdown("---")
    st.subheader("🔍 Predicted Sentiment (Lexicon-based)")
    
    predicted_sentiments = ['POSITIF', 'NETRAL', 'NEGATIF']
    predicted_counts = [1067, 565, 367]
    predicted_pct = [53.38, 28.26, 18.36]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.barh(predicted_sentiments, predicted_counts, color=colors)
        ax.set_xlabel('Count')
        ax.set_title('Predicted Sentiment Distribution')
        for i, (count, pct) in enumerate(zip(predicted_counts, predicted_pct)):
            ax.text(count + 20, i, f"{count} ({pct}%)", va='center')
        st.pyplot(fig)
    
    with col2:
        st.write("### Hasil Prediksi")
        for sentiment, count, pct in zip(predicted_sentiments, predicted_counts, predicted_pct):
            st.metric(f"{sentiment}", f"{count}", delta=f"{pct:.2f}%")

# ============================================================================
# SECTION 3: TEXT STATISTICS
# ============================================================================
elif section == "📝 Text Statistics":
    st.header("📝 Text Statistics & Analysis")
    st.markdown("---")
    
    # Calculate text statistics
    text_lengths = [len(str(text).split()) for text in df['text_clean'].astype(str)]
    char_lengths = [len(str(text)) for text in df['text_clean'].astype(str)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Words", f"{np.mean(text_lengths):.2f}")
    with col2:
        st.metric("Median Words", f"{np.median(text_lengths):.0f}")
    with col3:
        st.metric("Max Words", f"{np.max(text_lengths):.0f}")
    with col4:
        st.metric("Min Words", f"{np.min(text_lengths):.0f}")
    
    st.markdown("---")
    
    # Histogram
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Word Count Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(text_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(text_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(text_lengths):.1f}')
        ax.axvline(np.median(text_lengths), color='green', linestyle='--',
                   label=f'Median: {np.median(text_lengths):.1f}')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Text Length Distribution')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Character Count Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(char_lengths, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(char_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(char_lengths):.1f}')
        ax.set_xlabel('Number of Characters')
        ax.set_ylabel('Frequency')
        ax.set_title('Character Length Distribution')
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Detailed Statistics")
    
    stats_data = {
        'Metric': [
            'Mean',
            'Median',
            'Std Dev',
            'Min',
            'Max',
            'Q1 (25%)',
            'Q3 (75%)'
        ],
        'Words': [
            f"{np.mean(text_lengths):.2f}",
            f"{np.median(text_lengths):.0f}",
            f"{np.std(text_lengths):.2f}",
            f"{np.min(text_lengths):.0f}",
            f"{np.max(text_lengths):.0f}",
            f"{np.percentile(text_lengths, 25):.0f}",
            f"{np.percentile(text_lengths, 75):.0f}"
        ]
    }
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# ============================================================================
# SECTION 4: WORD FREQUENCY
# ============================================================================
elif section == "🔤 Word Frequency":
    st.header("🔤 Top Words Analysis")
    st.markdown("---")
    
    if 'word_frequency' in results:
        word_freq_df = results['word_frequency']
        
        # Slider untuk pilih jumlah words
        top_n = st.slider("Pilih jumlah top words:", min_value=5, max_value=50, value=20)
        
        top_words = word_freq_df.head(top_n)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Words Bar Chart")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_words)), top_words['frequency'].values, color='#3498db')
            ax.set_yticks(range(len(top_words)))
            ax.set_yticklabels(top_words['word'].values)
            ax.set_xlabel('Frequency')
            ax.set_title(f'Top {top_n} Most Frequent Words')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Statistics")
            st.metric("Total Unique Words", len(word_freq_df))
            st.metric("Most Frequent Word", word_freq_df.iloc[0]['word'])
            st.metric("Highest Frequency", word_freq_df.iloc[0]['frequency'])
            
            st.markdown("---")
            st.write("### Top 10 Words")
            top_10 = word_freq_df.head(10)[['word', 'frequency']]
            st.dataframe(top_10, use_container_width=True)
        
        # Full table
        st.markdown("---")
        st.subheader("Complete Word Frequency Table")
        st.dataframe(word_freq_df, use_container_width=True)

# ============================================================================
# SECTION 5: BIGRAM ANALYSIS
# ============================================================================
elif section == "🔗 Bigram Analysis":
    st.header("🔗 Bigram (2-Word Phrase) Analysis")
    st.markdown("---")
    
    if 'bigrams' in results:
        bigrams_df = results['bigrams']
        
        # Slider untuk pilih jumlah bigrams
        top_n = st.slider("Pilih jumlah top bigrams:", min_value=5, max_value=50, value=20)
        
        top_bigrams = bigrams_df.head(top_n)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Bigrams Bar Chart")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_bigrams)), top_bigrams['frequency'].values, color='#e74c3c')
            ax.set_yticks(range(len(top_bigrams)))
            ax.set_yticklabels(top_bigrams['bigram'].values, fontsize=9)
            ax.set_xlabel('Frequency')
            ax.set_title(f'Top {top_n} Most Frequent Bigrams')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Statistics")
            st.metric("Total Unique Bigrams", len(bigrams_df))
            st.metric("Most Frequent Bigram", bigrams_df.iloc[0]['bigram'])
            st.metric("Highest Frequency", bigrams_df.iloc[0]['frequency'])
            
            st.markdown("---")
            st.write("### Top 10 Bigrams")
            top_10_bigrams = bigrams_df.head(10)[['bigram', 'frequency']]
            st.dataframe(top_10_bigrams, use_container_width=True)
        
        # Full table
        st.markdown("---")
        st.subheader("Complete Bigram Frequency Table")
        st.dataframe(bigrams_df, use_container_width=True)

# ============================================================================
# SECTION 6: KEY INSIGHTS
# ============================================================================
elif section == "💡 Key Insights":
    st.header("💡 Key Insights & Recommendations")
    st.markdown("---")
    
    # Key findings
    st.subheader("🎯 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Sentiment Insights
        
        ✅ **Positive Sentiment Dominates**
        - 53.38% dari reviews bersifat POSITIF
        - Menunjukkan kepuasan pengguna tinggi
        - User experience generally good
        
        ⚠️ **Neutral Comments**
        - 28.26% neutral sentiment
        - Potential untuk improvement
        - Mixed feelings dari users
        
        ❌ **Negative Sentiment**
        - 18.36% negative sentiment
        - Specific issues perlu diperhatikan
        - Action items: customer support
        """)
    
    with col2:
        st.markdown("""
        ### Content Insights
        
        📊 **Text Characteristics**
        - Average text length: 12 words
        - Short, concise reviews
        - Mobile-friendly feedback
        
        🔤 **Most Discussed Topics**
        1. **Tokopedia Platform** (288x)
        2. **Product Availability** (274x)
        3. **Promotions/Discounts** (210x)
        
        🔗 **Common Phrases**
        - "pengguna baru" → new users
        - "tidak bisa" → payment/system issues
        - "gratis ongkir" → positive about shipping
        """)
    
    st.markdown("---")
    
    # Problem areas
    st.subheader("⚠️ Problem Areas Detected")
    
    problems = {
        'Issue': [
            'Payment Processing',
            'System Cancellation',
            'Delivery Issues',
            'Product Mismatch',
            'Seller Responsiveness'
        ],
        'Frequency': [50, 39, 42, 35, 28],
        'Priority': ['HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW'],
        'Action': [
            'Review payment gateway',
            'Debug system cancellation logic',
            'Partner with better couriers',
            'Improve product descriptions',
            'Train seller support'
        ]
    }
    
    st.dataframe(pd.DataFrame(problems), use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Short-term Actions (1-3 months)
        
        1. **Optimize Payment Gateway**
           - Fix "tidak bisa" issues
           - Add alternative payment methods
           - Reduce transaction failures
        
        2. **Improve Order Cancellation**
           - Review system logic
           - Reduce auto-cancellations
           - Better user communication
        
        3. **Shipping Enhancement**
           - Partner with reliable couriers
           - Track "kurir rekomendasi" issues
           - Speed up delivery
        """)
    
    with col2:
        st.markdown("""
        ### Long-term Strategy (3-12 months)
        
        1. **Product Catalog Quality**
           - Better descriptions
           - More accurate images
           - Detailed specifications
        
        2. **Seller Performance**
           - Training programs
           - Quality standards
           - Customer service metrics
        
        3. **New User Experience**
           - Onboarding optimization
           - "pengguna baru" support
           - First-purchase incentives
        """)
    
    st.markdown("---")
    
    # Opportunities
    st.subheader("🚀 Growth Opportunities")
    
    opportunities = """
    ✅ **Leverage Positive Feedback**
    - Highlight positive reviews in marketing
    - Use as testimonials
    - Reward positive reviewers
    
    ✅ **Expand Promotions**
    - Users mention "gratis ongkir" positively
    - Continue shipping promotions
    - Add seasonal offers
    
    ✅ **Community Building**
    - Engage with active reviewers
    - Create review rewards program
    - Build user community
    
    ✅ **Data-Driven Improvements**
    - Monitor sentiment trends
    - Real-time issue detection
    - Quick response system
    """
    
    st.info(opportunities)

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        <p>Tokopedia NLP Analysis Dashboard | Generated: January 12, 2026</p>
        <p>Data: 1,999 user reviews | Analysis: Sentiment, Features, Statistics</p>
    </div>
    """, unsafe_allow_html=True)
