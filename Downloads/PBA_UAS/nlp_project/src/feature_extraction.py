"""
Feature Extraction Module untuk NLP
Mendukung Bag of Words (BoW) dan TF-IDF
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Install with: pip install scikit-learn")


class TextFeatureExtractor:
    """
    Kelas untuk ekstraksi fitur dari teks
    Mendukung BoW (Bag of Words) dan TF-IDF
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Inisialisasi feature extractor
        
        Args:
            max_features: Maksimum jumlah features
            ngram_range: Range n-gram (unigram, bigram, etc)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.is_fitted = False
    
    def extract_bow(self, texts, fit=True, max_df=0.95, min_df=2):
        """
        Ekstraksi Bag of Words (BoW)
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer
            max_df: Ignore terms that appear in more than max_df documents
            min_df: Ignore terms that appear in fewer than min_df documents
            
        Returns:
            sparse matrix: BoW feature matrix
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required for BoW extraction")
            return None
        
        if fit:
            self.bow_vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                max_df=max_df,
                min_df=min_df,
                stop_words=self._get_stopwords(),
                lowercase=True
            )
            bow_matrix = self.bow_vectorizer.fit_transform(texts)
            self.feature_names = self.bow_vectorizer.get_feature_names_out()
        else:
            if self.bow_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Run with fit=True first.")
            bow_matrix = self.bow_vectorizer.transform(texts)
        
        return bow_matrix
    
    def extract_tfidf(self, texts, fit=True, max_df=0.95, min_df=2):
        """
        Ekstraksi TF-IDF (Term Frequency-Inverse Document Frequency)
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer
            max_df: Ignore terms that appear in more than max_df documents
            min_df: Ignore terms that appear in fewer than min_df documents
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn is required for TF-IDF extraction")
            return None
        
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                max_df=max_df,
                min_df=min_df,
                stop_words=self._get_stopwords(),
                lowercase=True,
                norm='l2',
                sublinear_tf=True
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Run with fit=True first.")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def get_bow_dataframe(self, texts, fit=True):
        """
        Get BoW sebagai DataFrame
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer
            
        Returns:
            DataFrame: BoW features
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        bow_matrix = self.extract_bow(texts, fit=fit)
        feature_names = self.bow_vectorizer.get_feature_names_out()
        
        df = pd.DataFrame(
            bow_matrix.toarray(),
            columns=feature_names
        )
        
        return df
    
    def get_tfidf_dataframe(self, texts, fit=True):
        """
        Get TF-IDF sebagai DataFrame
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer
            
        Returns:
            DataFrame: TF-IDF features
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        tfidf_matrix = self.extract_tfidf(texts, fit=fit)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )
        
        return df
    
    def get_top_features(self, matrix, feature_names, n=20):
        """
        Dapatkan top N features berdasarkan frequency/importance
        
        Args:
            matrix: Feature matrix (BoW or TF-IDF)
            feature_names: Array of feature names
            n: Number of top features
            
        Returns:
            DataFrame: Top features dengan scores
        """
        # Calculate mean score for each feature
        mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
        
        # Get top n indices
        top_indices = np.argsort(mean_scores)[-n:][::-1]
        
        # Create dataframe
        results = []
        for idx in top_indices:
            results.append({
                'feature': feature_names[idx],
                'score': round(mean_scores[idx], 6)
            })
        
        return pd.DataFrame(results)
    
    def extract_word_frequency(self, texts):
        """
        Ekstraksi frekuensi kata dari teks
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame: Word frequency
        """
        word_freq = Counter()
        
        for text in texts:
            words = str(text).lower().split()
            for word in words:
                if len(word) > 2:  # Skip short words
                    word_freq[word] += 1
        
        # Convert to DataFrame
        freq_df = pd.DataFrame(
            list(word_freq.most_common()),
            columns=['word', 'frequency']
        )
        freq_df['percentage'] = (freq_df['frequency'] / freq_df['frequency'].sum() * 100).round(2)
        
        return freq_df
    
    def extract_bigrams(self, texts, top_n=50):
        """
        Ekstraksi bigrams (2-word phrases) dari teks
        
        Args:
            texts: List of text documents
            top_n: Number of top bigrams to return
            
        Returns:
            DataFrame: Top bigrams dengan frequency
        """
        bigram_freq = Counter()
        
        for text in texts:
            words = str(text).lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(words[i]) > 2 and len(words[i+1]) > 2:  # Skip short words
                    bigram_freq[bigram] += 1
        
        # Convert to DataFrame
        bigram_df = pd.DataFrame(
            list(bigram_freq.most_common(top_n)),
            columns=['bigram', 'frequency']
        )
        bigram_df['percentage'] = (bigram_df['frequency'] / sum(bigram_freq.values()) * 100).round(2)
        
        return bigram_df
    
    def extract_text_statistics(self, texts):
        """
        Ekstraksi statistik teks
        
        Args:
            texts: List of text documents
            
        Returns:
            dict: Text statistics
        """
        text_lengths = [len(str(text).split()) for text in texts]
        char_lengths = [len(str(text)) for text in texts]
        
        return {
            'total_documents': len(texts),
            'avg_words_per_document': round(np.mean(text_lengths), 2),
            'median_words': int(np.median(text_lengths)),
            'min_words': int(np.min(text_lengths)),
            'max_words': int(np.max(text_lengths)),
            'std_words': round(np.std(text_lengths), 2),
            'avg_chars_per_document': round(np.mean(char_lengths), 2),
            'total_words': int(sum(text_lengths)),
        }
    
    def _get_stopwords(self):
        """
        Get Indonesian stopwords
        
        Returns:
            list: Indonesian stopwords
        """
        return {
            # Common Indonesian stopwords
            'yang', 'dan', 'atau', 'untuk', 'di', 'ke', 'dari', 'pada', 'dengan',
            'tanpa', 'oleh', 'saat', 'setelah', 'sebelum', 'dalam', 'lalu', 'juga',
            'karena', 'sebab', 'apakah', 'bagaimana', 'berapa', 'dimana', 'mengapa',
            'siapa', 'kapan', 'apa', 'ini', 'itu', 'ia', 'dia', 'kami', 'kita',
            'saya', 'kamu', 'kalian', 'mereka', 'ada', 'adalah', 'menjadi', 'dapat',
            'bisa', 'harus', 'akan', 'telah', 'sudah', 'pernah', 'jadi', 'tapi',
            'tetapi', 'namun', 'jika', 'bila', 'kalau', 'sebagai', 'sama', 'seperti',
            'antara', 'lebih', 'kurang', 'paling', 'sangat', 'agak', 'cukup', 'sekali',
            'banget', 'dulu', 'kemudian', 'lagi', 'terus', 'maka', 'masing', 'masing2',
            'nya', 'nya', 'nya', 'lah', 'lah', 'lah', 'kah', 'pun', 'dong', 'saja',
        }


# Testing
if __name__ == "__main__":
    extractor = TextFeatureExtractor()
    
    # Test texts
    test_texts = [
        "Produk sangat bagus dan pengiriman cepat!",
        "Barang rusak dan tidak sesuai dengan deskripsi.",
        "Produknya biasa saja, tidak spesial.",
        "Tokopedia bagus tapi promo terbatas.",
        "Pengiriman lambat dan customer service tidak responsif."
    ]
    
    print("="*80)
    print("FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Word frequency
    print("\n[1] WORD FREQUENCY")
    word_freq = extractor.extract_word_frequency(test_texts)
    print(word_freq.head(10))
    
    # Bigrams
    print("\n[2] BIGRAM FREQUENCY")
    bigrams = extractor.extract_bigrams(test_texts, top_n=10)
    print(bigrams)
    
    # Text statistics
    print("\n[3] TEXT STATISTICS")
    stats = extractor.extract_text_statistics(test_texts)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # BoW and TF-IDF
    if SKLEARN_AVAILABLE:
        print("\n[4] BAG OF WORDS (BoW)")
        bow_df = extractor.get_bow_dataframe(test_texts)
        print(f"Shape: {bow_df.shape}")
        print(f"Features: {list(bow_df.columns[:10])}")
        
        print("\n[5] TF-IDF")
        tfidf_df = extractor.get_tfidf_dataframe(test_texts)
        print(f"Shape: {tfidf_df.shape}")
        print(f"Features: {list(tfidf_df.columns[:10])}")
