"""
Sentiment Analysis Module untuk Tokopedia Dataset
Mendukung TextBlob dan Lexicon-based sentiment analysis
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not installed. Install with: pip install textblob")


class SentimentAnalyzer:
    """
    Kelas untuk analisis sentimen teks
    Mendukung metode: TextBlob, Lexicon-based
    """
    
    def __init__(self):
        """Inisialisasi sentiment analyzer"""
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.intensifiers = {'sangat', 'super', 'banget', 'sekali', 'sungguh', 'benar'}
        self.negators = {'tidak', 'bukan', 'gak', 'gag', 'jangan', 'nggak', 'tak', 'tiada'}
    
    def _load_positive_words(self):
        """Load daftar kata positif untuk Tokopedia"""
        return {
            # Produk & kualitas
            'bagus', 'baik', 'hebat', 'mantap', 'keren', 'sip', 'oke', 'okeh',
            'berkualitas', 'terbaik', 'super', 'excellent', 'good', 'great',
            'sempurna', 'memuaskan', 'rapi', 'lengkap', 'menarik', 'unik',
            'elegan', 'cantik', 'indah', 'gagah', 'mewah', 'premium', 'eksklusif',
            
            # Layanan & pengiriman
            'cepat', 'tepat', 'sampai', 'lancar', 'aman', 'nyaman', 'praktis',
            'mudah', 'sederhana', 'profesional', 'ramah', 'helpful', 'responsif',
            'cekas', 'berhasil', 'sukses', 'beruntung', 'gembira', 'puas',
            
            # Harga & promo
            'murah', 'hemat', 'diskon', 'gratis', 'ongkir', 'promo', 'deal',
            'terjangkau', 'ekonomis', 'worth', 'nilai', 'recommend',
            
            # Kepuasan
            'senang', 'suka', 'syukur', 'terima', 'sesuai', 'sepaket', 'lengkap',
            'persis', 'mirip', 'sama', 'cocok', 'pas', 'benar', 'tepat', 'akurat',
            
            # Rekomendasi
            'rekomen', 'rekomendasi', 'rekomendasikan', 'worth', 'penting',
            'harus', 'wajib', 'perlu', 'jangan', 'amat', 'lumayan',
            
            # Positif umum
            'membantu', 'berguna', 'bermanfaat', 'aman', 'terpercaya', 'resmi',
            'original', 'authentic', 'genuine', 'real', 'true', 'legit',
        }
    
    def _load_negative_words(self):
        """Load daftar kata negatif untuk Tokopedia"""
        return {
            # Masalah & keluhan
            'buruk', 'jelek', 'tidak', 'gak', 'tidak bisa', 'gak bisa', 'palsu',
            'rusak', 'cacat', 'pecah', 'hilang', 'kotor', 'bau', 'bekas',
            'abal', 'abal-abal', 'murahan', 'kw', 'kualitas rendah', 'murah murah',
            
            # Pengiriman & layanan
            'lama', 'lambat', 'terlambat', 'dikembalikan', 'dibatalkan', 'dibatalkan sistem',
            'hilang', 'rusak', 'tidak aman', 'semrawut', 'berantakan', 'mengecewakan',
            'tidak sesuai', 'tidak sama', 'berbeda', 'tidak seperti', 'tidak cocok',
            
            # Pembayaran & transaksi
            'error', 'failed', 'gagal', 'stuck', 'macet', 'terputus', 'disconnect',
            'tidak terverifikasi', 'tidak terproses', 'pending', 'tertunda',
            
            # Penjual & customer service
            'penipu', 'scam', 'bohong', 'membohongi', 'menipu', 'tidak jujur',
            'tidak responsif', 'kasar', 'seenaknya', 'tidak sopan', 'berbelanja',
            'tidak peduli', 'abaikan', 'mengabaikan', 'diabaikan',
            
            # Harga
            'mahal', 'overpriced', 'expensive', 'naik', 'kenaikan', 'membengkak',
            'tidak sebanding', 'rugi', 'sayang', 'mengecewakan', 'mengecewakan',
            
            # Negatif umum
            'salah', 'kesalahan', 'error', 'gagal', 'buntut', 'ketinggalan',
            'hilang', 'terjebak', 'tertipu', 'tertipu', 'bodoh', 'bego',
            'kesel', 'emosi', 'marah', 'kesal', 'sial', 'sialnya', 'sampah',
            'berantakan', 'berantakan', 'repot', 'ribet', 'rumit', 'kompleks',
            
            # Pengalaman negatif
            'kecewa', 'mengecewakan', 'mengecewakan', 'memilukan', 'nyesek',
            'nyeri', 'sakit', 'derita', 'penderitaan', 'menderita', 'sengsara',
            'kesal', 'keberatan', 'protes', 'komplain', 'keluhan',
        }
    
    def analyze_sentiment_textblob(self, text):
        """
        Analisis sentimen menggunakan TextBlob
        
        Args:
            text: Teks untuk dianalisis
            
        Returns:
            dict: {polarity, subjectivity, sentiment}
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'NETRAL', 'method': 'textblob'}
        
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'POSITIF'
            elif polarity < -0.1:
                sentiment = 'NEGATIF'
            else:
                sentiment = 'NETRAL'
            
            return {
                'polarity': round(polarity, 4),
                'subjectivity': round(subjectivity, 4),
                'sentiment': sentiment,
                'method': 'textblob'
            }
        except Exception as e:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'NETRAL', 'error': str(e), 'method': 'textblob'}
    
    def analyze_sentiment_lexicon(self, text):
        """
        Analisis sentimen berbasis lexicon
        
        Args:
            text: Teks untuk dianalisis
            
        Returns:
            dict: {positive_count, negative_count, sentiment, intensity}
        """
        text_lower = str(text).lower()
        
        # Count positive dan negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Check for intensifiers
        has_intensifier = any(word in text_lower for word in self.intensifiers)
        
        # Check for negators
        has_negator = any(word in text_lower for word in self.negators)
        
        # Hitung score
        score = positive_count - negative_count
        
        if has_intensifier:
            score *= 1.5
        
        if has_negator and score > 0:
            score *= 0.5
        
        # Determine sentiment
        if score > 0:
            sentiment = 'POSITIF'
        elif score < 0:
            sentiment = 'NEGATIF'
        else:
            sentiment = 'NETRAL'
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'score': round(score, 2),
            'sentiment': sentiment,
            'has_intensifier': has_intensifier,
            'has_negator': has_negator,
            'method': 'lexicon'
        }
    
    def analyze_sentiment_combined(self, text):
        """
        Analisis sentimen dengan kombinasi TextBlob dan Lexicon
        
        Args:
            text: Teks untuk dianalisis
            
        Returns:
            dict: Hasil gabungan dari kedua metode
        """
        textblob_result = self.analyze_sentiment_textblob(text)
        lexicon_result = self.analyze_sentiment_lexicon(text)
        
        # Combine hasil
        if TEXTBLOB_AVAILABLE and 'error' not in textblob_result:
            # Jika TextBlob available, gunakan kombinasi
            if textblob_result['polarity'] > 0.1 and lexicon_result['sentiment'] in ['POSITIF', 'NETRAL']:
                final_sentiment = 'POSITIF'
            elif textblob_result['polarity'] < -0.1 and lexicon_result['sentiment'] in ['NEGATIF', 'NETRAL']:
                final_sentiment = 'NEGATIF'
            else:
                final_sentiment = lexicon_result['sentiment']
        else:
            # Gunakan lexicon saja
            final_sentiment = lexicon_result['sentiment']
        
        return {
            'textblob': textblob_result,
            'lexicon': lexicon_result,
            'final_sentiment': final_sentiment,
            'confidence': abs(textblob_result.get('polarity', 0)) if TEXTBLOB_AVAILABLE else abs(lexicon_result['score']) / 10
        }
    
    def analyze_batch(self, texts, method='combined'):
        """
        Analisis batch teks
        
        Args:
            texts: List of texts
            method: 'textblob', 'lexicon', or 'combined'
            
        Returns:
            List of sentiment results
        """
        results = []
        
        for text in texts:
            if method == 'textblob':
                result = self.analyze_sentiment_textblob(text)
            elif method == 'lexicon':
                result = self.analyze_sentiment_lexicon(text)
            else:  # combined
                result = self.analyze_sentiment_combined(text)
            
            results.append(result)
        
        return results
    
    def get_sentiment_distribution(self, sentiments):
        """
        Hitung distribusi sentimen
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            dict: Distribution statistics
        """
        counter = Counter(sentiments)
        total = len(sentiments)
        
        distribution = {}
        for sentiment, count in counter.items():
            distribution[sentiment] = {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }
        
        return distribution


# Testing
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test TextBlob
    test_text1 = "Produk sangat bagus dan pengiriman cepat!"
    test_text2 = "Barang rusak dan tidak sesuai dengan deskripsi."
    test_text3 = "Produknya biasa saja."
    
    print("="*80)
    print("SENTIMENT ANALYZER TEST")
    print("="*80)
    
    for text in [test_text1, test_text2, test_text3]:
        print(f"\nText: {text}")
        result = analyzer.analyze_sentiment_combined(text)
        print(f"Lexicon Result: {result['lexicon']}")
        print(f"Final Sentiment: {result['final_sentiment']}")
