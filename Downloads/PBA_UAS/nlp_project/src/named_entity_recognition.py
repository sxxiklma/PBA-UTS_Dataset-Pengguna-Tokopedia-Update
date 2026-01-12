"""
Named Entity Recognition (NER) & POS Tagging Module
Untuk ekstraksi entitas dan analisis part-of-speech
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not installed. Install with: pip install spacy")
    print("    Then download model: python -m spacy download en_core_web_sm")


class NamedEntityRecognizer:
    """
    Kelas untuk Named Entity Recognition (NER)
    Ekstraksi entities seperti PERSON, ORG, GPE, DATE, LOCATION, dll
    """
    
    def __init__(self, model='en_core_web_sm'):
        """
        Inisialisasi NER
        
        Args:
            model: spaCy model name
        """
        self.model_name = model
        self.nlp = None
        self.entities_found = []
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model)
            except OSError:
                print(f"⚠️  Model {model} not found. Download with:")
                print(f"    python -m spacy download {model}")
    
    def extract_entities(self, text):
        """
        Ekstraksi entities dari teks
        
        Args:
            text: Text to extract entities from
            
        Returns:
            list: List of entities with type and text
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_entities_batch(self, texts):
        """
        Ekstraksi entities dari batch texts
        
        Args:
            texts: List of texts
            
        Returns:
            dict: Entities grouped by text index
        """
        if self.nlp is None:
            return {}
        
        batch_entities = {}
        
        for idx, text in enumerate(texts):
            entities = self.extract_entities(text)
            batch_entities[idx] = entities
            self.entities_found.extend(entities)
        
        return batch_entities
    
    def get_entity_distribution(self):
        """
        Hitung distribusi entity types
        
        Returns:
            dict: Distribution of entity types
        """
        if not self.entities_found:
            return {}
        
        labels = [ent['label'] for ent in self.entities_found]
        label_count = Counter(labels)
        
        distribution = {}
        total = len(labels)
        
        for label, count in label_count.items():
            distribution[label] = {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }
        
        return distribution
    
    def get_top_entities(self, top_n=20):
        """
        Get top N entities berdasarkan frequency
        
        Args:
            top_n: Number of top entities
            
        Returns:
            DataFrame: Top entities
        """
        if not self.entities_found:
            return pd.DataFrame()
        
        entity_texts = [ent['text'] for ent in self.entities_found]
        entity_counter = Counter(entity_texts)
        
        top_entities = []
        for text, count in entity_counter.most_common(top_n):
            top_entities.append({
                'entity': text,
                'count': count,
                'percentage': round((count / len(entity_texts)) * 100, 2)
            })
        
        return pd.DataFrame(top_entities)
    
    def extract_entities_by_type(self, label):
        """
        Get entities by specific type
        
        Args:
            label: Entity label (e.g., 'PERSON', 'ORG')
            
        Returns:
            list: Entities of specified type
        """
        return [ent['text'] for ent in self.entities_found if ent['label'] == label]
    
    def get_entity_statistics(self):
        """
        Get statistics tentang entities yang ditemukan
        
        Returns:
            dict: Entity statistics
        """
        if not self.entities_found:
            return {}
        
        labels = [ent['label'] for ent in self.entities_found]
        unique_entities = len(set(ent['text'] for ent in self.entities_found))
        
        return {
            'total_entities': len(self.entities_found),
            'unique_entities': unique_entities,
            'entity_types': len(set(labels)),
            'most_common_type': Counter(labels).most_common(1)[0][0] if labels else None
        }


class POS_Tagger:
    """
    Kelas untuk Part-of-Speech (POS) Tagging
    Mengidentifikasi kata (noun, verb, adjective, dll)
    """
    
    def __init__(self, model='en_core_web_sm'):
        """
        Inisialisasi POS Tagger
        
        Args:
            model: spaCy model name
        """
        self.model_name = model
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model)
            except OSError:
                print(f"⚠️  Model {model} not found")
    
    def tag_pos(self, text):
        """
        Tag parts-of-speech untuk teks
        
        Args:
            text: Text to tag
            
        Returns:
            list: List of (token, pos_tag) tuples
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return pos_tags
    
    def tag_pos_batch(self, texts):
        """
        Tag POS untuk batch texts
        
        Args:
            texts: List of texts
            
        Returns:
            dict: POS tags grouped by text index
        """
        if self.nlp is None:
            return {}
        
        batch_tags = {}
        
        for idx, text in enumerate(texts):
            tags = self.tag_pos(text)
            batch_tags[idx] = tags
        
        return batch_tags
    
    def extract_by_pos(self, text, pos_tag):
        """
        Extract words dengan specific POS tag
        
        Args:
            text: Text to extract from
            pos_tag: POS tag to extract (e.g., 'NOUN', 'VERB', 'ADJ')
            
        Returns:
            list: Words with specified POS tag
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        words = [token.text for token in doc if token.pos_ == pos_tag]
        
        return words
    
    def get_pos_distribution(self, texts):
        """
        Hitung distribusi POS tags
        
        Args:
            texts: List of texts
            
        Returns:
            dict: Distribution of POS tags
        """
        if self.nlp is None:
            return {}
        
        pos_counter = Counter()
        
        for text in texts:
            doc = self.nlp(text)
            for token in doc:
                pos_counter[token.pos_] += 1
        
        distribution = {}
        total = sum(pos_counter.values())
        
        for pos, count in pos_counter.items():
            distribution[pos] = {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }
        
        return distribution
    
    def get_dependency_tree(self, text):
        """
        Extract dependency tree dari teks
        
        Args:
            text: Text to analyze
            
        Returns:
            list: Dependency information
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text
            })
        
        return dependencies


class EntitySentimentAnalyzer:
    """
    Kelas untuk menganalisis sentiment berdasarkan entities
    """
    
    def __init__(self):
        """Inisialisasi entity sentiment analyzer"""
        self.positive_adjectives = {
            'bagus', 'baik', 'hebat', 'keren', 'cantik', 'indah', 'sempurna',
            'mantap', 'sip', 'oke', 'ok', 'excellent', 'great', 'good', 'super'
        }
        
        self.negative_adjectives = {
            'buruk', 'jelek', 'sering', 'lambat', 'rusak', 'cacat', 'palsu',
            'abal', 'kw', 'sedih', 'marah', 'kesal', 'bad', 'poor', 'terrible'
        }
    
    def analyze_entity_sentiment(self, text, entity, nlp=None):
        """
        Analisis sentiment dari context sekitar entity
        
        Args:
            text: Original text
            entity: Entity text to analyze
            nlp: spaCy nlp model (optional)
            
        Returns:
            dict: Sentiment analysis for entity
        """
        # Find entity in text
        if entity not in text:
            return None
        
        # Get context (simple approach)
        start_idx = text.find(entity)
        end_idx = start_idx + len(entity)
        
        # Extract 50 chars before and after
        context_start = max(0, start_idx - 50)
        context_end = min(len(text), end_idx + 50)
        context = text[context_start:context_end]
        
        # Count positive/negative words in context
        positive_count = sum(1 for adj in self.positive_adjectives if adj in context.lower())
        negative_count = sum(1 for adj in self.negative_adjectives if adj in context.lower())
        
        sentiment = 'NEUTRAL'
        if positive_count > negative_count:
            sentiment = 'POSITIVE'
        elif negative_count > positive_count:
            sentiment = 'NEGATIVE'
        
        return {
            'entity': entity,
            'context': context,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'sentiment': sentiment
        }


# Testing
if __name__ == "__main__":
    print("="*80)
    print("NAMED ENTITY RECOGNITION & POS TAGGING TEST")
    print("="*80)
    
    test_text = "Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California."
    
    if SPACY_AVAILABLE:
        # NER Test
        ner = NamedEntityRecognizer()
        
        if ner.nlp:
            print("\n[NER Test]")
            entities = ner.extract_entities(test_text)
            for ent in entities:
                print(f"  {ent['text']:20s} -> {ent['label']}")
            
            # POS Test
            print("\n[POS Tagging Test]")
            tagger = POS_Tagger()
            pos_tags = tagger.tag_pos(test_text)
            for token, pos in pos_tags:
                print(f"  {token:15s} -> {pos}")
        else:
            print("⚠️  spaCy model not available. Please install:")
            print("    pip install spacy")
            print("    python -m spacy download en_core_web_sm")
    else:
        print("⚠️  spaCy is required for NER and POS tagging")
        print("    Install with: pip install spacy")
