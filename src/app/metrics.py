from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet
try:
    wordnet.synsets('test')
except LookupError:
    # Якщо бази немає — завантажуємо
    nltk.download('wordnet')
    

class CosineSimilarity:
    """ Клас для обчислення косинусної подібності між двома текстами за допомогою TF-IDF векторизації.
    Використовує TfidfVectorizer для перетворення тексту у векторну форму з врахуванням
    уніграм та біграм, а також застосуванням сублінійного масштабування частоти термінів (TF).

    Attributes:
        vectorizer (TfidfVectorizer): Налаштований інструмент для перетворення тексту в матрицю TF-IDF.
    """
    def __init__(self):
        # Ініціалізуємо інструмент один раз при створенні об'єкта
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            use_idf=True,
            norm='l2',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            sublinear_tf=True,   # Apply sublinear TF scaling
            # You could also experiment with 'char' or 'char_wb' for character-level features
            analyzer='word'      
        )

    def calculate_cosine_similarity(self, text1, text2):
        """
        Обчислює косинусну подібність між двома вхідними рядками.

        Args:
            text1 (str): Перший текст для порівняння (наприклад, запит користувача).
            text2 (str): Другий текст для порівняння (наприклад, знайдений фрагмент документу).

        Returns:
            float: Значення подібності в діапазоні від 0.0 до 1.0, де 1.0 означає повну ідентичність.
        """
        tfidf = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
        return similarity[0][0]
    

class EnhancedSimilarity:
    """ """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_synonyms(self, word):
        """извлекает из WordNet синонимы указанного слова."""
        
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def preprocess_text(self, text):
        """преобразует весь текст в нижний регистр, лемматизи-
        рует его (получает корни слов) и удаляет стоп-слова (типичные слова, не не-
        сущие смысловой нагрузки) и знаки пунктуации."""
        
        doc = self.nlp(text.lower())
        lemmatized_words = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            lemmatized_words.append(token.lemma_)
        return lemmatized_words

    def expand_with_synonyms(self, words):
        """расширяет список слов, добавляя к ним синонимы"""
        
        expanded_words = words.copy()
        for word in words:
            expanded_words.extend(self.get_synonyms(word))
        return expanded_words

    def calculate_enhanced_similarity(self, text1, text2):
        # Preprocess and tokenize texts
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)

        # Expand with synonyms
        words1_expanded = self.expand_with_synonyms(words1)
        words2_expanded = self.expand_with_synonyms(words2)

        # Count word frequencies
        freq1 = Counter(words1_expanded)
        freq2 = Counter(words2_expanded)

        # Create a set of all unique words
        unique_words = set(freq1.keys()).union(set(freq2.keys()))

        # Create frequency vectors
        vector1 = [freq1[word] for word in unique_words]
        vector2 = [freq2[word] for word in unique_words]

        # Convert lists to numpy arrays
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # Calculate cosine similarity
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        return similarity





    
