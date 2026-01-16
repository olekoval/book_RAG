from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
