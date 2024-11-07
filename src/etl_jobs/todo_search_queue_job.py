# обновить (удалить) очередь поиска - не искать дважды: наполнить ее постгресом - нашли и поскрапили = 1
# search логика должна быть лучше - написать логику удаления говна из названия телефона - relevant tokens
# test case: "Смартфон Cubot Note 30 4GB/64GB" = Cubot Note 30 or Смартфон Cubot Note 30


import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    return text.lower()


def extract_meaningful_parts(corpus):
    # Preprocess the corpus
    preprocessed_corpus = [preprocess(name) for name in corpus]

    # Get word frequencies
    all_words = ' '.join(preprocessed_corpus).split()
    word_freq = Counter(all_words)

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)
    feature_names = vectorizer.get_feature_names_out()

    meaningful_parts = []

    for i, name in enumerate(corpus):
        words = name.split()
        meaningful_words = []

        for word in words:
            preprocessed_word = preprocess(word)

            # Check if the word is frequent and has high TF-IDF score
            if (word_freq[preprocessed_word] > 1 and
                    preprocessed_word in feature_names and
                    tfidf_matrix[i, feature_names.tolist().index(preprocessed_word)] > 0.1):
                meaningful_words.append(word)

        meaningful_parts.append(' '.join(meaningful_words))

    return meaningful_parts


# Example usage
corpus = [
    "Apple iPhone 12 Pro ABC123",
    "Samsung Galaxy S21 Ultra XYZ456",
    "Sony PlayStation 5 DEF789",
    "Microsoft Surface Pro 7 GHI101"
]

meaningful_parts = extract_meaningful_parts(corpus)

for original, meaningful in zip(corpus, meaningful_parts):
    print(f"Original: {original}")
    print(f"Meaningful: {meaningful}")
    print()