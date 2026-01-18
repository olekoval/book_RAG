from nltk.corpus import wordnet

word = "rag"
# Отримуємо всі синсети (значення) для слова
synsets = wordnet.synsets(word)

for syn in synsets:
    # Виводимо назву синсету, його визначення та всі синоніми в ньому
    print(f"Значення: {syn.name()} | Визначення: {syn.definition()}")
    lemmas = [lemma.name() for lemma in syn.lemmas()]
    print(f"  Синоніми: {lemmas}\n")
