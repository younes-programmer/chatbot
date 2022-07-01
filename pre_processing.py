import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


"""
    NLP pre-process
    tokenization : white space tokenizer
    steaming : work working worked works => work
    bag of words
"""


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # steaming words of a tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # using numpy or torch to make all words to 0
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1

    return bag


"""
# testing tokenization
msg = "Hi i'am your assistant bot"
tokenize_msg = tokenize(msg)
print(f"{msg}\n{tokenize_msg}")
"""

"""
# testing steaming
my_words = ['work', 'working', 'worked', 'works']
word_stemed = [stem(w) for w in my_words]
print(word_stemed)
"""

"""
# testing bag of words
all_words = ["hello", "hi", "i am", "you", "night", "to", "good", "glad", "see"]
my_tokenized_sentence = ["good", "night", "i am", "glad", "to", "see", "you"]
# expected bag of words [0, 0, 1, 1, 1, 1, 1, 1, 1]
print(bag_of_words(my_tokenized_sentence, all_words))
"""
