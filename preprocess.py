import os
import tensorflow as tf
import unicodedata
import re

class Preprocessor:
    def __init__(self, path_to_file=None):
        self.path_to_file = path_to_file

    def unicode_to_ascii(self, string):
        return ''.join(c for c in unicodedata.normalize('NFC', string) if unicodedata.category(c) != 'Mn')

    def preprocess(self, sentence):
        sentence = unicode_to_ascii(sentence.lower().strip())
        sentence = re.sub(r"([?.!,¿])", r" \1", sentence) # separate punctuation into its own token
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()
        sentence = '<start> ' + sentence + ' <end>'
        return sentence
