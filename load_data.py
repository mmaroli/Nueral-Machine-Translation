import os
import tensorflow as tf
from preprocess import Preprocessor



class DataLoader:
    """ Class to help loading data """
    def __init__(self, path_to_file=None):
        self.path_to_file = path_to_file

    def load_data_tf(self):
        path_to_zip = tf.keras.utils.get_file(
            'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True
        )
        self.path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

        preprocessor = Preprocessor(path_to_file=self.path_to_file)

        return self.path_to_file

    def load_data_from_file(self):
        if not self.path_to_file:
            raise ValueError("Path to file cannot be none.")

        preprocessor = Preprocessor(path_to_file=self.path_to_file)
        with open(self.path_to_file, encoding='UTF-8') as f:
            pairs = [ [preprocessor.preprocess(sentence) for sentence in line.strip().split('\t')] for line in f]
            return zip(*pairs)

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang) # creates index to word mapping

        tensor = lang_tokenizer.texts_to_sequences(lang) # list of sequences
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post') # pads sequence to be equal lengths, returns np array

        return tensor, lang_tokenizer

    def convert(self, lang, tensor):
        for t in tensor:
            if t !=0:
                print ("%d ----> %s" % (t, lang.index_word[t]))

    def load_dataset(self):
        targ_lang, inp_lang = self.load_data_from_file()
        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        targ_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer







dl = DataLoader(path_to_file='language-files/por.txt')
input_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = dl.load_dataset()
dl.convert(inp_lang_tokenizer, input_tensor[0])
dl.convert(targ_lang_tokenizer, targ_tensor[0])
