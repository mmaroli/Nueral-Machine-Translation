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

        Preprocessor(path_to_file=self.path_to_file)
        
        return self.path_to_file

    def load_data_from_file(self):
        if not self.path_to_file:
            raise ValueError("Path to file cannot be none.")

        Preprocessor(path_to_file=self.path_to_file)

        return self.path_to_file
