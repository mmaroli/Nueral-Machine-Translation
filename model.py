import tensorflow as tf
from load_data import DataLoader
from sklearn.model_selection import train_test_split



class Seq2Seq:
    def __init__(self):
        self.dl = DataLoader(path_to_file='language-files/por.txt')
        self.input_tensor, self.targ_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.dl.load_dataset()
        
