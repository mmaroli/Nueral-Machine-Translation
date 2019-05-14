import tensorflow as tf
from load_data import DataLoader
from sklearn.model_selection import train_test_split



class Seq2Seq:
    def __init__(self, batch_size):
        self.dl = DataLoader(path_to_file='language-files/por.txt')
        self.batch_size = batch_size
        self.input_tensor, self.targ_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.dl.load_dataset()
        self.input_tensor_train, self.input_tensor_val, self.targ_tensor_train, self.targ_tensor_val = train_test_split(self.input_tensor, self.targ_tensor, test_size=0.2)

    def create_tf_dataset(self):
        """ Creates dataset and stores in self.dataset member variable """
        BUFFER_SIZE = len(self.input_tensor_train)
        steps_per_epoch = len(self.input_tensor_train)//self.batch_size
        embedding_dim = 256
        units = 1024
        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.targ_tensor_train)).shuffle(BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)


if __name__ == '__main__':
    model = Seq2Seq(batch_size=64)
    model.create_tf_dataset()
    example_input_batch, example_target_batch = next(iter(model.dataset))
    print(example_input_batch.shape, example_target_batch.shape)
