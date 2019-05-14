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

class Encoder(tf.keras.Model):
    """
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

        # sample input
        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
        print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
        print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)




if __name__ == '__main__':
    model = Seq2Seq(batch_size=64)
    model.create_tf_dataset()
    example_input_batch, example_target_batch = next(iter(model.dataset))
    print(example_input_batch.shape, example_target_batch.shape)
