import tensorflow as tf
from load_data import DataLoader
from sklearn.model_selection import train_test_split
import time
import os



class Seq2Seq:
    def __init__(self, batch_size):
        self.dl = DataLoader(path_to_file='language-files/por.txt')
        self.batch_size = batch_size
        self.input_tensor, self.targ_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.dl.load_dataset()
        self.input_tensor_train, self.input_tensor_val, self.targ_tensor_train, self.targ_tensor_val = train_test_split(self.input_tensor, self.targ_tensor, test_size=0.2)
        self.steps_per_epoch = len(self.input_tensor_train//self.batch_size)
        print(f"vocab_inp_size: {len(self.inp_lang_tokenizer.word_index)+1}")
        print(f"vocab_targ_size: {len(self.targ_lang_tokenizer.word_index)+1}")
        # Encoder and Decoder
        self.encoder = Encoder(vocab_size=len(self.inp_lang_tokenizer.word_index)+1, embedding_dim=256, enc_units=1024, batch_sz=self.batch_size)
        self.decoder = Decoder(vocab_size=len(self.targ_lang_tokenizer.word_index)+1, embedding_dim=256, dec_units=1024, batch_sz=self.batch_size)
        # Optimizer and Loss Function
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        # Checkpoints
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

    def create_tf_dataset(self):
        """ Creates dataset and stores in self.dataset member variable """
        BUFFER_SIZE = len(self.input_tensor_train)
        steps_per_epoch = len(self.input_tensor_train)//self.batch_size
        embedding_dim = 256
        units = 1024
        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.targ_tensor_train)).shuffle(BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)


    def loss_function(self, real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real,0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0
        BATCH_SIZE = self.batch_size

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:,t], predictions, self.loss_object)

                dec_input = tf.expand_dims(targ[:,t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self):
        EPOCHS = 10
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}")

                if batch % 100 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}")

            if (epoch +1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print(f"Epoch {epoch+1} Loss {total_loss/self.steps_per_epoch}")
            print(f"Time take for 1 epoch {time.time()-start}")



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
        super(Encoder, self).__init__()
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
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh( self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights




##########
##########
##########

def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    BATCH_SIZE = 64

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:,t], predictions)

            dec_input = tf.expand_dims(targ[:,t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradients(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def train():
    EPOCHS = 10
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}")

        if (epoch +1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Epoch {epoch+1} Loss {total_loss/steps_per_epoch}")
        print(f"Time take for 1 epoch {time.time()-start}")










if __name__ == '__main__':
    BATCH_SIZE=256

    model = Seq2Seq(batch_size=BATCH_SIZE)
    model.create_tf_dataset()
    # example_input_batch, example_target_batch = next(iter(model.dataset))
    # print(example_input_batch.shape, example_target_batch.shape)

    # encoder = Encoder(vocab_size=17875, embedding_dim=50, enc_units=10, batch_sz=BATCH_SIZE)
    # sample_hidden = encoder.initialize_hidden_state()
    # sample_output, sample_hidden = encoder(example_input_batch,sample_hidden)
    # print(sample_output.shape, sample_hidden.shape)
    #
    # attention_layer = BahdanauAttention(10)
    # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    # print(attention_result.shape, attention_weights.shape)
    #
    # decoder = Decoder(vocab_size=11424, embedding_dim=50, dec_units=10, batch_sz=BATCH_SIZE)
    # sample_decoder_output, _, _ = decoder(tf.random.uniform((64,1)), sample_hidden, sample_output)
    # print(sample_decoder_output.shape)


    # # Optimizer and Loss Function
    # optimizer = tf.keras.optimizers.Adam()
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none'
    # )
    # # Checkpoints
    # checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer,
    #                                  encoder=encoder,
    #                                  decoder=decoder)

    model.train()
