import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense
from .BahdanauAttention import Attention

#-----------------------------------------------------------------------
# OneStepDecoder
#-----------------------------------------------------------------------
class Decoder(Model):
    
    def __init__(self,vocab_size, embedding_dim, lstm_units, input_length, batch_size):
        super().__init__()
        # Initialize decoder embedding layer, LSTM and any other objects needed
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.lstm_units = lstm_units
        self.batch_size = batch_size

        self.embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.input_length, mask_zero=True)

        self.lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True)
        self.fc = Dense(self.vocab_size)

        self.attention = Attention(self.lstm_units)


    def __call__(self, input_to_decoder, encoder_output, state_h, state_c, enc_mask):
        ''' Calling this function by passing decoder input for a single timestep, encoder output and encoder final states '''
        
        # shape: (batchsize, input_length, embedding dim)
        embedded_input = self.embedding(input_to_decoder)
        # shape: (batch_size, dec lstm units)
        context_vector, attention_weights = self.attention(state_h, encoder_output, enc_mask)
        # (batch_size, 1, dec lstm units)
        decoder_input = tf.concat([tf.expand_dims(context_vector, 1), embedded_input], axis=-1)
        # output shape: (batch size, input length, lstm units), state shape: (batch size, lstm units)
        decoder_output, dec_state_h, dec_state_c = self.lstm(decoder_input, initial_state=[state_h, state_c])
        # (batch_size, lstm units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        # (batch size, vocab size)
        output = self.fc(decoder_output)

        return output, dec_state_h, dec_state_c, attention_weights, context_vector
    
    def initialize_states(self):
      return (tf.zeros([self.batch_size, 2*self.lstm_units]), tf.zeros([self.batch_size, 2*self.lstm_units]))
