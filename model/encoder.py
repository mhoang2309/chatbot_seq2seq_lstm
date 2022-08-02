import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Concatenate

#-----------------------------------------------------------------------
# Encoder
#-----------------------------------------------------------------------
class Encoder(Model):

    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length, batch_size):
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size
        self.lstm_units = lstm_size
        self.input_length = input_length
        self.batch_size = batch_size
        self.enc_output = self.enc_state_h = self.enc_state_c = 0

        self.embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.input_length, mask_zero=True)

        self.lstm = LSTM(units=self.lstm_units, return_state=True, return_sequences=True)

        self.bidirectional = Bidirectional(self.lstm)

    def __call__(self, input_sequence, states):

        embedded_input = self.embedding(input_sequence)
        
        # mask for padding
        mask = self.embedding.compute_mask(input_sequence)

        self.enc_out, enc_fw_state_h, enc_bw_state_h, enc_fw_state_c, enc_bw_state_c = self.bidirectional(embedded_input, mask=mask)

        self.enc_state_h = Concatenate()([enc_fw_state_h, enc_bw_state_h])  # enc_state_h and c shape: (batch_size, 2*lstm_size)
        self.enc_state_c = Concatenate()([enc_fw_state_c, enc_bw_state_c])
        
        return self.enc_out, self.enc_state_h, self.enc_state_c, mask

    def initialize_states(self):
      return (tf.zeros([self.batch_size, 2*self.lstm_units]), tf.zeros([self.batch_size, 2*self.lstm_units]))
      