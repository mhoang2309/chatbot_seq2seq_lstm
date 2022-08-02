import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax

class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):

        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


#-----------------------------------------------------------------------
# Attention
#-----------------------------------------------------------------------
class Attention(tf.keras.layers.Layer):
    ''' Class the calculates score based on the scoring_function using Bahdanu attention mechanism. '''
    
    def __init__(self, att_units, scoring_function='concat'):
        super().__init__()
        self.scoring_function = scoring_function
        self.att_units = att_units

        # Initializing for 3 kind of losses
        if self.scoring_function=='dot':
            # Intialize variables needed for Dot score function here
            self.dot = tf.keras.layers.Dot(axes=[2,2])
        if scoring_function == 'general':
            # Intialize variables needed for General score function here
            self.wa = Dense(self.att_units)
        elif scoring_function == 'concat':
            # Intialize variables needed for Concat score function here
            self.wa = Dense(self.att_units, activation='tanh')
            self.va = Dense(1)
  
  
    def call(self,decoder_hidden_state,encoder_output, enc_mask):
        ''' Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs. '''
        
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=1)

        # mask from encoder
        enc_mask = tf.expand_dims(enc_mask, axis=-1)
        
        # score shape: (batch_size, input_length, 1)
        if self.scoring_function == 'dot':
            # Implementing Dot score function
            score = self.dot([encoder_output, decoder_hidden_state])
        elif self.scoring_function == 'general':
            # Implementing General score function here            
            score = tf.keras.layers.Dot(axes=[2, 2])([self.wa(encoder_output), decoder_hidden_state])
        elif self.scoring_function == 'concat':
            # Implementing General score function here
            decoder_output = tf.tile(decoder_hidden_state, [1, encoder_output.shape[1], 1])
            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))
            
        score = score + (tf.cast(tf.math.equal(enc_mask, False), score.dtype)*-1e9)
        
        # shape: (batch_size, input_length, 1)
        attention_weights = Softmax(axis=1)(score)
        enc_mask = tf.cast(enc_mask, attention_weights.dtype)
        
        # masking attention weights
        attention_weights = attention_weights * enc_mask

        context_vector = attention_weights * encoder_output
        # shape = (batch_size, dec lstm units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
        