from .encoder import Encoder
from .decoder import Decoder
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa
from .custom_lossfunction import custom_lossfunction
import time
import os
import math
from load_data.Load_Data import DataSet
from load_data.Data_Loader import Data_Processing

class Seq2Seq_LSTM(Model):
    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length, batch_size):
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.batch_size = batch_size

        self.encoder = Encoder(self.inp_vocab_size+1, self.embedding_size, self.lstm_size, self.input_length, self.batch_size)
        self.decoder = Decoder(self.inp_vocab_size+1, self.embedding_size, 2*self.lstm_size, self.input_length, self.batch_size)

        adam_optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tfa.optimizers.SWA(adam_optimizer)
        self.writer = tf.summary.create_file_writer('logs')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
   
    def predict(self, sentence, dataset:DataSet, data_processing:Data_Processing, path_mode=''):
        result = ''
        checkpoint_dir = path_mode + "model_seq2seq"
        sentence= dataset.clear_data(sentence)
        input_sentences_questions = data_processing.word_processing(sentence)
        input_sentence = [data_processing.word2id[text] for text in input_sentences_questions]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([input_sentence], data_processing.LSTM_UNITS, 
                                                                                dtype='int32',
                                                                                padding=data_processing.padding,
                                                                                truncating=data_processing.padding,
                                                                                value=data_processing.value_padding)

        inputs = tf.convert_to_tensor(inputs)
        
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        enc_hidden = self.encoder.initialize_states()
        enc_out, enc_hidden, enc_state, enc_mask = self.encoder(inputs, enc_hidden)

        dec_hidden = enc_hidden
        dec_state = enc_state
        dec_input = tf.expand_dims([data_processing.word2id["<start>"]] * 1, 1)
        temp = []
        for i in range(data_processing.LSTM_UNITS):
            predictions, dec_hidden, dec_state, _, _  = self.decoder(dec_input, enc_out, dec_hidden, dec_state, enc_mask)
            predicted_id = tf.argmax(predictions[0]).numpy()
            if data_processing.id2word[predicted_id] == '<end>':
                break 
            elif data_processing.id2word[predicted_id] == '<pad>' or data_processing.id2word[predicted_id] == '<start>':
                continue
            else:
                result += str(data_processing.id2word[predicted_id]) + ' '
                temp.append(data_processing.id2word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, temp

    def train(self, data_processing:Data_Processing, train_epochs, path_save=''):
        self.dataset = data_processing.from_tensor_slices(batch_size=self.batch_size)
        self.steps_per_epoch = len(data_processing.train_encoder_inp) // self.batch_size
        enc_hidden = self.encoder.initialize_states()
        checkpoint_dir = path_save + "model_seq2seq"
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        ckpt = tf.io.gfile.listdir(checkpoint_dir)
        if ckpt:
            print("Re-load pretrained model...")
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        start_time = time.time()
        epoch = 1                                                                                                                                                                               
        
        while epoch<=train_epochs:
            start_time_epoch = time.time()
            total_loss = 0
            step_loading = 50
            batch_len =  len(self.dataset.take(self.steps_per_epoch))
            x = step_loading / batch_len
            temp = True
            for batch, (inp_1, inp_2)  in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.training_step(inp_1, inp_2, enc_hidden, enc_hidden)
                if temp:
                    print(f'epoch: {epoch}\n')
                    temp = False
                total_loss += batch_loss
                i = math.ceil(batch * x)
                print("\033[F"+f'batch: {batch:0{len(str(batch_len))}d}/{batch_len} [{"="*(i-1)}{">"*(math.ceil((i+1)/(i+1)))}{"-"*(step_loading-math.floor(batch * x))}] batch_loss: {batch_loss:.4f}')

            step_time_epoch = (time.time() - start_time_epoch) / self.steps_per_epoch
            step_loss = total_loss / self.steps_per_epoch
            current_steps = self.steps_per_epoch
            epoch_time_total = (time.time() - start_time)
            print(f'Total training steps: {current_steps} | Total time: {epoch_time_total:.4f}\nepoch average time per step: {step_time_epoch:.4f} | average loss per step {step_loss:.4f}')
            if(epoch%10)==0 or epoch==1:
                test = os.listdir(checkpoint_dir+"/")
                for item in test:
                    if item.endswith(".data-00000-of-00001") or item.endswith(".index"):
                        os.remove( os.path.join(checkpoint_dir+"/", item ) )

                self.checkpoint.save(file_prefix=checkpoint_prefix)
                print('save model ...')
            epoch = epoch + 1
            with self.writer.as_default():
                tf.summary.scalar('loss', step_loss, step=epoch)

    @tf.function
    def training_step(self, input, targ, enc_state, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_out, enc_hidden, enc_state, enc_mask = self.encoder(input, enc_state)
            dec_hidden = enc_hidden
            dec_state = enc_state
            dec_input = tf.expand_dims([0] * self.batch_size, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, dec_state, _, _  = self.decoder(dec_input, enc_out, dec_hidden, dec_state, enc_mask)
                loss += custom_lossfunction(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)
        step_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return step_loss