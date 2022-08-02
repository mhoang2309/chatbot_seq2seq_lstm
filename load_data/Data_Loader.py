import random
from underthesea import word_tokenize
from gensim.utils import simple_preprocess
import fasttext 
import numpy as np
import tensorflow as tf

class Vocab():
    def __init__(self, chars=None):
        if chars is None:
            self.chars = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ\
                        bBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHi\
                        IìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗ\
                        ỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯ\
                        ừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 '
        else:
            self.chars = chars

        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        
        self.c2i = {c:i+4 for i, c in enumerate(self.chars)}
        self.i2c = {i+4:c for i, c in enumerate(self.chars)}

        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars

class Fasttext_Model():
    def __init__(self, path):
        self.words_not_found = []
        print("Loadding fasttext model ...")
        self.embeddings_index =  fasttext.load_model(path)
    
    def __call__(self, word2id, embedding_dim, nb_words):
        self.embedding_dim = embedding_dim
        # embedding matrix
        print('preparing embedding matrix...')
        # Với mỗi từ trong câu, lưu lại word vector phụ vụ để huấn luyện mô hình
        self.embedding_matrix = np.zeros((nb_words, self.embedding_dim))
        for word, i in word2id.items():
            if i >= nb_words:
                continue
            embedding_vector = self.embeddings_index[word]
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                self.words_not_found.append(word)
        print('number of null word embeddings: %d' % np.sum(np.sum(self.embedding_matrix, axis=1) == 0))
        return self.embedding_matrix

class Data_Processing():
    def __init__(self, lstm_units, max_nb_words, stopwords=None, fasttext_model=None, embedding_dim=300):
        self.stopwords = stopwords
        self.fasttext_model = fasttext_model
        self.LSTM_UNITS = lstm_units
        self.MAX_NB_WORDS = max_nb_words
        self.embedding_dim =embedding_dim

    def __call__(self, data, padding='pre', validation=False, return_value= True):
        self.validation =validation
        self.padding = padding
        input_sentences_questions = [self.word_processing(str(sentence)) for sentence in data[0]]
        input_sentences_answers = [self.word_processing(str(sentence)) for sentence in data[1]]

        input_sentences = input_sentences_questions + input_sentences_answers

        self._word2id_(input_sentences)   

        questions = [[self.word2id[word] for word in sentence] for sentence in input_sentences_questions]
        answers = [[self.word2id[word] for word in sentence] for sentence in input_sentences_answers]

        nb_words = min(self.MAX_NB_WORDS, len(self.word2id))
        if self.fasttext_model is not None:
            self.embedding_matrix = self.fasttext_model(self.word2id, self.embedding_dim, nb_words)
        else:
            self.embedding_matrix = None
        
        self.value_padding = self.word2id["<pad>"]

        if not self.validation:
            self.train_encoder_inp = tf.keras.preprocessing.sequence.pad_sequences(questions, self.LSTM_UNITS, 
                                                                                dtype='int32',
                                                                                padding=padding,
                                                                                truncating=padding,
                                                                                value=self.value_padding)

            self.train_decoder_inp = tf.keras.preprocessing.sequence.pad_sequences(answers, self.LSTM_UNITS, 
                                                                                    dtype='int32',
                                                                                    padding=padding,
                                                                                    truncating=padding,
                                                                                    value=self.value_padding)
            if return_value:
                return [self.train_encoder_inp, self.train_decoder_inp], self.embedding_matrix, len(self.word2id)

        else:
            if type(validation) is float:
                train, val = self.data_division(questions, answers, validation)
            else:
                train, val = self.data_division(questions, answers)

            self.train_encoder_inp = tf.keras.preprocessing.sequence.pad_sequences(train[0], self.LSTM_UNITS, 
                                                                                dtype='int32',
                                                                                padding=padding,
                                                                                truncating=padding,
                                                                                value=self.value_padding)

            self.train_decoder_inp = tf.keras.preprocessing.sequence.pad_sequences(train[1], self.LSTM_UNITS, 
                                                                                    dtype='int32',
                                                                                    padding=padding,
                                                                                    truncating=padding,
                                                                                    value=self.value_padding)
            
            self.val_encoder_inp = tf.keras.preprocessing.sequence.pad_sequences(val[0], self.LSTM_UNITS, 
                                                                                    dtype='int32',
                                                                                    padding=padding,
                                                                                    truncating=padding,
                                                                                    value=self.value_padding)

            self.val_decoder_inp = tf.keras.preprocessing.sequence.pad_sequences(val[1], self.LSTM_UNITS, 
                                                                                    dtype='int32',
                                                                                    padding=padding,
                                                                                    truncating=padding,
                                                                                    value=self.value_padding)
            if return_value:
                return [self.train_encoder_inp, self.train_decoder_inp], [self.val_encoder_inp, self.val_decoder_inp], self.embedding_matrix, len(self.word2id)

    def from_tensor_slices(self, batch_size=8, inp=None, out=None):
        if inp is None or out is None:
            data_train = tf.data.Dataset.from_tensor_slices((self.train_encoder_inp,self.train_decoder_inp)).shuffle(len(self.train_encoder_inp))
            data_train = data_train.batch(batch_size, drop_remainder=True)
            if self.validation:
                data_val = tf.data.Dataset.from_tensor_slices((self.val_encoder_inp,self.val_decoder_inp)).shuffle(len(self.val_encoder_inp))
                data_val = data_val.batch(batch_size, drop_remainder=True)
                return data_train, data_val
            else:
                return data_train

    def _word2id_(self, input_sentences):
        self.word2id = dict({'':0})
        self.max_words = 0 # maximum number of words in a sentence
        for sentence in input_sentences:
            for word in sentence:
                # Add words to word2id dict if not exist
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
            # If length of the sentence is greater than max_words, update max_words
            if len(sentence) > self.max_words:
                self.max_words = len(sentence)
        self.word2id["<pad>"] = len(self.word2id)   
        self.word2id["<start>"] = len(self.word2id)   
        self.word2id["<end>"] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def create_stopwords(self, path=None, original_stopwords=None):
        if path is None:
            with open(self.stopword, encoding="utf-8") as words:
                if original_stopwords is None:
                    return [w[:len(w) - 1] for w in words]
                else:
                    return [w[:len(w) - 1] for w in words] + original_stopwords
        else:
            with open(self.stopword, encoding="utf-8") as words:
                if original_stopwords is None:
                    return [w[:len(w) - 1] for w in words]
                else:
                    return [w[:len(w) - 1] for w in words] + original_stopwords

    def data_division(self, questions, answers, percent=0.2):
        val_encoder_inp = []
        val_decoder_inp = []
        train_encoder_inp =[]
        train_decoder_inp =[]
        list_items = random.choices(range(len(questions)), k=int(len(questions)*percent))
        for i in range(len(questions)):
            if i in list_items:
                val_encoder_inp.append(questions[i])
                val_decoder_inp.append(answers[i])
            else:
                train_encoder_inp.append(questions[i])
                train_decoder_inp.append(answers[i])
        return [train_encoder_inp, train_decoder_inp], [val_encoder_inp, val_decoder_inp]

    # sử dụng cơ bản 1 chút và tách từ
    def word_processing(self, sentence):
        if self.stopwords is not None:
            stopwords = self.create_stopwords()
            sentence = " ".join(simple_preprocess(sentence)) 
            #sentence = " ".join(sentence.split())
            #print(sentence)
            sentence = [word for word in word_tokenize(sentence.lower(), format="text").split() if word not in stopwords]
        else:
            sentence = [word for word in word_tokenize(sentence.lower(), format="text").split()]

        return [word for word in sentence if word != ""]