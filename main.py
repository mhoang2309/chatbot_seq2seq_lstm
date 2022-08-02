from tkinter.messagebox import NO
from load_data.Data_Loader import Data_Processing
from load_data.Load_Data import DataSet
from model.seq2seq_lstm import Seq2Seq_LSTM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_data = '/home/minerva/Desktop/model_lstm/data.txt'
    lstm_units = 64
    max_nb_words = 10000
    return_val = False
    batch_size = 64

    parser.add_argument("--path_data", default=path_data)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--path_save", default='')
    parser.add_argument("--path_model", default='')
    parser.add_argument("--stopwords", default=None)
    parser.add_argument("--fasttext_model", default=None)
    args = parser.parse_args()

    dataset = DataSet(path=path_data)
    data_processing = Data_Processing(lstm_units=lstm_units, max_nb_words=max_nb_words, stopwords=args.stopwords, fasttext_model=args.fasttext_model)
    data_processing(dataset(), validation=return_val, return_value=False)

    # data_train, embedding_matrix, QSN_WORD2ID_SIZE = data_processing.from_tensor_slices(batch_size=batch_size)

    model = Seq2Seq_LSTM(len(data_processing.word2id), data_processing.embedding_dim, lstm_units,\
                             data_processing.max_words, batch_size)
    if args.train:
        model.train(data_processing=data_processing, train_epochs=args.epochs, path_save=args.path_save)
    elif args.predict:

        print("=================>>>>>>>>>><<<<<<<<<==================")
        user_input = ''
        while (user_input != 'q'):
            print("--------------------------------------------------------------------")
            user_input = input("Please write something: ")
            print("--------------------------------------------------------------------")
            result, temp = model.predict(user_input, dataset=dataset, data_processing=data_processing, path_mode=args.path_model)
            print("Bot: ", result)
            print("Len: ", len(temp))
    else:
        print("done")

    
