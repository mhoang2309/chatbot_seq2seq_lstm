<div align="center"><br />
<img src="http://minerva.vn/wp-content/uploads/2020/06/logo.png" alt="Logo" style="-webkit-user-select: none;margin: auto;">

<h1>Research and Development</h1>
<h2>Model Chat Bot Sequence-to-Sequence</h2>
<h3 align="center">Long Short Term Memory (LSTM)</h3>
</div>

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://www.python.org/downloads/release/python-3913/)
[![Python](https://img.shields.io/badge/Tensorflow-2.6.0-critical)](https://tensorflow.org/) 

# Seq2Seq
- Sequence To Sequence model  giới thiệu trong [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078) sau đó trở thành model cho Dialogue System (Chatbot) và Machine Translation. Nó bao gồm hai RNNs (Recurrent Neural Network): Một Encoder và một Decoder. Encoder lấy một chuỗi (câu) làm input và xử lý các ký tự (các từ) trong mỗi lần. Nhiệm vụ của nó là chuyển một từ thành các vector có cùng kích cỡ, chỉ bao gồm các thông tin cần thiết và bỏ đi các thông tin cần thiết (có vẻ việc này được thực hiện tự động trong quá trình train).


- Mỗi hidden state (Một LSTM - một hình vuông trên ảnh đầu vào sẽ là một từ và một hidden state, đầu ra là hidden state và truyền qua LSTM cell tiếp theo) ảnh hưởng tới hidden state tiếp theo và hidden state cuối cùng có thể được nhìn thấy như là tổng kết của chuỗi. State này được gọi là context hoặc thought vector. Từ context (nghĩa là hidden state cuối cùng), the decoder sẽ tạo ra sequence (là câu trả lời được tạo ra), mỗi một từ một lần. Ở mỗi bước decoder sẽ bị ảnh hưởng bởi từ được tạo ra ở bước trước.


- Có một vài thử thách khi sử dụng model này. Cái mà phiền nhất là model không thể tự xử lý được chiều dài của câu văn. Nó là một điều phiền phức bởi hầu hết cho các ứng dụng seq2seq. Decoder dùng softmax cho toàn bộ từ vựng, lặp lại với mỗi từ của output. Điều này sẽ làm chậm quá trình train, mặc dù nếu phần cứng của bạn có khả năng xử lý được nó. Việc biểu diễn các từ là rất quan trọng. Bạn biểu diễn từ như thế nào? Sử dụng one-hot vector nghĩa là phải xử lý với vector lớn và one-hot vector không có ý nghĩa cho từ. Chúng ta sẽ phải đối mặt với các thử thách trên, từng cái một.
<div align="center">
  <img src="README_IMG/seq2seq_3.png?raw=true" alt="Mo hinh Seq2Seq"/>
  <h4>Mô hình Seq2Seq LSTM</h4>
</div>

- Trong bài này chúng ta sử dụng một mô hình nâng cấp hơn. Mô hình này sử dụng một lớp [Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional) cho phần encoder.
<div align="center">
  <img src="README_IMG/Deep-Dive-into-Bidirectional-LSTM-i2tutorials.jpg?raw=true" alt="Bidirectional LSTM"/>
  <h4>Bidirectional LSTM</h4>
</div>

# Xử lý data
- Bộ data cần có phần câu hỏi và câu trả lời có dạng:

|questions|answers|
| ------ | ------ |
|trưa nay cậu muốn đi ăn không ?|tất nhiên rồi, nhưng mình sẽ ăn gì ?|
|bạn nghĩ sao khi chúng ta đi ăn gà ?|mình nghĩ đó là ý kiến hay|
|chúng ta sẽ đi ăn ở đâu ?|chúng ta sẽ ăn ở KFC|
|...|...|

- Data được lưu dưới dạng *.txt  ( [data.txt](data/data.txt) )
<div>
  <img src="README_IMG/Screenshot from 2022-08-01 15-19-39.png?raw=true" alt="Data"/>
</div>

# Cài đặt

- Ubuntu
```ssh
git clone https://git.minerva.vn/hoangnm/chatbot-seq2seq-lstm.git
cd chatbot-seq2seq-lstm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Train model

#### Train với 500 epochs

```ssh
python main.py --train --path_data "data/data.txt" --epochs 500
```
#### Train với 500 epochs, bộ stopwords và fasttext_model
```ssh
python main.py --train --path_data "data/data.txt" --epochs 500 --stopwords "load_data/vietnamese-stopwords-dash.txt" --fasttext_model "cc.vi.300.bin"
```
#### Train với [stopwords](load_data/vietnamese-stopwords-dash.txt)
```ssh
python main.py --train --path_data "data/data.txt" --epochs 500 --stopwords "load_data/vietnamese-stopwords-dash.txt"
```
#### Train với fasttext_model [downloads cc.vi.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz)
```ssh
python main.py --train --path_data "data/data.txt" --epochs 500 --fasttext_model "cc.vi.300.bin"
```


<div>
  <img src="README_IMG/Screenshot from 2022-08-01 16-29-05.png?raw=true" alt="Training"/>
</div>

# Predict

```ssh
python main.py --predict
```
- Kết quả predict
<div>
  <img src="README_IMG/Screenshot from 2022-08-01 16-28-30.png?raw=true" alt="Predict"/>
</div>

### LƯU Ý

- Phiên bản CUDA trên cài trên GPU. Phải tương thích với version tensorflow
- Phiên bản Python.
- Model Seq2Seq với 500 epochs được lưu trong thư mục [model_seq2seq](model_seq2seq/)
### ĐÓNG GÓP

### LIÊN HỆ - THÔNG TIN

```BibTeX
{
  project      = {CHATBOT - Sequence-to-Sequence},
  author       = {Team R&D, Nguyễn Mạnh Hoàng},
  time         = {08/2022},
  url          = {https://git.minerva.vn/hoangnm/chatbot-seq2seq-lstm.git},
  organization = {Minerva Technology Solutions}
}
```
