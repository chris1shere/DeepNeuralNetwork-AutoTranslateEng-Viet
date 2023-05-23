# Deep Neural Network Machine Learning Auto-Translation: English to Vietnamese
## Overview
This project is aimed at developing a Deep Neural Network for Machine Learning Auto-Translation from English to Vietnamese. The main motivation behind this project is a fascination with the concept of machine understanding and the power of language translation, inspired by experiences with early chatbots. This system is designed to leverage the Sequence-to-Sequence (seq2seq) model and an attention mechanism to make effective and efficient translations.
## Architecture
The architecture of the model is based on the seq2seq model, including an encoder and a decoder.

- Encoder: The input English sentence is processed by the encoder to create a context vector that represents the sentence. This is achieved through several layers of LSTM (Long Short-Term Memory) cells.

- Decoder: The context vector from the encoder is used by the decoder to generate the translated sentence. The decoder also consists of LSTM cells and is also aided by the attention mechanism.

- Seq2Seq Model: The Seq2Seq model combines the encoder and the decoder. The model reads the input sentence word by word and produces the translated sentence.

- Attention Mechanism: The attention mechanism allows the model to focus on different parts of the input sequence while generating each word of the output sequence, thereby improving the accuracy of the model, especially for long sequences.
