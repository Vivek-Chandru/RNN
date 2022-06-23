# Text generation using Recurring Neural Networks

A deep learning model to generate text based on a dictionary generated from text inputs , the text inputs are also used to teach the model context and continuity within a specified window.

The model is a built using tensorflow keras sequential module. It consists of **GRUs (Gated recurrent units)** in the two hidden layers. 

Overfitting of data is tackled with **Early stopping** ,albeit for a very small patience value.

The code here takes two text inputs (Parfum_1.txt and Parfum_2.txt)

>**Note:** the model is built in Tensorflow 2.3.1





