#!/usr/bin/env python
# coding: utf-8

# # Projekt 3: Rekurrente neuronale Netze
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow import keras
from math import ceil



with open('parfum_1.txt',encoding='utf-8') as f:
    parfum_1_text = f.read()
with open('parfum_2.txt',encoding='utf-8') as f:
    parfum_2_text = f.read()



tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

parfum_12 = [parfum_1_text,parfum_2_text]
tokenizer.fit_on_texts(parfum_12)

max_id = len(tokenizer.word_index)

parfum_1_encoded,parfum_2_encoded = tokenizer.texts_to_sequences(parfum_12)
parfum_1_encoded = np.array(parfum_1_encoded, dtype='int64')-1
parfum_2_encoded = np.array(parfum_2_encoded,dtype='int64')-1

[parfum_1_decoded] = tokenizer.sequences_to_texts([parfum_1_encoded+1])

parfum_1_dataset = tf.data.Dataset.from_tensor_slices(parfum_1_encoded)
parfum_2_dataset = tf.data.Dataset.from_tensor_slices(parfum_2_encoded)

T=100
window_length=101


parfum_1_dataset=parfum_1_dataset.window(size=window_length,shift=1,drop_remainder=True)
parfum_2_dataset=parfum_2_dataset.window(size=window_length,shift=1,drop_remainder=True)


map_func = lambda window: window.batch(window_length)
parfum_1_dataset = parfum_1_dataset.flat_map(map_func)
parfum_2_dataset = parfum_2_dataset.flat_map(map_func)


parfum_dataset = parfum_1_dataset.concatenate(parfum_2_dataset)


tf.random.set_seed(0)
batch_size=32
parfum_dataset = parfum_dataset.repeat().shuffle(buffer_size=10000).batch(batch_size)


a = parfum_dataset.map(lambda t: (t[:,0:100],t[:,1:101]))


b = a.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch,depth=max_id),Y_batch))


c = b.prefetch(buffer_size=1)


l1=len(parfum_1_encoded)
l2=len(parfum_2_encoded)
l12=l1+l2-2*(window_length-1)

steps_per_epoch = ceil(l12/32)


model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=[None, max_id]),
        tf.keras.layers.GRU(units=128, return_sequences=True),
        tf.keras.layers.GRU(units=128, return_sequences=True),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=max_id, activation='softmax'))
    ])


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')


callback=keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
h= model.fit(
    c,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    callbacks=[callback]
)

model.save('parfum_model.h5')
