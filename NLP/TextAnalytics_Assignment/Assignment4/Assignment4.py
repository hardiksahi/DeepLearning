#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:07:49 2019

@author: hardiksahi
"""

import csv
import ast
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, History
from keras import regularizers
import numpy as np
from gensim.models import Word2Vec
import pandas as pd

# Set random seed
np.random.seed(0)

train_pos_path = 'train_pos_path'    
train_neg_path = 'train_neg_path'
val_pos_path = 'val_pos_path'
val_neg_path = 'val_neg_path'
test_pos_path = 'test_pos_path'
test_neg_path = 'test_neg_path'

vocab_size = 20000
sequence_size = 19
embedding_dim = 300

def getTextAndYDict(path_dict):
    return_dict = {}
    
    for key, path in path_dict.items():
        text_list = []
        y_list = []
        
        if key == train_pos_path or key == val_pos_path or key == test_pos_path:
            y_output = 1
        elif key == train_neg_path or key == val_neg_path or key == test_neg_path:
            y_output = 0
    
        with open(path,"r") as f:
            reader = csv.reader(f,delimiter="\n")
            for line in reader:
                text_list.append(' '.join(ast.literal_eval(line[0])))
                y_list.append(y_output)
        return_dict[key] = (text_list,y_list)
        
    return return_dict
 
def createEmbeddingMatrix(emb_model_path, word_index):
    word_emb_dict = {} #word: word2vec embedding [has all words for which we have word2vec embedding]
    
    word2vec_model = Word2Vec.load(emb_model_path)
    vocab = word2vec_model.wv.vocab
    for word in vocab.keys():
        word_emb_dict[word] = np.asarray(word2vec_model.wv[word], dtype='float32')
        
    #print("word_emb_dict len", len(word_emb_dict))
    
    word_emb_matrix = np.zeros((len(word_index)+1, embedding_dim))
    
    for word, index in word_index.items():
        emb = word_emb_dict.get(word)
        if emb is not None:
            word_emb_matrix[index] = emb
    
    return word_emb_matrix # This is a matrix where row numer corresponds to word index and vector is embedding.....
 
def createNetwork(optimizer, activation , l2, dropout, embedding_matrix, word_index):
    model = models.Sequential()
    
    embedding_layer = layers.Embedding(len(word_index)+1, embedding_dim, weights = [embedding_matrix], input_length = sequence_size, trainable = False, name='word_embedding_1')
    model.add(embedding_layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, name='dense_1', kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Dropout(dropout, name='dropout_1'))
    model.add(layers.Activation(activation=activation, name='activation_1'))
    model.add(layers.Dense(2, activation='softmax', name="output_layer"))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model          
    
if __name__ == "__main__":
    
    path_dict = {}
    
    input_path_train_pos_path = 'train_pos.csv'
    path_dict[train_pos_path] = input_path_train_pos_path
    
    
    input_path_train_neg_path = 'train_neg.csv'
    path_dict[train_neg_path] = input_path_train_neg_path
    
    
    input_path_val_pos_path = 'val_pos.csv'
    path_dict[val_pos_path] = input_path_val_pos_path
    
    
    input_path_val_neg_path = 'val_neg.csv'
    path_dict[val_neg_path] = input_path_val_neg_path
    
    
    input_path_test_pos_path = 'test_pos.csv'
    path_dict[test_pos_path] = input_path_test_pos_path
    
    
    input_path_test_neg_path = 'test_neg.csv'
    path_dict[test_neg_path] = input_path_test_neg_path
    
    text_y_dict = getTextAndYDict(path_dict)
    print("Created text_y_dict")
    
    combined_text_list = []
    
    for tup in text_y_dict.values():
        combined_text_list = combined_text_list+ tup[0]
    
    tokenizer = Tokenizer(num_words=vocab_size+1) # Toeknize input list of reviews...
    tokenizer.fit_on_texts(combined_text_list)
    
  
    train_text, train_y = text_y_dict[train_pos_path][0]+text_y_dict[train_neg_path][0], text_y_dict[train_pos_path][1]+text_y_dict[train_neg_path][1]
    val_text, val_y = text_y_dict[val_pos_path][0]+text_y_dict[val_neg_path][0], text_y_dict[val_pos_path][1]+text_y_dict[val_neg_path][1]
    test_text, test_y = text_y_dict[test_pos_path][0]+text_y_dict[test_neg_path][0], text_y_dict[test_pos_path][1]+text_y_dict[test_neg_path][1]
    
    
    # Create integer sequences 
    # 1. Train
    sequences_train = tokenizer.texts_to_sequences(train_text) # convert it into sequence of numbers [1 2 3] etc...
    data_train = pad_sequences(sequences_train, maxlen=sequence_size)
    y_cat_train = keras.utils.to_categorical(np.asarray(train_y))
    
    # 2. Val
    sequences_val = tokenizer.texts_to_sequences(val_text) # convert it into sequence of numbers [1 2 3] etc...
    data_val = pad_sequences(sequences_val, maxlen=sequence_size)
    y_cat_val = keras.utils.to_categorical(np.asarray(val_y))
    
    # 3. Test
    sequences_test = tokenizer.texts_to_sequences(test_text) # convert it into sequence of numbers [1 2 3] etc...
    data_test = pad_sequences(sequences_test, maxlen=sequence_size)
    y_cat_test = keras.utils.to_categorical(np.asarray(test_y))
    
    
    word_index = tokenizer.word_index # (word, index)
    
    word2vc_model_path = 'amazon_trained_word2vec.model'
    embedding_matrix = createEmbeddingMatrix(word2vc_model_path, word_index)
    
    # Shuffle train data..
    print("Shuffling data:")
    indices = np.arange(data_train.shape[0])
    np.random.shuffle(indices)
    data_train = data_train[indices]
    y_cat_train = y_cat_train[indices]
    
    indices = np.arange(data_val.shape[0])
    np.random.shuffle(indices)
    data_val = data_val[indices]
    y_cat_val = y_cat_val[indices]
    
    indices = np.arange(data_test.shape[0])
    np.random.shuffle(indices)
    data_test = data_test[indices]
    y_cat_test = y_cat_test[indices]
    
    dropout_prob_list = [0.1,0.2,0.3]
    l2_lambda_list = [0.00001, 0.0001, 0.001]
    activation_list = ['relu','sigmoid','tanh']

    
    best_val_acc_list = []
    best_model_list = []
    
    optimizer = 'adam'
    best_activation_func = None
    val_acc = 0
    val_acc_best = 0
    
    df = pd.DataFrame()
    configuration_list = []
    val_list = []
    
    
    for activation in activation_list:
        model = createNetwork(optimizer, activation, 0, 0,embedding_matrix, word_index)
        history = History()
        callbacks = [history, EarlyStopping(monitor='val_loss', patience=2)]
        history = model.fit(data_train,y_cat_train,epochs=50,batch_size=512,validation_data=(data_val, y_cat_val),verbose=1, callbacks=callbacks)
        history_index = len(history.history['loss'])-2
        val_acc = history.history['val_acc'][history_index-1]
        print("Val set accuracy for activation => ", activation, "(Everything else constant) is ::", val_acc)
        
        config_name = 'activation='+activation
        configuration_list.append(config_name)
        val_list.append(val_acc)
        
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            best_activation_func = activation
    
    best_val_acc_list.append(val_acc_best) 
    best_model_list.append(model)
    print("Best activation funciton is ", best_activation_func)
    
    val_acc = 0
    val_acc_best = 0
    best_l2_lambda = 0
    
    for l2_lambda in l2_lambda_list:
        model = createNetwork(optimizer, best_activation_func, l2_lambda, 0,embedding_matrix, word_index)
        history = History()
        callbacks = [history, EarlyStopping(monitor='val_loss', patience=2)]
        history = model.fit(data_train,y_cat_train,epochs=50,batch_size=512,validation_data=(data_val, y_cat_val),verbose=1, callbacks=callbacks)
        history_index = len(history.history['loss'])-2
        val_acc = history.history['val_acc'][history_index-1]
        print("Val set accuracy for l2_lambda => ", l2_lambda, "(Everything else constant) is ::", val_acc)
        
        config_name = 'activation='+best_activation_func+', l2=' + str(l2_lambda)
        configuration_list.append(config_name)
        val_list.append(val_acc)
        
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            best_l2_lambda = l2_lambda
            #best_model = model
            
    best_val_acc_list.append(val_acc_best) 
    best_model_list.append(model)
    print("Best l2 value", best_l2_lambda)
    
    
    val_acc = 0
    val_acc_best = 0
    best_dropout_prob = 0
    
    for dropout_prob in dropout_prob_list:
        model = createNetwork(optimizer, best_activation_func, best_l2_lambda, dropout_prob,embedding_matrix, word_index)
        history = History()
        callbacks = [history, EarlyStopping(monitor='val_loss', patience=2)]
        history = model.fit(data_train,y_cat_train,epochs=50,batch_size=512,validation_data=(data_val, y_cat_val),verbose=1, callbacks=callbacks)
        history_index = len(history.history['loss'])-2
        val_acc = history.history['val_acc'][history_index-1]
        print("Val set accuracy for dropout_prob => ", dropout_prob, "(Everything else constant) is ::", val_acc)
        
        config_name = 'activation='+best_activation_func+', l2=' + str(best_l2_lambda)+', dropout='+str(dropout_prob)
        configuration_list.append(config_name)
        val_list.append(val_acc)
        
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            best_dropout_prob = dropout_prob
            
    
    best_val_acc_list.append(val_acc_best) 
    best_model_list.append(model)
    print("Best dropout prob value", best_dropout_prob)
    
    df['Configuration'] = configuration_list
    df['Validation accuracy'] = val_list
    
    print("Displaying validation accuracy for all 9 cases:")
    print(df)
    
    
    # Choosing best model from validation set
    best_val_acc_overall = np.argmax(best_val_acc_list)
    best_model = best_model_list[best_val_acc_overall]
    
    loss_test_best, accuracy_test_best = best_model.evaluate(data_test, y_cat_test, verbose=0)
    
    print("Test set accuracy using best model", accuracy_test_best)
    #print("Best combination of paramaters: activation=>", best_activation_func, ", l2_lambda=>", str(best_l2_lambda), " dropout_prob=>", str(best_dropout_prob))
    
    
    
    
    
    
    
    
    
    
    