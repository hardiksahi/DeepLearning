#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:58:52 2019

@author: hardiksahi
"""
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import sys
import ast
import csv

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class SentenceGenerator:
    def __init__(self,path_list):
        self.path_list = path_list # /Users/hardiksahi/LastTerm/TextAnalytics/Assignments/Data/textstyletransferdata/sentiment
    
    def __iter__(self):
        for path in self.path_list:
            with open(path,"r") as f:
                reader = csv.reader(f,delimiter="\n")
                for line in reader:
                    #print(line[0])
                    yield ast.literal_eval(line[0])
                

if __name__ == "__main__":
    pathList = []
    input_path_train_pos_path = sys.argv[1]
    pathList.append(input_path_train_pos_path)
    
    input_path_train_neg_path = sys.argv[2]
    pathList.append(input_path_train_neg_path)
    
    input_path_val_pos_path = sys.argv[3]
    pathList.append(input_path_val_pos_path)
    
    input_path_val_neg_path = sys.argv[4]
    pathList.append(input_path_val_neg_path)
    
    input_path_test_pos_path = sys.argv[5]
    pathList.append(input_path_test_pos_path)
    
    input_path_test_neg_path = sys.argv[6]
    pathList.append(input_path_test_neg_path)
    
    

    sentences = SentenceGenerator(pathList)

    epoch_logger = EpochLogger()

    model = Word2Vec(sentences, min_count = 1, iter = 100, seed = 42, callbacks = [epoch_logger], size=300)
    model.save('/tmp/amazon_trained_word2vec')


    # Loading save model
    loaded_model = Word2Vec.load('/tmp/amazon_trained_word2vec')

    print("\n Most similar to good.. \n")
    print(loaded_model.wv.most_similar(positive='good',topn=20))

    print("\n Most similar to bad.. \n")
    print(loaded_model.wv.most_similar(positive='bad', topn=20))


        
        