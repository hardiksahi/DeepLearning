#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:51:01 2019

@author: hardiksahi
"""

import pandas as pd
import numpy as np
import sys
import ast
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import csv


def createDataInFormat(input_path_pos, input_path_neg):
    
    #Positive csv
    pos_start_list = []
    with open(input_path_pos,"r") as f:
        reader = csv.reader(f,delimiter="\n")
        for line in reader:
            pos_start_list.append(line[0])
    
    pos_list = [ast.literal_eval(reviewStr) for reviewStr in pos_start_list]
    pos_text_list_x = [' '.join(review) for review in pos_list]
    pos_text_list_y = [1]*len(pos_text_list_x)
    
    #Negative csv
    neg_start_list = []
    with open(input_path_neg,"r") as f:
        reader = csv.reader(f,delimiter="\n")
        for line in reader:
            neg_start_list.append(line[0])
    
    neg_list = [ast.literal_eval(reviewStr) for reviewStr in neg_start_list]
    neg_text_list_x = [' '.join(review) for review in neg_list]
    neg_text_list_y = [0]*len(neg_text_list_x)
    
    combined_list_x = pos_text_list_x+neg_text_list_x
    combined_list_y = pos_text_list_y+neg_text_list_y
    
    combined_df = pd.DataFrame()
    combined_df['reviews'] = combined_list_x
    combined_df['labels'] = combined_list_y
    
    combined_df_shuffle = combined_df.sample(frac=1).reset_index(drop=True)
    
    return combined_df_shuffle
    
    
    
if __name__ == "__main__":
    input_path_train_pos_path = sys.argv[1]
    input_path_train_neg_path = sys.argv[2]  
    train_combined_df_shuffle = createDataInFormat(input_path_train_pos_path, input_path_train_neg_path)
    
    input_path_val_pos_path = sys.argv[3]
    input_path_val_neg_path = sys.argv[4]
    val_combined_df_shuffle = createDataInFormat(input_path_val_pos_path, input_path_val_neg_path)
    
    input_path_test_pos_path = sys.argv[5]
    input_path_test_neg_path = sys.argv[6]
    test_combined_df_shuffle = createDataInFormat(input_path_test_pos_path, input_path_test_neg_path)
    
    alpha_list = np.arange(0.1,1.01, 0.1).tolist()
    #print("Shuffled train..")
    
    gram_list = [(1,1),(2,2),(1,2)]
    
    for gram_range in gram_list:
        start_gram = gram_range[0]
        end_gram = gram_range[1]
        print("\n")
        if start_gram == 1 and end_gram == 1:
            print("====Start analysis for unigram features====")
        elif start_gram == 2 and end_gram == 2:
            print("====Start analysis for bigram features====")
        elif start_gram == 1 and end_gram == 2:
            print("====Start analysis for unigram+bigram features====")
        
        #Count vectorizer...
        count_vectorizer = CountVectorizer(ngram_range=(start_gram,end_gram))
        count_vectorizer.fit(train_combined_df_shuffle['reviews'].tolist())
        
        print("Creating vocab..")
        X_train = count_vectorizer.transform(train_combined_df_shuffle['reviews'])
        Y_train = train_combined_df_shuffle['labels']

        X_val = count_vectorizer.transform(val_combined_df_shuffle['reviews'])
        Y_val = val_combined_df_shuffle['labels']
        
        accuracy_list = []
        print("Finding best alpha using Val set..")
        for alpha in alpha_list:
            #Train the model
            clf = MultinomialNB(alpha = alpha)
            clf.fit(X_train, Y_train)
            #Validate the model
            Y_val_pred = clf.predict(X_val)
            #print("Y_val_pred shape", Y_val_pred.shape)
            acc = accuracy_score(Y_val, Y_val_pred)
            #print("acc for: ",alpha,"::" , acc)
            accuracy_list.append(acc)
        
        max_acc_index = np.argmax(accuracy_list)
        max_acc_alpha = alpha_list[max_acc_index]
        print("Best alpha::", max_acc_alpha)
        
        
        #Final trained model
        print("Training on Train dataset using best alpha:", max_acc_alpha)
        clf_final = MultinomialNB(alpha = max_acc_alpha)
        clf_final.fit(X_train, Y_train)
        
        
        print("Evaluating on Test set..")
        X_test = count_vectorizer.transform(test_combined_df_shuffle['reviews'])
        Y_test = test_combined_df_shuffle['labels']
        Y_test_pred = clf_final.predict(X_test)
    
        acc_test = accuracy_score(Y_test, Y_test_pred)
        print("Accuracy on Test set ==>", acc_test)
        
        if start_gram == 1 and end_gram == 1:
            print("====End analysis for unigram features====")
        elif start_gram == 2 and end_gram == 2:
            print("====End analysis for bigram features====")
        elif start_gram == 1 and end_gram == 2:
            print("====End analysis for unigram+bigram features====")
        
        
    
    
    
    
    
    
    
    