#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:58:43 2019

@author: hardiksahi
"""

# =============================================================================
# # Command to run:
# # python3 Assignment1.py path_to_data
# =============================================================================



#import argparse
import re
import pandas as pd
import numpy as np
import sys
from gensim.parsing.preprocessing import STOPWORDS


def tokenize_corpus(data):
    return [spl for spl in data.split('\n') if spl]

# Remove punctuation
def remove_punctuation(string):
    return re.sub(r'[!"#$%&()*+\/:;<=>@[\\\]^`{}|~\t\n]', " ", string)

#Remove stopwords
def remove_stopwords(string, stopword_list):
    return [w for w in string.split() if w not in stopword_list]
    
if __name__ == "__main__":
    input_path = sys.argv[1]
    stopword_list = list(STOPWORDS)
    
    open_data = open(input_path, "r")
    data = open_data.read()
    data_lower = data.lower()
    
    split_newline_data = tokenize_corpus(data_lower)
    
    reviews_with_stop = []
    reviews_without_stop = []
    
    
    for r in split_newline_data:
        punc_removed = remove_punctuation(r)
        remove_apostrophe = re.sub(r'[\']'," ' ",punc_removed)
        removecomma = re.sub(r'[,]'," , ",remove_apostrophe) # Giving space before/after apostrope
        removefullstop = re.sub(r'[.]'," . ",removecomma) #Giving space before/after comma
        removehyphen = re.sub(r'[-]'," - ",removefullstop)# Giving space before/after fullstop
        stopword_removed = remove_stopwords(removehyphen, stopword_list) # Giving space before/after hyphen
        
        # Split by space..
        reviews_with_stop.append(removehyphen.split())
        reviews_without_stop.append(stopword_removed)
        
        
    combined_df = pd.DataFrame()
    combined_df['reviews_with_stop'] = reviews_with_stop
    combined_df['reviews_without_stop'] = reviews_without_stop
    
    np.random.seed(100)
    comb_train_df, comb_val_df, comb_test_df = np.split(combined_df, [int(.8*len(combined_df)), int(.9*len(combined_df))])
    
    train_list = comb_train_df['reviews_with_stop'].tolist()
    df = pd.DataFrame(data={"col1": train_list})
    df.to_csv("./train.csv", sep=',',index=False)
    
    val_list = comb_val_df['reviews_with_stop'].tolist()
    df = pd.DataFrame(data={"col1": val_list})
    df.to_csv("./val.csv", sep=',',index=False)
    
    test_list = comb_test_df['reviews_with_stop'].tolist()
    df = pd.DataFrame(data={"col1": test_list})
    df.to_csv("./test.csv", sep=',',index=False)
    
    
    train_list_no_stopword = comb_train_df['reviews_without_stop'].tolist()
    df = pd.DataFrame(data={"col1": train_list_no_stopword})
    df.to_csv("./train_no_stopword.csv", sep=',',index=False)
    
    
    val_list_no_stopword = comb_val_df['reviews_without_stop'].tolist()
    df = pd.DataFrame(data={"col1": val_list_no_stopword})
    df.to_csv("./val_no_stopword.csv", sep=',',index=False)
    
    
    test_list_no_stopword = comb_test_df['reviews_without_stop'].tolist()
    df = pd.DataFrame(data={"col1": test_list_no_stopword})
    df.to_csv("./test_no_stopword.csv", sep=',',index=False)

    
    