# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 07:19:09 2020

@author: lcalv
"""


import numpy as np
import pandas as pd
import pathlib
import os
import gensim
from nltk.tokenize import word_tokenize

read_from = 'C:\\mallet\\lemmatized'

lemmatized_texts = []
read_from_dir = pathlib.Path(read_from)
with os.scandir(read_from_dir) as files:
    for file in files:
        f = open(file,encoding='utf-8')
        lines = f.readlines()
        print(lines)
        if lines != []:
            text_tok = word_tokenize(lines[0])
            lemmatized_texts.append(text_tok)
lemmatized_texts_orig = lemmatized_texts

# Create dictionary
D_train = gensim.corpora.Dictionary(lemmatized_texts)
len_init = len(D_train)

# Filtering
nr_min_doc = 5 # Minumun number of documents to keep a term in the dictionary
nr_max_prop = 0.75 # Maximum proportion of documents in which a term can appear to be kept in the dictionary
D_train.filter_extremes(no_below=nr_min_doc, no_above=nr_max_prop, keep_n=25000)
dictionary_filtered_list = []
[dictionary_filtered_list.append(D_train.get(i)) for i in np.arange(0,len(D_train),1)] # convertir a set

# SAVE EACH NEW IN A TXT
iter = 0
for i in np.arange(0,len(lemmatized_texts_orig),1):
    if iter == 0:
        iter = 1
    text_toks = lemmatized_texts_orig[i]
    final_text = []
    [final_text.append(text_tok) for text_tok in text_toks if (text_tok in dictionary_filtered_list)]
    join_final_text = ' '.join(final_text)
    arr = np.asarray(join_final_text, dtype = object).reshape(1,-1)
    text_to_save = 'C:\\mallet\\data_news_txt_lem_J\\' + 'new' + str(iter) + '.txt'
    np.savetxt(text_to_save, arr, fmt='%s', encoding='utf-8')
    iter = iter + 1
    print(iter)