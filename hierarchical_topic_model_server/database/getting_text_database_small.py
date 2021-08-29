# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:40:55 2020

@author: lcalv
"""

import gensim
from nltk.tokenize import word_tokenize
import numpy as np

file = 'C:\\mallet\\S24Ever_corpus_small.txt'

texts = []
with open(file, 'r',  encoding='utf-8') as f:
    for line in f:
        line_split = line.split(" ")
        line_split = line_split[2:-1]
        line_split =  ' '.join(line_split)
        text_tok = word_tokenize(line_split)
        texts.append(text_tok)
        
texts_orig = texts

print(texts[0])

# Create dictionary
D_train = gensim.corpora.Dictionary(texts)
len_init = len(D_train)

# Filtering
nr_min_doc = 5 # Minumun number of documents to keep a term in the dictionary
nr_max_prop = 0.75 # Maximum proportion of documents in which a term can appear to be kept in the dictionary
D_train.filter_extremes(no_below=nr_min_doc, no_above=nr_max_prop, keep_n=25000)
dictionary_filtered_list = []
[dictionary_filtered_list.append(D_train.get(i)) for i in np.arange(0,len(D_train),1)] # convertir a set

# SAVE EACH NEW IN A TXT
iter = 0
for i in np.arange(0,len(texts_orig),1):
    if iter == 0:
        iter = 1
    text_toks = texts_orig[i]
    final_text = []
    [final_text.append(text_tok) for text_tok in text_toks if (text_tok in dictionary_filtered_list)]
    join_final_text = ' '.join(final_text)
    arr = np.asarray(join_final_text, dtype = object).reshape(1,-1)
    text_to_save = 'C:\\mallet\\S24Ever_corpus_small\\' + 'new' + str(iter) + '.txt'
    np.savetxt(text_to_save, arr, fmt='%s', encoding='utf-8')
    iter = iter + 1
    print(iter)
