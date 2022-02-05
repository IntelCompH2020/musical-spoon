# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             READ_DATA                                  ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import pandas as pd
import numpy as np

import configparser

# Gensim
import gensim
from nltk.tokenize import word_tokenize
# NLTK
from base_dm_sql import BaseDMsql

##############################################################################
#                              CONFIG FILE                                   #
##############################################################################
file = 'config_database.ini'
config = configparser.ConfigParser()
config.read(file)

dbSERVER = config['credentials']['dbSERVER']
dbUSER = config['credentials']['dbUSER']
dbPASS = config['credentials']['dbPASS']
dbTABLENAME = config['production']['dbTABLENAME']
dbNAME = config['credentials']['dbNAME']

num_txt_files = config['production']['num_txt_files']
num_texts = 20

DB = BaseDMsql(db_name=dbNAME, db_connector='mysql', path2db=None,
               db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS)

# READ DATA
for df in DB.readDBchunks(dbTABLENAME, 'id', chunksize=50000, selectOptions='textLemmatized', limit=int(num_txt_files)*num_texts, filterOptions=None, verbose=True):
    news_df = pd.DataFrame(df, columns=['textLemmatized'])


# Get lemmatized texts
lemmatized_texts = []
for i in np.arange(0, len(news_df), 1):
    text = news_df.values[i].tolist()
    text_str = ', '.join(text)
    text_tok = word_tokenize(text_str)
    lemmatized_texts.append(text_tok)

# Create dictionary
D_train = gensim.corpora.Dictionary(lemmatized_texts)
len_init = len(D_train)

# Filtering
nr_min_doc = 5  # Minumun number of documents to keep a term in the dictionary
# Maximum proportion of documents in which a term can appear to be kept in the dictionary
nr_max_prop = 0.75
D_train.filter_extremes(no_below=nr_min_doc,
                        no_above=nr_max_prop, keep_n=25000)
dictionary_filtered_list = []
[dictionary_filtered_list.append(D_train.get(i)) for i in np.arange(
    0, len(D_train), 1)]  # convertir a set


# SAVE EACH NEW IN A TXT
iter = 0
for i in np.arange(0, len(news_df), 1):
    if iter == 0:
        iter = 1
    text = news_df.values[i].tolist()
    text_str = ','.join(text)
    text_toks = word_tokenize(text_str)
    final_text = []
    [final_text.append(text_tok) for text_tok in text_toks if (
        text_tok in dictionary_filtered_list)]
    join_final_text = ' '.join(final_text)
    arr = np.asarray(join_final_text, dtype=object).reshape(1, -1)
    text_to_save = 'C:\\mallet\\data_news_txt_all3\\' + \
        'new' + str(iter) + '.txt'
    np.savetxt(text_to_save, arr, fmt='%s', encoding='utf-8')
    iter = iter + 1
    print(news_df.values[i])

# data_news_txt_ohne_lem
