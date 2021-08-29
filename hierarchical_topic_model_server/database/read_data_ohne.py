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
for df in DB.readDBchunks(dbTABLENAME, 'id', chunksize=50000, selectOptions='text', limit=int(num_txt_files)*num_texts, filterOptions=None, verbose=True):
    news_df  = pd.DataFrame(df, columns = ['text'])
    

# SAVE EACH NEW IN A TXT
iter = 0
for i in np.arange(0,len(news_df),1):
    if news_df.values[i].tolist() != [""]:
        if iter == 0:
            iter = 1
        text_to_save = 'C:\\mallet\\data_news_txt_ohne_lem\\' + 'new_ohne_lem' + str(iter) + '.txt'
        np.savetxt(text_to_save, news_df.values[i], fmt='%s', encoding='utf-8')
        iter = iter + 1
    print(news_df.values[i])
    
