# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             LOAD_DATA                                  ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np
import json

import configparser
from bs4 import BeautifulSoup


from aux_funcs_db import add_to_list_from_dict, preprocessing, keep_unique, ensureUtf, normalize
from base_dm_sql import BaseDMsql
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk import Counter
import pickle


##############################################################################
#                              CONFIG FILE                                   #
##############################################################################
file = 'config_database.ini'
config = configparser.ConfigParser()
config.read(file)

dbSERVER = config['credentials']['dbSERVER']
dbUSER = config['credentials']['dbUSER']
dbPASS = config['credentials']['dbPASS']
dbTABLENAME = config['testing']['dbTABLENAME']
dbNAME = config['credentials']['dbNAME']

columns = ['id', 'anteTitle', 'breadCrumbRef', 'commentOptions', 'contentEndDate', 'contentInitDate',
           'contentType', 'essntialInfo1', 'essentialInfo2', 'expirationDate', 'frontSummary',
           'frontTitle', 'htmlShortUrl', 'htmlUrl', 'ids', 'image', 'imageSEO', 'language', 
           'comentariosRef', 'encuestaDestacadaRef', 'encuestasRelacionadasRef', 'encuestasTotemRef',
           'estadisticasRef', 'galeriasRelacionadasRef', 'galeriasTotemRef', 'tagsRef', 'longTitle',
           'mainCategory', 'mainCategoryLang', 'mainCategoryRef', 'multimediaTotemRef', 
           'newsEspecialesRef', 'newsRelatedRef', 'numVisits', 'otherTopicsRef', 'popHistoric',
           'popularity', 'code', 'description' , 'publicationDate', 'publicationDateTimestamp', 
           'refreshSeconds', 'relatedByLangRef', 'rightModule', 'shortTitle', 'ctvId', 
           'facebook', 'firma', 'googlePlus', 'name', 'numPublications', 'photo', 'publicationDate2',
           'twitter', 'numComentarios', 'numCompartidas', 'summary', 'tabTitle', 'text',
           'textLemmatized', 'tickerNews', 'tickerSports', 'title', 'uri']

num_txt_files = config['production']['num_txt_files']
num_texts = 20

ips = [] # List to create order ids for each entry in the database
info_all = [] # List of lists being each of the lists a new entry in the database, 
              # whose components are each of the columns defined in columns

ohne = []
ohne_row = []

# Each txt file is a dict with 20 texts, so we iterate 20 times per txt file, 
# getting a total of 20*16054 = 321.080 texts
txt_files_index = np.arange(1, int(num_txt_files)+1,1)   
for i in txt_files_index:
    file = 'C:\\mallet\\raw-data\\' + str(i) + '.txt'
    with open(file, encoding="utf8") as json_file:
        data = json.load(json_file,encoding="utf-8")
    dict1 = data.get('page')
    dict2 = dict1.get('items')
    
    txt_index = np.arange(0,num_texts,1)
    for j in txt_index:
        info_row = [] # List that defines a new entry in the database
        
        dict3 = dict2[j]
        
        # ID
        if i == 1:
            new_id = (i*j).item()
        else:
            last = (i-1)*20
            new_id = (ips[last-1] + j).item()
        ips.append(new_id)
        info_row.append(new_id)
        
        # ANTETITLE
        anteTitle_str ='anteTitle'
        add_to_list_from_dict(dict3, anteTitle_str, info_row)
        
        # BREADCRUMBREF
        breadCrumbRef_str ='breadCrumbRef'
        add_to_list_from_dict(dict3, breadCrumbRef_str, info_row)
      
        # COMMENT OPTIONS
        commentOptions_str ='commentOptions'
        add_to_list_from_dict(dict3, commentOptions_str, info_row)
      
        # CONTENT END DATE
        contentEndDate_str ='contentEndDate'
        add_to_list_from_dict(dict3, contentEndDate_str, info_row)
        
        # CONTENT INIT DATE 
        contentInitDate_str ='contentInitDate'
        add_to_list_from_dict(dict3, contentInitDate_str, info_row)
        
        # CONTET TYPE
        contentType_str ='contentType'
        add_to_list_from_dict(dict3, contentType_str, info_row)
        
        # ESSENTIAL INFO DICT
        essentialInfo = dict3['essentialInfo']
        
        # ESSENTIAL INFO 1
        info_str ='info'
        add_to_list_from_dict(essentialInfo, info_str, info_row) 
        
        # ESSENTIAL INFO 2
        photo_str ='photo'
        add_to_list_from_dict(essentialInfo, photo_str, info_row)  
       
        # EXPIRATION DATE
        expirationDate_str ='expirationDate'
        add_to_list_from_dict(dict3, expirationDate_str, info_row)  
    
        # FRONT SUMMARY
        frontSummary_str ='frontSummary'
        add_to_list_from_dict(dict3, frontSummary_str, info_row)
        
        # FRONT TITLE
        frontTitle_str ='frontTitle'
        add_to_list_from_dict(dict3, frontTitle_str, info_row)
        
        # HTML SHORT URL
        htmlShortUrl_str ='htmlShortUrl'
        add_to_list_from_dict(dict3, htmlShortUrl_str, info_row)
       
        # HTML URL 
        htmlUrl_str ='htmlUrl'
        add_to_list_from_dict(dict3, htmlUrl_str, info_row)
        
        # IDS 
        id_str ='id'
        add_to_list_from_dict(dict3, id_str, info_row)

        # IMAGE
        image_str ='image'
        add_to_list_from_dict(dict3, image_str, info_row)
        
        # IMAGE SEO
        imageSEO_str ='imageSEO'
        add_to_list_from_dict(dict3, imageSEO_str, info_row)
   
        # LANGUAGE 
        language_str ='language'
        add_to_list_from_dict(dict3, imageSEO_str, info_row)
   
        # LINKS DICT   
        links = dict3['links']
        
        # COMENTARIOS REF
        comentariosRef_str ='comentariosRef'
        add_to_list_from_dict(links, comentariosRef_str, info_row)
    
        # ENCUESTA DESTACADA REF 
        encuestaDestacadaRef_str ='encuestaDestacadaRef'
        add_to_list_from_dict(links, encuestaDestacadaRef_str, info_row)
        
        # ENCUESTAS RELACIONADAS REF
        encuestasRelacionadasRef_str ='encuestasRelacionadasRef'
        add_to_list_from_dict(links, encuestasRelacionadasRef_str, info_row)
       
        # ENCUESTAS TOTEMREF
        encuestasTotemRef_str ='encuestasTotemRef'
        add_to_list_from_dict(links, encuestasTotemRef_str, info_row)
     
        # ESTADISTICAS REF
        estadisticasRef_str ='estadisticasRef'
        add_to_list_from_dict(links, estadisticasRef_str, info_row)
        
        # GALERIAS RELACIONADAS REF
        galeriasRelacionadasRef_str ='galeriasRelacionadasRef'
        add_to_list_from_dict(links, galeriasRelacionadasRef_str, info_row)
   
        # GALERIAS TOTEM REF
        galeriasTotemRef_str ='galeriasTotemRef'
        add_to_list_from_dict(links, galeriasTotemRef_str, info_row)
     
        # TAGS REF   
        tagsRef_str ='tagsRef'
        add_to_list_from_dict(links, tagsRef_str, info_row)
       
        # LONG TITLE 
        longTitle_str ='longTitle'
        add_to_list_from_dict(dict3, normalize(longTitle_str), info_row)
        
        # MAIN CATEGORY
        mainCategory_str ='mainCategory'
        add_to_list_from_dict(dict3, mainCategory_str, info_row)
      
        # MAIN CATEGORY LANG  
        mainCategoryLang_str ='mainCategoryLang'
        add_to_list_from_dict(dict3, mainCategoryLang_str, info_row)
       
        # MAIN CATEGORY REF 
        mainCategoryRef_str ='mainCategoryRef'
        add_to_list_from_dict(dict3, mainCategoryRef_str, info_row)
     
        # MULTIMEDIA TOTEM REF 
        multimediaTotemRef_str ='multimediaTotemRef'
        add_to_list_from_dict(dict3, multimediaTotemRef_str, info_row)
        
        # MEWS ESPECIALES REF
        newsEspecialesRef_str ='newsEspecialesRef'
        add_to_list_from_dict(dict3, newsEspecialesRef_str, info_row)
       
        # NEWS RELATED REF
        newsRelatedRef_str ='newsRelatedRef'
        add_to_list_from_dict(dict3, newsRelatedRef_str, info_row)
        
        # NUM VISITS
        numVisits_str ='numVisits'
        add_to_list_from_dict(dict3, numVisits_str, info_row)
        
        # OTHER TOPICS REF
        otherTopicsRef_str ='otherTopicsRef'
        add_to_list_from_dict(dict3, otherTopicsRef_str, info_row)
         
        # POP HISTORIC 
        popHistoric_str ='popHistoric'
        add_to_list_from_dict(dict3, popHistoric_str, info_row)
      
        # POPULARITY 
        popularity_str ='popularity'
        add_to_list_from_dict(dict3, popularity_str, info_row)
        
        # PUB STATE DICT
        pubState = dict3['pubState']
        
        # CODE 
        code_str ='code'
        add_to_list_from_dict(pubState, code_str, info_row)
       
        # DESCRIPTION 
        description_str ='description'
        add_to_list_from_dict(pubState, description_str, info_row)
        
        # PUBLICATION DATE
        publicationDate_str ='publicationDate'
        add_to_list_from_dict(dict3, publicationDate_str, info_row)
     
        # PUBLICATION DATE TIMESTAMP  
        publicationDateTimestamp_str ='publicationDateTimestamp'
        add_to_list_from_dict(dict3, publicationDateTimestamp_str, info_row)
        
        # REFRESH SECONDS
        refreshSeconds_str ='refreshSeconds'
        add_to_list_from_dict(dict3, refreshSeconds_str, info_row)
        
        # RELATED BY LANG REF
        relatedByLangRefs_str ='relatedByLangRef'
        add_to_list_from_dict(dict3, relatedByLangRefs_str, info_row)
        
        # RIGHT MODULE
        rightModule_str ='rightModule'
        add_to_list_from_dict(dict3, rightModule_str, info_row)
       
        # SHORT TITLE 
        shortTitle_str ='shortTitle'
        add_to_list_from_dict(dict3, normalize(shortTitle_str), info_row)
       
        # SIGN DICT
        sign = dict3['sign']
        
        # CTV ID
        ctvId_str ='ctvId'
        add_to_list_from_dict(sign, ctvId_str, info_row)  
        
        # FACEBOOK 
        facebook_str ='facebook'
        add_to_list_from_dict(sign, facebook_str, info_row) 
        
        # FIRMA
        firma_str ='firma'
        add_to_list_from_dict(sign, firma_str, info_row) 
        
        # GOOGLE PLUS        
        googlePlus_str ='googlePlus'
        add_to_list_from_dict(sign, googlePlus_str, info_row) 
    
        # NAME
        name_str ='name'
        add_to_list_from_dict(sign, name_str, info_row) 
    
        # NUM PUBLICATIONS
        numPublications_str ='numPublications'
        add_to_list_from_dict(sign, numPublications_str, info_row) 
       
        # PHOTO 
        photo_str ='photo'
        add_to_list_from_dict(sign, photo_str, info_row) 
        
        # PUBLICATION DATE 2
        publicationDate2_str ='publicationDate'
        add_to_list_from_dict(sign, photo_str, info_row) 
      
        # TWITTER 
        twitter_str ='twitter'
        add_to_list_from_dict(sign, photo_str, info_row) 
       
        # STATISTICS DICT
        statistics = dict3['statistics']
        
        # NUM COMETARIOS
        numComentarios_str ='numComentarios'
        add_to_list_from_dict(statistics, numComentarios_str, info_row)
        
        # NUM COMPARTIDAS
        numCompartidas_str ='numCompartidas'
        add_to_list_from_dict(statistics, numCompartidas_str, info_row)
        
        # SUMMARY
        summary_str ='summary'
        add_to_list_from_dict(dict3, summary_str, info_row)
        
        # TAB TITLE
        tabTitle_str ='tabTitle'
        add_to_list_from_dict(dict3, normalize(tabTitle_str), info_row)
      
        
        text = dict3['text']
        if text != None:
            #tokens_lemmatized = preprocessing(text)
            [text_lemmatized_str, text_bef_lem]  = preprocessing(ensureUtf(text))
            soup = BeautifulSoup(text,'html.parser')
            text_orig= soup.get_text()
            text_orig = ensureUtf(text)
            #text_imp= keep_unique(tokens_reduced, text_orig)
        info_row.append(text_bef_lem)
        ohne_row.append(text_bef_lem)
        info_row.append(text_lemmatized_str)
        print(text_bef_lem)
        print("")
        print(text_lemmatized_str)
        print("")

        # TICKER DICT
        ticker = dict3['ticker']
        
        # TICKER NEWS 
        tickerNews_str ='tickerNews'
        add_to_list_from_dict(ticker, tickerNews_str, info_row)
      
        # TICKER SPORTS
        tickerSports_str ='tickerSports'
        add_to_list_from_dict(ticker, tickerSports_str, info_row)
        
        # TITLE 
        title_str ='title'
        add_to_list_from_dict(dict3, title_str, info_row)
        
        # URI
        uri_str ='uri'
        add_to_list_from_dict(dict3, uri_str, info_row)
        
    info_all.append(info_row)
    ohne.append(ohne_row)

# # # DATABASE CONNECTION
# DB = BaseDMsql(db_name=dbNAME, db_connector='mysql', path2db=None,
#                 db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS)

# # INSERT DATA INTO THE DATABASE
# DB.insertInTable(dbTABLENAME, columns, info_all, chunksize=1000, verbose=True)

# #SHOW DATABASE SAVED INFORMATION
# [cols, n_rows] = DB.getTableInfo(dbTABLENAME)
# print(cols)
# print(n_rows)
