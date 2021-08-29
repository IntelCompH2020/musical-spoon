# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:22:33 2021

@author: lcalv
"""
import sys, os
import pickle
import numpy as np
from gensim import models
import gensim
import string
import pathlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from Model import Model
import pandas as pd
from scipy import sparse


def get_models_submodels_values(sources):
    results = []
    for s in range(len(sources)):
        print(len(results))
        print("Processing model " , s)
        print(sources[s])
        infile = open(pathlib.Path(sources[s]),'rb') # -->> coger el dic
        model = pickle.load(infile)
        
        # Root model
        coherence = model.avg_coherence
        nr_docs_root = len(model.thetas) # Nr of documents of the root model
        thr_root = -1
        
        # Get average entropy of the root model
        if isinstance(model.thetas, sparse.csr_matrix):
            theta_aux = np.array(model.thetas)
            xcorr = np.corrcoef(theta_aux.T)
            theta_aux += 1e-12
            doc_entropy = -np.sum(theta_aux * np.log(theta_aux),axis=1)
            doc_entropy = doc_entropy/np.log(model.num_topics)
        else:
            #Si la matriz no es sparse le sumamos una pequeña constante para prevenir
            #divisiones por cero
            model.thetas = np.array(model.thetas)
            xcorr = np.corrcoef(model.thetas.T)
            if np.min(model.thetas) < 1e-12:
                model.thetas += 1e-12
            doc_entropy = -np.sum(model.thetas * np.log(model.thetas),axis=1)
            doc_entropy = doc_entropy/np.log(model.num_topics)
            model_entropy = np.sum(doc_entropy)/len(doc_entropy)
            
        training_time = model.training_time

        results.append([model.model_name, -1, nr_docs_root, -1, model.num_topics, thr_root, coherence, model_entropy, training_time])

        # Get info for each of the submodels
        for i in range(len(model.topics_models)):
            if str(type(model.topics_models[i])) == "<class 'Model.Model'>":
                print(str(i) + "/" + str(len(model.topics_models)))

                # Append to model name
                submodel_name = model.topics_models[i].model_name

                # Append number of documents 
                submodel_nr_documents = len(model.topics_models[i].thetas)

                # Topic from which the submodel comes
                submodel_topic_from = model.topics_models[i].model_name.split("Topic_")[1].split("_")[0]

                # Get threhsolds (if it is the case)
                if "v2" in model.topics_models[i].model_name:
                    submodel_threshold = float(model.topics_models[i].model_name.split("v2_")[1].split("_")[0])
                else:
                    submodel_threshold = 0

                # Get corpus ----> CAMBIAR PARA MODELOS DEL SERVIDOR
                submodel_path = model.topics_models[i].model_path
                submodel_corpus = pathlib.Path(submodel_path) / "submodel.txt"
                corpus = []
                with open(submodel_corpus, 'rb') as f:
                    for myline in f: 
                        a = str(myline).split("0 ")[1].split("\\")[0].split(" ")
                        corpus.append(a)
    
                submodel_coherence = model.topics_models[i].avg_coherence

                # Get entropy
                if isinstance(model.topics_models[i].thetas, sparse.csr_matrix):
                    theta_aux = np.array(model.topics_models[i].thetas)
                    xcorr = np.corrcoef(theta_aux.T)
                    theta_aux += 1e-12
                    doc_entropy = -np.sum(theta_aux * np.log(theta_aux),axis=1)
                    doc_entropy = doc_entropy/np.log(model.topics_models[i].num_topics)
                    print(doc_entropy)
                    print(len(doc_entropy))
                else:
                    #Si la matriz no es sparse le sumamos una pequeña constante para prevenir
                    #divisiones por cero
                    model.topics_models[i].thetas = np.array(model.topics_models[i].thetas)
                    xcorr = np.corrcoef(model.topics_models[i].thetas.T)
                    if np.min(model.topics_models[i].thetas) < 1e-12:
                        model.topics_models[i].thetas += 1e-12
                    doc_entropy = -np.sum(model.topics_models[i].thetas * np.log(model.topics_models[i].thetas),axis=1)
                    doc_entropy = doc_entropy/np.log(model.topics_models[i].num_topics)
                    submodel_entropy = np.sum(doc_entropy)/len(doc_entropy)
                
                training_time = model.topics_models[i].training_time
                
                results.append([submodel_name, s, submodel_nr_documents, submodel_topic_from, model.topics_models[i].num_topics, submodel_threshold, submodel_coherence, submodel_entropy, training_time])
                   
    df = pd.DataFrame(results, columns=['modelName', 'modelFrom', 'nrDocs', 'topicFrom', 'topicsTrainWith', 'thr', 'cohr', 'entro', 'trainingTime'])
    for i in df.index:
        if("v2" in df["modelName"][i]):
            if(df["thr"][i] == 0):
                thr = float(df["modelName"][i].split("v2_")[1].split("_")[0])
                df["thr"][i] = thr
    df = df.drop_duplicates()
    return df

def save_in_pickle(structure, pickle_to_save_in):
  with open(pickle_to_save_in, 'wb') as f:
    pickle.dump(structure, f)

num_topics_root_model = 10
persistence_path = "/export/usuarios01/lbartolome/projectNIH/peristence"
df1_pickle = "/export/usuarios01/lbartolome/projectNIH/df_" + str(num_topics_root_model) + ".pickle"
df2_pickle = "/export/usuarios01/lbartolome/projectNIH/df_" + str(num_topics_root_model) + "_avg.pickle"
df3_pickle = "/export/usuarios01/lbartolome/projectNIH/df_" + str(num_topics_root_model) + "_root.pickle"

sources = []
for x in os.listdir(persistence_path): #añadir sólo si endswith 10 topics para el mío
    if x.endswith(str(5) + "_topics.pickle"):
        print(pathlib.Path(persistence_path) / x)
        sources.append(pathlib.Path(persistence_path) / x)   
print(sources)

df2 = get_models_submodels_values(sources)
save_in_pickle(df2, df1_pickle)
