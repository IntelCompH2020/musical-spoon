# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:33:21 2021

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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-topics", help="Mostrar información de depuración", action="store_true")
parser.add_argument("-f", "--file", help="Nombre de archivo a procesar")
args = parser.parse_args()

def get_models_submodels_values(sources):
    results = []
    for s in range(len(sources)):
        print(len(results))
        print("Processing model " , s)
        print(sources[s])
        infile = open(pathlib.Path(sources[s]),'rb') # -->> coger el dic
        model = pickle.load(infile)
        
        #Root model
        results.append([model.model_name, -1, len(model.thetas), -1, model.num_topics, -1, model.sizes, 
                        model.avg_coherence, model.document_entropy, model.training_time])

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

                submodel_coherence = model.topics_models[i].avg_coherence

                submodel_entropy = model.topics_models[i].document_entropy
                
                training_time = model.topics_models[i].training_time
                
                submodel_sizes = model.topics_models[i].sizes
                
                results.append([submodel_name, s, submodel_nr_documents, submodel_topic_from, model.topics_models[i].num_topics,
                                submodel_threshold, submodel_sizes, submodel_coherence, submodel_entropy, training_time])
                   
    df = pd.DataFrame(results, columns=['modelName', 'modelFrom', 'nrDocs', 'topicFrom', 'topicsTrainWith', 'thr', 'sizes', 'cohr', 'entro', 'trainingTime'])
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

num_topics_root_model = args[0]
project_path = args[1]
persistence_path = "/export/usuarios01/lbartolome/" + project_path + "/peristence"
df1_pickle = "/export/usuarios01/lbartolome/" + project_path + "/df_" + str(num_topics_root_model) + "_avg.pickle"
df3_pickle = "/export/usuarios01/lbartolome/" + project_path + "/df_" + str(num_topics_root_model) + "_root.pickle"

sources = []
for x in os.listdir(persistence_path): #añadir sólo si endswith 10 topics para el mío
    if x.endswith(str(num_topics_root_model) + "_topics.pickle"):
        print(pathlib.Path(persistence_path) / x)
        sources.append(pathlib.Path(persistence_path) / x)   
print(sources)

df2 = get_models_submodels_values(sources)
save_in_pickle(df2, df1_pickle)
