# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:04:48 2021

@author: lcalv
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
import sys 
import os
import numpy as np
import pandas as pd
import pathlib
import configparser
from time import gmtime, strftime
from shutil import rmtree
import pickle
import logging
import time as timer
import logger

#import gensim
#import pyLDAvis.gensim as gensimvis
#import pyLDAvis
#from gensim.models.wrappers import LdaMallet
#import xml.etree.ElementTree as ET
from colorama import init, Fore, Back, Style

#import matplotlib.pyplot as plt


from random import randrange

sys.path.append(os.path.abspath(".."))
sys.setrecursionlimit(10**6)

# You local imports
from Topic import Topic
from Model import Model
from init_mallet import train_a_model, train_a_submodel, create_submodels
from database.aux_funcs_db import ensureUtf
from auxiliary_functions import xml_dir, indent

import gc
##############################################################################

##############################################################################
#                             VARIABLES                                      #
##############################################################################
project_path = "D:\\project_cord_all_v2" #/projectNIH
source_path = "C:\\mallet\\cord19.txt" #"/mallet/cord19.txt"
model_ids = "model_ids.txt"
time = strftime("_%Y-%m-%d_%H-%M-%S", gmtime())
num_topics_root = 16
logging_file = "D:\\project_cord_all_v2\\log_root_" + str(num_topics_root) 
##############################################################################

logging.basicConfig(filename=logging_file, format='%(asctime)s %(message)s', filemode='w') 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

##############################################################################
#                       AUXILIARY FUNCTIONS                                  #
##############################################################################
def create_model():
    # Path to the models in project folder
    models_dir = pathlib.Path(project_path , "models")
    
    # Available models 
    model_name = "model" + time + "_" + str(randrange(100))
    if not(os.path.isdir((models_dir / model_name).as_posix())):
        # 1.1. Create model's folder
        model_selected = (models_dir / model_name).as_posix()
        os.makedirs(model_selected)
        return model_selected, model_name
    else:
        print("nothing")
        return "",""
    
    
    
def train_model(route_to_model, name, nr_topics):
       
    ## 1. Get route to model, route to persistence and model's name 
    route_to_persistence = pathlib.Path(project_path, "persistence")
    
    # 2.1. Create model object
    model = Model("", "", 0, [],[], [], [], [],[], 0)
    
    # 2.2 Get the nr of topics selected by the user
    model.set_nr_topics(nr_topics)
    print("the selected nr of topics is " + str(nr_topics))
    
    # 2.3. Train the model (create an object of type model)
    train_a_model(source_path, route_to_model, model)
    
    # 2.4. Rename the initial model's folder so it is considered the nr
    # of topics with which is trained
    new_name_path = route_to_model + "_" + str(model.num_topics) + "_topics"
    new_name = name + "_" + str(model.num_topics) + "_topics"
    os.rename(route_to_model, new_name_path)
    model.set_name(new_name)
    model.set_path(new_name_path)
        
    # 2.5. Save the model object created into the persitence file
    name_persistance = name + "_" +  str(model.num_topics) + "_topics" + ".pickle"
    filename_persistence =  pathlib.Path(route_to_persistence,name_persistance).as_posix()
    outfile = open(filename_persistence,'wb')
    pickle.dump(model,outfile)
    outfile.close()

    return filename_persistence, new_name

def train_save_submodels(route_to_model, route_to_persistence, model_name, model_for_expansion, selected_topic, nr_topics, version, thr):

    # Path to the models in project folder, model path, persistence path 
    # and model name
    route_to_models = pathlib.Path(project_path, "models")
    
    ## 1. Load the model from the persitence file
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
    

    ## 3. Select model/submodel for expansion
    models_list = []
    models_paths = []
    model.print_model(models_list, models_paths, True, '---', False)
    print(model_for_expansion)
    for i in np.arange(0,len(models_list),1):
        if models_list[i] == model_for_expansion:
            print( models_list[i])
            model_selected_path = models_paths[i]
            model_selected_name = models_list[i]

    if version == "v1":
        time_rnd = time + "_" + str(randrange(100)) + "_v1"
        submodels_paths, submodels_names = create_submodels([selected_topic], model_selected_path, time_rnd, version, model, thr)
        print("Generating submodels with HTM v1")
    elif version == "v2":
        time_rnd = time + "_" + str(randrange(100)) + "_v2_" + "{:.1f}".format(thr)
        submodels_paths, submodels_names = create_submodels([selected_topic], model_selected_path, time_rnd, version, model, thr)
        print("Generating submodels with HTM v2")
    else:
        print("No HTM version has been given.")
        return
    
    ## 5. Train submodels
    num_topics_all = []
    for i in np.arange(0,len(submodels_paths),1):
        # 5.1 Create submodel object
        submodel = Model("", "", 0, [], [], [], [], [], [], 0)
        submodel.set_nr_topics(nr_topics)
        num_topics_all.append(submodel.num_topics)
        # 5.3 Train the submodel (create an object of type model)
        train_a_submodel(str(submodels_names[i]), str(submodels_paths[i]), submodel)
        # 5.4 Add the name of the submodel to the submodel object
        submodel.set_name(str(submodels_names[i]))
        submodel.set_path(str(submodels_paths[i]))
        # 5.4.1 Check if the model selected for expansion is directly the 
        # model selected in option 2 (father model). If yes, the submodel
        # is directly append to the topics_models lists of the father model
        submodel.add_to_father(model_selected_name, model)
        submodel.set_fathers(model_selected_name, model)
        
        for i in np.arange(0,len(submodels_paths),1):
            file = pathlib.Path(submodels_paths[i], model_ids)
            topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
            topic_ids = topics_ids_df.values[:,0].tolist()
            
        num_docs_model = len(model.thetas)
        submodel.set_n_docs_father(num_docs_model)

        saving = True
        for i in np.arange(0,len(submodels_paths),1):
            if saving:
                new_submodel_path = submodels_paths[i] + "_" + str(num_topics_all[i]) + "_topics"
                new_submodel_name = submodels_names[i] + "_" + str(num_topics_all[i]) + "_topics"
                model.rename_child(submodels_names[i], new_submodel_name, new_submodel_path)
                os.rename(submodels_paths[i], new_submodel_path)
                print(Fore.GREEN + "Submodel " + '"' + new_submodel_name + '"' + "was saved." + Fore.WHITE)
            # 6.2.4 If the answer is no, we remove both the submodel's folder 
            # and the submodel object
            else:
                  rmtree(submodels_paths[i])
                  model.delete_child(submodels_names[i])
        outfile = open(route_to_persistence,'wb')
        pickle.dump(model,outfile)
        outfile.close()
        
        gc.collect()
    return

##############################################################################


num_htm = 5

for htm in range(num_htm):
    # 1. Create new HTM (folder of the root model)
    start_time = timer.time()
    model_path, model_name = create_model()
    
    # 2. Train root model
    route_to_persistence, model_name = train_model(model_path, model_name, num_topics_root)
    training_time = timer.time() - start_time
    print("Training root model nr ", htm)
    logger.debug("root", 0, str(num_topics_root), str(training_time))
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
    
    num_topics_sub = [4,5,6,7,8]
    for j in range(len(num_topics_sub)):
        for i in np.arange(0, len(model.topics_models), 1):
            if str(type(model.topics_models[i])) == "<class 'Topic.Topic'>":
                topic_id = model.topics_models[i].id_topic
                for thr in np.arange(0.1, 1, 0.1):
                    start_time = timer.time()
                    train_save_submodels(model_path, route_to_persistence, model_name, model_name, topic_id, num_topics_sub[j], "v2", thr)
                    training_time = timer.time() - start_time
                    logger.debug("v2", "{:.1f}".format(thr), str(num_topics_sub[j]), str(training_time))
                start_time = timer.time()
                train_save_submodels(model_path, route_to_persistence, model_name, model_name, topic_id, num_topics_sub[j], "v1", "")
                training_time = timer.time() - start_time
                logger.debug("v1", str(0), str(num_topics_sub[j]), str(training_time))
        print("Submodel trained with ", num_topics_sub[j])