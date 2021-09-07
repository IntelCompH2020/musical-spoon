# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:50:02 2021

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
import time as timer

import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import xml.etree.ElementTree as ET
from colorama import init, Fore, Back, Style

import matplotlib.pyplot as plt

from random import randrange

sys.path.append(os.path.abspath(".."))
sys.setrecursionlimit(10**6)

# You local imports
from Topic import Topic
from Model import Model
from init_mallet import train_a_model, train_a_submodel, create_submodels
from database.aux_funcs_db import ensureUtf
from auxiliary_functions import xml_dir, indent
from PyQt5 import QtWidgets


config_file =  'config_project.ini' 
config = configparser.ConfigParser()
config.read(config_file)

project_path = config['files']['project_path']
source_path = config['files']['source_path']
model_ids = config['out-documents']['model_ids']
mallet_path = config['mallet']['mallet_path']

time = strftime("_%Y-%m-%d_%H-%M-%S", gmtime())

def create_model():
    # Path to the models in project folder
    models_dir = pathlib.Path(project_path , "models")
    route_to_persistence = pathlib.Path(project_path, "persistence")
    
    # Available models 
    models = [model.name for model in models_dir.iterdir() if model.is_dir()]
    model_name = "model" + time + "_" + str(randrange(100))
    if not(os.path.isdir((models_dir / model_name).as_posix())):
        # 1.1. Create model's folder
        model_selected = (models_dir / model_name).as_posix()
        os.makedirs(model_selected)
        # 1.2. Save model's folder into config file
        config.read(config_file)
        config.set('models', 'model_selected', model_selected)
        config.set('models', 'model_name', model_name)
        with open(config_file, 'w') as configfile:
            config.write(configfile)
    else:
        return
    
def list_models():
    # Reload config file just in case changes are found
    config.read(config_file)
    project_path = config['files']['project_path']
    models_dir = pathlib.Path(project_path , "models")
    print(models_dir)
        
    # Available models 
    models = [model.name for model in models_dir.iterdir() if model.is_dir()]
    models_list = []
    models_name = []
    models_name_str = []
    model_nr = 0
    for model in models:
        model_nr = model_nr + 1
        models_name_str.append(model.__str__())
        models_list.append(pathlib.Path(models_dir, model).as_posix())
        models_name.append(model)
   
    return models_name_str

def select_model(model_name):
    # Reload config file just in case changes are found
    config.read(config_file)
    project_path = config['files']['project_path']
    # Path to the models in project folder
    models_dir = pathlib.Path(project_path , "models")
    route_to_persistence = pathlib.Path(project_path, "persistence")
    
    # Available models 
    models = [model.name for model in models_dir.iterdir() if model.is_dir()]
    models_list = []
    models_name = []
    models_name_str = []
    model_nr = 0
    model_selected_nr = 0
    for model in models:
        model_nr = model_nr + 1
        models_name_str.append(model.__str__())
        models_list.append(pathlib.Path(models_dir, model).as_posix())
        models_name.append(model)
        if model_name == model.__str__():
            model_selected_nr = model_nr
    
    config.read(config_file)
    config.set('models', 'model_selected', models_list[model_selected_nr-1])
    config.set('models', 'model_name',  models_name[model_selected_nr-1])
    persis_name =  models_name[model_selected_nr-1] + ".pickle"
    config.set('models', 'persistence_selected',  pathlib.Path(route_to_persistence,persis_name).as_posix())
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    
    #print(models_list[model_selected_nr-1] + " loaded.")
        
def train_model(nr_topics):
       
    ## 1. Get route to model, route to persistence and model's name 
    config.read(config_file)
    route_to_model = config['models']['model_selected']
    route_to_persistence = pathlib.Path(project_path, "persistence")
    name  =  config['models']['model_name']
    
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
    config.read(config_file)
    config.set('models', 'model_selected', new_name_path)
    config.set('models', 'model_name', new_name)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
        
    # 2.5. Save the model object created into the persitence file
    name_persistance = name + "_" +  str(model.num_topics) + "_topics" + ".pickle"
    filename_persistence =  pathlib.Path(route_to_persistence,name_persistance).as_posix()
    outfile = open(filename_persistence,'wb')
    pickle.dump(model,outfile)
    outfile.close()
    config.read(config_file)
    config.set('models', 'persistence_selected', filename_persistence)
    with open(config_file, 'w') as configfile:
        config.write(configfile)

    return 

def show_topic_model_description(model_selected):
    """Shows the topic's chemical description from the model selected 
       by the user in option 2 and all its submodels.
    """
    # Model path and model name
    route_to_model = config['models']['model_selected']
    model_name = config['models']['model_name']
    route_to_persistence = config['models']['persistence_selected']
    
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)

    
    ## 1. Check if a model has been trained already. Otherwise, its
    # chemical description can not be shown
    #models,model_nr,models_paths =  Model.list_models_names(route_to_model, False)
    models = []
    models_paths =  []
    model.print_model(models, models_paths, True, '---', False)

    if (models == []) or (not (models[-1].endswith("topics"))):
        print("Any model has been trained yet.")
        print("Go to option 2 in order to trained the model selected in option 1.")
        return []
    else:
        for i in np.arange(0,len(models),1):
            if models[i] == model_selected:
                model_selected_path = models_paths[i]
        #model_selected_path = models_paths[model_selected]
        file = pathlib.Path(model_selected_path,model_ids)
        
        if not(os.path.isfile((file).as_posix())):
            print("")
            print("The model " + '"' +  model_selected_path + '"' + " has not been trained yet.")
            print("Go to option 2 in order to train it if it is a model, and to option 3 in case it is a submodel.")
        else:
            ## 4. Print the description
            topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
            topic_ids = topics_ids_df.values[:,0].tolist()
            # print
        return topic_ids
    
def show_topics_to_expand(model_selected):
    # Path to the models in project folder, model path, persistence path 
    # and model name
    route_to_models = pathlib.Path(project_path, "models")
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']
    model_name = config['models']['model_name']
    
    ## 1. Load the model from the persitence file
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
    
    models = []
    models_paths =  []
    model.print_model(models, models_paths, True, '---', False)
    for i in np.arange(0,len(models),1):
        if models[i] == model_selected:
            model_selected_path = models_paths[i]
    file = pathlib.Path(model_selected_path, model_ids).as_posix()
    topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
    topic_ids = topics_ids_df.values[:,0].tolist()
   
    return topic_ids

def show_topics_to_expand_general(model_selected, model):
    models = []
    models_paths = []
    model.print_model(models, models_paths, True, '---', False)
    for i in np.arange(0, len(models), 1):
        if models[i] == model_selected:
            model_selected_path = models_paths[i]
    file = pathlib.Path(model_selected_path, model_ids).as_posix()
    topics_ids_df = pd.read_csv(file, sep="\t", header=None)
    topic_ids = topics_ids_df.values[:, 0].tolist()

    return topic_ids

def train_save_submodels(model_for_expansion, selected_topic, nr_topics, app, version, thr):

    # Path to the models in project folder, model path, persistence path 
    # and model name
    route_to_models = pathlib.Path(project_path, "models")
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']
    model_name = config['models']['model_name']
    
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
   
    # print(Fore.GREEN + "The model/submodel selected is: " + model_selected_name + Fore.WHITE)
    
    ## 4. Create subfiles
    
    # 4.3 Create the submodel files
    #time_rnd = time + "_" + str(randrange(100))
    
    if version == "v1":
        time_rnd = time + "_" + str(randrange(100)) + "_v1"
        submodels_paths, submodels_names = create_submodels([selected_topic], model_selected_path, time_rnd, version, model, thr)
        print("Generating submodels with HTM v1")
    elif version == "v2":
        time_rnd = time + "_" + str(randrange(100)) + "_v2_" + str(thr)
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
            file = pathlib.Path(submodels_paths[i],model_ids)
            topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
            topic_ids = topics_ids_df.values[:,0].tolist()
            
        num_docs_model = len(model.thetas)
        submodel.set_n_docs_father(num_docs_model)

        saving = True
        for i in np.arange(0,len(submodels_paths),1):
            if saving:
            # if button_reply_delete_model == QtWidgets.QMessageBox.Yes:
                print("entra")
                new_submodel_path = submodels_paths[i] + "_" + str(num_topics_all[i]) + "_topics"
                new_submodel_name = submodels_names[i] + "_" + str(num_topics_all[i]) + "_topics"
                app.new_submodel = new_submodel_name
                model.rename_child(submodels_names[i], new_submodel_name, new_submodel_path)
                os.rename(submodels_paths[i], new_submodel_path)
                print("")
                print("saved")
                #print(Fore.GREEN + "Submodel " + '"' + new_submodel_name + '"' + "was saved." + Fore.WHITE)
                print("")
            # 6.2.4 If the answer is no, we remove both the submodel's folder 
            # and the submodel object
            else:
                  rmtree(submodels_paths[i])
                  model.delete_child(submodels_names[i])
        
        outfile = open(route_to_persistence,'wb')
        pickle.dump(model,outfile)
        outfile.close()
            
    # return [submodels_paths, submodels_names , topic_ids, num_topics_all]
    return

def save_submodel(submodels_paths, submodels_names, num_topics_all, saving):
    # Path to the models in project folder, model path, persistence path 
    # and model name    
    route_to_persistence = config['models']['persistence_selected']
    
    ## 1. Load the model from the persitence file
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
    
    for i in np.arange(0,len(submodels_paths),1):
        if saving:
            new_submodel_path = submodels_paths[i] + "_" + str(num_topics_all[i]) + "_topics"
            new_submodel_name = submodels_names[i] + "_" + str(num_topics_all[i]) + "_topics"
            model.rename_child(submodels_names[i], new_submodel_name, new_submodel_path)
            print("")
            print(Fore.GREEN + "Submodel " + '"' + new_submodel_name + '"' + "was saved." + Fore.WHITE)
            print("")
        # 6.2.4 If the answer is no, we remove both the submodel's folder 
        # and the submodel object
        else:
              rmtree(submodels_paths[i])
              model.delete_child(submodels_names[i])
    ## The model is saved in the persistance file
    outfile = open(route_to_persistence,'wb')
    pickle.dump(model,outfile)
    outfile.close()
    
    return new_submodel_name

def change_description(model_selected, topic, description):
    # Path to the models in project folder, model path, persistence path 
    # and model name
    route_to_persistence = config['models']['persistence_selected']
    
    ## 1. Load the model from the persitence file
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
   
    # Only those models that are children from the model selected in option
    # 2, as well as such a model, are shown
    models_list = []
    models_paths = []
    model.print_model(models_list, models_paths, True, '---', False)
    
    for i in np.arange(0,len(models_list),1):
        if models_list[i] == model_selected:
            model_selected_path = models_paths[i]
            model_selected_name = models_list[i]  

    #print(Fore.GREEN + "The model/submodel selected is: " + model_selected_name + Fore.WHITE)
    
    # 3.3 Look for the submodel object within the model
    model_selected = model.look_for_model(model_selected_name)
    
  
    ## 5. 
    # 5.1 Add the description in the submodel and update the submodel 
    # object within the model object
   
    model_selected.set_one_topic_description(model_selected_path, model_ids, topic, description)
    
    model.update_submodel(model_selected)
    
    ## 6. Save the model in the persistance file
    outfile = open(route_to_persistence,'wb')
    pickle.dump(model,outfile)
    outfile.close()
    
    # New description
    file = pathlib.Path(model_selected_path, model_ids).as_posix()
    topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
    topic_ids = topics_ids_df.values[:,0].tolist()
    
    
    return topic_ids

def generatePyLavis(model_to_plot_str):
    route_to_persistence = config['models']['persistence_selected']

    ## 1. Load the model from the persitence file
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

    models = []
    models_paths = []
    model.print_model(models, models_paths, True, '---', False)

    if models == []:
        print("Any model has been trained yet.")
        print("Go to option 2 in order to trained the model selected in option 1.")
        return

    for i in np.arange(0, len(models), 1):
        if models[i] == model_to_plot_str:
            model_to_plot_path = models_paths[i]

    model_to_plot = model.look_for_model(model_to_plot_str)

    params = extract_params(os.path.join(model_to_plot_path, 'topic-state.gz'))
    alpha = [float(x) for x in params[0][1:]]
    beta = params[1]
    print("{}, {}".format(alpha, beta))

    df_lda = state_to_df(os.path.join(model_to_plot_path, 'topic-state.gz'))
    df_lda['type'] = df_lda.type.astype(str)
    df_lda[:10]

    # Get document lengths from statefile
    docs = df_lda.groupby('#doc')['type'].count().reset_index(name='doc_length')

    # Get vocab and term frequencies from statefile
    vocab = df_lda['type'].value_counts().reset_index()
    vocab.columns = ['type', 'term_freq']
    vocab = vocab.sort_values(by='type', ascending=True)

    phi_df = df_lda.groupby(['topic', 'type'])['type'].count().reset_index(name='token_count')
    phi_df = phi_df.sort_values(by='type', ascending=True)
    phi = pivot_and_smooth(phi_df, beta, 'topic', 'type', 'token_count')

    theta_df = df_lda.groupby(['#doc', 'topic'])['topic'].count().reset_index(name='topic_count')
    theta = pivot_and_smooth(theta_df, alpha, '#doc', 'topic', 'topic_count')

    data = {'topic_term_dists': phi,
            'doc_topic_dists': theta,
            'doc_lengths': list(docs['doc_length']),
            'vocab': list(vocab['type']),
            'term_frequency': list(vocab['term_freq'])
            }

    vis_data = pyLDAvis.prepare(**data)
    pyLDAvis.display(vis_data)

    file = pathlib.Path(model_to_plot_path, "pyLDAvis.html").as_posix()
    pyLDAvis.save_html(vis_data, file)

    return


def delete_model(model_to_delete):
    # Path to the models in project folder, model path and 
    # persitance path
    models_dir = pathlib.Path(project_path, "models")
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']
    
    rmtree(pathlib.Path(route_to_model))
    
    if model_to_delete.endswith("topics"):
        os.remove(route_to_persistence)
            
    return True


def delete_submodel(model_to_delete_str):
    deleted = False
    # Path to the models in project folder, model path and 
    # persitance path
    models_dir = pathlib.Path(project_path, "models")
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']
    
    ## 1. Load the model from the persitance file
    infile = open(route_to_persistence,'rb')
    model = pickle.load(infile)
    infile.close()
    
    ## 2. Check that at least one submodel has been trained
    models,model_nr,models_paths =  Model.list_models_names(route_to_model, False)
            
    if models == []:
        print("There are no submodels to delete.")
        return deleted
    else:
        ## 3. Find the submodel to delete
        ## Only the submodels children from model selected in option 2 are
        ## available for deleting
        models_list = []
        models_paths = []
        model.print_model(models_list, models_paths, False, '---', True)
        
        for i in np.arange(0,len(models_list),1):
            if models_list[i] == model_to_delete_str:
                to_delete_path = models_paths[i]
                to_delete_model = models_list[i]
     
        ## 4.1. Remove submodel's path in project folder
        rmtree(to_delete_path)
        ## 4.2. Remove submodel object from the root model object
        model.delete_child(to_delete_model)
        
        print("The model " + '"' +  to_delete_model + '"' + " was deleted.")
        
        ## 5. Save the model in the persistance file
        outfile = open(route_to_persistence,'wb')
        pickle.dump(model,outfile)
        outfile.close()
        deleted = True
       
        return deleted
    
def get_model_xml(path):
    ret = xml_dir(pathlib.Path(path))
    indent(ret)
    return ret

def configure_project_folder(path2project):
    path2project = pathlib.Path(path2project)
    print(path2project)
    models_dir = path2project / "models"
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    persistence_dir = path2project / "persistence"
    if not os.path.isdir(persistence_dir):
        os.mkdir(persistence_dir)

def progress_fn(n):
    print("%d%% done" % n)


def clearQTreeWidget(tree):
    iterator = QtWidgets.QTreeWidgetItemIterator(tree, QtWidgets.QTreeWidgetItemIterator.All)
    while iterator.value():
        iterator.value().takeChildren()
        iterator += 1
    i = tree.topLevelItemCount()
    while i > -1:
        tree.takeTopLevelItem(i)
        i -= 1


def printTree(xml_ret, treeWidget):
    treeWidget.setColumnCount(1)
    treeWidget.setHeaderHidden(True)
    a = QtWidgets.QTreeWidgetItem([xml_ret.tag])
    treeWidget.addTopLevelItem(a)

    def displayTree(a, s):
        for child in s:
            branch = QtWidgets.QTreeWidgetItem([child.tag])
            a.addChild(branch)
            print("branch", branch)
            displayTree(branch, child)
        if s.text is not None:
            content = s.text
            print("content", s.text)
            a.addChild(QtWidgets.QTreeWidgetItem([content]))

    displayTree(a, xml_ret)

def get_pickle(model_selected, project_path):
        project_path = pathlib.Path(project_path)
        models_dir = (project_path / "persistence").as_posix()
        with os.scandir(models_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    infile = open(entry.path, 'rb')
                    model = pickle.load(infile)
                    if model.model_name == model_selected:
                        return entry.path
                    else:
                        for i in np.arange(0, len(model.topics_models), 1):
                            if str(type(model.topics_models[i])) == "<class 'Model.Model'>":
                                if model.topics_models[i].model_name == model_selected:
                                    return entry.path


def get_root_path(model_selected, project_path):
    project_path = pathlib.Path(project_path)
    models_dir = (project_path / "persistence").as_posix()
    with os.scandir(models_dir) as entries:
        for entry in entries:
            if entry.is_file():
                infile = open(entry.path, 'rb')
                model = pickle.load(infile)
                if model.model_name == model_selected:
                    return model.model_path
                else:
                    for i in np.arange(0, len(model.topics_models), 1):
                        if str(type(model.topics_models[i])) == "<class 'Model.Model'>":
                            if model.topics_models[i].model_name == model_selected:
                                return model.model_path

def plot_diagnostics(list_diagnostics_id, measurement, measurement2, xaxis, yaxis, title, figure_to_save):
    # version = name.split("v2_")[1].split("_")[0]
    x = []
    y = []
    valueY = ""
    plt.figure()
    for el in list_diagnostics_id:
        tree = ET.parse(el[0])
        root = tree.getroot()
        model_name = el[1]
        topic_id = el[2]
        if measurement2 == "threshold":
            valueY = model_name.split("v2_")[1].split("_")[0]
        elif measurement2 == "topics":
            valueY = model_name.split("_")[-2]
        else:
            valueY = [child.get(measurement2) for child in root if child.tag == 'topic' and child.get('id') == topic_id][0]
        value_measurement = [child.get(measurement) for child in root if child.tag == 'topic' and child.get('id') == topic_id][0]
        x.append(float(valueY))
        y.append(float(value_measurement))

    plt.plot(x, y)
    plt.xlabel(xaxis), plt.ylabel(yaxis), plt.title(title)
    if figure_to_save:
        plt.savefig(figure_to_save)
    return x,y

def extract_params(statefile):
    """Extract the alpha and beta values from the statefile.

    Args:
        statefile (str): Path to statefile produced by MALLET.
    Returns:
        tuple: alpha (list), beta
    """
    with gzip.open(statefile, 'r') as state:
        params = [x.decode('utf8').strip() for x in state.readlines()[1:3]]
    return (list(params[0].split(":")[1].split(" ")), float(params[1].split(":")[1]))


def state_to_df(statefile):
    """Transform state file into pandas dataframe.
    The MALLET statefile is tab-separated, and the first two rows contain the alpha and beta hypterparamters.

    Args:
        statefile (str): Path to statefile produced by MALLET.
    Returns:
        datframe: topic assignment for each token in each document of the model
    """
    return pd.read_csv(statefile,
                       compression='gzip',
                       sep=' ',
                       skiprows=[1, 2]
                       )


def pivot_and_smooth(df, smooth_value, rows_variable, cols_variable, values_variable):
    """
    Turns the pandas dataframe into a data matrix.
    Args:
        df (dataframe): aggregated dataframe
        smooth_value (float): value to add to the matrix to account for the priors
        rows_variable (str): name of dataframe column to use as the rows in the matrix
        cols_variable (str): name of dataframe column to use as the columns in the matrix
        values_variable(str): name of the dataframe column to use as the values in the matrix
    Returns:
        dataframe: pandas matrix that has been normalized on the rows.
    """
    matrix = df.pivot(index=rows_variable, columns=cols_variable, values=values_variable).fillna(value=0)
    matrix = matrix.values + smooth_value

    normed = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)

    return pd.DataFrame(normed)