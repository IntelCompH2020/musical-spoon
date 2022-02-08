"""
Created on Sat Feb  6 11:50:02 2021

@author: lcalv
******************************************************************************
***                              AUX_MODEL                                 ***
******************************************************************************
Module that contains functionalities callable from gui.py, which consists of the
class governing the whole application management. With this, it is possible to
disengage the processing (i.e. models/submodels training and creation, pyLDAvis
generation, etc.) of the windows, widgets, and dialogs conforming and controlling.
Thus, this module contains a method for each of the functionalities provided.
Besides, it includes some auxiliary functions, such as methods for managing the
configuration of the project folder or the display of the hierarchical topics
models in a hierarchic way.
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
import configparser
import os
import pathlib
import pickle

import sys
import xml.etree.ElementTree as ET
from random import randrange
from shutil import rmtree
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
from colorama import Fore
from PyQt5 import QtWidgets
import sklearn

from htms.auxiliary_functions import indent, xml_dir
from htms.init_mallet import create_submodels, train_a_model, train_a_submodel
from htms.model import Model

##############################################################################
#                                CONFIG                                      #
##############################################################################
sys.setrecursionlimit(10**6)
time = strftime("_%Y-%m-%d_%H-%M-%S", gmtime())

config_file = os.path.dirname(__file__) + '/../config_project.ini'
config = configparser.ConfigParser()
config.read(config_file)

project_path = config['files']['project_path']
source_path = config['files']['source_path']
model_ids = config['out-documents']['model_ids']
mallet_path = config['mallet']['mallet_path']


##############################################################################
#                                FUNCTIONS                                   #
##############################################################################
def create_model():
    """Creates a model in the "models" folder within the project folder specified by the user, which is read from the configuration file.
    """
    # Path to the models in project folder
    models_dir = pathlib.Path(project_path, "models")

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
    """Lists all the models (root models, each root model referring to a Hierarchical topic model), and all the submodels under it.

    Returns:
    --------
        * List[str]: List of names, each name referring to a model / submodel.
    """
    # Reload config file just in case changes are found
    config.read(config_file)
    project_path = config['files']['project_path']
    models_dir = pathlib.Path(project_path, "models")
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
    """It writes in the configuration file's "selected_model", "model_name" and "persistence_selected" fields the details of the model given by the user.

    Args:
    -----
        * model_name (str): Name of the model selected by the user.
    """
    # Reload config file just in case changes are found
    config.read(config_file)
    project_path = config['files']['project_path']
    # Path to the models in project folder
    models_dir = pathlib.Path(project_path, "models")
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
    persis_name = models_name[model_selected_nr-1] + ".pickle"
    config.set('models', 'persistence_selected',  pathlib.Path(
        route_to_persistence, persis_name).as_posix())
    with open(config_file, 'w') as configfile:
        config.write(configfile)


def train_model(nr_topics):
    """It trains a root model via LDA Mallet.

    Args:
    -----
       * nr_topics (int): Number of topics to train the model with.
    """
    # 1. Get route to model, route to persistence and model's name
    config.read(config_file)
    route_to_model = config['models']['model_selected']
    route_to_persistence = pathlib.Path(project_path, "persistence")
    name = config['models']['model_name']

    # 2.1. Create model object
    model = Model("", "", 0, [], [], [], [], [], [], 0)

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
    name_persistance = name + "_" + \
        str(model.num_topics) + "_topics" + ".pickle"
    filename_persistence = pathlib.Path(
        route_to_persistence, name_persistance).as_posix()
    outfile = open(filename_persistence, 'wb')
    pickle.dump(model, outfile)
    outfile.close()
    config.read(config_file)
    config.set('models', 'persistence_selected', filename_persistence)
    with open(config_file, 'w') as configfile:
        config.write(configfile)

    return


def show_topic_model_description(model_selected):
    """Shows the topic's chemical description from the model selected 
       by the user in option 2 and all its submodels.

    Args:
    -----
       * model_selected (str): Name of the model selected by the user to show its topics' 
                               chemical description.

    Returns:
    --------
       * List[str]: List of strings, each string being of the form: 
                    "Topic XX - (topic description) - topic's chemical description". 
                    The size of the list is given by the number of topics with which the model selected was trained with.
    """
    route_to_persistence = config['models']['persistence_selected']
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)

    # Check if a model has been trained already. Otherwise, its
    # chemical description can not be shown
    models = []
    models_paths = []
    model.print_model(models, models_paths, True, '---', False)

    if (models == []) or (not (models[-1].endswith("topics"))):
        print("Any model has been trained yet.")
        print("Go to option 2 in order to trained the model selected in option 1.")
        return []
    else:
        for i in np.arange(0, len(models), 1):
            if models[i] == model_selected:
                model_selected_path = models_paths[i]
        file = pathlib.Path(model_selected_path, model_ids)

        if not(os.path.isfile((file).as_posix())):
            print("")
            print("The model " + '"' + model_selected_path +
                  '"' + " has not been trained yet.")
            print("Go to option 2 in order to train it if it is a model, and to option 3 in case it is a submodel.")
        else:
            # Print the description
            topics_ids_df = pd.read_csv(file, sep="\t", header=None)
            topic_ids = topics_ids_df.values[:, 0].tolist()
        return topic_ids


def show_topics_to_expand(model_selected):
    """From a model selected by the user, its shows all the topics that are available for expansion, as well as its chemical description.

    Args:
    -----
       * model_selected (str): Model selected by the user.

    Returns:
    --------
       * List[str]: List of strings, each string being of the form: 
                    "Topic XX - (topic description) - topic's chemical description". 
                    The size of the list is given by the number of topics with which the model selected was trained with.
    """
    # Load the model from the persistence file
    route_to_persistence = config['models']['persistence_selected']
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

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


def show_topics_to_expand_general(model_selected, model):
    """From a model selected by the user, which does not need to be a child of the model previously selected by the user in the train/select vie, its shows all the topics that are available for expansion, as well as its chemical description.

    Args:
    -----
       * model_selected (str): Model selected by the user.
       * model (Model): Model structure under which a model is going to be searched.

    Returns:
    --------
        * List[str]: List of strings, each string being of the form: 
                     "Topic XX - (topic description) - topic's chemical description". 
                     The size of the list is given by the number of topics with which the model selected was trained with.
    """
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
    """It trains a submodel with either HTM-WS or HTM-DS, depending on the user's choice.

    Args:
    -----
       *  model_for_expansion (str): Topic model that has been selected for expansion, i.e. 
                                     for creating a child submodel from it.
       * selected_topic (int):       Topic of the selected model ("model_for_expansion") that 
                                     has been selected to generate the submodel from.
       * nr_topics (int):            Number of topics to train the submodel with.
       * app (UI_MainWindow):        GUI
       * version (str):              Either "V1" or "V2", indicationg which HTM version is     
                                     going to be used for the training of the model.
       * thr (float):                Threshold for the generation of the submodel's reduced 
                                     corpus, in case version V2 (i.e. HTM-DS) has been selected by the user.
    """
    # 1. Load model from persistence file
    route_to_persistence = config['models']['persistence_selected']
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

    # 2.Select model/submodel for expansion
    models_list = []
    models_paths = []
    model.print_model(models_list, models_paths, True, '---', False)
    print(model_for_expansion)
    for i in np.arange(0, len(models_list), 1):
        if models_list[i] == model_for_expansion:
            print(models_list[i])
            model_selected_path = models_paths[i]
            model_selected_name = models_list[i]
    # 3. Create submodel files
    if version == "v1":
        time_rnd = time + "_" + str(randrange(100)) + "_v1"
        submodels_paths, submodels_names = create_submodels(
            [selected_topic], model_selected_path, time_rnd, version, model, thr)
        print("Generating submodels with HTM v1")
    elif version == "v2":
        time_rnd = time + "_" + str(randrange(100)) + "_v2_" + str(thr)
        submodels_paths, submodels_names = create_submodels(
            [selected_topic], model_selected_path, time_rnd, version, model, thr)
        print("Generating submodels with HTM v2")
    else:
        print("No HTM version has been given.")
        return

    # 4. Train submodels
    num_topics_all = []
    for i in np.arange(0, len(submodels_paths), 1):
        # 4.1 Create submodel object
        submodel = Model("", "", 0, [], [], [], [], [], [], 0)
        submodel.set_nr_topics(nr_topics)
        num_topics_all.append(submodel.num_topics)
        # 4.2 Train the submodel (create an object of type model)
        train_a_submodel(str(submodels_names[i]), str(
            submodels_paths[i]), submodel)
        # 4.3 Add the name of the submodel to the submodel object
        submodel.set_name(str(submodels_names[i]))
        submodel.set_path(str(submodels_paths[i]))
        # 4.3.1 Check if the model selected for expansion is directly the
        # model selected in option 2 (father model). If yes, the submodel
        # is directly append to the topics_models lists of the father model
        submodel.add_to_father(model_selected_name, model)
        submodel.set_fathers(model_selected_name, model)

        for i in np.arange(0, len(submodels_paths), 1):
            file = pathlib.Path(submodels_paths[i], model_ids)
            topics_ids_df = pd.read_csv(file, sep="\t", header=None)
            topic_ids = topics_ids_df.values[:, 0].tolist()

        num_docs_model = len(model.thetas)
        submodel.set_n_docs_father(num_docs_model)

        # 5. Save submodel
        saving = True
        for i in np.arange(0, len(submodels_paths), 1):
            if saving:
                new_submodel_path = submodels_paths[i] + \
                    "_" + str(num_topics_all[i]) + "_topics"
                new_submodel_name = submodels_names[i] + \
                    "_" + str(num_topics_all[i]) + "_topics"
                app.new_submodel = new_submodel_name
                model.rename_child(
                    submodels_names[i], new_submodel_name, new_submodel_path)
                os.rename(submodels_paths[i], new_submodel_path)
                print("")
                print("saved")
                print("")
            else:
                rmtree(submodels_paths[i])
                model.delete_child(submodels_names[i])

        outfile = open(route_to_persistence, 'wb')
        pickle.dump(model, outfile)
        outfile.close()

    return


def change_description(model_selected, topic, description):
    """Changes the description of a topic within a topic model by a string that is specified by the user.

    Args:
    -----
        * model_selected (str): Name of the model which is going to have the description of 
                                one of its topics changed.
        * topic (int):          Id of the topic whose description is going to be changed
        * description (str):    Description that is going to be given to the selected topic.

    Returns:
    --------
        * List[str]: Updated list of strings, each string being of the form: 
                   "Topic XX - (topic description) - topic's chemical description". 
                   The size of the list is given by the number of topics with which the model selected was trained with.
    """
    # 1. Load the model from the persitence file
    route_to_persistence = config['models']['persistence_selected']
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

    # Only those models that are children from the model selected in option
    # 2, as well as such a model, are shown
    models_list = []
    models_paths = []
    model.print_model(models_list, models_paths, True, '---', False)

    for i in np.arange(0, len(models_list), 1):
        if models_list[i] == model_selected:
            model_selected_path = models_paths[i]
            model_selected_name = models_list[i]

    # Look for the submodel object within the model
    model_selected = model.look_for_model(model_selected_name)

    # Add the description in the submodel and update the submodel
    # object within the model object
    model_selected.set_one_topic_description(
        model_selected_path, model_ids, topic, description)

    model.update_submodel(model_selected)

    # Save the model in the persistance file
    outfile = open(route_to_persistence, 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # New description
    file = pathlib.Path(model_selected_path, model_ids).as_posix()
    topics_ids_df = pd.read_csv(file, sep="\t", header=None)
    topic_ids = topics_ids_df.values[:, 0].tolist()

    return topic_ids


def generatePyLavis(model_to_plot_str):
    """It generate the PyLDAvis graph a model.

    Args:
       *  model_to_plot_str (str): Model to get the PyLDAvis representation of.
    """
    # 1. Load the model from the persitence file
    route_to_persistence = config['models']['persistence_selected']
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

    models = []
    models_paths = []
    model.print_model(models, models_paths, True, '---', False)

    if not models:
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
    docs = df_lda.groupby('#doc')['type'].count(
    ).reset_index(name='doc_length')

    # Get vocab and term frequencies from statefile
    vocab = df_lda['type'].value_counts().reset_index()
    vocab.columns = ['type', 'term_freq']
    vocab = vocab.sort_values(by='type', ascending=True)

    phi_df = df_lda.groupby(['topic', 'type'])[
        'type'].count().reset_index(name='token_count')
    phi_df = phi_df.sort_values(by='type', ascending=True)
    phi = pivot_and_smooth(phi_df, beta, 'topic', 'type', 'token_count')

    theta_df = df_lda.groupby(['#doc', 'topic'])[
        'topic'].count().reset_index(name='topic_count')
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
    """Deletes a model (i.e. root model). Since the root model is the main reference of the hierarchical topic model to which it belongs, all the submodels that are located under it are deleted as well. Hence, both the whole folder with name equals to "model_to_delete" and its respective persistence file are erased from the project folder.

    Args:
    -----
        * model_to_delete (str): Name of the model that the user wants to delete.

    Returns:
    --------
        * boolean: True indicating that the model has been removed.
    """
    # Model path and persitance path
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']

    rmtree(pathlib.Path(route_to_model))

    if model_to_delete.endswith("topics"):
        os.remove(route_to_persistence)

    return True


def delete_submodel(model_to_delete_str):
    """Deletes a submodel specified by the user and all the other submodels that are located under it in the hierarchy of the HTM.

    Args:
    -----
        * model_to_delete_str (str): Name of the submodel to be deleted.

    Returns:
    --------
        * boolean: True indicating that the submodel has been deleted.
    """
    deleted = False
    # Path to the models in project folder, model path and
    # persitance path
    models_dir = pathlib.Path(project_path, "models")
    route_to_model = config['models']['model_selected']
    route_to_persistence = config['models']['persistence_selected']

    # 1. Load the model from the persitance file
    infile = open(route_to_persistence, 'rb')
    model = pickle.load(infile)
    infile.close()

    # 2. Check that at least one submodel has been trained
    models, model_nr, models_paths = Model.list_models_names(
        route_to_model, False)

    if models == []:
        print("There are no submodels to delete.")
        return deleted
    else:
        # 3. Find the submodel to delete
        # Only the submodels children from model selected in option 2 are
        # available for deleting
        models_list = []
        models_paths = []
        model.print_model(models_list, models_paths, False, '---', True)

        for i in np.arange(0, len(models_list), 1):
            if models_list[i] == model_to_delete_str:
                to_delete_path = models_paths[i]
                to_delete_model = models_list[i]

        # 4.1. Remove submodel's path in project folder
        rmtree(to_delete_path)
        # 4.2. Remove submodel object from the root model object
        model.delete_child(to_delete_model)

        print("The model " + '"' + to_delete_model + '"' + " was deleted.")

        # 5. Save the model in the persistance file
        outfile = open(route_to_persistence, 'wb')
        pickle.dump(model, outfile)
        outfile.close()
        deleted = True

        return deleted


def get_model_xml(path):
    """Gets a XML ET.Element in which the hierarchical structure of all HTMs contained in the project folder.

    Args:
    -----
        * path (str): String referring to the path to the "models" directoy in the project 
                      folder.

    Returns:
    --------
        * ET.Element: Pretty printed XML ElementTree with the project folder's HTMs structure.
    """
    ret = xml_dir(pathlib.Path(path))
    indent(ret)
    return ret


def configure_project_folder(path2project):
    """It configures the project folder, that is, it creates the "models" and "persistence" directories within it.

    Args:
    -----
        * path2project (str): String referring to the path of the project folder.
    """
    path2project = pathlib.Path(path2project)
    print(path2project)
    models_dir = path2project / "models"
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    persistence_dir = path2project / "persistence"
    if not os.path.isdir(persistence_dir):
        os.mkdir(persistence_dir)


def progress_fn(n):
    """Funtion to print status.

    Args:
    -----
        * n (int): Status to print.
    """
    print("%d%% done" % n)


def clearQTreeWidget(tree):
    """Removes all the elements of a QTreeWidget.

    Args:
    -----
        * tree (QTreeWidget): QTreeWidget whose elements are desired to be removed.
    """
    iterator = QtWidgets.QTreeWidgetItemIterator(
        tree, QtWidgets.QTreeWidgetItemIterator.All)
    while iterator.value():
        iterator.value().takeChildren()
        iterator += 1
    i = tree.topLevelItemCount()
    while i > -1:
        tree.takeTopLevelItem(i)
        i -= 1


def printTree(xml_ret, treeWidget):
    """Displays the elements of a XML ET.Element into a QTreeWidget.

    Args:
    -----
        * xml_ret (ET.Element):     ET.Element whose elements are going to be displayed in the 
                                    QTreeWidget.
        * treeWidget (QTreeWidget): QTreeWidget object to visualize the items on.
    """
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
    """Gets the pickle in which a model or submodel object is saved.

    Args:
    -----
        * model_selected (str): Name of the model or submodel whose persistence file is     
                                desired to be acquired.
        * project_path (str):   Path to the project folder.

    Returns:
    --------
        * pathlib.Path: Path of the persistence file of the model / submodel.
    """
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
    """Gets the persistence file that is associated with a root model.

    Args:
    -----
        * model_selected (str): Name of the model whose persistence file is desired to be 
                                acquired.
        * project_path (str):   Path to the project folder.

    Returns:
    --------
        * pathlib.Path: Persistence file associated with the asked root model.
    """
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
    """Plots a diagnostics graph, i.e. depending on the measures that the user selects, the corresponding scores per topic are extracted from the XML file for each of the models located at the Selected models to plot table, and its average value is calculated and added to the graph. So as to render the figure into the application, the class FigureCanvasQTAgg56 from matplotlib, which is in charge of creating the rendering canvas and drawing the figure on it, is utilized.

    Args:
    -----
        * list_diagnostics_id (List[List]): List of lists of the form "diagnostics_path, 
                                            model_name, topic_id", referring to the path where the Mallet diagnostic file for the model with name "model_name" is located, the name of the topic model that is desired to be represented, and the topic from such a topic model that is desired to be included in the comparisson graph.
        * measurement (str):                Measurement on the X axis.
        * measurement2 (str):               Measurement on the Y axis.
        * xaxis (str):                      Label of the X axis.
        * yaxis (str):                      Label of the Y axis.
        * title (str):                      Title of the graph.
        * figure_to_save (boolean):         Boolean describing whether the user wants to save 
                                            the graph within the project folder.

    Returns:
    --------
        * List[float]: List of values for the X axis.
        * List[float]: List of values for the Y axis.
    """
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
            valueY = [child.get(measurement2) for child in root if child.tag ==
                      'topic' and child.get('id') == topic_id][0]
        value_measurement = [child.get(
            measurement) for child in root if child.tag == 'topic' and child.get('id') == topic_id][0]
        x.append(float(valueY))
        y.append(float(value_measurement))

    plt.plot(x, y)
    plt.xlabel(xaxis), plt.ylabel(yaxis), plt.title(title)
    if figure_to_save:
        plt.savefig(figure_to_save)
    return x, y


def extract_params(statefile):
    """Extract the alpha and beta values from the statefile.
    Taken from: http://dissertation.jeriwieringa.com/notebooks/6-Evaluate/pyldavis_and_mallet/

    Args:
    -----
        * statefile (str): Path to statefile produced by MALLET.

    Returns:
    --------
        * tuple: alpha (list), beta
    """
    with gzip.open(statefile, 'r') as state:
        params = [x.decode('utf8').strip() for x in state.readlines()[1:3]]
    return (list(params[0].split(":")[1].split(" ")), float(params[1].split(":")[1]))


def state_to_df(statefile):
    """Transform state file into pandas dataframe.
    The MALLET statefile is tab-separated, and the first two rows contain the alpha and beta hypterparamters.
    Taken from: http://dissertation.jeriwieringa.com/notebooks/6-Evaluate/pyldavis_and_mallet/

    Args:
    -----
        * statefile (str): Path to statefile produced by MALLET.
    Returns:
    --------
        * datframe: topic assignment for each token in each document of the model
    """
    return pd.read_csv(statefile,
                       compression='gzip',
                       sep=' ',
                       skiprows=[1, 2]
                       )


def pivot_and_smooth(df, smooth_value, rows_variable, cols_variable, values_variable):
    """
    Turns the pandas dataframe into a data matrix.
    Taken from: http://dissertation.jeriwieringa.com/notebooks/6-Evaluate/pyldavis_and_mallet/

    Args:
    -----
        * df (dataframe):       aggregated dataframe
        * smooth_value (float): value to add to the matrix to account for the priors
        * rows_variable (str):  name of dataframe column to use as the rows in the matrix
        * cols_variable (str):  name of dataframe column to use as the columns in the matrix
        * values_variable(str): name of the dataframe column to use as the values in the matrix

    Returns:
    --------
        * dataframe: pandas matrix that has been normalized on the rows.
    """
    matrix = df.pivot(index=rows_variable, columns=cols_variable,
                      values=values_variable).fillna(value=0)
    matrix = matrix.values + smooth_value

    normed = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)

    return pd.DataFrame(normed)
