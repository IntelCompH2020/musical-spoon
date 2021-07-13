# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS MODEL                                ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import pandas as pd
import numpy as np
import os
from shutil import rmtree
import pathlib
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import pickle

class Model():
    """
    Attributes:
    ----------
        * model_name
        * num_topics     = Number of topics with which the model is trained.
        * topics_models  = list of Topic or Model objects.
        * sizes          = Topics' sizes.
        * doc_ids        = Ids of the documents from which the model is built.
        * thetas         = Topic proportions of each document
                          (output-doc-topics).
        * dictionary     = List with all the words conforming the model.
        * father_models  = List containing the all the models from which
                           a submodel is expanded. In case it is the root model,
                           this list is empty
    """

    def __init__(self, model_name, model_path, num_topics, topics_models, sizes, doc_ids, thetas, dictionary,
                 father_models, n_docs_father):
        self.model_name = model_name
        self.model_path = model_path
        self.num_topics = num_topics
        self.topics_models = topics_models
        self.sizes = sizes
        self.doc_ids = doc_ids
        self.thetas = thetas
        self.dictionary = dictionary
        self.father_models = father_models
        self.n_docs_father = n_docs_father

    def set_after_trained_parameters(self, topics_models, dictionary, topic_keys_weight, file):
        """Set all parameters of a model, with the exception of the number of topics
        
        Parameters:
        ----------
            * topics_models     - List structure in which the topic objects, 
                                  or sub-model obejcts in case the model is
                                  extended, are saved.
            * dictionary        - List structure containig all the words that
                                  conform the model.
            * topic_keys_weight - List sturcture containing the size (weight)
                                  of each topic.
            * file              - File containing the mallet output
                                  "output-doc-topics".
        """
        doc_topics_df = pd.read_csv(file, sep="\t", header=None)
        doc_ids = doc_topics_df.values[:, 0].tolist()
        thetas = doc_topics_df.values[:, 2:].tolist()
        sizes = topic_keys_weight
        self.topics_models = topics_models
        self.sizes = sizes
        self.doc_ids = doc_ids
        self.thetas = thetas
        self.dictionary = dictionary
        return

    def set_topic_description(self, route_to_model, model_ids):
        print("write the name you want to give for each topic.")
        print("If there is a topic for which you do not want to give a description, press Enter.")
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Topic.Topic'>":
                while True:
                    try:
                        print("")
                        topic_name = str(input("Topic " + str(self.topics_models[i].id_topic) + ": "))
                        self.topics_models[i].set_description_name(topic_name)
                        break
                    except:
                        print("You must insert a word describing the topic.")
        # Create document "model_ids"
        topic_ids = []
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Topic.Topic'>":
                if self.topics_models[i].get_description_name() == "":
                    topic_ids.append("* TOPIC " + str(self.topics_models[i].get_topics()) + " -> " + str(
                        self.topics_models[i].get_description()))
                else:
                    topic_ids.append("* TOPIC " + str(self.topics_models[i].get_topics()) + ": '" + str(
                        self.topics_models[i].get_description_name()) + "'" + " -> " + str(
                        self.topics_models[i].get_description()))
            text_to_save = pathlib.Path(route_to_model, model_ids)
            np.savetxt(text_to_save, topic_ids, fmt='%s', encoding='utf-8')
        print("")

    def set_one_topic_description(self, route_to_model, model_ids, topic_id, topic_name):
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Topic.Topic'>":
                if self.topics_models[i].id_topic == topic_id:
                    self.topics_models[i].set_description_name(topic_name)
            # Create document "model_ids"
        topic_ids = []
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Topic.Topic'>":
                # reducir para que no haga el bloque completo
                if self.topics_models[i].get_description_name() == "":
                    topic_ids.append("* TOPIC " + str(self.topics_models[i].get_topics()) + " -> " + str(
                        self.topics_models[i].get_description()))
                else:
                    topic_ids.append("* TOPIC " + str(self.topics_models[i].get_topics()) + ": '" + str(
                        self.topics_models[i].get_description_name()) + "'" + " -> " + str(
                        self.topics_models[i].get_description()))
            text_to_save = pathlib.Path(route_to_model, model_ids)
            np.savetxt(text_to_save, topic_ids, fmt='%s', encoding='utf-8')
    
    def get_name(self):
        """Gets the name of the model
        
        Return:
        ----------
        * model_name      - Model name
        """
        return self.model_name

    def set_name(self, name):
        """Sets the name of the model
        
        Parameters:
        ----------
        * name      - Model name
        """
        self.model_name = name

    def get_path(self):
        """Gets the path of the model
        
        Return:
        ----------
        * model_path      - Model path
        """
        return self.model_path

    def set_path(self, path):
        """Sets the path of the model
        
        Parameters:
        ----------
        * path      - Model path
        """
        self.model_path = path

    def set_nr_topics(self, nr_topics):
        """Sets the nr of topics of the model
        
        Parameters:
        ----------
        * nr_topics      - Number of topics selected by the user
        """
        self.num_topics = nr_topics
        
    def get_n_docs_father(self):
        """Gets the number of docs of the father model.
        
        Return:
        ----------
        * n_docs_father      - Nr of docs of the father model
        """
        return self.n_docs_father

    def set_n_docs_father(self, n_docs_father):
        """Sets the number of docs of the father model.
        
        Parameters:
        ----------
        * n_docs_father      - Nr of docs of the father model
        """
        self.n_docs_father = n_docs_father

    def add_to_father(self, model_selected, model):
        """Given a submodel, look which is the model from which it comes from.
           When it is found, such a submodel is added to the topics_models 
           list of the found model
        
        Parameters:
        ----------
        * model_selected - Name of the model from which the submodel comes from
        * model          - Father model
        """
        if model.model_name == model_selected:
            model.topics_models.append(self)
            for i in model.topics_models:
                print(i)
            return
        else:
            for i in np.arange(0, len(model.topics_models), 1):
                if str(type(model.topics_models[i])) == "<class 'Model.Model'>":
                    if model.topics_models[i].model_name == model_selected:
                        model.topics_models[i].topics_models.append(self)
                        return
                    else:
                        self.add_to_father(model_selected, model.topics_models[i])
    
    def rename_child(self, old_name, new_name, new_path):
        """Given a submodel, look which is the model from which it comes from.
           When it is found, such a submodel is added to the topics_models 
           list of the found model
        
        Parameters:
        ----------
        * model_selected - Name of the model from which the submodel comes from
        * model          - Father model
        """
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                if self.topics_models[i].model_name == old_name:
                    self.topics_models[i].set_name(new_name)
                    self.topics_models[i].set_path(new_path)
                    return
                else:
                    self.topics_models[i].rename_child(old_name, new_name, new_path)

    def delete_child(self, model_to_delete):
        """Given a submodel, look which is the model from which it comes from.
           When it is found, such a submodel is deleted from the topics_models 
           list of the found model
        
        Parameters:
        ----------
        * model_to_delete - Name of the model to be deleted
        * model          - Father model
        """
        for i in np.arange(0, len(self.topics_models), 1):
            if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                if self.topics_models[i].model_name == model_to_delete:
                    self.topics_models.pop(i)
                    print("removed")
                    return
                else:
                    self.topics_models[i].delete_child(model_to_delete)
        return

    def set_fathers(self, father_model, model):
        """Given a submodel, it sets a list containing all of its father models.
        Parameters:
        ----------
        * father_model   - Name of the model from which the submodel is extended
        * model          - Father model
        """
        if father_model == model.model_name:
            self.father_models.append(model.model_name)
            return
        else:
            for i in np.arange(0, len(model.topics_models), 1):
                if str(type(model.topics_models[i])) == "<class 'Model.Model'>":
                    if model.topics_models[i].model_name == father_model:
                        self.father_models.append(model.topics_models[i].model_name)
                        if model.topics_models[i].father_models != []:
                            for i in model.topics_models[i].father_models:
                                self.father_models.append(i)
                    else:
                        self.set_fathers(father_model, model.topics_models[i])
        return

    def look_for_model(self, name):
        """It recursively loos for a model by name.
        Parameters:
        ----------
        * name - Name of the model to look for
        """

        if self.model_name == name:
            return self
        else:
            for i in np.arange(0, len(self.topics_models), 1):
                if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                    result = self.topics_models[i].look_for_model(name)
                    if result:
                        return result
            return None

    def update_submodel(self, new_submodel):
        """Given a route model, it updates one of its submodels.
        
      Parameters:
      ----------
      * new_submodel - Submodel object to update in the route model
      """
        if self.model_name == new_submodel.model_name:
            return
        else:
            for i in np.arange(0, len(self.topics_models), 1):
                if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                    if self.topics_models[i].model_name == new_submodel.model_name:
                        self.topics_models[i] = new_submodel
                    else:
                        self.topics_models[i].update_submodel(new_submodel)
        return

    def print_model(self, models_names, models_paths, route_model, ident, toPrint):
        """It recursively paints all the submodels' names that are bound to a
      model. When route_model is set to True, the model is printed as well.

      Parameters:
      ----------
      * models_names - List to save the names of the model/submodels
      * models_paths - List to save the paths of the model/submodels
      * route_model  - Boolean indicating whether the model is printed
      * ident        - Character for make the enumeration
      """
        if route_model:
            models_names.append(self.model_name)
            models_paths.append(self.model_path)
            if toPrint:
                print(Fore.GREEN + "[" + str(len(models_names) - 1) + "] " + Fore.WHITE + self.model_name)
            route_model = False
            self.print_model(models_names, models_paths, route_model, ident, toPrint)

        else:
            for i in np.arange(0, len(self.topics_models), 1):
                if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                    models_names.append(self.topics_models[i].model_name)
                    models_paths.append(self.topics_models[i].model_path)
                    model_ident = ident * len(self.topics_models[i].father_models)
                    if toPrint:
                        print(Fore.GREEN + model_ident + "[" + str(len(models_names) - 1) + "] " + Fore.WHITE +
                              self.topics_models[i].model_name)
                    self.topics_models[i].print_model(models_names, models_paths, route_model, ident, toPrint)
        if not toPrint:
            return models_names
        return

    def get_models_to_expand_in_gui(self, models_names, models_paths, route_model, ident):
        """It recursively paints all the submodels' names that are bound to a
      model. When route_model is set to True, the model is printed as well.

      Parameters:
      ----------
      * models_names - List to save the names of the model/submodels
      * models_paths - List to save the paths of the model/submodels
      * route_model  - Boolean indicating whether the model is printed
      * ident        - Character for make the enumeration
      """
        if route_model:
            models_names.append(self.model_name)
            models_paths.append(self.model_path)
            print(Fore.GREEN + "[" + str(len(models_names) - 1) + "] " + Fore.WHITE + self.model_name)
            route_model = False
            self.print_model(models_names, models_paths, route_model, ident)
        else:
            for i in np.arange(0, len(self.topics_models), 1):
                if str(type(self.topics_models[i])) == "<class 'Model.Model'>":
                    models_names.append(self.topics_models[i].model_name)
                    models_paths.append(self.topics_models[i].model_path)
                    model_ident = ident * len(self.topics_models[i].father_models)
                    print(Fore.GREEN + model_ident + "[" + str(len(models_names) - 1) + "] " + Fore.WHITE +
                          self.topics_models[i].model_name)
                    self.topics_models[i].print_model(models_names, models_paths, route_model, ident)
        return

    def ask_topics(self, msg, range_int, num_topics):
        """Ask for the a value and controls that it is in the allow range.
        
        Parameters:
        ----------
        * msg       - String message that is gonna be asked for inserting an input.
        * range_int - Range in which the asked input can be .
        * variable  - Variable whose value is being asked.
        
        """
        while True:
            try:
                num = int(input(msg))
                if (num < 1) or (num > range_int):
                    print(
                        "Invalid number. The " + num_topics + " must be a number between 1 and " + str(range_int) + ".")
                else:
                    self.num_topics = num
                    return
            except:
                print("You must insert a number. The " + num_topics + " must be a number between 1 and " + str(
                    range_int) + ".")

    def ask_ids(self, range_ids):
        """ Asks the number and ids of the topics from which the submodels are 
            going to be created in create_submodels (task_manager).
         
        Parameters:
        ----------
        * range_ids - Id range of the submodels.
    
        Returns
        -------
        * num_models   - Number of models created.
        * topics_ids   - The selected topics' ids for creating each corresponding
                         submodel.
        """
        while True:
            try:
                print("")
                num_models = int(input("How many submodels do you want to create? "))
                if (num_models < 1) or (num_models > range_ids):
                    print("Invalid number. The number of submodels must be between 1 and " + str(range_ids) + ".")
                else:
                    break
            except:
                print("You must insert a number. The number of submodels must be a number between 1 and " + str(
                    range_ids) + ".")
        topic_ids = []
        for i in np.arange(0, num_models, 1):
            while True:
                try:
                    print("")
                    id = int(input("Select the topic id of the model from which you want to create the submodel " + str(
                        i + 1) + ": "))
                    if (range_ids < 0) or (id > range_ids - 1):
                        print("Invalid number. The topic id must be a number between 0 and " + str(range_ids - 1) + ".")
                    else:
                        topic_ids.append(id)
                        break
                except:
                    print("Invalid number. The topic id must be a number between 0 and " + str(range_ids - 1) + ".")
        print("")
        return num_models, topic_ids

    @staticmethod
    def list_models_names(route, printing):
        """Given a route, it recursively founds all the model/submodels
           contained in it.
           When printing is set to True, it prints the folder names of
           the model/submodels found.

        Parameters:
        ----------
        * route     - Route to the model/submodels
        * printing  - Boolean indicating whether the models are printed or not
        """
        models_list = []
        models_paths = []
        model_nr = 0
        for nombre_directorio, dirs, ficheros in os.walk(route):
            for dir_ in dirs:
                if printing:
                    print(Fore.GREEN + "[" + str(model_nr) + "] " + Fore.WHITE + dir_)
                model_nr = model_nr + 1
                models_list.append(dir_)
                model_dir = os.path.join(nombre_directorio, dir_)
                models_paths.append(model_dir)
                Model.list_models_names(dir_, printing)
        return models_list, model_nr, models_paths
