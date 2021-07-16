# -*- coding: utf-8 -*-
"""
@author: Jesús Cid Sueiro
@modified_author: lcalv
******************************************************************************
***                             TASK MANAGER                               ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import sys 
import os
import re
import numpy as np
import pandas as pd
import pathlib
import configparser
from time import gmtime, strftime
from shutil import rmtree
import pickle
from colorama import init, Fore, Back, Style


from random import randrange

sys.path.append(os.path.abspath(".."))
sys.setrecursionlimit(10**6)

# You local imports
from .base_taskmanager import baseTaskManager
from Topic import Topic
from Model import Model
from init_mallet import train_a_model, train_a_submodel, create_submodels

config_file = 'config_project.ini'
config = configparser.ConfigParser()
config.read(config_file)

project_path = config['files']['project_path']
source_path = config['files']['source_path']
model_ids = config['out-documents']['model_ids']

time = strftime("_%Y-%m-%d_%H-%M-%S", gmtime())

class TaskManager(baseTaskManager):
    """
    This is an example of task manager that extends the functionality of
    the baseTaskManager class for a specific example application
    This class inherits from the baseTaskManager class, which provides the
    basic method to create, load and setup an application project.
    The behavior of this class might depend on the state of the project, in
    dictionary self.state, with the followin entries:
    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, path2source=None,
                 config_fname='parameters.yaml', metadata_fname='metadata.pkl',
                 set_logs=True):
        """
        Opens a task manager object.
        Parameters
        ----------
        path2project : str
            Path to the application project
        config_fname : str, optional (default='parameters.yaml')
            Name of the configuration file
        metadata_fname : str or None, optional (default=None)
            Name of the project metadata file.
            If None, no metadata file is used.
        set_logs : bool, optional (default=True)
            If True logger objects are created according to the parameters
            specified in the configuration file
        path2source : pathlib.Path
            Path to the source data.
        """
        super().__init__(path2project, config_fname=config_fname,
                         metadata_fname=metadata_fname, set_logs=set_logs)

        self.f_struct = {'models': 'models',
                         'persistence': 'persistence'}

        return 

    def select_model(self):
        """Ask the user whether he/she wants to use an old model for the
          training; otherwise, it is asked if he wants to create a new model.
          A new model being created means that a folder inside 
          ".../project/moldels" is created, being the folder name 
          "model_year-month-day_hour-min-secs". Once the model is trained in
          option 3, this folder will be renamed to 
          "model_year-month-day_hour-min-secs_XX_topics."
        """
        # Path to the models in project folder
        models_dir = pathlib.Path(project_path , "models")
        route_to_persistence = pathlib.Path(project_path, "persistence")
        # Available models 
        models = [model.name for model in models_dir.iterdir() if model.is_dir()]
        
        ## 1. If there are not old models to load, it is only possible to create
        # a new model
        if models == []:
            print("")
            print("There are no old models to load.")
            while True:
                try:
                    answer = str(input("Do you want to create a new model?" + Fore.GREEN + " [yes/no]" + Fore.WHITE + ": "))
                    if (answer == "yes") or (answer == "no"):
                        break
                    else:
                        print("You must insert 'yes' or 'no'.")
                except:
                    print("You must insert 'yes' or 'no'.")
            if answer == 'yes':
                model_name = "model" + time + "_" + str(randrange(100))
                if not(os.path.isdir((models_dir / model_name).as_posix())):
                    # 1.1. Create model's folder
                    model_selected = (models_dir / model_name).as_posix()
                    os.makedirs(model_selected)
                    print("")
                    print(Fore.GREEN + "Model " + '"' +  model_name + '"' + " created." + Fore.WHITE)
                    # 1.2. Save model's folder into config file
                    config.read(config_file)
                    config.set('models', 'model_selected', model_selected)
                    config.set('models', 'model_name', model_name)
                    with open(config_file, 'w') as configfile:
                        config.write(configfile)
            else:
                return
        ## 2. If there are old models to load
        else:
            while True:
                try:
                    answer = str(input("Do you want to create a new model or use and old one?" + Fore.GREEN + " [new/old]" + Fore.WHITE + ": "))
                    if (answer == "new") or (answer == "old"):
                        break
                    else:
                        print("You must insert 'new' or 'old'.")
                except:
                    print("You must insert 'new' or 'old'.")
            # 2.1. If the user wants to create a new model
            if answer == 'new':
                model_name = "model"+ time + "_" + str(randrange(100))
                if not(os.path.isdir((models_dir / model_name).as_posix())):
                    ## Create model's folder
                    model_selected = (models_dir / model_name).as_posix()
                    os.makedirs(model_selected)
                    print("")
                    print(Fore.GREEN + "Model " + '"' +  model_selected + '"' +  " created." + Fore.WHITE)
                    ## Save model's folder into config file
                    config.read(config_file)
                    config.set('models', 'model_selected', model_selected)
                    config.set('models', 'model_name', model_name)
                    with open(config_file, 'w') as configfile:
                        config.write(configfile)
            # 2.2. If the user wants to load an old model
            else:
                # Show the user which the available models are
                print("")
                print("The available models are listed below: ")
                models_list = []
                models_name = []
                model_nr = 0
                for model in models:
                    model_nr = model_nr + 1
                    print(Fore.GREEN + "["+ str(model_nr)+"] " + Fore.WHITE + model.__str__())
                    models_list.append(pathlib.Path(models_dir, model).as_posix())
                    models_name.append(model)
                # 2.2.1. Ask the user to select one
                while True:
                    try:
                       print("")
                       model_selected = int(input("Select the model you want to load: "))
                       if (model_selected <1) or (model_selected > len(models_list)):
                           print("You must insert a number between 1 and " + str(len(models_list)) + ".")
                       else:
                           break
                    except:
                        print("You must insert a number between 1 and " + str(len(models_list)) + ".")
                # 2.2.2. Save model's folder into config file
                config.read(config_file)
                config.set('models', 'model_selected', models_list[model_selected-1])
                config.set('models', 'model_name',  models_name[model_selected-1])
                persis_name =  models_name[model_selected-1] + ".pickle"
                config.set('models', 'persistence_selected',  pathlib.Path(route_to_persistence,persis_name).as_posix())
                with open(config_file, 'w') as configfile:
                    config.write(configfile)
                print("")
                print(Fore.GREEN + "Model " + '"' +  models_list[model_selected-1] + '"' + " loaded." + Fore.WHITE)
                return

    def train_model(self):
        """Asks the user how many topics want to use for creating the submodel,
        creates the mallet file and trains the model with the specified number
        of topics.
        Once the model is trained, the original model folder is renamed to 
        "model_year-month-day_hour-min-secs_XX_topics." and a folder with the 
        same name in ".../project/persistance" is created. In this folder, 
        the model object will be saved using pickle module.
        """
        # Path to the models in project folder
        models_dir = pathlib.Path(project_path, "models")
        # Available models 
        models = [model.name for model in models_dir.iterdir() if model.is_dir()]
        
        ## 1. Controlling that a model has been selected first.
        if models == []:
            print("No model has been selected yet.")
            print("Go to option 1 in order to select the model you want to train.")
        else:
            ## 2. Get route to model, route to persistence and model's name 
            config.read(config_file)
            route_to_model = config['models']['model_selected']
            route_to_persistence = pathlib.Path(project_path, "persistence")
            name  =  config['models']['model_name']
            
            # 2.1. Create model object
            model = Model("", "", 0, [],[], [], [], [],[])
            
            # 2.2 Ask the number of topics and control that is inside the range
            # available and set this number in the model object
            range_topics = 100
            model.ask_topics("Write the number of topics you want to use for training the model: ", range_topics, "Number of topics" )
            
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
            name_persistance = name+ "_" +  str(model.num_topics) + "_topics" + ".pickle"
            filename_persistence =  pathlib.Path(route_to_persistence,name_persistance).as_posix()
            outfile = open(filename_persistence,'wb')
            pickle.dump(model,outfile)
            outfile.close()
            config.read(config_file)
            config.set('models', 'persistence_selected', filename_persistence)
            with open(config_file, 'w') as configfile:
                config.write(configfile)
           
            print("")
            print(Fore.GREEN + "Model " + '"' +  new_name + '"' + " was trained with " + str(model.num_topics) + " topics." + Fore.WHITE)
        return 
 
    def train_submodels(self):
        """Trains one or more submodels that are made by expanding the model
           selected in option 2, or any other submodel, being such a submodel
           also a child of the model from 2.
           It asks the user from which model/submodel he/she wants to create
           the submodel, and shows the chemical description of the model/
           submodel selected. Then, it is asked from which topic he/she wants
           to create the submodel, and consequently, a folder for each submodel
           and a txt file containing all the words that belong for the topic 
           selected for that particular submodel is created.
           Once the submodels' folders, with their respective txt files have 
           beeen created, each submodel is trained.
           Lately, it will be shown to the user the chemical description of each
           of the submodels created and it will be asked whether he/she wants
           to save each of the submodels. If the answer is positive, the 
           submodel folder will be updated to contain the nr of topics that was
           used for the training and the object will be added to the list of
           topics_models of the correspondin model or submodel from which it 
           comes from.
        """
        # Availabe range of topics
        range_topics = 50
        
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
        
        # 2. Check if there is a model available to make the expansion from
        if os.listdir(route_to_models) == []:
            print("You must train the model first.")
            return
        
        ## 3. Select model/submodel for expansion
        print("Select the model/submodel from which you want to create a submodel.")
        print("Available models/submodels are listed below.")
        print("")
        
        # 3.1 Show all the models available in ".../project/persistance".
        # Only those models that are children from the model selected in option
        # 2, as well as such a model, are shown
        models_list = []
        models_paths = []
        model.print_model(models_list, models_paths, True, '---', True)
        
        # 3.2 Ask the user to selected the model/submodel 
        while True:
            try:
                print("")
                model_selected = int(input("Select the number referring to the model/submodel from which you want to create a submodel: "))
                model_selected_path = models_paths[model_selected]
                model_selected_name = models_list[model_selected]
                break
            except:
                print("You must insert a number between 0 and " + str(len(models_paths)-1) + ".")
        
        print("")
        print(Fore.GREEN + "The model/submodel selected is: " + model_selected_name + Fore.WHITE)
        
        ## 4. Create subfiles
        # 4.1 Show the chemical description of the available model's topics
        print("")
        print("The available topics and their chemical description are listed below: ")
        print("")
        file = pathlib.Path(model_selected_path, model_ids).as_posix()
        topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
        topic_ids = topics_ids_df.values[:,0].tolist()
        for i in np.arange(0, len(topic_ids),1):
            print(topic_ids[i])
    
        # 4.2 Ask the topics' ids
        num_models, ids = model.ask_ids(len(topic_ids))
        
        # 4.3 Create the submodel files
        time_rnd = time + "_" + str(randrange(100))
        submodels_paths, submodels_names = create_submodels(ids, model_selected_path,time_rnd)
        
        ## 5. Train submodels
        num_topics_all = []
        for i in np.arange(0,len(submodels_paths),1):
            # 5.1 Create submodel object
            submodel = Model("", "", 0, [], [], [], [], [], [])
            # 5.2 Ask the number of topics
            submodel.ask_topics("Write the number of topics you want to use for training the submodel " + '"' + submodels_names[i] + '": ', range_topics, "Number of topics" )
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

        ## 6. Ask if the trained submodels want to be saved
        for i in np.arange(0,len(submodels_paths),1):
            # 6.1 The chemical description of the submodels' topcis is shown
            print("The chemical description of the sumbmodel " + submodels_names[i] + " is shown below:")
            file = pathlib.Path(submodels_paths[i],model_ids)
            topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
            topic_ids = topics_ids_df.values[:,0].tolist()
            print("")
            for j in np.arange(0, len(topic_ids),1):
                print(topic_ids[j])
            print("")
            
            # 6.2 We ask the user whether he/she wants to save each submodel
            while True:
                try:
                    saving = str(input("Do you want to save the submodel " + submodels_names[i] + "? " + Fore.GREEN + "[yes/no]" + Fore.WHITE + ": "))
                    if (saving == "yes") or (saving == "no"):
                        break
                    else:
                        print("You must insert 'yes' or 'no'.")
                except:
                    print("You must insert 'yes' or 'no'.")
            
            # 6.2.3 If the answer is yes, we rename the submodel's folder so the 
            # number of topics is considered
            if saving == "yes":
                new_submodel_path = submodels_paths[i] + "_" + str(num_topics_all[i]) + "_topics"
                new_submodel_name = submodels_names[i] + "_" + str(num_topics_all[i]) + "_topics"
                os.rename(submodels_paths[i], new_submodel_path)
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
      
        return 
    
    def show_topic_model_description(self):
        """Shows the topic's chemical description from the model selected 
           by the user in option 2 and all its submodels.
        """
        # Model path and model name
        route_to_model = config['models']['model_selected']
        model_name = config['models']['model_name']
        route_to_persistence = config['models']['persistence_selected']
        
        infile = open(route_to_persistence,'rb')
        model = pickle.load(infile)
        infile.close()
        
        ## 1. Check if a model has been trained already. Otherwise, its
        # chemical description can not be shown
        models,model_nr,models_paths =  Model.list_models_names(route_to_model, False)
        models.append(model_name)
        
        if (models == []) or (not (models[-1].endswith("topics"))):
            print("Any model has been trained yet.")
            print("Go to option 2 in order to trained the model selected in option 1.")
            return
        else:
            ## 2. Show available model/submodels
            print("")
            print("Available model/submodels are listed below: ")
            models_list = []
            models_paths = []
            model.print_model(models_list, models_paths, True, '---', True)
            
            ## 3. Select model/submodel to show its description
            while True:
                try:
                    print("")
                    model_selected = int(input("Select the number referring to the model/submodel from which you want to see the description: "))
                    model_selected_path = models_paths[model_selected]
                    break
                except:
                    print("You must insert a number between 0 and " + str(len(models_paths)-1) + ".")
            
            file = pathlib.Path(model_selected_path,model_ids)
            
            if not(os.path.isfile((file).as_posix())):
                print("")
                print("The model " + '"' +  model_selected_path + '"' + " has not been trained yet.")
                print("Go to option 2 in order to train it if it is a model, and to option 3 in case it is a submodel.")
            else:
                ## 4. Print the description
                topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
                topic_ids = topics_ids_df.values[:,0].tolist()
                print("")
                for i in np.arange(0, len(topic_ids),1):
                    print(topic_ids[i])
        return
    
    def delete_submodel(self):
        """Deletes amy submodel selected by the user, which is a child of the 
           model selected in option 2.
        """
        # Path to the models in project folder, model path and 
        # persitance path
        models_dir = pathlib.Path(project_path, "models")
        route_to_model = config['models']['model_selected']
        route_to_persistence = config['models']['persistence_selected']
        
        ## 1. Check that a model has been trained
        try: 
            os.listdir(route_to_model) == []
        except:
            print("You must have trained a model first in order to delete it..")
            print("Go to option 1 to select a model and to option 2 to train it.")
            return            
        
        ## 2. Load the model from the persitance file
        infile = open(route_to_persistence,'rb')
        model = pickle.load(infile)
        infile.close()
        
        ## 3. Check that at least one submodel has been trained
        models,model_nr,models_paths =  Model.list_models_names(route_to_model, False)
                
        if models == []:
            print("There are no submodels to delete.")
            return
        else:
            ## 4. Show available submodels
            print("")
            print("Available submodels are listed below: ")
            
            ## Only the submodels children from model selected in option 2 are
            ## available for deleting
            models_list = []
            models_paths = []
            model.print_model(models_list, models_paths, False, '---', True)
    
            while True:
                try:
                    print("")
                    to_delete_nr = int(input("Insert the number referring to the submodel you want to delete: "))
                    to_delete_path = models_paths[to_delete_nr]
                    to_delete_model = models_list[to_delete_nr]
                    break
                except:
                    print("You must insert a number between 0 and " + str(len(models_list)-1) + ".")
            
            ## 5.1. Remove submodel's path in project folder
            rmtree(to_delete_path)
            ## 5.2. Remove submodel object from the root model object
            model.delete_child(to_delete_model)
            
            print("The model " + '"' +  to_delete_model + '"' + " was deleted.")
            
            ## 6. Save the model in the persistance file
            outfile = open(route_to_persistence,'wb')
            pickle.dump(model,outfile)
            outfile.close()
        return
    
    def set_topic_description(self):
        
        # Path to the models in project folder, model path, persistence path 
        # and model name
        route_to_models = pathlib.Path(project_path, "models")
        route_to_persistence = config['models']['persistence_selected']
        
        ## 1. Load the model from the persitence file
        infile = open(route_to_persistence,'rb')
        model = pickle.load(infile)
        infile.close()
        
        # 2. Check if there is a model available to make the expansion from
        if os.listdir(route_to_models) == []:
            print("You must train the model first.")
            return
        
        ## 3. Show all the models available in ".../project/persistance".
        print("Select the model/submodel to which you want to give a name to the topics.")
        print("Available models/submodels are listed below.")
        print("")
        
        # Only those models that are children from the model selected in option
        # 2, as well as such a model, are shown
        models_list = []
        models_paths = []
        model.print_model(models_list, models_paths, True, '---', True)
      
        
        # 3.2 Ask the user to selected the model/submodel 
        while True:
            try:
                print("")
                model_selected = int(input("Select the number referring to the model/submodel to which you want to give a name to the topics: "))
                model_selected_path = models_paths[model_selected]
                model_selected_name = models_list[model_selected]
                break
            except:
                print("You must insert a number between 0 and " + str(len(models_paths)-1) + ".")
        
        print("")
        print(Fore.GREEN + "The model/submodel selected is: " + model_selected_name + Fore.WHITE)
        
        # 3.3 Look for the submodel object within the model
        model_selected = model.look_for_model(model_selected_name)
        
        ## 4. 
        # 4.1 Show the chemical description of the available model's topics
        print("")
        print("The available topics and their chemical description are listed below: ")
        print("")
        file = pathlib.Path(model_selected_path, model_ids).as_posix()
        topics_ids_df = pd.read_csv(file, sep = "\t", header = None)
        topic_ids = topics_ids_df.values[:,0].tolist()
        for i in np.arange(0, len(topic_ids),1):
            print(topic_ids[i])
        
        ## 5. 
        # 5.1 Add the description in the submodel and update the submodel 
        # object within the model object
        model_selected.set_topic_description(model_selected_path, model_ids)
        model.update_submodel(model_selected)
        
        ## 6. Save the model in the persistance file
        outfile = open(route_to_persistence,'wb')
        pickle.dump(model,outfile)
        outfile.close()