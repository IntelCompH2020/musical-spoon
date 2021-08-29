# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:56:19 2021

@author: lcalv
"""

import configparser
import os
import pathlib
import pickle
import sys
import numpy as np
import matplotlib
import threading
import time
import traceback
import IPython

from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, QUrl, QTimer, Qt, QObject, QRunnable, QThreadPool
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtGui import QIcon, QMovie, QPixmap, QPainter

from Model import *
from aux_model import create_model, list_models, select_model, train_model, show_topic_model_description, \
    show_topics_to_expand, train_save_submodels, save_submodel, change_description, generatePyLavis, \
    delete_submodel, delete_model, get_model_xml
from Worker import Worker
from MessagesGui import MessagesGui

config_file = 'config_project.ini'
config = configparser.ConfigParser()
config.read(config_file)

project_path = config['files']['project_path']
source_path = config['files']['source_path']
model_ids = config['out-documents']['model_ids']
pldavis = config['out-documents']['pldavis']
default_project_path = config['default']['project_path']



class UI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI_MainWindow, self).__init__()

        uic.loadUi("UIS/musicalSpoonV2.ui", self)
        self.setGeometry(100, 60, 2000, 1600)

        self.centralwidget.setGeometry(100, 60, 2000, 1600)

        # self.setStyleSheet(stylesheet)

        if not pathlib.Path(project_path, "models").is_dir():
            self.configure_project_folder(project_path)

        # INFORMATION BUTTONS
        self.infoButtonSelectDataset.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectDataset.setToolTip(MessagesGui.INFO_SELECT_DATASET)

        self.infoButtonLoadFiles.setIcon(QIcon('Images/help2.png'))
        self.infoButtonLoadFiles.setToolTip(MessagesGui.INFO_LOAD_FILES)

        self.infoButtonTrainModel.setIcon(QIcon('Images/help2.png'))
        self.infoButtonTrainModel.setToolTip(MessagesGui.INFO_TRAIN_MODEL)

        self.infoButtonSelectSubmodel.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectSubmodel.setToolTip(MessagesGui.INFO_SELECT_SUBMODEL)

        self.infoButtoSelectTopicExpand.setIcon(QIcon('Images/help2.png'))
        self.infoButtoSelectTopicExpand.setToolTip(MessagesGui.INFO_SELECT_TOPIC_TO_EXPAND)

        self.infoButtonShowDescription.setIcon(QIcon('Images/help2.png'))
        self.infoButtonShowDescription.setToolTip(MessagesGui.INFO_SHOW_DESCRIPTION)

        # CONFIGURATION
        self.treeViewSelectDataset.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeViewSelectDataset.customContextMenuRequested.connect(self.context_menu)
        self.treeViewSelectDataset.doubleClicked.connect(self.clicked_select_dataset)
        self.modelTreeView = QtWidgets.QFileSystemModel()
        self.modelTreeView.setRootPath((QtCore.QDir.rootPath()))

        self.pushButtonSelectDataset.setIcon(QIcon('Images/folder.png'))
        self.pushButtonSelectDataset.clicked.connect(self.show_datasets)

        self.tableWidgetGeneralSettings.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetGeneralSettings.resizeColumnsToContents()
        self.tableWidgetGeneralSettings.setRowCount(2)
        self.tableWidgetGeneralSettings.setColumnCount(2)

        self.tableWidgetMalletSettings.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetMalletSettings.resizeColumnsToContents()
        self.tableWidgetMalletSettings.setRowCount(4)
        self.tableWidgetMalletSettings.setColumnCount(2)

        self.pushButtonApplySettings.clicked.connect(self.apply_changes_settings)

        self.pushButtonResetSettings.clicked.connect(self.set_default_settings)

        self.tableWidgetOutputFiles.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetOutputFiles.resizeColumnsToContents()
        self.tableWidgetOutputFiles.setRowCount(5)
        self.tableWidgetOutputFiles.setColumnCount(2)

        self.tableWidgetModelFiles.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetModelFiles.resizeColumnsToContents()
        self.tableWidgetModelFiles.setRowCount(2)
        self.tableWidgetModelFiles.setColumnCount(2)
        self.tableWidgetModelFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(str("")))
        self.tableWidgetModelFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(str("")))

        # SELECT MODEL / TRAIN MODEL
        self.column_listSelectModel.clicked.connect(self.clicked_selected_model)
        self.outputSelectedModel.setStyleSheet('color:green')
        self.CreateNewModel.clicked.connect(self.clicked_create_model)
        self.RemoveModel.clicked.connect(self.clicked_delete_model)

        # TRAIN
        self.inputNrTopics = self.findChild(QtWidgets.QComboBox, "InsertNumberTopicsModel")

        self.TrainModelButton.clicked.connect(self.clicked_train_model)

        self.plainTextTrainModel = self.findChild(QtWidgets.QPlainTextEdit, "plainTextTrainModel")

        # TRAIN SUBMODEL
        self.treeViewShowModelsToExpand.clicked.connect(self.clicked_model_to_expand_selected)

        self.TrainSubmodel_2.clicked.connect(self.clicked_train_submodel)

        self.tableWidgetTrainSubmodel.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetTrainSubmodel.resizeColumnsToContents()

        # SHOW DESCRIPTION
        self.treeWidgetSelectModelToSeeDescription.clicked.connect(self.clicked_select_model_to_see_description)

        self.plainTextShow = self.findChild(QtWidgets.QPlainTextEdit, "plainTextShow")

        self.pushButtonSeeDescription.clicked.connect(self.clicked_see_topics_description)

        # PYLDAVIS
        self.layoutPlot = self.findChild(QtWidgets.QVBoxLayout, "layoutPlot")

        self.pushButtonPlotPyLDAvis.clicked.connect(self.clicked_plot_pyldavis)

        # INSERT TOPIC'S NAME
        self.treeViewShowModelsToExpand.clicked.connect(self.clicked_model_to_change_description)

        self.ShowButton.clicked.connect(self.clicked_apply_changes)

        self.pushButtonResetDescriptions.clicked.connect(self.clicked_reset_changes)

        # TABLE TOPICS INSERT
        self.tableWidgetTrainSubmodel.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetTrainSubmodel.resizeColumnsToContents()

        # DELETE SUBMODEL
        self.treeViewShowModelsToExpand.clicked.connect(self.clicked_selected_model_to_delete)

        self.pushButtonToDeleteSubmodel.clicked.connect(self.clicked_delete_submodel)

        # MODEL PARAMETERS THAT ARE SAVED TO PASS TO THREADS
        self.num_training_topics = 5
        self.model_to_expand = ""
        self.topic_to_expand = 5
        self.model_to_plot = ""
        self.new_submodel = ""
        self.model_to_train = ""
        self.web = None

        self.initialize_settings()
        self.show_models()
        self.show()

        # THREADS FOR EXECECUTING PARALELLY TRAIN MODELS, TRAIN SUBMODELS, AND PYLDAVIS
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.worker = None
        self.loading_window = None

        self.Btn_Toggle.clicked.connect(lambda: self.toggleMenu(250, True))
        self.Btn_Toggle.setIcon(QIcon('Images/menu.png'))

        ## PAGES
        ########################################################################

        # PAGE 1
        self.pushButtonConfiguration_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage))
        # self.pushButtonConfiguration_2.clicked.connect(lambda: self.label_title_bar_top.setText("|CONFIGURATION"))
        self.pushButtonConfiguration_2.setIcon(QIcon('Images/settings.png'))

        # PAGE 2
        self.pushButtonSelectModel_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage2))
        # self.pushButtonSelectModel_2.clicked.connect(lambda: self.label_title_bar_top.setText("|SELECT MODEL"))
        self.pushButtonSelectModel_2.setIcon(QIcon('Images/new.png'))

        # PAGE 3
        self.pushButtonTrainSubmodel_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage3))
        # self.pushButtonTrainSubmodel_2.clicked.connect(lambda: self.label_title_bar_top.setText("|SHOW MODELS"))
        self.pushButtonTrainSubmodel_2.setIcon(QIcon('Images/create.png'))

        # PAGE 4
        self.pushButtonShowDescription_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage4))
        # self.pushButtonShowDescription_2.clicked.connect(lambda: self.label_title_bar_top.setText("|EDIT MODEL"))
        self.pushButtonShowDescription_2.setIcon(QIcon('Images/show.png'))

    def toggleMenu(self, maxWidth, enable):
        if enable:
            # GET WIDTH
            width = self.frame_left_menu.width()
            maxExtend = maxWidth
            standard = 70

            # SET MAX WIDTH
            if width == 70:
                widthExtended = maxExtend
                # SHOW TEXT INSTEAD OF ICON
                self.pushButtonConfiguration_2.setText(' Configuration')
                self.pushButtonSelectModel_2.setText('Select model')
                self.pushButtonShowDescription_2.setText('Show model')
                self.pushButtonTrainSubmodel_2.setText('Edit model')

            else:
                widthExtended = standard
                self.pushButtonConfiguration_2.setText('')
                self.pushButtonSelectModel_2.setText('')
                self.pushButtonShowDescription_2.setText('')
                self.pushButtonTrainSubmodel_2.setText('')

            # ANIMATION
            self.animation = QtCore.QPropertyAnimation(self.frame_left_menu, b"minimumWidth")
            self.animation.setDuration(400)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
            self.animation.start()

    @QtCore.pyqtSlot()
    def start_animation(self):
        print("signal started")
        self.loading_window = QtWidgets.QDialog()
        # @ TODO : hacer modal
        self.loading_window.setWindowFlags(QtCore.Qt.SplashScreen)
        self.loading_window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.loading_window.setStyleSheet("background-color:#DFE6ED")
        # self.loading_window.exec_()  # blocks all other windows until this window is closed.

        movie = QtGui.QMovie('Images/ZC9Y.gif', cacheMode=QtGui.QMovie.CacheAll)

        movie_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        movie_label.setStyleSheet("border: 0px;")
        movie_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        movie_label.setFixedSize(500, 500)
        movie_label.setMovie(movie)
        movie_label.setStyleSheet("background-color:#DFE6ED")

        vbox = QtWidgets.QVBoxLayout(self.loading_window)
        vbox.addWidget(movie_label)
        self.worker.signals.finished.connect(self.loading_window.close)
        self.loading_window.show()

        movie.start()

    def progress_fn(self, n):
        print("%d%% done" % n)

    def execute_to_train_model(self, progress_callback):
        train_model(self.num_training_topics)
        return "Done."

    def execute_to_train_submodel(self, progress_callback):
        train_save_submodels(self.model_to_expand, self.topic_to_expand, self.num_training_topics, self)
        return "Done."

    def execute_to_get_pyldavis(self, progress_callback):
        generatePyLavis(self.model_to_plot)

    def execute_apply_changes(self, progress_callback):
        item = self.treeViewShowModelsToExpand.currentItem()
        print(self.tableWidgetTrainSubmodel.rowCount())
        for i in np.arange(0, self.tableWidgetTrainSubmodel.rowCount(), 1):
            new_name = self.tableWidgetTrainSubmodel.item(i, 1).text()
            print(new_name)
            change_description(item.text(0), i, new_name)

    def execute_reset_changes(self, progress_callback):
        item = self.treeViewShowModelsToExpand.currentItem()
        for i in np.arange(0, self.tableWidgetTrainSubmodel.rowCount(), 1):
            change_description(str(item.text(0)), i, "")
            item_topic_name = QtWidgets.QTableWidgetItem("")
            self.tableWidgetTrainSubmodel.setItem(i, 1, item_topic_name)

    def execute_in_thread(self, function, function_output):
        # Pass the function to execute
        self.worker = Worker(function)  # Any other args, kwargs are passed to the run function
        self.worker.signals.started.connect(self.start_animation)
        self.worker.signals.finished.connect(function_output)

        # Execute
        self.threadpool.start(self.worker)

    def set_default_model_parameters(self):
        self.num_training_topics = 0
        self.model_to_expand = ""
        self.topic_to_expand = 0
        self.model_to_plot = ""
        self.new_submodel = ""
        self.model_to_train = ""

    def initialize_settings(self):
        config.read(config_file)
        first_menu_source_path = config['default']['source_path']
        first_menu_project_path = config['default']['project_path']
        first_menu_mallet_path = config['default']['mallet_path']

        self.tableWidgetGeneralSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(first_menu_source_path)))
        self.tableWidgetGeneralSettings.setItem(1, 1, QtWidgets.QTableWidgetItem(str(first_menu_project_path)))
        self.tableWidgetMalletSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(first_menu_mallet_path)))

        return

    def set_default_settings(self):
        # Project settings
        source_path = config['default']['source_path']
        project_path = config['default']['project_path']

        # Mallet settings
        mallet_path = config['default']['mallet_path']
        optimize_interval = config['default']['optimize_interval']
        model_mallet = config['default']['model_mallet']
        submodel_mallet = config['default']['submodel_mallet']

        # Output documents
        topic_state = config['default']['topic_state']
        topic_keys = config['default']['topic_keys']
        doc_topics = config['default']['doc_topics']
        topic_word_weights = config['default']['topic_word_weights']
        model_ids = config['default']['model_ids']

        self.tableWidgetGeneralSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(source_path)))
        self.tableWidgetGeneralSettings.setItem(1, 1, QtWidgets.QTableWidgetItem(str(project_path)))

        self.tableWidgetMalletSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(mallet_path)))
        self.tableWidgetMalletSettings.setItem(1, 1, QtWidgets.QTableWidgetItem(str(optimize_interval)))
        self.tableWidgetMalletSettings.setItem(2, 1, QtWidgets.QTableWidgetItem(str(model_mallet)))
        self.tableWidgetMalletSettings.setItem(3, 1, QtWidgets.QTableWidgetItem(str(submodel_mallet)))

        self.tableWidgetOutputFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(str(topic_state)))
        self.tableWidgetOutputFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(str(topic_keys)))
        self.tableWidgetOutputFiles.setItem(2, 1, QtWidgets.QTableWidgetItem(str(doc_topics)))
        self.tableWidgetOutputFiles.setItem(3, 1, QtWidgets.QTableWidgetItem(str(topic_word_weights)))
        self.tableWidgetOutputFiles.setItem(4, 1, QtWidgets.QTableWidgetItem(str(model_ids)))

        config.read(config_file)
        config.set('files', 'source_path', source_path)
        config.set('files', 'project_path', project_path)
        config.set('mallet', 'mallet_path', mallet_path)
        config.set('mallet', 'optimize_interval', optimize_interval)
        config.set('mallet', 'model_mallet', model_mallet)
        config.set('mallet', 'submodel_mallet', submodel_mallet)
        config.set('out-documents', 'topic_state', topic_state)
        config.set('out-documents', 'topic_keys', topic_keys)
        config.set('out-documents', 'doc_topics', doc_topics)
        config.set('out-documents', 'topic_word_weights', topic_word_weights)
        config.set('out-documents', 'model_ids', model_ids)

        with open(config_file, 'w') as configfile:
            config.write(configfile)

        self.lineEditOutputSelectDataset.setStyleSheet('color:green')
        self.lineEditOutputSelectDataset.setText("All values were restore to the default settings.")

        self.show_models()
        self.refresh()

        return

    def apply_changes_settings(self):
        new_source_path = self.tableWidgetGeneralSettings.item(0, 1).text()
        new_project_path = self.tableWidgetGeneralSettings.item(1, 1).text()

        # Mallet settings
        new_mallet_path = self.tableWidgetMalletSettings.item(0, 1).text()
        new_optimize_interval = self.tableWidgetMalletSettings.item(1, 1).text()
        new_model_mallet = self.tableWidgetMalletSettings.item(2, 1).text()
        new_submodel_mallet = self.tableWidgetMalletSettings.item(3, 1).text()

        # Output documents
        new_topic_state = self.tableWidgetOutputFiles.item(0, 1).text()
        new_topic_keys = self.tableWidgetOutputFiles.item(1, 1).text()
        new_doc_topics = self.tableWidgetOutputFiles.item(2, 1).text()
        new_topic_word_weights = self.tableWidgetOutputFiles.item(3, 1).text()
        new_model_ids = self.tableWidgetOutputFiles.item(4, 1).text()

        config.read(config_file)
        config.set('files', 'source_path', new_source_path)
        config.set('files', 'project_path', new_project_path)
        config.set('mallet', 'mallet_path', new_mallet_path)
        config.set('mallet', 'optimize_interval', new_optimize_interval)
        config.set('mallet', 'model_mallet', new_model_mallet)
        config.set('mallet', 'submodel_mallet', new_submodel_mallet)
        config.set('out-documents', 'topic_state', new_topic_state)
        config.set('out-documents', 'topic_keys', new_topic_keys)
        config.set('out-documents', 'doc_topics', new_doc_topics)
        config.set('out-documents', 'topic_word_weights', new_topic_word_weights)
        config.set('out-documents', 'model_ids', new_model_ids)

        with open(config_file, 'w') as configfile:
            config.write(configfile)

        # Update project folder
        self.configure_project_folder(new_project_path)
        if default_project_path != new_project_path:
            config.read(config_file)
            config.set('models', 'model_selected', " ")
            config.set('models', 'persistence_selected', " ")
            config.set('models', 'model_name', " ")
            with open(config_file, 'w') as configfile:
                config.write(configfile)
            self.tableWidgetModelFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
            self.tableWidgetModelFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(""))

        self.show_models()
        self.refresh()

        # self.lineEditOutputSelectDataset.setStyleSheet('color:green')
        self.lineEditOutputSelectDataset.setText("Changes were saved in the configuration file.")  # add mess

        return

    def configure_project_folder(self, path2project):
        path2project = pathlib.Path(path2project)

        # ######################
        # Project file structure:
        # Default file and folder names for the folder
        # structure of the project.
        f_struct = {'models': 'models',
                    'persistence': 'persistence'}

        if f_struct is not None:
            f_struct.update(f_struct)

        print("FSTRCUT: fstruct created")

        # In the following, we assume that all files in self.f_struct are
        # subfolders of self.path2project
        for d in f_struct:
            path2d = path2project / f_struct[d]
            if not path2d.exists():
                path2d.mkdir()

    def show_datasets(self):
        config.read(config_file)
        path = pathlib.Path(pathlib.Path("\mallet").parent.absolute(), "mallet").as_posix()
        self.treeViewSelectDataset.setModel(self.modelTreeView)
        self.treeViewSelectDataset.setRootIndex(self.modelTreeView.index(path))
        self.treeViewSelectDataset.setSortingEnabled(True)
        return

    def context_menu(self):
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open file")
        open.triggered.connect(self.open_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())
        return

    def open_file(self):
        index = self.treeViewSelectDataset.currentIndex()
        file_path = self.modelTreeView.filePath(index)
        os.startfile(file_path)

    def clicked_select_dataset(self):
        index = self.treeViewSelectDataset.currentIndex()
        dataset_selected_path = self.modelTreeView.filePath(index)
        self.lineEditOutputSelectDataset.setStyleSheet('color:green')
        self.lineEditOutputSelectDataset.setText("'" +
                                                 str(dataset_selected_path) + "' was selected as dataset.")
        button_info_select_dataset = QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                                                       "'" + str(dataset_selected_path) + "' was "
                                                                                                          "selected as dataset.")
        self.tableWidgetGeneralSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(dataset_selected_path)))

    def show_models(self):
        if pathlib.Path(project_path, "models").is_dir():
            self.column_listSelectModel.clear()
            models = list_models()
            if models:
                for model_nr in np.arange(0, len(models), 1):
                    self.column_listSelectModel.insertItem(model_nr, models[model_nr])

        if self.web:
            self.web.setParent(None)

    def printTree(self, xml_ret, treeWidget):
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

    def clearQTreeWidget(self, tree):
        iterator = QtWidgets.QTreeWidgetItemIterator(tree, QtWidgets.QTreeWidgetItemIterator.All)
        while iterator.value():
            iterator.value().takeChildren()
            iterator += 1
        i = tree.topLevelItemCount()
        while i > -1:
            tree.takeTopLevelItem(i)
            i -= 1

    def show_models_to_expand(self):
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        route_to_model = config['models']['model_selected']

        self.clearQTreeWidget(self.treeViewShowModelsToExpand)
        if pathlib.Path(route_to_model).is_dir():
            ret = get_model_xml(route_to_model)
            self.printTree(ret, self.treeViewShowModelsToExpand)

    def show_models_to_see_description(self):
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_model = config['models']['model_selected']

        self.clearQTreeWidget(self.treeWidgetSelectModelToSeeDescription)
        self.clearQTreeWidget(self.treeViewShowModelsToExpand)

        if pathlib.Path(route_to_model).is_dir():
            ret = get_model_xml(route_to_model)
            self.printTree(ret, self.treeWidgetSelectModelToSeeDescription)
            self.printTree(ret, self.treeViewShowModelsToExpand)

    def refresh(self):
        self.show_models_to_expand()
        self.show_models_to_see_description()
        self.plainTextShow.clear()

    def clicked_create_model(self):
        create_model()
        self.show_models()

    def clicked_delete_model(self):
        models = list_models()
        if not models:
            self.outputSelectedModel.setStyleSheet('color:red')
            self.outputSelectedModel.setText(
                "There are not models to delete.")
        else:
            if not self.column_listSelectModel.currentItem():
                button_info_no_selected_model_to_delete = QtWidgets.QMessageBox.warning(self,
                                                                                        'MusicalSpoon message',
                                                                                        "A model to delete must "
                                                                                        "be selected first.")
                return
            else:
                model_to_delete_str = str(self.column_listSelectModel.currentItem().text())

            deleted = delete_model(model_to_delete_str)
            if deleted:
                self.outputSelectedModel.setStyleSheet('color:green')
                self.outputSelectedModel.setText(
                    "The model " + '"' + model_to_delete_str + '"' + " was deleted.")
            else:
                button_info_no_model_deleted = QtWidgets.QMessageBox.warning(self,
                                                                             'MusicalSpoon message',
                                                                             "An error occurred while "
                                                                             "deleting the selected model.")
            self.set_default_model_parameters()
        self.show_models()

    def clicked_selected_model(self):
        item = self.column_listSelectModel.currentItem()
        self.model_to_train = str(item.text())
        self.outputSelectedModel.setText("You have selected : " + str(item.text()))
        select_model(str(item.text()))
        # Actualize lists
        if str(item.text()).endswith("topics"):
            self.refresh()

        config.read(config_file)
        model_selected = config['models']['model_selected']
        persistence_selected = config['models']['persistence_selected']
        self.tableWidgetModelFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(str(model_selected)))
        self.tableWidgetModelFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(str(persistence_selected)))

        if self.web:
            self.web.setParent(None)

    def clicked_train_model(self):
        # Checked that a model has been selected
        models_dir = pathlib.Path(project_path, "models")
        # Available models
        models = [model.name for model in models_dir.iterdir() if model.is_dir()]
        if not models:
            self.outputTrainModel.setStyleSheet('color:red')
            self.outputTrainModel.setText("No model has been selected yet. Select first the model you want to train.")

        if not self.InsertNumberTopicsModel.text():
            button_info_no_number_topics = QtWidgets.QMessageBox.warning(self,
                                                                         'MusicalSpoon message',
                                                                         "You must select the number of "
                                                                         "topics you want to use for"
                                                                         "training the model.")
            return
        else:
            self.num_training_topics = self.InsertNumberTopicsModel.text()
        if not self.model_to_train:
            button_info_no_model_to_train = QtWidgets.QMessageBox.warning(self,
                                                                          'MusicalSpoon message',
                                                                          "You must select a model to train first.")
            return
        self.plainTextTrainModel.clear()
        self.outputTrainModel.setStyleSheet('color:blue')
        self.outputTrainModel.setText(
            "The selected model is being trained with " + self.num_training_topics + " topics.")

        # THREAD -> Start thread to train the model while the loading page is being shown
        self.execute_in_thread(self.execute_to_train_model, self.do_after_train)

    def do_after_train(self):
        # Print the message that the model was trained
        config.read(config_file)
        new_model_name = config['models']['model_name']
        new_persistence = config['models']['persistence_selected']
        self.tableWidgetModelFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(str(new_model_name)))
        self.tableWidgetModelFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(str(new_persistence)))

        self.outputTrainModel.setStyleSheet('color:green')
        self.outputTrainModel.setText(
            "The model " + new_model_name + " was trained with " + self.num_training_topics + " topics.")

        button_info_train_model = QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                                                    "The model " + new_model_name + " was trained "
                                                                                                    "with " +
                                                                    self.num_training_topics + " topics.")
        # Actualize lists
        self.show_models()
        self.refresh()
        self.set_default_model_parameters()
        # When trained, show the obtain topics
        route_to_model = config['models']['model_selected']
        file = pathlib.Path(route_to_model, model_ids)
        topics_ids_df = pd.read_csv(file, sep="\t", header=None)
        topic_ids = topics_ids_df.values[:, 0].tolist()
        for i in topic_ids:
            text = i + "\n"
            self.plainTextTrainModel.insertPlainText(text)
        self.InsertNumberTopicsModel.setText("")

    def clicked_model_to_expand_selected(self):
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        item = self.treeViewShowModelsToExpand.currentItem()
        model_selected = item.text(0)
        # self.outputSelectedModel.setStyleSheet('color:green')
        self.outputSelectedModel.setText("You have selected : " + str(model_selected))

        topic_ids = show_topics_to_expand(str(model_selected))

        # column 0 = topic nr
        # column 1 = description
        # column 2 = words
        self.tableWidgetTrainSubmodel.clearContents()
        self.tableWidgetTrainSubmodel.setRowCount(len(topic_ids))
        self.tableWidgetTrainSubmodel.setColumnCount(3)
        model_to_change = model.look_for_model(str(model_selected))

        list_names = []
        list_description = []
        for i in np.arange(0, len(model_to_change.topics_models), 1):
            if str(type(model_to_change.topics_models[i])) == "<class 'Topic.Topic'>":
                list_description.append(
                    model_to_change.topics_models[i].get_description()[0][0])
                if model_to_change.topics_models[i].get_description_name() == "":
                    list_names.append("")
                else:
                    list_names.append(
                        model_to_change.topics_models[i].get_description_name())

        for i in np.arange(0, len(list_names), 1):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            item_topic_nr.setFlags(item_topic_nr.flags() ^ Qt.ItemIsEditable)
            self.tableWidgetTrainSubmodel.setItem(i, 0, item_topic_nr)
            item_topic_name = QtWidgets.QTableWidgetItem(str(list_names[i]))
            self.tableWidgetTrainSubmodel.setItem(i, 1, item_topic_name)
            item_topic_description = QtWidgets.QTableWidgetItem(str(list_description[i]))
            item_topic_description.setFlags(item_topic_description.flags() ^ Qt.ItemIsEditable)
            self.tableWidgetTrainSubmodel.setItem(i, 2, item_topic_description)

        self.tableWidgetTrainSubmodel.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetTrainSubmodel.resizeColumnsToContents()

    def clicked_train_submodel(self):
        self.plainTextEditSubmodelTrained.clear()
        route_to_models = pathlib.Path(project_path, "models")
        # 1. Check if there is a model available to make the expansion from
        print(os.listdir(route_to_models))
        if not os.listdir(route_to_models):
            self.outputCreatedSubmodel.setStyleSheet('color:red')
            self.outputCreatedSubmodel.setText("You must train the model first.")
            return

        # Set model to expand, topic to expand and number of topic so they are accessible from the Thread

        if not self.treeViewShowModelsToExpand.currentItem():
            button_info_no_model_to_expand = QtWidgets.QMessageBox.warning(self,
                                                                           'MusicalSpoon message',
                                                                           "You must select from which model you want "
                                                                           "to "
                                                                           "create the submodel first.")
            return
        else:
            item = self.treeViewShowModelsToExpand.currentItem()
            self.model_to_expand = item.text(0)
        r = self.tableWidgetTrainSubmodel.currentRow()
        if not self.tableWidgetTrainSubmodel.item(r, 0):
            button_info_no_topic_to_expand = QtWidgets.QMessageBox.warning(self,
                                                                           'MusicalSpoon message',
                                                                           "You must the select the topic whose "
                                                                           "vocabulary you want to use for creating "
                                                                           "the submodel.")
            return
        else:
            self.topic_to_expand = int(self.tableWidgetTrainSubmodel.item(r, 0).text())
        if not self.InsertNumberTopicsSubmodel.text():
            button_info_no_number_topics = QtWidgets.QMessageBox.warning(self,
                                                                         'MusicalSpoon message',
                                                                         "You must insert the number of topics that "
                                                                         "you want to use for training the submodel.")
            return
        else:
            self.num_training_topics = int(self.InsertNumberTopicsSubmodel.text())

        # Train submodels
        self.outputCreatedSubmodel.clear()
        self.outputCreatedSubmodel.setStyleSheet('color:blue')
        self.outputCreatedSubmodel.setText("A submodel for the topic" + str(
            self.topic_to_expand) + " of the model " + self.model_to_expand + " is being trained.")
        self.execute_in_thread(self.execute_to_train_submodel, self.do_after_train_submodels)

    def do_after_train_submodels(self):
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        submodel = model.look_for_model(self.new_submodel)  # @TODO -> esto tiene que ser el submodelo
        route_to_submodel = submodel.model_path
        file = pathlib.Path(route_to_submodel, model_ids)
        topics_ids_df = pd.read_csv(file, sep="\t", header=None)
        topic_ids = topics_ids_df.values[:, 0].tolist()

        # Get topic ids
        self.plainTextEditSubmodelTrained.clear()
        for i in topic_ids:
            text = i + "\n"
            self.plainTextEditSubmodelTrained.insertPlainText(text)

        self.outputCreatedSubmodel.setStyleSheet('color:green')
        self.outputCreatedSubmodel.setText("A submodel for the topic " + str(self.topic_to_expand) + " from " +
                                           self.model_to_expand + " was trained with " + str(
            self.num_training_topics) + " topics.")

        button_reply_delete_model = QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                                                      "A submodel for the topic " + str(
                                                                          self.topic_to_expand) + " from " +
                                                                      self.model_to_expand + " was trained with " + str(
                                                                          self.num_training_topics) + " topics.")

        # Actualize lists
        self.tableWidgetTrainSubmodel.clearContents()
        self.refresh()
        self.set_default_model_parameters()
        self.InsertNumberTopicsSubmodel.setText("")

    def clicked_select_model_to_see_description(self):
        item = self.treeWidgetSelectModelToSeeDescription.currentItem()
        # self.outputSelectedModelToShow.setStyleSheet('color:green')
        self.outputSelectedModel.setText("You have selected : " + item.text(0))
        if self.web:
            self.web.setParent(None)

    def clicked_see_topics_description(self):
        self.plainTextShow.clear()
        item = self.treeWidgetSelectModelToSeeDescription.currentItem()

        topic_ids = show_topic_model_description(item.text(0))
        if not topic_ids:
            # self.outputSelectedModelToShow.setStyleSheet('color:red')
            self.outputSelectedModel.setText("Any model has been trained yet.")  # pasar a warnings
        else:
            for i in topic_ids:
                text = i + "\n"
                self.plainTextShow.insertPlainText(text)

    def clicked_model_to_change_description(self):
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        item = self.treeViewShowModelsToExpand.currentItem()

        topic_ids = show_topic_model_description(item.text(0))
        if not topic_ids:
            self.outputCreatedSubmodel.setStyleSheet('color:red')
            self.outputCreatedSubmodel.setText("Any model has been trained yet.")
        else:

            ## meter en la tabla
            # column 0 = topic nrv
            # column 1 = description
            # column 2 = words
            self.tableWidgetTrainSubmodel.clearContents()
            self.tableWidgetTrainSubmodel.setRowCount(len(topic_ids))
            self.tableWidgetTrainSubmodel.setColumnCount(3)

            model_to_change = model.look_for_model(item.text(0))

            list_names = []
            list_description = []
            for i in np.arange(0, len(model_to_change.topics_models), 1):
                if str(type(model_to_change.topics_models[i])) == "<class 'Topic.Topic'>":  ## si es tópico
                    list_description.append(
                        model_to_change.topics_models[i].get_description()[0][0])
                    if model_to_change.topics_models[i].get_description_name() == "":
                        list_names.append("")
                    else:
                        list_names.append(
                            model_to_change.topics_models[i].get_description_name())

            for i in np.arange(0, len(list_names), 1):
                item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
                self.tableWidgetTrainSubmodel.setItem(i, 0, item_topic_nr)
                item_topic_name = QtWidgets.QTableWidgetItem(str(list_names[i]))
                self.tableWidgetTrainSubmodel.setItem(i, 1, item_topic_name)
                item_topic_description = QtWidgets.QTableWidgetItem(str(list_description[i]))
                self.tableWidgetTrainSubmodel.setItem(i, 2, item_topic_description)

            self.tableWidgetTrainSubmodel.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.AdjustToContents)
            self.tableWidgetTrainSubmodel.resizeColumnsToContents()

    def clicked_apply_changes(self):
        self.execute_in_thread(self.execute_apply_changes, self.after_apply_changes)

    def after_apply_changes(self):
        item = self.treeViewShowModelsToExpand.currentItem()
        self.outputCreatedSubmodel.setStyleSheet('color:green')
        self.outputCreatedSubmodel.setText("Topics' description of model " + item.text(0) + " have been updated.")
        return

    def clicked_reset_changes(self):
        self.execute_in_thread(self.execute_reset_changes, self.after_reset_changes)

    def after_reset_changes(self):
        item = self.treeViewShowModelsToExpand.currentItem()
        self.outputCreatedSubmodel.setStyleSheet('color:green')
        self.outputCreatedSubmodel.setText("Topics' description of model " + "''" + item.text(0) + "'' were "
                                                                                                   "changed to "
                                                                                                   "its default "
                                                                                                   "value.")

    def clicked_plot_pyldavis(self):
        self.model_to_plot = str(self.treeWidgetSelectModelToSeeDescription.currentItem().text(0))

        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)
        model_plot_pyldavis = model.look_for_model(self.model_to_plot)
        route_to_model_plot_pyldavis = model_plot_pyldavis.model_path
        file_pldavis = pathlib.Path(route_to_model_plot_pyldavis, pldavis).as_posix()

        if not os.path.exists(file_pldavis):
            self.execute_in_thread(self.execute_to_get_pyldavis, self.after_show_plot_pyldavis)
        else:
            self.after_show_plot_pyldavis()

    def after_show_plot_pyldavis(self):
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        model_plot_pyldavis = model.look_for_model(self.model_to_plot)
        route_to_model_plot_pyldavis = model_plot_pyldavis.model_path
        file_pldavis = pathlib.Path(route_to_model_plot_pyldavis, pldavis).as_posix()

        if self.web:
            self.web.setParent(None)
        self.web = QWebEngineView()
        self.web.load(QUrl.fromLocalFile(file_pldavis))
        self.layoutPlot.addWidget(self.web)
        self.web.show()

    def clicked_selected_model_to_delete(self):
        self.plainTextEditSubmodelTrained.clear()
        return

    def clicked_delete_submodel(self):
        model_to_delete_str = str(self.treeViewShowModelsToExpand.currentItem().text(0))

        if model_to_delete_str.startswith("model"):
            button_reply_delete_model = QtWidgets.QMessageBox.question(self, 'MusicalSpoon message',
                                                                       "Are you sure you want to delete the submodel " +
                                                                       '"' + model_to_delete_str + '"' + "? If you "
                                                                                                         "delete this "
                                                                                                         "model, "
                                                                                                         "all its "
                                                                                                         "children "
                                                                                                         "submodels "
                                                                                                         "will be "
                                                                                                         "deleted as "
                                                                                                         "well.",
                                                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        else:
            button_reply_delete_model = QtWidgets.QMessageBox.question(self, 'MusicalSpoon message',
                                                                       "Are you sure you want to delete the submodel " +
                                                                       '"' + model_to_delete_str + '"' + "?",
                                                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if button_reply_delete_model == QtWidgets.QMessageBox.Yes:

            if model_to_delete_str.startswith("model"):
                deleted = delete_model(model_to_delete_str)
            else:
                route_to_model = config['models']['model_selected']
                try:
                    os.listdir(route_to_model) == []
                except:
                    self.outputCreatedSubmodel.setStyleSheet('color:red')
                    self.outputCreatedSubmodel.setText(
                        "You must have trained a model first in order to delete it.")
                    return
                deleted = delete_submodel(model_to_delete_str)
            if not deleted:
                self.outputCreatedSubmodel.setStyleSheet('color:red')
                self.outputCreatedSubmodel.setText(
                    "There are no submodels to delete.")
                return
            else:
                self.outputCreatedSubmodel.setStyleSheet('color:green')
                self.outputCreatedSubmodel.setText(
                    "The submodel " + '"' + model_to_delete_str + '"' + " was deleted.")
                self.show_models_to_expand()
                self.show_models()
                self.refresh()
        return
