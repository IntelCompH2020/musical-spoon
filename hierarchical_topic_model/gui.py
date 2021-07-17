# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:56:19 2021

@author: lcalv
"""

import configparser
import pickle
import time

from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QUrl, Qt, QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from MessagesGui import MessagesGui
from Model import *
from Worker import Worker
from aux_model import create_model, list_models, select_model, train_model, show_topic_model_description, \
    show_topics_to_expand, train_save_submodels, change_description, generatePyLavis, \
    delete_submodel, delete_model, get_model_xml, configure_project_folder, \
    clearQTreeWidget, printTree, show_topics_to_expand_general, get_root_path, get_pickle, plot_diagnostics

from styleGrey import styleGrey
from styleDarkOrange import styleDarkOrange

config_file = 'config_project.ini'
config = configparser.ConfigParser()
config.read(config_file)

project_path = config['files']['project_path']
source_path = config['files']['source_path']
model_ids = config['out-documents']['model_ids']
pldavis = config['out-documents']['pldavis']
diagnostics = config['out-documents']['diagnosis']
default_project_path = config['default']['project_path']


class UI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, version):
        super(UI_MainWindow, self).__init__()

        # Load UI and configure default geometry of the window
        ########################################################################
        if version == "v1":
            self.version = "v1"
        else:
            self.version = "v2"

        uic.loadUi("UIS/musicalSpoonV2.ui", self)

        self.setGeometry(100, 60, 2000, 1600)
        self.centralwidget.setGeometry(100, 60, 2000, 1600)
        self.animation = QtCore.QPropertyAnimation(self.frame_left_menu, b"minimumWidth")

        # Set style of the window
        ########################################################################
        # self.setStyleSheet(styleDarkOrange.STYLE)

        # Configure project folder
        ########################################################################
        if not pathlib.Path(project_path, "models").is_dir():
            configure_project_folder(project_path)

        # INFORMATION BUTTONS
        ########################################################################
        self.infoButtonSelectDataset.setIcon(QIcon('Images/help2.png'))
        self.infoButtonLoadFiles.setIcon(QIcon('Images/help2.png'))
        self.infoButtonTrainModel.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectSubmodel_4.setIcon(QIcon('Images/help2.png'))
        self.infoButtoSelectTopicExpand_4.setIcon(QIcon('Images/help2.png'))
        self.infoButtonShowDescription.setIcon(QIcon('Images/help2.png'))
        self.infoButtonDiagnostics.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectModelDiagnostic.setIcon(QIcon('Images/help2.png'))
        self.infoButtonDragTopicDiagnostic.setIcon(QIcon('Images/help2.png'))

        self.infoButtonSelectDataset.setToolTip(MessagesGui.INFO_SELECT_DATASET)
        self.infoButtonLoadFiles.setToolTip(MessagesGui.INFO_LOAD_FILES)
        self.infoButtonTrainModel.setToolTip(MessagesGui.INFO_TRAIN_MODEL)
        self.infoButtonSelectSubmodel_4.setToolTip(MessagesGui.INFO_SELECT_SUBMODEL)
        self.infoButtoSelectTopicExpand_4.setToolTip(MessagesGui.INFO_SELECT_TOPIC_TO_EXPAND)
        self.infoButtonShowDescription.setToolTip(MessagesGui.INFO_SHOW_DESCRIPTION)
        self.infoButtonDiagnostics.setToolTip(MessagesGui.INFO_DIAGNOSTICS)
        self.infoButtonSelectModelDiagnostic.setToolTip(MessagesGui.SELECT_COMPARE)
        self.infoButtonDragTopicDiagnostic.setToolTip(MessagesGui.DRAG_COMPRARE)


        # CONFIGURE ELEMENTS IN THE "CONFIGURATION VIEW"
        ########################################################################
        self.treeViewSelectDataset.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeViewSelectDataset.customContextMenuRequested.connect(self.context_menu)
        self.treeViewSelectDataset.doubleClicked.connect(self.clicked_select_dataset)
        self.modelTreeView = QtWidgets.QFileSystemModel(nameFilterDisables=False)
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

        self.tableWidgetHTMVersion.doubleClicked.connect(self.clicked_changeHTM)

        # CONFIGURE ELEMENTS IN THE "SELECT ROUTE MODEL VIEW"
        ########################################################################
        self.column_listSelectModel.clicked.connect(self.clicked_selected_model)
        self.createNewModel.clicked.connect(self.clicked_create_model)
        self.removeModel.clicked.connect(self.clicked_delete_model)
        self.trainModel.clicked.connect(self.clicked_train_model)
        self.treeViewShowModelsToExpand_4.clicked.connect(self.clicked_model_to_change_description)
        self.buttonApplyChanges.clicked.connect(self.clicked_apply_changes)
        self.buttonResetDescriptions.clicked.connect(self.clicked_reset_changes)
        # DELETE SUBMODEL
        self.treeViewShowModelsToExpand_4.clicked.connect(self.clicked_selected_model_to_delete)
        self.pushButtonToDeleteSubmodel_4.clicked.connect(self.clicked_delete_submodel)

        # CONFIGURE ELEMENTS IN THE "EDIT MODEL VIEW"
        ########################################################################
        self.treeViewShowModelsToExpand_4.clicked.connect(self.clicked_model_to_expand_selected)
        self.buttonTrainSub.clicked.connect(self.clicked_train_submodel)
        self.treeViewShowModelsToExpand_4.clicked.connect(self.clicked_model_to_change_description)
        self.buttonApplyChanges.clicked.connect(self.clicked_apply_changes)
        self.buttonResetDescriptions.clicked.connect(self.clicked_reset_changes)

        # CONFIGURE ELEMENTS IN THE "SHOW MODELS"
        ########################################################################
        self.treeWidgetSelectModelToSeeDescription.clicked.connect(self.clicked_select_model_to_see_description)
        self.pushButtonSeeDescription.clicked.connect(self.clicked_see_topics_description)
        self.pushButtonPlotPyLDAvis.clicked.connect(self.clicked_plot_pyldavis)

        # CONFIGURE ELEMENTS IN THE "SHOW Diagnostics"
        ########################################################################
        self.pushButtonDiagnostics.clicked.connect(self.clicked_plot_diagnosis)

        # CONFIGURE ELEMENTS IN THE "DRAW Diagnostics"
        ########################################################################
        self.treeWidgetSelectModelToDiagnostic.clicked.connect(self.clicked_showTopicsDragDiagnosis)
        self.pushButtonPlotDiagnosisGraph.clicked.connect(self.click_draw_diagnosis)
        self.pushButtonClearDiagnostics.clicked.connect(self.clear_tables_graph_diagnostics)

        # MODEL PARAMETERS THAT ARE SAVED TO PASS TO THREADS
        ########################################################################
        self.num_training_topics = 5
        self.model_to_expand = ""
        self.topic_to_expand = 5
        self.model_to_plot = ""
        self.new_submodel = ""
        self.model_to_train = ""
        self.web = None
        self.webExpand = None
        self.web_diag = None
        self.webExpand_diag = None
        self.list_names = []
        self.list_description = []
        self.model_get_ids = None
        self.threshold = ""

        self.figure = plt.figure()
        # this is the Canvas Widget that
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.layoutPlot_5.addWidget(self.canvas)

        self.initialize_settings()
        self.show_models()
        self.show()

        # THREADS FOR EXECUTING PARALLEL TRAIN MODELS, TRAIN SUBMODELS, AND PYLDAVIS
        ########################################################################
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.worker = None
        self.loading_window = None

        # TOGGLE MENU
        ########################################################################
        self.Btn_Toggle.clicked.connect(lambda: self.toggleMenu(250))
        self.Btn_Toggle.setIcon(QIcon('Images/menu.png'))

        # PAGES
        ########################################################################
        # PAGE 1: Configuration: Change dataset, see porject settings
        self.pushButtonConfiguration_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage))
        self.pushButtonConfiguration_2.setIcon(QIcon('Images/settings.png'))

        # PAGE 2: Create new model/select model /train model /delete model
        self.pushButtonSelectModel_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage2))
        self.pushButtonSelectModel_2.setIcon(QIcon('Images/new.png'))

        # PAGE 3: Create/train submodel /add description to topics
        self.pushButtonTrainSubmodel_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage3))
        self.pushButtonTrainSubmodel_2.setIcon(QIcon('Images/create.png'))

        # PAGE 4: See topics' description / generate and plot PyLDAvis
        self.pushButtonShowDescription_2.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage4))
        self.pushButtonShowDescription_2.setIcon(QIcon('Images/show.png'))

        # PAGE 5: See PyLDAvis plot expanded
        self.pushButtonShowPylavisBig.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage5))
        self.pushButtonShowPylavisBig.setIcon(QIcon('Images/expand.png'))

        # PAGE 6: See diagnostics
        self.pushButtonShowDiagnosis.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage6))
        self.pushButtonShowDiagnosis.setIcon(QIcon('Images/diagnostic_white2.png'))

        # PAGE 7: See diagnostics plot expanded
        self.pushButtonShowDiagnosticsBig.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage7))
        self.pushButtonShowDiagnosticsBig.setIcon(QIcon('Images/expand.png'))

        # PAGE 8: Draw diagnostics graphs
        self.pushButtonDraw.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage8))
        self.pushButtonDraw.setIcon(QIcon('Images/draw_white.png'))

    def toggleMenu(self, maxWidth):
        """Method to control the movement of the Toggle menu located on the
        left. When collapsed, only the icon for each of the options is shown;
        when expanded, both icons and name indicating the description of the
        functionality are shown.

        Parameters:
        ----------
        * maxWidth  -  Maximum width to which the toggle menu is going to be
                       expanded.
        """
        # GET WIDTH
        width = self.frame_left_menu.width()
        maxExtend = maxWidth
        standard = 70

        # SET MAX WIDTH
        if width == 70:
            widthExtended = maxExtend
            # SHOW TEXT INSTEAD OF ICON
            self.pushButtonConfiguration_2.setText('Configuration')
            self.pushButtonSelectModel_2.setText('Select model')
            self.pushButtonShowDescription_2.setText('Show model')
            self.pushButtonTrainSubmodel_2.setText('Edit model')
            self.pushButtonShowDiagnosis.setText('Diagnostics')
            self.pushButtonDraw.setText('Draw graph')
            self.label_logo.setFixedSize(widthExtended, widthExtended)

        else:
            widthExtended = standard
            self.pushButtonConfiguration_2.setText('')
            self.pushButtonSelectModel_2.setText('')
            self.pushButtonShowDescription_2.setText('')
            self.pushButtonTrainSubmodel_2.setText('')
            self.pushButtonShowDiagnosis.setText('')
            self.pushButtonDraw.setText('')
            self.label_logo.setFixedSize(widthExtended, widthExtended)

        # ANIMATION
        self.animation = QtCore.QPropertyAnimation(self.frame_left_menu, b"minimumWidth")
        self.animation.setDuration(400)
        self.animation.setStartValue(width)
        self.animation.setEndValue(widthExtended)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()

    @QtCore.pyqtSlot()
    def start_animation(self):
        """Method that controls the animation of the loading bar during
        high computational operations, like the training of a model. It is
        started when a process is executed in the secondary thread, and it
        stops when the thread finished signal is triggered..

        """
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

    def execute_to_train_model(self, progress_callback):
        """Method to control the execution of the training of a model.
        """
        train_model(self.num_training_topics)
        return "Done."

    def execute_to_train_submodel(self, progress_callback):
        """Method to control the execution of the training of a submodel.
        """
        train_save_submodels(self.model_to_expand, self.topic_to_expand, self.num_training_topics, self, self.version,
                             self.threshold)
        return "Done."

    def execute_to_get_pyldavis(self, progress_callback):
        """Method to control the execution of the generation of a PyLDAVis.
        """
        generatePyLavis(self.model_to_plot)

    def execute_apply_changes(self, progress_callback):
        """Method to control the change the description of the topics of a
        model/submodel .
        """
        item = self.treeViewShowModelsToExpand_4.currentItem()
        # Get names from the right table and show in the left one
        for i in np.arange(0, self.tableWidgetNewTopicName_4.rowCount(), 1):
            print(self.tableWidgetNewTopicName_4.item(i, 0))
            new_name = self.tableWidgetNewTopicName_4.item(i, 0).text()
            change_description(item.text(0), i, new_name)
            self.tableWidgetTrainSubmodel_4.setItem(i, 1, QtWidgets.QTableWidgetItem(new_name))
            self.tableWidgetNewTopicName_4.setItem(i, 0, QtWidgets.QTableWidgetItem(""))

    def execute_reset_changes(self, progress_callback):
        """Method to control the resetting of the default topics' description
        of a model/submodel (default value = no decscription).
        """
        item = self.treeViewShowModelsToExpand_4.currentItem()
        for i in np.arange(0, self.tableWidgetTrainSubmodel_4.rowCount(), 1):
            change_description(str(item.text(0)), i, "")
            self.tableWidgetTrainSubmodel_4.setItem(i, 1, QtWidgets.QTableWidgetItem(""))
            self.tableWidgetNewTopicName_4.setItem(i, 0, QtWidgets.QTableWidgetItem(""))

    def execute_get_topic_ids(self, progress_callback):
        """Method to get the topics associated to the model/submodel saved
        in "self.model_get_ids".
        """
        self.list_names = []
        self.list_description = []
        for i in np.arange(0, len(self.model_get_ids.topics_models), 1):
            if str(type(self.model_get_ids.topics_models[i])) == "<class 'Topic.Topic'>":
                self.list_description.append(
                    self.model_get_ids.topics_models[i].get_description()[0][0])
                if self.model_get_ids.topics_models[i].get_description_name() == "":
                    self.list_names.append("")
                else:
                    self.list_names.append(
                        self.model_get_ids.topics_models[i].get_description_name())

    def execute_in_thread(self, function, function_output, animation):
        """Method to execute a function in the secondary thread, while showing
        an animation at the time the function is being executed if animation is
        set to true. When finished, it forces the execution of the method to be
        executed after the function executing in a thread is completed.".

        Parameters:
        ----------
        * function         - Function to be executed in thread
        * function_output  - Function to be executed afte the thread
        * animation        - If true, it shows a loading bar when the funtion
                             in thread is being executed.
        """
        # Pass the function to execute
        self.worker = Worker(function)  # Any other args, kwargs are passed to the run function
        if animation:
            self.worker.signals.started.connect(self.start_animation)
        self.worker.signals.finished.connect(function_output)

        # Execute
        self.threadpool.start(self.worker)

    def set_default_model_parameters(self):
        """Method to initialize the model pararameters that are save as
        attributes of UI_MainWindow to be passed to the executions in threads".
        """
        self.num_training_topics = 0
        self.model_to_expand = ""
        self.topic_to_expand = 0
        self.model_to_plot = ""
        self.new_submodel = ""
        self.model_to_train = ""
        self.threshold = "0.4"

    def initialize_settings(self):
        """Method to initialize table "Model file settings" from the Configuration
        view using the information saved in the configuration file.
        """
        config.read(config_file)
        first_menu_source_path = config['default']['source_path']
        first_menu_project_path = config['default']['project_path']
        first_menu_mallet_path = config['default']['mallet_path']

        self.tableWidgetGeneralSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(first_menu_source_path)))
        self.tableWidgetGeneralSettings.setItem(1, 1, QtWidgets.QTableWidgetItem(str(first_menu_project_path)))
        self.tableWidgetMalletSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(first_menu_mallet_path)))

    def set_default_settings(self):
        """Method to reset the values of the tables in the Configuration
        view using the default information saved in the configuration file.
        """
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

        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "All values were restore to the default settings.")
        # self.logging.info('All values were restore to the default settings.')

        self.show_models()
        self.refresh()

    def apply_changes_settings(self):
        """Method to save in the configuration file the changes made by the user
        in the configuration file.
        """
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
        configure_project_folder(new_project_path)
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

        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message', "Changes were saved in the configuration file.")

        return

    def show_datasets(self):
        """Method to list all the possible datasets contained in the mallet
        folder of the user being executed the app.
        """
        config.read(config_file)
        path = pathlib.Path(pathlib.Path("\mallet").parent.absolute(), "mallet").as_posix()
        self.treeViewSelectDataset.setModel(self.modelTreeView)
        self.treeViewSelectDataset.setRootIndex(self.modelTreeView.index(path))
        self.modelTreeView.setNameFilters(["*" + ".txt"])
        self.treeViewSelectDataset.setSortingEnabled(True)
        return

    def context_menu(self):
        """Method control the opening of file in a text editor when an element
        from the dataset list in the mallet directory is clicked with the right
        mouse button.
        """
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open file")
        open.triggered.connect(self.open_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        """Method to open the content of a file in a text editor when called
        by context_menu.
        """
        index = self.treeViewSelectDataset.currentIndex()
        file_path = self.modelTreeView.filePath(index)
        os.startfile(file_path)

    def clicked_select_dataset(self):
        """Method to control the selection of a new dataset by the user when
        he/she makes double clicked into one of the item of the dataset list in
        the mallet directory.
        """
        index = self.treeViewSelectDataset.currentIndex()
        dataset_selected_path = self.modelTreeView.filePath(index)
        self.statusBar().showMessage("'" + str(dataset_selected_path) + "' was selected as dataset.", 10000)

        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "'" + str(dataset_selected_path) +
                                          "' was selected as dataset.")

        self.tableWidgetGeneralSettings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(dataset_selected_path)))

    def clicked_changeHTM(self):
        r = self.tableWidgetHTMVersion.currentRow()
        if r == 0:
            self.version = "v1"
            self.statusBar().showMessage("The algorithm of the underlying HTM was changed to version 1.", 10000)
        elif r == 1:
            self.version = "v2"
            self.statusBar().showMessage("The algorithm of the underlying HTM was changed to version 2.", 10000)

    def show_models(self):
        """Method to list all the model contained in the project folder in the
        "Select model" view.
        """
        if pathlib.Path(project_path, "models").is_dir():
            self.column_listSelectModel.clear()
            models = list_models()
            if models:
                for model_nr in np.arange(0, len(models), 1):
                    self.column_listSelectModel.insertItem(model_nr, models[model_nr])

        if self.web:
            self.web.setParent(None)
        if self.webExpand:
            self.webExpand.setParent(None)
        if self.web_diag:
            self.web_diag.setParent(None)
        if self.webExpand_diag:
            self.webExpand_diag.setParent(None)

    def show_models_to_expand(self):
        """Method to list all the models available for expansion in the
        "Edit model" view.
        """
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_model = config['models']['model_selected']

        clearQTreeWidget(self.treeViewShowModelsToExpand_4)
        if pathlib.Path(route_to_model).is_dir():
            ret = get_model_xml(route_to_model)
            printTree(ret, self.treeViewShowModelsToExpand_4)

    def show_models_to_see_description(self):
        """Method to list all the train model/submodels to show its topics'
        description in the "Show model description" view
        """
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_model = config['models']['model_selected']

        clearQTreeWidget(self.treeWidgetSelectModelToSeeDescription)
        clearQTreeWidget(self.treeViewShowModelsToExpand_4)
        clearQTreeWidget(self.treeWidgetSelectModelToSeeDiagnostics)
        clearQTreeWidget(self.treeWidgetSelectModelToDiagnostic)

        if pathlib.Path(route_to_model).is_dir():
            ret = get_model_xml(route_to_model)
            printTree(ret, self.treeWidgetSelectModelToSeeDescription)
            printTree(ret, self.treeViewShowModelsToExpand_4)
            printTree(ret, self.treeWidgetSelectModelToSeeDiagnostics)

        project_dir = pathlib.Path(project_path)
        all_models = (project_dir / "models").as_posix()
        if pathlib.Path(all_models).is_dir():
            ret = get_model_xml(all_models)
            printTree(ret, self.treeWidgetSelectModelToDiagnostic)

    def refresh(self):
        """Method to clear lists and reload information in "column_listSelectModel",
        "treeViewShowModelsToExpand_4" and "treeWidgetSelectModelToSeeDescription",
        in the "Select model", "Edit model" and "Show model description" views,
        respectively.
        """
        self.show_models_to_expand()
        self.show_models_to_see_description()
        self.tableWidgetShowModels.clearContents()
        self.tableWidgetTrainSubmodel_4.clearContents()
        self.tableWidgetNewTopicName_4.clearContents()

    def clicked_create_model(self):
        """Method to control the creation of a new model (not trained yet). It
        creates a new model and updates the list "column_listSelectModel" to
        include the new model.
        """
        create_model()
        self.show_models()

    def clicked_delete_model(self):
        """Method to control the deletion of a model selected by the user. It
        deletes a model and updates the list "column_listSelectModel".
        """
        models = list_models()
        if not models:
            self.statusBar().showMessage("There are not models to delete.", 10000)
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "There are not models to delete.")
        else:
            if not self.column_listSelectModel.currentItem():
                QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "A model to delete must be selected first.")
                return
            else:
                model_to_delete_str = str(self.column_listSelectModel.currentItem().text())

                button_reply_delete_model = QtWidgets.QMessageBox.question(self, 'MusicalSpoon message',
                                                                           "Are you sure you want to delete the model " +
                                                                           '"' + model_to_delete_str + '"' + "?",
                                                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

                if button_reply_delete_model == QtWidgets.QMessageBox.Yes:
                    deleted = delete_model(model_to_delete_str)
                    if deleted:
                        self.statusBar().showMessage("The model " + '"' + model_to_delete_str + '"' + " was deleted.",
                                                     10000)
                    else:
                        QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "An error occurred while "
                                                                                    "deleting the selected model.")
                    self.set_default_model_parameters()
        self.show_models()
        self.refresh()

    def clicked_selected_model(self):
        """Method to control the change in the selection of the main model. When
        updated, the lists "treeViewShowModelsToExpand_4" and
        "treeWidgetSelectModelToSeeDescription" are also updated to include the
        the new model selected, as well as the submodels in it contained.
        """
        self.tableWidgetModelTrained.clearContents()
        item = self.column_listSelectModel.currentItem()
        self.model_to_train = str(item.text())
        self.statusBar().showMessage("You have selected : " + str(item.text()), 10000)
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
        if self.webExpand:
            self.webExpand.setParent(None)
        if self.web_diag:
            self.web_diag.setParent(None)
        if self.webExpand_diag:
            self.webExpand_diag.setParent(None)

    def clicked_train_model(self):
        """Method to control training of a model It checks whether the parameters
        selected for training the model are adequate and calls the execution in
        thread in case the parameters are satisfied.
        """
        self.inicio = time.time()
        # Checked that a model has been selected
        models_dir = pathlib.Path(project_path, "models")
        # Available models
        models = [model.name for model in models_dir.iterdir() if model.is_dir()]
        if not models or not self.model_to_train:
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "No model has been selected yet. Select first the model you want to train.")

        if not self.insertNumberTopicsModel.text():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select the number of topics you want to use for training the model.")
            return
        elif not self.insertNumberTopicsModel.text().isdigit():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "The number of topics must be a number.")
            return
        else:
            self.num_training_topics = self.insertNumberTopicsModel.text()

        self.tableWidgetModelTrained.clearContents()
        self.statusBar().showMessage(
            "The selected model is being trained with " + self.num_training_topics + " topics.")

        # THREAD -> Start thread to train the model while the loading page is being shown
        #################################################################################
        self.execute_in_thread(self.execute_to_train_model, self.do_after_train, True)

    def do_after_train(self):
        """Method to show the results of training a model.
        """
        # Reload config file just in case changes are found
        config.read(config_file)
        new_model_name = config['models']['model_name']
        new_persistence = config['models']['persistence_selected']
        infile = open(new_persistence, 'rb')
        model = pickle.load(infile)

        self.tableWidgetModelFiles.setItem(0, 1, QtWidgets.QTableWidgetItem(str(new_model_name)))
        self.tableWidgetModelFiles.setItem(1, 1, QtWidgets.QTableWidgetItem(str(new_persistence)))

        self.insertNumberTopicsModel.setText("")
        fin = time.time()
        print("Tiempo training")
        print(fin - self.inicio)  # 1.0005340576171875

        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "The model ''" + new_model_name + "'' was trained with " + self.num_training_topics + " topics.")

        # Actualize lists
        self.show_models()
        self.refresh()
        self.set_default_model_parameters()

        # When trained, show the obtain topics
        route_to_model = config['models']['model_selected']
        file = pathlib.Path(route_to_model, model_ids)
        topics_ids_df = pd.read_csv(file, sep="\t", header=None)
        topic_ids = topics_ids_df.values[:, 0].tolist()

        #####################
        # column 0 = topic nr
        # column 1 = words
        #####################
        self.tableWidgetModelTrained.clearContents()
        self.tableWidgetModelTrained.setRowCount(len(topic_ids))
        self.tableWidgetModelTrained.setColumnCount(2)
        new_trained_model = model.look_for_model(str(new_model_name))

        list_description = []
        for i in np.arange(0, len(new_trained_model.topics_models), 1):
            if str(type(new_trained_model.topics_models[i])) == "<class 'Topic.Topic'>":
                list_description.append(
                    new_trained_model.topics_models[i].get_description()[0][0])

        for i in np.arange(0, len(list_description), 1):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.tableWidgetModelTrained.setItem(i, 0, item_topic_nr)
            item_topic_description = QtWidgets.QTableWidgetItem(str(list_description[i]))
            self.tableWidgetModelTrained.setItem(i, 1, item_topic_description)

        self.tableWidgetModelTrained.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetModelTrained.resizeColumnsToContents()

    def clicked_model_to_expand_selected(self):
        """Method to control the selection of a model to expand (i.e. to create
        a submodel out of it). It shows in the status bar the model that has
        been selected by the user and shows the a table containing the information
        of that model/submodel; that is, the number of topics and the description
        associated to each of those topics. By double clicking in any row belonging
        to the latter table, the user will choose the topics from which he/she
        wants to create the submodel.
        """
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        if not self.treeViewShowModelsToExpand_4.currentItem().text(0).lower().startswith("model") and \
                not self.treeViewShowModelsToExpand_4.currentItem().text(0).lower().startswith("submodel"):
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "You must select an item. no an empty space.")
            return
        item = self.treeViewShowModelsToExpand_4.currentItem()
        model_selected = item.text(0)
        self.statusBar().showMessage("You have selected : " + str(model_selected), 10000)
        # self.logging.info("Model " + str(model_selected) + " was selected for expansion")
        topic_ids = show_topics_to_expand(str(model_selected))
        ##########################
        # column 0 = topic nr
        # column 1 = description
        # column 2 = words
        ##########################
        self.tableWidgetTrainSubmodel_4.clearContents()
        self.tableWidgetNewTopicName_4.clearContents()
        self.tableWidgetTrainSubmodel_4.setRowCount(len(topic_ids))
        self.tableWidgetNewTopicName_4.setRowCount(len(topic_ids))
        self.tableWidgetTrainSubmodel_4.setColumnCount(3)
        self.tableWidgetNewTopicName_4.setColumnCount(1)
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
            self.tableWidgetTrainSubmodel_4.setItem(i, 0, item_topic_nr)
            self.tableWidgetNewTopicName_4.setItem(i, 0, item_topic_nr)
            item_topic_name = QtWidgets.QTableWidgetItem(str(list_names[i]))
            self.tableWidgetTrainSubmodel_4.setItem(i, 1, item_topic_name)
            item_topic_description = QtWidgets.QTableWidgetItem(str(list_description[i]))
            self.tableWidgetTrainSubmodel_4.setItem(i, 2, item_topic_description)

        self.tableWidgetTrainSubmodel_4.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetTrainSubmodel_4.resizeColumnsToContents()

        self.tableWidgetNewTopicName_4.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetNewTopicName_4.resizeColumnsToContents()

    def clicked_train_submodel(self):
        """Method to control the training of a submodel. It checks whether the
                parameters selected for training the model are adequate and calls the
                execution in thread in case the parameters are satisfied
                """
        self.inicio = time.time()
        self.tableWidgetSubmodelTrained_4.clearContents()
        route_to_models = pathlib.Path(project_path, "models")

        # Check if there is a model available to make the expansion from
        if not os.listdir(route_to_models):
            self.statusBar().showMessage("You must train the model first.", 10000)
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "You must train the model first.")
            # self.logging.warning("Submodel training was not proceeded because no father model has been trainied yet")
            return

        # Set model to expand, topic to expand and number of topic so they are accessible from the Thread
        if not self.treeViewShowModelsToExpand_4.currentItem():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select from which model you want to create the submodel first.")
            # self.logging.warning("Submodel training was not proceeded because no model for expansion was selected")
            return
        else:
            item = self.treeViewShowModelsToExpand_4.currentItem()
            self.model_to_expand = item.text(0)
        r = self.tableWidgetTrainSubmodel_4.currentRow()
        if not self.tableWidgetTrainSubmodel_4.item(r, 0):
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select the topic whose vocabulary you want to use for creating the submodel.")
            # self.logging.warning(
            #    "Submodel training was not proceeded because no topic for creating the submodel from was selected")
            return
        else:
            self.topic_to_expand = int(self.tableWidgetTrainSubmodel_4.item(r, 0).text())
        print(self.version)
        if self.version == "v2":
            if not self.InsertThreshold.text():
                QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                              "You must insert the threshold that indicates how representative a "
                                              "topic in a document must be to keep it in the new submodel's corpus.")
                return
            else:
                if float(self.InsertThreshold.text()) < 0 or float(self.InsertThreshold.text()) > 1:
                    QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                                  "The threshold must be between 0 and 1.")
                    return
                else:
                    self.threshold = float(self.InsertThreshold.text())

        if not self.InsertNumberTopicsSubmodel_8.text():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must insert the number of topics that you want to use for training the submodel.")
            # self.logging.warning(
            #    "Submodel training was not proceeded because the number of topcis to train the submodel was not selected.")
            return
        elif not self.InsertNumberTopicsSubmodel_8.text().isdigit():
            # self.logging.warning(
            #    "Submodel training did not proceeded because the number of topics selected was not a number.")
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "The number of topics must be a number.")
            return
        else:
            self.num_training_topics = int(self.InsertNumberTopicsSubmodel_8.text())

        # Train submodels
        self.statusBar().showMessage("A submodel for the topic '" + str(
            self.topic_to_expand) + "' of the model " + self.model_to_expand + "' is being trained.")

        # THREAD -> Start thread to train the submodel while the loading page is being shown
        #################################################################################
        self.execute_in_thread(self.execute_to_train_submodel, self.do_after_train_submodels, True)

    def do_after_train_submodels(self):
        """Method to show the results of training a submodel.
        """
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        submodel = model.look_for_model(self.new_submodel)
        route_to_submodel = submodel.model_path
        file = pathlib.Path(route_to_submodel, model_ids)
        topics_ids_df = pd.read_csv(file, sep="\t", header=None)
        topic_ids = topics_ids_df.values[:, 0].tolist()

        # Get topic ids
        self.tableWidgetSubmodelTrained_4.clearContents()
        ##########################
        # column 0 = topic nr
        # column 1 = words
        ##########################
        self.tableWidgetSubmodelTrained_4.clearContents()
        self.tableWidgetSubmodelTrained_4.setRowCount(len(topic_ids))
        self.tableWidgetSubmodelTrained_4.setColumnCount(2)

        # Show new model in the trained model table
        self.model_get_ids = submodel
        self.execute_in_thread(self.execute_get_topic_ids, self.do_after_get_topicsIds_train_submodel, False)

    def do_after_get_topicsIds_train_submodel(self):
        """Method to get the topics description of the submodel trained. It is
        executed in the secondary thread in order to speed up computation, but
        does not show a loading bar while it is being executed since it is quite
        fast.
        """
        for i in np.arange(0, len(self.list_description), 1):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.tableWidgetSubmodelTrained_4.setItem(i, 0, item_topic_nr)
            item_topic_description = QtWidgets.QTableWidgetItem(str(self.list_description[i]))
            self.tableWidgetSubmodelTrained_4.setItem(i, 1, item_topic_description)

        self.tableWidgetSubmodelTrained_4.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetSubmodelTrained_4.resizeColumnsToContents()

        self.statusBar().showMessage("A submodel for the topic " + str(self.topic_to_expand) + " from " +
                                     self.model_to_expand + " was trained with " + str(
            self.num_training_topics) + " topics.", 10000)

        fin = time.time()
        print("Time train submodel")
        print(fin - self.inicio)

        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "A submodel for the topic " + str(self.topic_to_expand) + " from " +
                                          self.model_to_expand + " was trained with " + str(
                                              self.num_training_topics) + " topics.")

        # Actualize lists
        self.refresh()
        self.set_default_model_parameters()
        self.InsertNumberTopicsSubmodel_8.setText("")
        self.InsertThreshold.setText("")

    def clicked_select_model_to_see_description(self):
        """Method to control the selection of a model to show its description
        in the "Show description" view. When a model/submodel is clicked, both
        PyLDAvis graph and table to show topics' descriptions are cleared. It
        shows the model selected in the status bar.
        """
        self.tableWidgetShowModels.clearContents()
        if not self.treeWidgetSelectModelToSeeDescription.currentItem().text(0).lower().startswith("m") and \
                not self.treeWidgetSelectModelToSeeDescription.currentItem().text(0).lower().startswith("s"):
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "You must select an item. no an empty space.")
            return

        item = self.treeWidgetSelectModelToSeeDescription.currentItem()
        self.statusBar().showMessage("You have selected : " + item.text(0), 10000)
        self.tableWidgetShowModels.clearContents()
        if self.web:
            self.web.setParent(None)
        if self.webExpand:
            self.webExpand.setParent(None)
        if self.web_diag:
            self.web_diag.setParent(None)
        if self.webExpand_diag:
            self.webExpand_diag.setParent(None)

    def clicked_see_topics_description(self):
        """Method to control the displaying of a model/submodel to see its
        description int the "Show model description" view when the button
        "See topics' description" is cliked by the user.
        """
        self.inicio = time.time()
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        item = self.treeWidgetSelectModelToSeeDescription.currentItem()
        if item is None:
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You need to select first a model or submodel to see its description.")
        else:
            topic_ids = show_topic_model_description(item.text(0))
            if not topic_ids:
                # self.logging.info("No model has been trained yet.")
                QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "No model has been trained yet.")

            else:
                # Put into the table
                ##########################
                # column 0 = topic nr
                # column 1 = description
                # column 2 = words
                ##########################
                self.tableWidgetShowModels.clearContents()
                self.tableWidgetShowModels.setRowCount(len(topic_ids))
                self.tableWidgetShowModels.setColumnCount(3)

                model_to_show = model.look_for_model(item.text(0))
                self.model_get_ids = model_to_show
                self.execute_in_thread(self.execute_get_topic_ids, self.do_after_getIds_see_description, False)

    def do_after_getIds_see_description(self):
        """Method to show the description of the topics belonging to a model/
        submodel in the "Show model description" view.
        """
        for i in np.arange(0, len(self.list_names), 1):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.tableWidgetShowModels.setItem(i, 0, item_topic_nr)
            item_topic_name = QtWidgets.QTableWidgetItem(str(self.list_names[i]))
            self.tableWidgetShowModels.setItem(i, 1, item_topic_name)
            item_topic_description = QtWidgets.QTableWidgetItem(str(self.list_description[i]))
            self.tableWidgetShowModels.setItem(i, 2, item_topic_description)

        self.tableWidgetShowModels.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetShowModels.resizeColumnsToContents()
        print("Tiempo ver description")
        fin = time.time()
        print(fin - self.inicio)

    def clicked_model_to_change_description(self):
        """Method to control the insertion of a name to describe one or several
        topics from a model/submodel by the user. The topics' names are written
        in the right middle table fromt the "Edit model" view, and once the user
        clickes the button "Apply", the new topics' descriptions are shown in the
        left middle table.
        """
        # Reload config file just in case changes are found
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        if not self.treeViewShowModelsToExpand_4.currentItem().text(0).lower().startswith("model") and \
                not self.treeViewShowModelsToExpand_4.currentItem().text(0).lower().startswith("submodel"):
            return
        item = self.treeViewShowModelsToExpand_4.currentItem()
        topic_ids = show_topic_model_description(item.text(0))
        if not topic_ids:
            # self.logging.warning("The description change was not proceeded since no model has been trained yet.", 10000)
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "No model has been trained yet.")

        else:

            # Put into the table
            ##########################
            # column 0 = topic nr
            # column 1 = description
            # column 2 = words
            ##########################
            self.tableWidgetTrainSubmodel_4.clearContents()
            self.tableWidgetNewTopicName_4.clearContents()
            self.tableWidgetTrainSubmodel_4.setRowCount(len(topic_ids))
            self.tableWidgetNewTopicName_4.setRowCount(len(topic_ids))
            self.tableWidgetTrainSubmodel_4.setColumnCount(3)
            self.tableWidgetNewTopicName_4.setColumnCount(1)

            model_to_change = model.look_for_model(item.text(0))
            self.model_get_ids = model_to_change
            self.execute_in_thread(self.execute_get_topic_ids, self.do_after_getIds_change_description, False)

    def do_after_getIds_change_description(self):
        """Method to show the results of changing the topics' name of a model/
        submodel.
        """
        for i in np.arange(0, len(self.list_names), 1):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.tableWidgetTrainSubmodel_4.setItem(i, 0, item_topic_nr)
            self.tableWidgetNewTopicName_4.setItem(i, 0, item_topic_nr)
            item_topic_name = QtWidgets.QTableWidgetItem(str(self.list_names[i]))
            self.tableWidgetTrainSubmodel_4.setItem(i, 1, item_topic_name)
            item_topic_description = QtWidgets.QTableWidgetItem(str(self.list_description[i]))
            self.tableWidgetTrainSubmodel_4.setItem(i, 2, item_topic_description)

        self.tableWidgetTrainSubmodel_4.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetTrainSubmodel_4.resizeColumnsToContents()

        self.tableWidgetNewTopicName_4.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidgetNewTopicName_4.resizeColumnsToContents()

    def clicked_apply_changes(self):
        """Method to save  the topics' name of a model/
        submodel in both the pickle and file "topic_ids".
        """
        self.execute_in_thread(self.execute_apply_changes, self.after_apply_changes, True)

    def after_apply_changes(self):
        """Method to show the results of having applied the changes of the topics'
        namew in a model/submodel".
        """
        item = self.treeViewShowModelsToExpand_4.currentItem()
        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "Topics' description of model " + item.text(0) + " have been updated.")
        # self.logging.info("Topics' description of model " + item.text(0) + " have been updated.")

    def clicked_reset_changes(self):
        """Method to reset to no name of the topics' name of a model/
        submodel in both the pickle and file "topic_ids".
        """
        self.execute_in_thread(self.execute_reset_changes, self.after_reset_changes, True)

    def after_reset_changes(self):
        """Method to show the results reseting the changes of the topics'
        namew in a model/submodel".
        """
        item = self.treeViewShowModelsToExpand_4.currentItem()
        QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                          "Topics' description of model " + "''" + item.text(
                                              0) + "'' were changed to its default value.")
        # self.logging.info(
        #    "Topics' description of model " + "''" + item.text(0) + "'' were changed to its default value.")

    def clicked_plot_pyldavis(self):
        """Method to control the generation of the PyLDAVis".
        """
        self.inicio = time.time()
        self.model_to_plot = str(self.treeWidgetSelectModelToSeeDescription.currentItem().text(0))

        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)
        model_plot_pyldavis = model.look_for_model(self.model_to_plot)
        route_to_model_plot_pyldavis = model_plot_pyldavis.model_path
        file_pldavis = pathlib.Path(route_to_model_plot_pyldavis, pldavis).as_posix()

        if not os.path.exists(file_pldavis):
            self.execute_in_thread(self.execute_to_get_pyldavis, self.after_show_plot_pyldavis, True)
        else:
            self.after_show_plot_pyldavis()

    def after_show_plot_pyldavis(self):
        """Method to show the generated PyLDAVis".
        """
        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        model_plot_pyldavis = model.look_for_model(self.model_to_plot)
        route_to_model_plot_pyldavis = model_plot_pyldavis.model_path
        file_pldavis = pathlib.Path(route_to_model_plot_pyldavis, pldavis).as_posix()

        if self.web:
            self.web.setParent(None)
        if self.webExpand:
            self.webExpand.setParent(None)
        self.web = QWebEngineView()
        self.web.load(QUrl.fromLocalFile(file_pldavis))
        self.layoutPlot.addWidget(self.web)
        self.web.show()

        self.webExpand = QWebEngineView()
        self.webExpand.load(QUrl.fromLocalFile(file_pldavis))
        self.layoutPlot_2.addWidget(self.webExpand)
        self.webExpand.show()

        fin = time.time()
        print("Tiempo Pyldavis")
        print(fin - self.inicio)  # 1.0005340576171875

    def clicked_plot_diagnosis(self):
        """Method to control the generation of the PyLDAVis".
        """
        self.inicio = time.time()
        self.model_to_plot = str(self.treeWidgetSelectModelToSeeDiagnostics.currentItem().text(0))

        config.read(config_file)
        route_to_persistence = config['models']['persistence_selected']
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)
        model_plot_pyldavis = model.look_for_model(self.model_to_plot)
        route_to_model_plot_pyldavis = model_plot_pyldavis.model_path
        file_diagnostics = pathlib.Path(route_to_model_plot_pyldavis, diagnostics).as_posix()

        if not os.path.exists(file_diagnostics):
            print("The file was not generated with training.")
        else:
            if self.web_diag:
                self.web_diag.setParent(None)
            if self.webExpand_diag:
                self.webExpand_diag.setParent(None)

            self.web_diag = QWebEngineView()
            self.web_diag.load(QUrl.fromLocalFile(file_diagnostics))
            self.layoutPlotDiagnosis.addWidget(self.web_diag)
            self.web_diag.show()

            self.webExpand_diag = QWebEngineView()
            self.webExpand_diag.load(QUrl.fromLocalFile(file_diagnostics))
            self.layoutPlotDiagnosisExpand.addWidget(self.webExpand_diag)
            self.webExpand_diag.show()

            fin = time.time()
            print("Tiempo Pyldavis")
            print(fin - self.inicio)  # 1.0005340576171875
        return

    def clicked_selected_model_to_delete(self):
        """Method to link the model/submodel selected in the treeWidget
        "treeViewShowModelsToExpand_4" with its deletion.
        """
        self.tableWidgetSubmodelTrained_4.clearContents()
        return

    def clicked_delete_submodel(self):
        """Method to control the deletion of a model or submodel in the
        "Edit model" view.
        """
        model_to_delete_str = str(self.treeViewShowModelsToExpand_4.currentItem().text(0))

        if model_to_delete_str.startswith("model"):
            QtWidgets.QMessageBox.question(self, 'MusicalSpoon message',
                                           "Are you sure you want to delete the submodel " + '"' + model_to_delete_str + '"' + "? If you "
                                                                                                                               "delete this model, all its children submodels will be deleted as  well.",
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
                    self.statusBar().showMessage("You must have trained a model first in order to delete it.", 10000)
                    QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                                  "You must have trained a model first in order to delete it.")
                    return
                deleted = delete_submodel(model_to_delete_str)
            if not deleted:
                self.statusBar().showMessage("There are no submodels to delete.", 10000)
                QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "There are no submodels to delete.")
                return
            else:
                QtWidgets.QMessageBox.information(self, 'MusicalSpoon message',
                                                  "The submodel " + '"' + model_to_delete_str + '"' + " was deleted.")
                # self.logging.info("The submodel " + '"' + model_to_delete_str + '"' + " was deleted.")
                self.show_models_to_expand()
                self.show_models()
                self.refresh()
        return

    def clicked_showTopicsDragDiagnosis(self):

        config.read(config_file)
        project_path = config['files']['project_path']

        if self.treeWidgetSelectModelToDiagnostic.currentItem().text(0).lower == "models":
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "You must select an item within 'models'.")
            return
        if not self.treeWidgetSelectModelToDiagnostic.currentItem().text(0).lower().startswith("model") and \
                not self.treeWidgetSelectModelToDiagnostic.currentItem().text(0).lower().startswith("submodel"):
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message', "You must select an item, no an empty space.")
            return

        item = self.treeWidgetSelectModelToDiagnostic.currentItem()
        model_selected = item.text(0)

        self.statusBar().showMessage("You have selected : " + str(model_selected), 10000)
        # self.logging.info("Model " + str(model_selected) + " was selected for expansion")

        route_to_persistence = get_pickle(model_selected, project_path)
        infile = open(route_to_persistence, 'rb')
        model = pickle.load(infile)

        topic_ids = show_topics_to_expand_general(str(model_selected), model)
        ##########################
        # column 0 = model / topic nr / topic description
        ##########################
        self.tableDragFrom.clearContents()
        self.tableDragFrom.clearContents()
        self.tableDragFrom.setRowCount(len(topic_ids))
        self.tableDragFrom.setRowCount(len(topic_ids))
        self.tableDragFrom.setColumnCount(1)

        model_to_show = model.look_for_model(str(model_selected))

        list_description = []
        for i in np.arange(0, len(model_to_show.topics_models), 1):
            if str(type(model_to_show.topics_models[i])) == "<class 'Topic.Topic'>":
                list_description.append(
                    model_selected + " / " + str(i) + " / " + model_to_show.topics_models[i].get_description()[0][0])

        for i in np.arange(0, len(list_description), 1):
            item_topic = QtWidgets.QTableWidgetItem(str(list_description[i]))
            self.tableDragFrom.setItem(i, 0, item_topic)

        self.tableDragFrom.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableDragFrom.resizeColumnsToContents()

        self.tableDragFrom.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableDragFrom.resizeColumnsToContents()
        return

    def click_draw_diagnosis(self):

        if not self.comboBox.currentText():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select a measurement for the Y axis to be represented in the graph.")
        measurement = str(self.comboBox.currentText())

        if not self.comboBox2.currentText():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select a measurement for the X axis to be represented in the graph.")
        measurement2 = str(self.comboBox2.currentText())
        if not self.xaxis_name.text() or not self.yaxis_name.text() or not self.title_name.text():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "X-axis, Y-axis and Title must be filled up in order to proceed with the "
                                          "graph generation.")
            return
        text_xaxis = self.xaxis_name.text()
        text_yaxis = self.yaxis_name.text()
        text_title = self.title_name.text()
        save = self.name_save_plot.text()

        config.read(config_file)
        project_path = config['files']['project_path']

        figure_to_save = ""
        if save:
            figures_save = (pathlib.Path(project_path)) / "Figures"
            if not os.path.isdir(figures_save):
                os.mkdir(figures_save)
            figure_to_save = figures_save / text_title
            print(figure_to_save)

        diagnostics_paths_id_all = []
        for i in np.arange(0, self.tableDragTo.rowCount(), 1):
            model_name = self.tableDragTo.item(i, 0).text().split("/")[0][:-1]
            topic_id = self.tableDragTo.item(i, 0).text().split("/")[1][1]
            model_path = get_root_path(model_name, project_path)
            diagnostics_path = ((pathlib.Path(model_path)) / "diagnostics.xml").as_posix()
            diagnostics_paths_id_all.append([diagnostics_path, model_name, topic_id])

        x, y = plot_diagnostics(diagnostics_paths_id_all, measurement, measurement2, text_xaxis, text_yaxis, text_title, figure_to_save)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(text_title)
        ax.set_xlabel(text_xaxis)
        ax.set_ylabel(text_yaxis)
        # refresh canvas
        self.canvas.draw()

    def clear_tables_graph_diagnostics(self):
        self.tableDragFrom.clearContents()
        self.tableDragTo.clearContents()
        self.figure.clear()