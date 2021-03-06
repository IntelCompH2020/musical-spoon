# -*- coding: utf-8 -*-
"""

@author: lcalv
******************************************************************************
***                                 MAIN                                   ***
******************************************************************************
The entry point of the application, that contains the definition of the PreConfig
class, namely, the GUI’s initiation page. It extends from the QDialog class
and loads the UI file in which the design for this window has been implemented.
Besides, it controls the transition to the main application window once the user
has set the mandatory three input parameters.
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from pathlib import Path
import configparser

from gui.gui import *
from gui.styles.styleGrey import styleGrey
from gui.messages_gui import MessagesGui
from PyQt5.QtGui import QIcon


class PreConfig(QDialog):
    def __init__(self, widget):
        super(PreConfig, self).__init__()
        loadUi("gui/UIS/menuConfig.ui", self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("MusicalSpoon")
        self.setStyleSheet(styleGrey.STYLE)

        # Get home in any op
        self.home = str(Path.home())
        self.projectFolder = ""
        self.databaseFile = ""
        self.malletPath = ""
        self.version = "v1"  # default version is v1
        self.widget = widget

        self.selectProjectFolder.clicked.connect(self.getProjectFolder)
        self.selectDatabase.clicked.connect(self.getDatabaseFile)
        self.selectMalletPath.clicked.connect(self.getMalletPath)
        self.start.clicked.connect(self.startApplication)
        self.checkBoxV1.clicked.connect(self.getVersionHTM)
        self.checkBoxV2.clicked.connect(self.getVersionHTM)

        # Info buttons
        self.infoButtonV1.setIcon(QIcon('gui/Images/help2.png'))
        self.infoButtonV2.setIcon(QIcon('gui/Images/help2.png'))
        self.infoButtonV1.setToolTip(MessagesGui.INFO_HTM_V1)
        self.infoButtonV2.setToolTip(MessagesGui.INFO_HTM_V2)

    def getProjectFolder(self):
        self.projectFolder = QFileDialog.getExistingDirectory(
            self, 'Select directory', self.home)
        self.showProjectFolder.setText(self.projectFolder)

    def getDatabaseFile(self):
        self.databaseFile = QFileDialog.getOpenFileName(
            self, "Open File", "~", "Text Files (*.txt)")[0]
        self.showDatabaseFolder.setText(self.databaseFile)

    def getMalletPath(self):
        self.malletPath = QFileDialog.getOpenFileName(
            self, 'Select executable', self.home)[0]
        self.showMalletPath.setText(self.malletPath)

    def getVersionHTM(self):
        if self.checkBoxV1.isChecked() and self.checkBoxV2.isChecked():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must only check one of the version; only one hierarchical topic model "
                                          "algorithm can be used at a time")
            self.checkBoxV1.setChecked(False)
            self.checkBoxV2.setChecked(False)
        elif self.checkBoxV1.isChecked():
            self.version = "v1"
        elif self.checkBoxV2.isChecked():
            self.version = "v2"
        return

    def startApplication(self):
        # Write in the config file, also in the default values
        if self.projectFolder == "" or self.databaseFile == "" or self.malletPath == "":
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "The three input parameters must be filled in to proceed to the main menu.")
            return

        if self.checkBoxV1.isChecked() and self.checkBoxV2.isChecked():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must only check one of the version; only one hierarchical topic model "
                                          "algorithm can be used at a time")
        elif not self.checkBoxV1.isChecked() and not self.checkBoxV2.isChecked():
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "You must select a version of the hierarchical topic model algorithm to use.")
            return

        config_file = 'config_project.ini'
        config = configparser.ConfigParser()
        config.read(config_file)

        config.set('files', 'project_path', self.projectFolder)
        config.set('default', 'project_path', self.projectFolder)
        config.set('files', 'source_path', self.databaseFile)
        config.set('default', 'source_path', self.databaseFile)
        config.set('mallet', 'mallet_path', self.malletPath)
        config.set('default', 'mallet_path', self.malletPath)

        with open(config_file, 'w') as configfile:
            config.write(configfile)

        # Change to gui
        mainWindow = UI_MainWindow(self.version)
        self.widget.addWidget(mainWindow)
        self.widget.setCurrentIndex(self.widget.currentIndex() + 1)
        self.widget.showMaximized()
        return


def main():
    # Main
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo.png"))
    widget = QtWidgets.QStackedWidget()
    widget.setWindowTitle("MusicalSpoon")
    width = widget.frameGeometry().width()
    height = widget.frameGeometry().height()
    print(height)
    print(width)
    configWindow = PreConfig(widget)
    widget.addWidget(configWindow)
    widget.showMaximized()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
