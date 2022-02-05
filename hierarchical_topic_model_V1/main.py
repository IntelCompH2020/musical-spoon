import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from pathlib import Path
import configparser

from gui import *
from PyQt5.QtGui import QIcon


class PreConfig(QDialog):
    def __init__(self):
        super(PreConfig, self).__init__()
        loadUi("UIS/menuConfig.ui", self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("MusicalSpoon")
        self.setStyleSheet(styleGrey.STYLE)
        font = QtGui.QFont('Arial')
        font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setPixelSize(10)
        self.setFont(font)

        # Get home in any op
        self.home = str(Path.home())
        self.projectFolder = ""
        self.databaseFile = ""
        self.malletPath = ""

        self.selectProjectFolder.clicked.connect(self.getProjectFolder)
        self.selectDatabase.clicked.connect(self.getDatabaseFile)
        self.selectMalletPath.clicked.connect(self.getMalletPath)
        self.start.clicked.connect(self.startApplication)

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

    def startApplication(self):
        # Write in the config file, also in the default values
        if self.projectFolder == "" or self.databaseFile == "" or self.malletPath == "":
            QtWidgets.QMessageBox.warning(self, 'MusicalSpoon message',
                                          "The three input parameters must be filled in to proceed to the main menu.")
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
        mainWindow = UI_MainWindow()
        widget.addWidget(mainWindow)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        widget.resize(1680, 960)
        return


# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, False)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

# Main
app = QApplication(sys.argv)
app.setWindowIcon(QIcon("logo.png"))
widget = QtWidgets.QStackedWidget()
widget.setWindowTitle("MusicalSpoon")
width = widget.frameGeometry().width()
height = widget.frameGeometry().height()
print(height)
print(width)
configWindow = PreConfig()
widget.addWidget(configWindow)
widget.resize(1540, 880)
widget.show()
sys.exit(app.exec_())
