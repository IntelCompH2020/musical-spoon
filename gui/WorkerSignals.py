"""
Created on Tue Mar  2 13:15:57 2021

@author: lcalv
******************************************************************************
***                        CLASS WORKER SIGNALS                            ***
******************************************************************************
Module that defines the signals that are available from a running
worker thread, the supported signals being “finished” (there is no more data to process), “error”, “result” (object data returned from processing) and “progress” (a
numerical indicator of the progress that has been achieved at a particular moment).
It also has been created based on the analogous class provided by:
https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################

from PyQt5 import QtCore


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
