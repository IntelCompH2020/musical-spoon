# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:19:34 2021

@author: lcalv
******************************************************************************
***                           CLASS WORKER                                 ***
******************************************************************************
Module that inherits from QRunnable and is used to handler worker
thread setup, signals and wrap-up. It has been created based on the analogous
class provided by:
https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/.
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################

import sys
import traceback
from PyQt5 import QtCore

from gui.worker_signals import WorkerSignals


class Worker(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        try:
            self.signals.started.emit()
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            print("final signal emitted")
            self.signals.finished.emit()  # Done
