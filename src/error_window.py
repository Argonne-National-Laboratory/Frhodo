#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import sys, logging, traceback
from logging.handlers import RotatingFileHandler

from qtpy.QtWidgets import QApplication, QDialog
from qtpy import uic, QtCore, QtGui

path = {}

class Error_Window(QDialog):
    def __init__(self, app, path, error_text):
        super().__init__()
        uic.loadUi(str(path['main']/'UI'/'error_window.ui'), self)
        self.setWindowIcon(QtGui.QIcon(str(path['main']/'UI'/'graphics'/'main_icon.png')))
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint |
                            QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        
        error_text = error_text.rstrip('\n')
        self.error_text_box.setText(error_text)
        
        self.copy_button.clicked.connect(self.copy)
        self.close_button.clicked.connect(self.closeEvent)
        self.installEventFilter(self)
        
        self.app = app

        self.exec_()
    
    def copy(self):
        def deselectAll():
            cursor = self.error_text_box.textCursor()
            cursor.clearSelection()
            self.error_text_box.setTextCursor(cursor)
            
        self.error_text_box.selectAll()
        self.error_text_box.copy()
        deselectAll()
    
    def eventFilter(self, obj, event):
        # intercept enter, space and escape
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() in [QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space]:
                self.close_button.click()
                return True
                    
        return super().eventFilter(obj, event)
    
    def closeEvent(self, event):    # TODO: Not closing the program properly
        #QApplication.quit() # some errors can be recovered from, maybe I shouldn't autoclose the program
        self.app.quit() # some errors can be recovered from, maybe I shouldn't autoclose the program

def excepthookDecorator(app, parent_path, shut_down):
    path = parent_path
    
    def excepthook(type, value, tback):
        shut_down['bool'] = True

        # log the exception
        path['log'] = path['appdata']/'error.log'
        
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        
        log_handler = RotatingFileHandler(path['log'], mode='a', maxBytes=1/2*1024*1024, # maximum of 512 kB
                                         backupCount=1, encoding=None, delay=0)        # maximum of 2 error files
        log_handler.setFormatter(log_formatter)
        log_handler.setLevel(logging.DEBUG)

        app_log = logging.getLogger('root')
        app_log.setLevel(logging.DEBUG)
        app_log.addHandler(log_handler)
        
        text = "".join(traceback.format_exception(type, value, tback))   
        app_log.error(text)

        # call the default handler
        sys.__excepthook__(type, value, tback)
        
        Error_Window(app, path, text)    
        
    return excepthook