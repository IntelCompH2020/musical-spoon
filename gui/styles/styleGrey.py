# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:56:19 2021

@author: lcalv
******************************************************************************
***                        CLASS STYLE GREY                                ***
******************************************************************************
Module that comprises the CSS style that defines the blue-grey style
that characterizes the application.
"""


class styleGrey:
    STYLE = """
    QDialog {
        background-color:#DFE6ED;
    }

    QLineEdit {
        background-color: #EBF2FA;
        border-radius: 5px;
        border: 2px solid #EBF2FA;
        padding-left: 10px;
    }
    QLineEdit:hover {
        border: 2px solid #A3A8AD;
    }
    QLineEdit:focus {
        border: 2px solid #A3A8AD;
    }

    QListWidget {	
        background-color: #EBF2FA;
        padding: 10px;
        border-radius: 5px;
        gridline-color: #DFE6ED;
        border-bottom: 1px solid #DFE6ED;
    }
    QListWidget::item{
        border-color: #DFE6ED;
        padding-left: 5px;
        padding-right: 5px;
        gridline-color: #DFE6ED;
    }
    QListWidget::item:selected{
        background-color: #C7CDDA;
    }
    QScrollBar:horizontal {
        border: none;
        background: #A3A8AD;
        height: 14px;
        margin: 0px 21px 0 21px;
        border-radius: 0px;
    }
    QScrollBar:vertical {
        border: none;
        background: #A3A8AD;
        width: 14px;
        margin: 21px 0 21px 0;
        border-radius: 0px;
    }
    QHeaderView::section{
        Background-color: #A3A8AD;
        max-width: 30px;
        border: 1px solid #A3A8AD;
        border-style: none;
        border-bottom: 1px solid #DFE6ED;
        border-right: 1px solid #DFE6ED;
    }
    QListWidget::horizontalHeader {	
        background-color: #A3A8AD;
    }
    QHeaderView::section:horizontal
    {
        border: 1px solid #DFE6ED;
        background-color: #A3A8AD;
        padding: 3px;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
    }
    QHeaderView::section:vertical
    {
        border: 1px solid #A3A8AD;
    }

    QTreeWidget {	
        background-color: #EBF2FA;
        padding: 10px;
        border-radius: 5px;
        gridline-color: #DFE6ED;
        border-bottom: 1px solid #DFE6ED;
    }
    QTreeWidget::item{
        border-color: #DFE6ED;
        padding-left: 5px;
        padding-right: 5px;
        gridline-color: #DFE6ED;
    }
    QTreeWidget::item:selected{
        background-color: #C7CDDA;
    }
    QScrollBar:horizontal {
        border: none;
        background: #A3A8AD;
        height: 14px;
        margin: 0px 21px 0 21px;
        border-radius: 0px;
    }
    QScrollBar:vertical {
        border: none;
        background: #A3A8AD;
        width: 14px;
        margin: 21px 0 21px 0;
        border-radius: 0px;
     }
    QHeaderView::section{
        Background-color: #A3A8AD;
        max-width: 30px;
        border: 1px solid #A3A8AD;
        border-style: none;
        border-bottom: 1px solid #DFE6ED;
        border-right: 1px solid #DFE6ED;
    }
    QTreeWidget::horizontalHeader {	
        background-color: #A3A8AD;
    }
    QHeaderView::section:horizontal
    {
        border: 1px solid #DFE6ED;
        background-color: #A3A8AD;
        padding: 3px;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
    }
    QHeaderView::section:vertical
    {
        border: 1px solid #A3A8AD;
    }

    QTreeView {	
        background-color: #EBF2FA;
        padding: 10px;
        border-radius: 5px;
        gridline-color: #DFE6ED;
        border-bottom: 1px solid #DFE6ED;
    }
    QTreeView::item{
        border-color: #DFE6ED;
        padding-left: 5px;
        padding-right: 5px;
        gridline-color: #DFE6ED;
    }
    QTreeView::item:selected{
        background-color: #C7CDDA;
    }
    QScrollBar:horizontal {
        border: none;
        background: #A3A8AD;
        height: 14px;
        margin: 0px 21px 0 21px;
        border-radius: 0px;
    }
    QScrollBar:vertical {
        border: none;
        background: #A3A8AD;
        width: 14px;
        margin: 21px 0 21px 0;
        border-radius: 0px;
     }
    QHeaderView::section{
        Background-color: #A3A8AD;
        max-width: 30px;
        border: 1px solid #A3A8AD;
        border-style: none;
        border-bottom: 1px solid #DFE6ED;
        border-right: 1px solid #DFE6ED;
    }
    QTreeView::horizontalHeader {	
        background-color: #A3A8AD;
    }
    QHeaderView::section:horizontal
    {
        border: 1px solid #DFE6ED;
        background-color: #A3A8AD;
        padding: 3px;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
    }
    QHeaderView::section:vertical
    {
        border: 1px solid #A3A8AD;
    }

    QTableWidget {	
        background-color: rgb(235, 242, 250);
        padding: 10px;
        border-radius: 5px;
        gridline-color: #DFE6ED;
        border-bottom: 1px solid #DFE6ED;
    }
    QTableWidget::item{
        border-color: #DFE6ED;
        padding-left: 5px;
        padding-right: 5px;
        gridline-color: #DFE6ED;
    }
    QTableWidget::item:selected{
        background-color: #C7CDDA;
    }
    QScrollBar:horizontal {
        border: none;
        background: #A3A8AD;
        height: 14px;
        margin: 0px 21px 0 21px;
        border-radius: 0px;
    }
    QScrollBar:vertical {
        border: none;
        background: #A3A8AD;
        width: 14px;
        margin: 21px 0 21px 0;
        border-radius: 0px;
     }
    QHeaderView::section{
        Background-color: #A3A8AD;
        max-width: 30px;
        border: 1px solid #A3A8AD;
        border-style: none;
        border-bottom: 1px solid #DFE6ED;
        border-right: 1px solid #DFE6ED;
    }
    QTableWidget::horizontalHeader {	
        background-color: #A3A8AD;
    }
    QHeaderView::section:horizontal
    {
        border: 1px solid #DFE6ED;
        background-color: #A3A8AD;
        padding: 3px;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
    }
    QHeaderView::section:vertical
    {
        border: 1px solid #A3A8AD;
    }

    QPushButton {
        background-color: #A3A8AD;
    }
    QPushButton:hover {
        background-color: #A3A8AD;
    }
    QPushButton:pressed {	
        background-color: #A3A8AD;
    }

    """
