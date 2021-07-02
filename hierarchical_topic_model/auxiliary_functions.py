# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                        AUXILIARY_FUNCTIONS                             ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import re
import subprocess
import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET

##############################################################################
#                                FUNCTIONS                                   #
##############################################################################
def remove_matches(text, to_remove):
    """Replaces characters that match not wanted patterns inside a text. 
     
    Parameters:
    ----------
    * text      - Expression to be transformed.
    * to_remove - List of patterns that will be replaced.

    Returns
    -------
    * text      - The replaced text.
    """
    for item in to_remove:
        text = re.sub(item, '', text)
    return text

def replace(text,to_replace):
    """Replace a pattern inside a text. 
       
    Parameters:
    ----------
    * text       - Expression to be transformed.
    * to_replace - List of contractions that will be replaced.
    
    Returns
    -------
    * text - The replaced text.
    """
    for(raw,rep) in to_replace:
        regex = re.compile(raw)
        text = regex.sub(rep,text)
    return text

def cmd(command):
    """Executes a cmd commnad. 
    
    Parameters:
    ----------
    * command - Command to be executed.
    
    """
    p = subprocess.run(command, shell=True, capture_output=True)
    #print(p.stdout.decode())
    #print(p.returncode)
    #print(p.stderr)
    return

def xml_dir(pth, et_element=None):
    if et_element is None:
        et_element = ET.Element(pth.name)
    else:
        et_element = ET.SubElement(et_element, pth.name)

    for directory in (fle for fle in pth.iterdir() if fle.is_dir()):
        print(directory)
        xml_dir(directory, et_element)

    return et_element



def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i