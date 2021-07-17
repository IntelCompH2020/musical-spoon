# -*- coding: utf-8 -*-
"""
@author: Jesús Cid Sueiro
@modified_author: lcalv
******************************************************************************
***                             MAIN MENU                                  ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################

import os
import pathlib
import argparse
import configparser

# ########################
# Main body of application
# ########################

# ####################
# Read input arguments

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing project")
parser.add_argument('--source', type=str, default='../source_data',
                    help="path to the source data folder")
args = parser.parse_args()

# Read project_path
project_path = args.p
if args.p is None:
    while project_path is None or project_path == "":
        project_path = input('-- Write the path to the project to load or '
                            'create: ')
if os.path.isdir(args.p):
    option = 'load'
else:
    option = 'create'
active_options = None
query_needed = False

##############################################################################
#                              CONFIG FILE                                   #
##############################################################################
# Write paths in config file
file ='config_project.ini'
config = configparser.ConfigParser()
config.read(file)
config.set('files', 'source_path', args.source)
config.set('files', 'project_path', (pathlib.Path(project_path)).__str__())
with open(file, 'w') as configfile:
    config.write(configfile)
print(args.source)
print((pathlib.Path(project_path)).__str__())

##############################################################################
#                           LOCAL IMPORTS                                    #
##############################################################################
# -> They need to be here
from menu_code.menu_navigator.menu_navigator import MenuNavigator
from menu_code.task_manager import TaskManager

# # Create task manager object
tm = TaskManager(project_path, path2source=args.source)

# # ########################
# # Prepare user interaction
# # ########################

path2menu = pathlib.Path('config', 'options_menu.yaml')

paths2data = {'model': pathlib.Path(project_path,'models')}

# ##############
# Call navigator
# ##############
menu = MenuNavigator(tm, path2menu, paths2data)
menu.front_page(title="Hierarchical Topic Moedlling Using Mallet")
menu.navigate(option, active_options)


