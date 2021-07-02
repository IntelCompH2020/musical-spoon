# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:37:31 2021

@author: lcalv
"""


class MessagesGui:
    INFO_SELECT_DATASET = 'Select a dataset for training the model. You can open the\n files by left clicking with ' \
                          'the mouse and select open file. '

    INFO_LOAD_FILES = 'Create a new model or select a new model to use as father\n for the model. Once you have ' \
                      'create the model, you must\n click over one of the models in order to select it. '

    INFO_TRAIN_MODEL = 'For the model above selected, choose the number of topics\n that you want to use for training ' \
                       'the model, and then click\n train. Then wait until the model is trained. It may take a little. '

    INFO_SELECT_SUBMODEL = 'Select the model/submodel that you want to expand.'

    INFO_SELECT_TOPIC_TO_EXPAND = 'Select the topic belonging to the model/submodel above selected\n from which you ' \
                                  'want to create the submodel. The words belonging\n to a submodel will be those ' \
                                  'that belong to the chosen topic in the \n selected model/submodel. Then, ' \
                                  'select the number of topics to \n you want to use for the training and click train ' \
                                  'submodel. '

    INFO_SHOW_DESCRIPTION = 'Select a model/submodel from which you want to see its description.\n On the right you ' \
                            'can see a PLDAVIS diagram of the selected model. '

    INFO_INSERT_DESCRIPTION = 'Select the model for which you want to insert\n the description for one or more of its ' \
                              'topics. '

    INFO_DELETE_SUBMODEL = 'Select the submodel that you want to delete.\n For deleting ' \
                           'a model you must go to the second tab. '
