# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS TOPIC                                ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

class Topic():
    """
    Attributes:
    ----------
        * id_topic
        * betas            = topic distribution over the vocabulary; 
                             topic-word-weights-file: unnormalized weights for 
                             every topic and word type.
        * description      = chemical description of the topic (list)
                             output-topic-keys: Textual description of the topics
                             for the given number of words selected .
        * description_name = title given by the user to describe the topic
    """
    
    def __init__(self, id_topic, betas, description, description_name, coherence, entropy):
        self.id_topic = id_topic
        self.betas = betas
        self.description = description
        self.description_name = description_name
        self.coherence = coherence
        self.entropy = entropy
    
    def __str__(self):
        return str(self.id_topic)
    
    @classmethod
    def save_betas(cls, file, topic_ids):
        """Sets the value of all topic's arrays of betas (from a model)
           Since the weights provided in the file are unnormalized, it 
           normalizes them before creating each topic's array of betas
        
        Parameters:
        ----------
            * file        - Path to the file containing the unnormalized weights
                            for every topic and word .
            * topic_ids   - List structure containing all the topics's ids
        Returns:
        -------
            * dictionary - List structure containig all the words that conform
                           the model.
            * betas_all - List structure containing a np.array for each topic's betas
        """
        col_names = ["topic", "word", "weight"]
        df = pd.read_csv(file, names=col_names, sep = "\t")
        topic_values = df.values[:,0].tolist()
        word_values = df.values[:,1].tolist()
        weight_values = df.values[:,2].tolist()
        
        betas_all = []
        words_all = []
        # For each topic
        for j in np.arange(0,len(topic_ids),1):
            # Ids that are equal to that topic
            betas = np.array([],dtype = float)
            words = []
            for i in np.arange(0,len(topic_values),1):
                if topic_values[i] == topic_ids[j]:
                    betas = np.append(betas, weight_values[i])
                    words.append(word_values[i])
            betas_norm = normalize(betas[:,np.newaxis], axis = 0, norm='l1').ravel()
            betas_all.append(betas_norm)
            words_all.append(words)
        dictionary = words_all[0]
        return dictionary, betas_all
    
    
    @classmethod
    def save_desciptions(cls, file, topic_ids):
        """Save the value of all topic's description (from a model)
        
        Parameters:
        ----------
            * file               - Path to the file containing the unnormalized
                                   weights for every topic and word 
            * topic_ids          - List structure containing all the topics's ids
        Returns:
        ----------
            * topic-keys-weights - List sturcture containing the size (weight)
                                   of each topic.
            * description_all    - List structure containing a list for each 
                                   topic's chemical description
        """
        topics_keys_df = pd.read_csv(file, sep = "\t", header = None)
        topic_ids_keys = topics_keys_df.values[:,0].tolist()
        topic_keys_weight = topics_keys_df.values[:, 1].tolist()
        topic_keys_description = topics_keys_df.values[:,2:].tolist()
        description_all = []
        # For each topic
        for j in np.arange(0,len(topic_ids),1):
            # Ids that are equal to that topic
            description = []
            for i in np.arange(0,len(topic_ids_keys),1):
                if topic_ids_keys[i] == topic_ids[j]:
                    description.append(topic_keys_description[i])
            description_all.append(description)
        return topic_keys_weight, description_all
                    
        
    def create_topics(num_topics, topic_word_weights, topic_keys):
        """Creates a list containing all the topics of the model.
        
        Parameters:
        ----------
            * num_topics         - Number of topics with which the model 
                                   is being trained.
            * topic_word_weights - Path to the file containing the unnormalized
                                   weights for every topic and word .
            * topic-keys         - Path to the file containing the textual 
                                   description of the topics for the given 
                                   number of words selected.
        Returns:
        -------
            * topics -             List structure in which the topic objects 
                                   are going to be saved.
            * dictionary -         List structure containig all the words that
                                   conform the model.
            * topic-keys-weights-  List sturcture containing the size (weight)
                                   of each topic.
        """
        topics = []
        topic_ids = np.arange(0,int(num_topics),1)
        
        dictionary, betas_all = Topic.save_betas(topic_word_weights, topic_ids)
        
        if np.min(betas_all) < 1e-12:
            betas_all += 1e-12
        topic_entropies = -np.sum(betas_all * np.log(betas_all), axis=1)
        topic_entropies = topic_entropies/np.log(len(betas_all[0]))
        
        topic_keys_weight, description_all = Topic.save_desciptions(topic_keys,topic_ids)
        for i in np.arange(0,len(betas_all),1):
            topic = Topic(i, betas_all[i], description_all[i], "", 0, topic_entropies[i])
            topics.append(topic)
        return topics, dictionary, topic_keys_weight

    def get_description(self):
        """Gets topic's description.
        
        Returns:
        -------
            * description - Topic's description.
        """
        return self.description
    
    def get_topics(self):
        """Gets topic's id.
        
        Returns:
        -------
            * description - Topic's id.
        """
        return self.id_topic
    
    def get_description_name(self):
        """Gets topic's description name.
        
        Returns:
        -------
            * description_name - Topic's description name.
        """
        return self.description_name
    
    def set_description_name(self, name):
        """Sets topic's description name.
        
        Returns:
        -------
            * description_name - Topic's description name.
        """
        self.description_name = name
    
    def set_coherence(self, coherence):
        """Sets topic's coherence .
    
        """
        self.coherence = coherence
    
    def set_entropy(self, entropy):
        """Sets topic's coherence .
    
        """
        self.entropy = entropy