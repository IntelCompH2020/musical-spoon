# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             EXEC_IN_SERVER                             ***
******************************************************************************
A script to extract the results from a pickle file of hierarchical topic
models according to the specifications of "exec_in_server.py" and save them
in a dataframe for posterior analysis.
"""
import os
import pathlib
import pickle
import xml.etree.ElementTree as ET
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import models

##############################################################################
#                                IMPORTS                                     #
##############################################################################


##############################################################################
# Load corpus
corpus_to_load = 'C:\\Users\\lcalv\\OneDrive\\Documentos\\MASTER\TFM_salud\\cord19.pickle'
persistence_path = "D:\\project_cord_all_v2\\persistence"
df1_pickle = "D:\\project_cord_all_v2\\df.pickle"
df2_pickle = "D:\\project_cord_all_v2\\df2.pickle"
df_final_results_pickle = "D:\\project_cord_all_v2\\df_final_results.pickle"
infile = open(corpus_to_load, 'rb')
corpus = pickle.load(infile)
##############################################################################

##############################################################################
##############################################################################
# Function to get submodels generated with v1 or v2
def get_characteristics_submodel(str_start, str_end, str_not_end, version, submodel_names, submodel_coherences, submodel_entropies, submodel_nr_documents, submodel_threshold):
    submodel_indexes = []
    for i in range(len(submodel_names)):
        if (submodel_names[i].startswith(str_start) and submodel_names[i].endswith(str_end) and not submodel_names[i].endswith(str_not_end)):
            submodel_indexes.append(i)
    submodel_coherences_ = np.array(submodel_coherences)
    submodel_coherences_ = submodel_coherences_[submodel_indexes]
    submodel_coherences_ = list(map(float, submodel_coherences_))

    submodel_names_ = [i for i in submodel_names if (i.startswith(
        str_start) and i.endswith(str_end) and not i.endswith(str_not_end))]

    submodel_entropies_all = np.array(submodel_entropies)
    submodel_entropies_all = submodel_entropies_all[submodel_indexes]
    submodel_entropies_all = list(submodel_entropies_all)
    submodel_entropies_ = []
    for i in submodel_entropies_all:
        s = list(map(float, i))
        submodel_entropies_.append(sum(s)/len(s))

    submodel_nr_docs_ = np.array(submodel_nr_documents)
    submodel_nr_docs_ = submodel_nr_docs_[submodel_indexes]
    submodel_nr_docs = list(map(int, submodel_nr_docs_))

    if version == "v2":
        submodel_thrs_ = np.array(submodel_threshold)
        submodel_thrs_ = submodel_thrs_[submodel_indexes]
        submodel_thrs_ = list(map(float, submodel_thrs_))
    else:
        submodel_thrs_ = 0

    return submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_


def get_submodel_values(file, corpus):
    # Load model
    infile = open(pathlib.Path(file), 'rb')  # -->> coger el dic
    model = pickle.load(infile)
    # Get the topic words from the model
    topics = []
    for i in model.topics_models:
        if str(type(i)) == "<class 'Topic.Topic'>":
            topic = i.description[0][0].split(" ")
            topic = topic[0:len(topic)-1]
            topics.append(topic)
    # Dictionary creation
    D = gensim.corpora.Dictionary(corpus)
    # Calculate average coherence of the root model
    cm = models.CoherenceModel(
        topics=topics,
        texts=corpus,
        dictionary=D,
        coherence="c_v")
    coherence = cm.get_coherence()
    nr_docs_root = len(model.thetas)  # Nr of documents of the root model
    thr_root = 0
    # Get average entropy of the root model
    diagnostics_path = ((pathlib.Path(model.model_path)) /
                        "diagnostics.xml").as_posix()
    tree = ET.parse(diagnostics_path)
    root = tree.getroot()
    entropies_root = [child.get("document_entropy")
                      for child in root if child.tag == 'topic']
    entropy_root_f = list(map(float, entropies_root))
    entropy_root = sum(entropy_root_f)/len(entropy_root_f)
    # Get info for each of the submodels
    submodel_corpora = []
    submodel_dics = []
    submodel_topics = []
    submodel_names = []
    submodel_coherences = []
    submodel_topic_from = []
    submodel_threshold = []
    submodel_entropies = []
    submodel_nr_documents = []
    id = 0
    for i in model.topics_models:
        if str(type(i)) == "<class 'Model.Model'>":
            print(str(id) + "/" + str(len(model.topics_models)))
            # Append to model name
            submodel_names.append(i.model_name)

            # Append number of documents
            submodel_nr_documents.append(len(i.thetas))

            # Topic from which the submodel comes
            topic = i.model_name.split("Topic_")[1].split("_")[0]
            submodel_topic_from.append(topic)

            # Get threhsolds (if it is the case)
            if "v2" in i.model_name:
                thr = i.model_name.split("v2_")[1].split("_")[0]
                submodel_threshold.append(thr)
            else:
                submodel_threshold.append("v1")

            # Get corpus
            submodel_path = i.model_path
            submodel_corpus = pathlib.Path(submodel_path) / "submodel.txt"
            corpus = []
            with open(submodel_corpus, 'rb') as f:
                for myline in f:
                    a = str(myline).split("0 ")[1].split("\\")[0].split(" ")
                    corpus.append(a)
            submodel_corpora.append(corpus)

            # Build dictionary
            D = gensim.corpora.Dictionary(corpus)
            submodel_dics.append(D)
            topics = []
            for j in i.topics_models:
                if str(type(j)) == "<class 'Topic.Topic'>":
                    topic = j.description[0][0].split(" ")
                    topic = topic[0:len(topic)-1]
                    topics.append(topic)
                    submodel_topics.append(topics)

            # Get average coherence
            cm = models.CoherenceModel(
                topics=topics,
                texts=corpus,
                dictionary=D,
                coherence="c_v")
            coherence = cm.get_coherence()
            submodel_coherences.append(coherence)

            # Get entropy
            diagnostics_path = (
                (pathlib.Path(i.model_path)) / "diagnostics.xml").as_posix()
            tree = ET.parse(diagnostics_path)
            root = tree.getroot()
            entropies = [child.get("document_entropy")
                         for child in root if child.tag == 'topic']
            submodel_entropies.append(entropies)
        id += 1
    return submodel_names, submodel_coherences, submodel_threshold, submodel_entropies, submodel_nr_documents, model


def parse_to_results(results, id_model, submodel_names, submodel_coherences, submodel_threshold, submodel_entropies, submodel_nr_documents, model):
    # Get each of the submodels characterisitcs
    for i in np.arange(0, model.num_topics, 1):
        # V2 - 4 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "4_topics", "v1_4_topics", "v2",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V2- 5 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "5_topics", "v1_5_topics", "v2",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V2 - 6 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "6_topics", "v1_6_topics", "v2",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V2 - 7 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "7_topics", "v1_7_topics", "v2",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V2 - 8 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "8_topics", "v1_8_topics", "v2",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        #####################################

        # V1 - 4 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "4_topics", "v2_4_topics", "v1",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V1- 5 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "5_topics", "v2_5_topics", "v1",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V1 - 6 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "6_topics", "v2_6_topics", "v1",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V1 - 7 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "7_topics", "v2_7_topics", "v1",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])

        # V1 - 8 topics
        submodel_coherences_, submodel_names_, submodel_entropies_, submodel_nr_docs_, submodel_thrs_ = get_characteristics_submodel("SubmodelFromTopic_" + str(i), "8_topics", "v2_8_topics", "v1",
                                                                                                                                     submodel_names, submodel_coherences, submodel_entropies,
                                                                                                                                     submodel_nr_documents, submodel_threshold)
        results.append([submodel_coherences_, submodel_names_,
                       submodel_entropies_, submodel_nr_docs_, submodel_thrs_, id_model])
    return results


def save_in_pickle(structure, pickle_to_save_in):
    with open(pickle_to_save_in, 'wb') as f:
        pickle.dump(structure, f)

###############################################################################
###############################################################################


sources = []
# añadir sólo si endswith 10 topics para el mío
for x in os.listdir(persistence_path):
    if x.endswith("10_topics.pickle"):
        print(pathlib.Path(persistence_path) / x)
        sources.append(pathlib.Path(persistence_path) / x)
print(sources)

results = []
id_model = 1
for i in range(len(sources)):
    print("Processing model " + str(id_model))
    submodel_names, submodel_coherences, submodel_threshold, submodel_entropies, submodel_nr_documents, model = get_submodel_values(
        sources[i], corpus)
    results = parse_to_results(results, id_model, submodel_names, submodel_coherences,
                               submodel_threshold, submodel_entropies, submodel_nr_documents, model)
    id_model += 1
df = pd.DataFrame(results, columns=[
                  'Coherences', 'Names', 'Entro', 'NrDocs', 'Thr', 'Model'])
save_in_pickle(df, df1_pickle)

results_all_all = []
for i in df.index:
    row_names = df["Names"][i]
    for j in range(len(row_names)):
        if(df["Thr"][i] == 0):
            results_all_all.append([df["Coherences"][i][j], df["Names"][i][j],
                                   df["Entro"][i][j], df["NrDocs"][i][j], df["Thr"][i], df["Model"][i]])
        else:
            results_all_all.append([df["Coherences"][i][j], df["Names"][i][j],
                                   df["Entro"][i][j], df["NrDocs"][i][j], df["Thr"][i][j], df["Model"][i]])
df2 = pd.DataFrame(results_all_all, columns=[
                   'Coherences', 'Names', 'Entro', 'NrDocs', 'Thr', 'Model'])

for i in df2.index:
    if("v2" in df2["Names"][i]):
        if(df2["Thr"][i] == 0):
            thr = df2["Names"][i].split("v2_")[1].split("_")[0]
            df2["Thr"][i] = thr

df2 = df2.drop_duplicates()
save_in_pickle(df2, df2_pickle)


iter_topics = [4, 5, 6, 7, 8]
iter_models = [1, 2, 3, 4, 5]

final_results = []
for topic_trained in iter_topics:
    for topic_from in np.arange(0, 5, 1):
        starts = "SubmodelFromTopic_" + str(topic_from)
        for thr in np.arange(0.1, 1, 0.1):
            ending = "{:.1f}".format(thr) + "_" + \
                str(topic_trained) + "_topics"
            cohrs = df2[(df2.Thr > 0) & (df2.Names.str.endswith(ending)) & (
                df2.Names.str.startswith(starts))]["Coherences"]
            entro = df2[(df2.Thr > 0) & (df2.Names.str.endswith(ending)) & (
                df2.Names.str.startswith(starts))]["Entro"]
            nrdocs = df2[(df2.Thr > 0) & (df2.Names.str.endswith(ending)) & (
                df2.Names.str.startswith(starts))]["NrDocs"]
            final_results.append([thr, topic_from, topic_trained, cohrs.mean(
            ), cohrs.std(), entro.mean(), entro.std(), nrdocs.mean(), nrdocs.std()])
        ending = "v1_" + str(topic_trained) + "_topics"
        cohrs = df2[(df2.Thr == 0) & (df2.Names.str.endswith(ending)) & (
            df2.Names.str.startswith(starts))]["Coherences"]
        entro = df2[(df2.Thr == 0) & (df2.Names.str.endswith(ending))
                    & (df2.Names.str.startswith(starts))]["Entro"]
        nrdocs = df2[(df2.Thr == 0) & (df2.Names.str.endswith(ending)) & (
            df2.Names.str.startswith(starts))]["NrDocs"]
        thr = 0
        final_results.append([thr, topic_from, topic_trained, cohrs.mean(
        ), cohrs.std(), entro.mean(), entro.std(), nrdocs.mean(), nrdocs.std()])

df_final_results = pd.DataFrame(final_results, columns=[
                                'Thr', 'TopFrom', 'TrainTops', 'Cohr_m', 'Cohr_std', 'Entro_m', 'Entro_std', 'NrDocs_m', 'NrDocs_std'])
save_in_pickle(df_final_results, df_final_results_pickle)
