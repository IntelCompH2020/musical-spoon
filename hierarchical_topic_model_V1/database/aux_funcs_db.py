# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                            AUX_FUNCS_DB                                ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
import langid
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, Counter, sent_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk import Counter
import numpy as np

from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()
stopwords_es = stopwords.words('spanish')
moreStopWords = ['entonces', 'según', 'acorde', 'si']
stopwords_es.extend(moreStopWords)


##############################################################################
#                                FUNCTIONS                                   #
##############################################################################
def tf_idf(text):
    """
    Computes the TF-IDF of a text - using every sentence as a separate "document".
    Returns a list of tuples with words and weights.
    """
    texts = [preprocessing(sentence) for sentence in sent_tokenize(text)]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    tfidf_weights = {dictionary.get(id): value
                     for doc in corpus_tfidf
                     for id, value in doc}
    sorted_tfidf_weights = sorted(tfidf_weights.items(), key=lambda w: w[1])

    return sorted_tfidf_weights


def replace(text, patterns):
    for(raw, rep) in patterns:
        regex = re.compile(raw)
        text = regex.sub(rep, text)
    return text


def add_to_list_from_dict(dict_original, content_str, content_list):
    """Takes an element from a dictionary and adds it to a list, 
       replacing None elements with an empty string.
       Used for give the structure to the elements of the news when
       adding them to the database

    Parameters:
    ----------
    * dict_original - Dictionary to take the elements from 
    * content_str - String refering to the key of the dictionary
    * content_list - List in which the elements are going to be saved

    """
    content = dict_original[content_str]
    if content == None:
        content = ""
    content_list.append(content)


def is_spanish(text):
    """CChecks if the context of a given text is written in Spanish. In case it is, it returns true. Otherwise, it returns false.

    Parameters:
    ----------
    * text - Text whose content's language is going to be checked

    Returns:
    --------
    * True if the content of the text is written in Spanish.

    """

    lang = langid.classify(text)
    if lang and 'es' in lang[0]:
        return True
    return False


def ensureUtf(s):
    """Ensure that the string given as parameter follows UTF-8. In case it does not, it converts it to such a format and returns the converted string.

    Parameters:
    ----------
    * s - String to be checked

    Returns:
    --------
    * s - the UTF-8 transformed string
    """
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as e:
        if e.reason == 'surrogates not allowed':
            s = s.encode('utf-8', "backslashreplace").decode('utf-8')
    return s


def normalize(s):
    """It replaces in the given string the characters defined in "Replacements".

    Parameters:
    ----------
    * s - String to be checked

    Returns:
    --------
    * s - the UTF-8 transformed string
    """
    replacements = (
        ("à", "a"),
        ("è", "e"),
        ("ì", "i"),
        ("ò", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


def preprocessing(text):
    """Eliminates all the html tags from the given text, performs tokeniztion
       and homogenization, and cleans the text in the sense of eliminating all 
       the spanish stopwords. After all the former, put all the words back to 
       one string.

    Parameters:
    ----------
    * text - Text to preprocess

    Returns
    -------
    * text_lemmatizedstr: The preprocessed text as a string
    """
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = ensureUtf(text)
    if is_spanish(text):
        text_clean = normalize(text)

        # Tokenization
        text_tokens = word_tokenize(text_clean)
        # Homogeneization
        # Remove numbers
        tok_no_nums = [w for w in text_tokens if not w.isnumeric()]
        # Remove words which contains numbers and letters at the same time- Ex.: 5kg, 10001..
        tok_only_letters = [w for w in tok_no_nums if w.isalpha()]
        # Remove single caracters
        tok_no_single_char = [w for w in tok_only_letters if len(w) > 1]
        # Convert to lowercase words
        tok_lowercase = [w.lower() for w in tok_no_single_char if w.isalnum()]
        # Make biagrams
        bigram = gensim.models.Phrases(text_tokens, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        tok_bigrammed = bigram_mod[tok_lowercase]
        # Make triagrams
        trigram = gensim.models.Phrases(bigram[text_tokens], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        tok_trigrammed = trigram_mod[tok_bigrammed]
        text_bef_lem = (" ".join(tok_trigrammed))
        # Lemmatization
        tokens_lemmatized = [lemmatizer.lemmatize(
            tok) for tok in tok_trigrammed]
        # Cleaning
        tokens_clean = [
            token for token in tokens_lemmatized if token not in stopwords_es]
        # Tokens to string
        text_lemmatized_str = (" ".join(tokens_clean))
        return text_lemmatized_str, text_bef_lem
    else:
        return [""], [""]
