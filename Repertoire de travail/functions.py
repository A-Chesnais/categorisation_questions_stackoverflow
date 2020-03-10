#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:23:10 2019

@author: antoinechesnais
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from wordcloud import WordCloud
from IPython.display import display

def words_freq(questions):
    # Initialisation du counter, fitting du modèle au données et transformation de celles ci
    count_vectorizer = CountVectorizer(stop_words=None)
    sparse_questions = count_vectorizer.fit_transform(questions)

    # Création d'un tableau qui contient les fréquences de chaque mots
    words_count = np.squeeze(np.asarray(sparse_questions.sum(axis=0)))

    # Création d'un dataframe contenant les mots et leur fréquences
    df_words_count = pd.DataFrame([count_vectorizer.get_feature_names(), words_count]).T
    df_words_count.columns = ['word', 'count']
    df_words_count.sort_values(by='count', inplace=True, ascending=False)
    df_words_count['rank'] = range(0,len(df_words_count),1)
    display(df_words_count.head(100))

    # Création d'un bag of words
    plt.figure(figsize=(12,8))
    word_viz = dict(zip(df_words_count.word.values,df_words_count['count'].values))
    wordcloud = WordCloud(max_words=500)
    fig = wordcloud.generate_from_frequencies(frequencies=word_viz)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return df_words_count

def tags_freq(tags, vectorizer):
    # fitting du modèle au données et transformation de celles ci
    sparse_tags = vectorizer.fit_transform(tags)

    # Création d'un tableau qui contient les fréquences de chaque mots
    tags_count = np.squeeze(np.asarray(sparse_tags.sum(axis=0)))

    # Création d'un dataframe contenant les mots et leur fréquences
    df_tags_count = pd.DataFrame([vectorizer.get_feature_names(), tags_count]).T
    df_tags_count.columns = ['tags', 'count']
    df_tags_count.sort_values(by='count', inplace=True, ascending=False)
    df_tags_count['rank'] = range(0,len(df_tags_count),1)
    display(df_tags_count.head(100))

    # Création d'un bag of words
    plt.figure(figsize=(12,8))
    tags_viz = dict(zip(df_tags_count.tags.values,df_tags_count['count'].values))
    wordcloud = WordCloud(max_words=500)
    fig = wordcloud.generate_from_frequencies(frequencies=tags_viz)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return df_tags_count, sparse_tags

def tags_questions_coverage(sparse, tags_frequencies, max_tags, min_tags):
    '''
    '''
    tags_to_select = list(tags_frequencies[:max_tags].index)
    reduced_sparce = sparse[:,tags_to_select]
    questions_coverage = reduced_sparce[reduced_sparce.getnnz(1)>(min_tags-1)].shape[0] / sparse.shape[0] * 100
    print('{:03.2f}'.format(questions_coverage) + ' questions have at least ' + str(min_tags) + ' tag(s)')
    return questions_coverage, reduced_sparce.getnnz(1)>(min_tags-1);


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def get_top_words(model, model_type, feature_names, n_top_words, tags_list):
    
    results =pd.DataFrame(columns = ['topic', 'key_words'])
    
    if model_type == 'lda':
        model_components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    else:
        model_components = model.components_
    
    for topic_idx, topic in enumerate(model_components):
        results = results.append({'topic': topic_idx,
                                  'key_words': dict(zip([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]],
                                  topic[topic.argsort()[:-n_top_words - 1:-1]]))}, ignore_index=True)
        
    results['potential_tags'] = results['key_words'].apply(lambda x: [(keyword,weight) for (keyword,weight) in x.items() if keyword in tags_list])

    results['potential_tags_weighted'] = results['potential_tags'].apply(lambda x: dict(x))
    
    print('Total number of tags :')
    tags_number = results['potential_tags_weighted'].apply(lambda x: list(x.keys()))
    print(len(set(tags_number.sum())))
    
    results.drop(labels=['potential_tags'], axis=1, inplace=True)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(results)
        
    return results

def save_unsupervivised_model(features, vectorizer, target, potential_tags , filename):
    
    model = {}
    model['features'] = {'data': features.data.tolist(), 'indices': features.indices.tolist(), 
                         'indptr': features.indptr.tolist(), 'shape': features.shape}
    model['vocabulary'] = dict([key, value.item()] for key, value in vectorizer.vocabulary_.items())
    model['target'] = target.tolist()
    model['topic_tags'] = potential_tags['potential_tags_weighted'].to_dict()
    
    json_txt = json.dumps(model)
    with open(filename, 'w') as file:
        file.write(json_txt)

def load_unsupervivised_model(filename):
    
    with open(filename, 'r') as file:
        model = json.load(file)
        
    features = csr_matrix((np.asarray(model['features']['data']), np.asarray(model['features']['indices']), 
                           np.asarray(model['features']['indptr'])), shape=model['features']['shape'])
    vocabulary = model['vocabulary']
    target = np.asarray(model['target'])
    tags = model['topic_tags']
    
    return features, target, vocabulary, tags