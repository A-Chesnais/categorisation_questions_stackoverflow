# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:37:53 2019

@author: Antoine
"""

import requests
import time
from datetime import datetime, timezone
from joblib import load

# Librairies de manipulation de données
import pandas as pd

# Librairies et modules de traitement du texte et du language naturel
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('en_core_web_lg')

# Chargement des modèles
SGDC_LDA_Model = load('SGDC_LDA_Model.joblib')
SGDC_NMF_Model = load('SGDC_NMF_Model.joblib')
SGDC_supervised_Model = load('SGDC_supervised_Model.joblib')
vocabulary = pd.read_csv('models_tags_vocabulary.csv', header=0)

def data_retrieve(date):
    
    data =pd.DataFrame()
    p = 1
    has_more =True
    
    # Construction de la requête à envoyer à l'API
    start_url = 'https://api.stackexchange.com/2.2/search/advanced?pagesize=100'
    date_start_url = '&fromdate='
    date_end_url = '&todate='
    end_url = '&accepted=True&site=stackoverflow&filter=withbody&page='

    year =  int(date[6:10])  
    month = int(date[3:5])
    day = int(date[0:2])

    date_start = int(datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    date_end = int(datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    
    core_url = start_url + date_start_url + str(date_start) + date_end_url \
                + str(date_end) + end_url
                
    # Récupération des résultats page par page
    while has_more==True:
    
        r = requests.get(core_url+str(p))
        json_data = r.json()    
        has_more = json_data['has_more']
        p +=1
        temp_data = pd.DataFrame(json_data['items'])
        data = data.append(temp_data, ignore_index=True)
        time.sleep(0.05)
        print('page : ' + str(p))
        
    # Sélection uniquement du titre, du coprs de la question et des tags associés
    data_ligth = data.loc[:,['title', 'body','tags']]
        
    return data_ligth

def questions_cleaning(data):
    
    # Retrait du code
    data['body_text'] = data.body.apply(lambda x: re.sub('<code>([\s\S]*?)</code>', '', x))
    
    # Retrait des blockquote
    data['body_text'] = data.body_text.apply(lambda x: re.sub('<blockquote>([\s\S]*?)</blockquote>', '', x))
    
    # Retrait des liens
    data['body_text'] = data.body_text.apply(lambda x: re.sub('<a[\s\S]*?</a>', '', x))
    
    # Retrait du texte entre les balises complémentaires
    data['body_text'] = data.body_text.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    data['body_text'] = data.body_text.apply(lambda x: re.sub("<.*?>", '', x))
    
    # Combinaison du corps de la question et de son titre
    questions = data['title'] + ' ' + data['body_text']
    questions = questions.str.lower()
    
    # Détection de formats techniques (adresses, extensions, languages ...) et retrait de ceux ci
    data['files_extensions'] = questions.apply(lambda x: re.findall("\.[a-z]+", x))
    data['files_extensions'] = data['files_extensions'].apply(lambda x: ' '.join(x))
    alpha_questions = questions.apply(lambda x: re.sub("((?:[a-z0-9]*\.)+[a-z0-9]+)", ' ', x))
    
    data['C++'] = questions.apply(lambda x: re.findall("c\+\+", x))
    data['C++'] = data['C++'].apply(lambda x: ' '.join(x))
    alpha_questions = alpha_questions.apply(lambda x: re.sub("c\+\+", ' ', x))
    
    data['C#'] = questions.apply(lambda x: re.findall("c\#", x))
    data['C#'] = data['C#'].apply(lambda x: ' '.join(x))
    alpha_questions = alpha_questions.apply(lambda x: re.sub("c\#", ' ', x))
                                                             
    # Retrait du reste des caractères non alphanumériques (sauf apostrophe)
    alpha_questions = alpha_questions.apply(lambda x: re.sub("[^a-zA-Z']", ' ' , x))

    # Supression d'éventuels espaces en trop
    alpha_questions = alpha_questions.apply(lambda x: re.sub("\s+", ' ', x))

    # Retrait des stopwords :
    stopwords_list = nltk.corpus.stopwords.words('English')
    tokenized_questions = alpha_questions.apply(lambda x: word_tokenize(x))
    question_wth_stopwords = tokenized_questions.apply(lambda x: ' '.join([t for t in x if t not in stopwords_list]))

    # Lemmatisation et retrait versb, adverbes et adjectifs
    docs = question_wth_stopwords.tolist()
    lemmetized_questions = []
    forbiden_POS = ['ADJ', 'VERB', 'ADV']
    documents = nlp.pipe(docs, disable=["parser", "ner", 'textcat'])
    for doc in documents:
        tokens = [token.lemma_ for token in doc if token.pos_ not in forbiden_POS]
        lemmetized_questions.append(' '.join(tokens))
    data['spacy_lemmetized_text'] = lemmetized_questions

    # Retrait du vocabulaire partagé
    words_to_remove = pd.read_csv('words_to_remove.csv', header=None)[0].tolist()
    tokenized_questions = data['spacy_lemmetized_text'].apply(lambda x: word_tokenize(x))
    question_VF = tokenized_questions.apply(lambda x: ' '.join([t for t in x if t not in words_to_remove]))
    data['cleaned_questions'] = question_VF
    
    #Sauvegarde des données nettoyées
    data.to_csv('cleaned_dataset.csv', index=False)

def load_questions(data) :
    
    data.fillna('', inplace=True)
    
    # Retourne uniquement un set de questions composées du titre et du corps nettoyés,
    # avec également les extensions de fichiers ou l'apparition de language C
    questions = data['cleaned_questions'] + ' ' + data['files_extensions'] + ' ' + data['C++'] + ' ' + data['C#']
                    
    questions = questions.apply(lambda x: re.sub("\s+", ' ', x))
    
    return questions

def tags_suggestions(questions, data, index):
    
    question = questions[[index]]
    title = data.title[index]
    true_tags = data.tags[index]
    body = data.body[index]
    LDA_tags = SGDC_LDA_Model.predict(question)
    NMF_tags = SGDC_NMF_Model.predict(question)
    supervised_tags =SGDC_supervised_Model.predict(question)
    
    print('Titre de la question :')
    print(title)
    print('\n Tags utilisés :')
    print(true_tags)
    print('\n Tags suggérés :')
    print('LDA : ' + str(vocabulary.LDA[LDA_tags.nonzero()[1]].tolist()))
    print('NMF : ' + str(vocabulary.NMF[NMF_tags.nonzero()[1]].tolist()))
    print('Supervised : ' + str(vocabulary.supervised[supervised_tags.nonzero()[1]].tolist()))
    print('\n Corps de la question :')
    print(body)

    


                    

                                             