#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:24:48 2019

@author: antoinechesnais

Programme permettant de collecter des données à partir de l'API stackOverflow 
de Juilet a Octobre 2019
Pensez à inserer votre cle dans la requête (remplacer le texte INSERT_YOUR_API_KEY_HERE)
"""
# Import des librairies
import requests
import pandas as pd
import time

# Definition de variables
data =pd.DataFrame()
p = 1
has_more =True
core_url = 'https://api.stackexchange.com/2.2/search/advanced?key=INSERT_YOUR_API_KEY_HERE&pagesize=100&fromdate=1561939200&todate=1572566399&accepted=True&site=stackoverflow&filter=withbody&page='

# Boucle d'appel a l'API StackExchange pour collecter les resultats page par page
# Continue d'appeler la page suivante tant que des donnees
# sont disponibles (has_more==True)
while has_more==True:
    
    # Envoi de la requete et collecte sous format json
    r = requests.get(core_url+str(p))
    json_data = r.json()
    
    # Update de la variable de verification de resultats supplementaires
    has_more = json_data['has_more']
    
    # incrementation de l'index de la page a collecter
    p +=1
    
    # transformation des resultats en Dataframe
    # Et concatenation avec les pages precedentes
    temp_data = pd.DataFrame(json_data['items'])
    data = data.append(temp_data, ignore_index=True)
    
    # Temporisation entre deux appels a l'API pour ne pas exceder les quotas
    time.sleep(0.05)
    print('page : ' + str(p) + ' and ' + str(json_data['quota_remaining']) + ' requests remaining')

# Conservation uniquement du titre, du corps et des tags
# associes a une question
data_ligth = data.loc[:,['title', 'body','tags']]

# Sauvegarde des resultats au format CSV
data_ligth.to_csv('data_stackoverflow_july_october.csv', index=False)
