#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:24:48 2019

@author: antoinechesnais

Programme permettant de collecter des données à partir de l'API stackOverflow
Pensez à insérer votre clé dans la requête (remplacer le texte INSERT_YOUR_API_KEY_HERE)
"""

import requests
import pandas as pd
import time

data =pd.DataFrame()
p = 1
has_more =True
core_url = 'https://api.stackexchange.com/2.2/search/advanced?key=INSERT_YOUR_API_KEY_HERE&pagesize=100&fromdate=1561939200&todate=1572566399&accepted=True&site=stackoverflow&filter=withbody&page='


while has_more==True:
    
    r = requests.get(core_url+str(p))
    json_data = r.json()    
    has_more = json_data['has_more']
    p +=1
    temp_data = pd.DataFrame(json_data['items'])
    data = data.append(temp_data, ignore_index=True)
    time.sleep(0.05)
    print('page : ' + str(p) + ' and ' + str(json_data['quota_remaining']) + ' requests remaining')

data_ligth = data.loc[:,['title', 'body','tags']]

data_ligth.to_csv('data_stackoverflow_july_october.csv', index=False)