*** Comment se décompose ce répertoire ? ***

Ce répertoire contient 2 dossiers :

- Dossier de travail : Ce dossier contient :
		       Les notebooks de nettoyage des données, d'analyse supervisée et d'analyse non supervisée.
		       Un fichier de fonctions personnalisées nécessaire au fonctionnement des notebooks
		       Le script utilisé pour la collecte des données depuis l'API StackExchange

- Démo : Contient tous les éléments (notebook, modèles, ressources) nécessaires pour tester les modèles issus de l'étude



*** Comment utiliser ce répertoire ? ***

Pour tester les modèles :

Se rendre dans le répertoire 'Démo' et ouvrir le notebook 'Tags_suggestion_demo'.
Suivre ensuite les instructions du notebook.
Le notebook téléchargera les données issues de stackoverflow sur la journée renseignée et les sauvegardera une fois nettoyées.

Les librairies nécessaires sont indiquées au début du fichier 'functions_demo'.
A noter que pour la partie nettoyage, les librairies suivantes sont nécessaires :

- NLTK : utilisation pour retrait stopwords et tokenisation. 
         (si les modules nécessaires ne sont pas installés, le message d'erreur indiquera la procédure d'installation)
- Spacy : utilisation pour la lemmatisation et retrait POS.
	  Le module 'en_core_web_lg' doit être téléchargé.

Egalement, les fichiers .joblib des modèles ont été réalisés avec les versions de librairies
suivantes, et il est nécessaire d'avoir les mêmes dans l'environnement de travail :
- Python 3.7.5
- joblib 0.14.0
- Numpy 1.17.3
- Scikit Learn 0.21.3

Pour tester la démarche en entier :

** Attention les résultats peuvent être différents de ceux obtenus dans les notebooks au format HTML **
** car les données, même prises sur la même période peuvent évoluer (Edition, acceptation réponse ...) **

Se rendre dans le dossier 'Répertoire de travail':

- Lancer le script 'stackoverflow_data_collect'. Pensez à insérer votre clé dans la requête afin de ne pas atteindre la limite journalière de requêtes

Cela ira collecter les données auprès de l'API StackExchange et les sauvegardera dans une fichier .csv 'data_stackoverflow_july_october'
A noter que cette opération peut prendre du temps.

- Ouvrir le notebook 'Nettoyage_VF', décommenter les lignes de sauvegarde des fichiers et l'éxécuter.

Cela prendra en entrée les données téléchargées à l'étape précédente et les nettoiera.
Plusieurs fichiers seront sauvegardés au format .csv à cette étape :
- Les données nettoyées : 'cleaned_dataset'
- Une liste de tags : 'most_common_tags'
- Une liste de mots à retirer : 'words_to_remove'

- Ouvrir le notebook 'modèle_supervisé', décommenter les lignes de sauvegarde des fichiers et l'éxécuter.

Cela générera toute l'étude supervisée et sauvegardera le modèle supervisé 'SGDC_supervised_Model.joblib'

- Ouvrir le notebook 'modèle non supervisé', décommenter les lignes de sauvegarde des fichiers et l'éxécuter.

Cela générera toute l'étude non supervisée et sauvegardera plusieurs modèles :
- Les résultats en sortie d'étude non supervisée (lda_model.json et nmf_model.json)
- Les modèles supervisés issus des cibles LDA et NMF ('SGDC_LDA_Model.joblib' et 'SGDC_NMF_Model.joblib')



*** Quelles sont les librairies / modules nécessaires ? ***

Manipulation des données:
Pandas
Numpy
scipy

Traitement du texte :
re
BeautifulSoup
NLTK
Spacy (module 'en_core_web_lg')

Machine learning :
sklearn

Visualisation :
Matplotlib
Seaborn
Wordcloud

Autres :
json
joblib
requests
time
datetime


	
