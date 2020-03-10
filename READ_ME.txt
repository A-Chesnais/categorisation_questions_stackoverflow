*** Comment se d�compose ce r�pertoire ? ***

Ce r�pertoire contient 3 dossiers :

- Dossier de travail : Ce dossier contient :
		       Les notebooks de nettoyage des donn�es, d'analyse supervis�e et d'analyse non supervis�e.
		       Un fichier de fonctions personnalis�es n�cessaire au fonctionnement des notebooks
		       Le script utilis� pour la collecte des donn�es depuis l'API StackExchange

- D�mo : Contient tous les �l�ments (notebook, mod�les, ressources) n�cessaires pour tester les mod�les issus de l'�tude



*** Comment utiliser ce r�pertoire ? ***

Pour tester les mod�les :

Se rendre dans le r�pertoire 'D�mo' et ouvrir le notebook 'Tags_suggestion_demo'.
Suivre ensuite les instructions du notebook.
Le notebook t�l�chargera les donn�es issues de stackoverflow sur la journ�e renseign�e et les sauvegardera une fois nettoy�es.

Les librairies n�cessaires sont indiqu�es au d�but du fichier 'functions_demo'.
A noter que pour la partie nettoyage, les librairies suivantes sont n�cessaires :

- NLTK : utilisation pour retrait stopwords et tokenisation. 
         (si les modules n�cessaires ne sont pas install�s, le message d'erreur indiquera la proc�dure d'installation)
- Spacy : utilisation pour la lemmatisation et retrait POS.
	  Le module 'en_core_web_lg' doit �tre t�l�charg�.

Egalement, les fichiers .joblib des mod�les ont �t� r�alis�s avec les versions de librairies
suivantes, et il est n�cessaire d'avoir les m�mes dans l'environnement de travail :
- Python 3.7.5
- joblib 0.14.0
- Numpy 1.17.3
- Scikit Learn 0.21.3

Pour tester la d�marche en entier :

** Attention les r�sultats peuvent �tre diff�rents de ceux obtenus dans les notebooks au format HTML **
** car les donn�es, m�me prises sur la m�me p�riode peuvent �voluer (Edition, acceptation r�ponse ...) **

Se rendre dans le dossier 'R�pertoire de travail':

- Lancer le script 'stackoverflow_data_collect'. Pensez � ins�rer votre cl� dans la requ�te afin de ne pas atteindre la limite journali�re de requ�tes

Cela ira collecter les donn�es aupr�s de l'API StackExchange et les sauvegardera dans une fichier .csv 'data_stackoverflow_july_october'
A noter que cette op�ration peut prendre du temps.

- Ouvrir le notebook 'Nettoyage_VF', d�commenter les lignes de sauvegarde des fichiers et l'�x�cuter.

Cela prendra en entr�e les donn�es t�l�charg�es � l'�tape pr�c�dente et les nettoiera.
Plusieurs fichiers seront sauvegard�s au format .csv � cette �tape :
- Les donn�es nettoy�es : 'cleaned_dataset'
- Une liste de tags : 'most_common_tags'
- Une liste de mots � retirer : 'words_to_remove'

- Ouvrir le notebook 'mod�le_supervis�', d�commenter les lignes de sauvegarde des fichiers et l'�x�cuter.

Cela g�n�rera toute l'�tude supervis�e et sauvegardera le mod�le supervis� 'SGDC_supervised_Model.joblib'

- Ouvrir le notebook 'mod�le non supervis�', d�commenter les lignes de sauvegarde des fichiers et l'�x�cuter.

Cela g�n�rera toute l'�tude non supervis�e et sauvegardera plusieurs mod�les :
- Les r�sultats en sortie d'�tude non supervis�e (lda_model.json et nmf_model.json)
- Les mod�les supervis�s issus des cibles LDA et NMF ('SGDC_LDA_Model.joblib' et 'SGDC_NMF_Model.joblib')



*** Quelles sont les librairies / modules n�cessaires ? ***

Manipulation des donn�es:
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


	