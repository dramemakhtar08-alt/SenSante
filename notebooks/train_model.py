import pandas as pd
import numpy as np

#Charger le dataset
df  = pd.read_csv("data/patients_dakar.csv")

#Vérifier les dimensions*
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes :  {list(df.columns)}")
print(f"\nDiagnostics :  \n{df['diagnostic'].value_counts()}")


# Préparer les features et la cible
from sklearn.preprocessing import LabelEncoder
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder ()
le_region = LabelEncoder ()
df['sexe_encoded'] = le_sexe.fit_transform (df['sexe'])
df['region_encoded'] = le_region.fit_transform (df['region'])
# Definir les features (X) et la cible (y)
feature_cols = ['age','sexe_encoded','temperature','tension_sys', 'toux','fatigue','maux_tete','region_encoded']
X = df[feature_cols]
y = df['diagnostic']
print (f" Features : {X.shape }") # (500 , 8)
print (f" Cible : {y.shape }") # (500 ,)


# Séparer les données
from sklearn.model_selection import train_test_split
# 80% pour l'entrainement , 20% pour le test
X_train , X_test , y_train , y_test = train_test_split (
X, y,
test_size = 0.2 , # 20% pour le test
random_state = 42 , # Pour avoir les memes resultats a chaque fois
stratify = y # Garder les memes proportions de diagnostics
)
print (f" Entrainement : { X_train.shape[0]} patients ")
print (f" Test : { X_test.shape[0]} patients ")




from sklearn.ensemble import RandomForestClassifier
# Creer le modele
model = RandomForestClassifier (
n_estimators = 100 , # 100 arbres de decision
random_state = 42 # Reproductibilite
)
# Entrainer sur les donnees d' entrainement
model.fit ( X_train , y_train )
print (" Modele entraine !")
print (f" Nombre d'arbres : { model.n_estimators }")
print (f" Nombre de features : { model.n_features_in_ }")
print (f" Classes : { list ( model.classes_ )}")



# Predire sur les donnees de test
y_pred = model.predict ( X_test )
# Comparer les 10 premieres predictions avec la realite
comparison = pd.DataFrame ({
'Vrai diagnostic ': y_test.values [:10] ,
'Prediction ': y_pred [:10]
})
print ( comparison )


from sklearn.metrics import accuracy_score
accuracy = accuracy_score ( y_test , y_pred )
print (f"\n Accuracy : { accuracy :.2%} ")


from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Matrice de confusion
cm = confusion_matrix ( y_test , y_pred , labels = model.classes_ )
print (" Matrice de confusion :")
print (cm)
# Rapport de classification
print ("\nRapport de classification :")
print ( classification_report ( y_test , y_pred ))


# Visualiser avec seaborn
plt.figure ( figsize =(8 , 6))
sns.heatmap (cm , annot =True , fmt ='d', cmap ='Blues',
xticklabels = model.classes_ ,
yticklabels = model.classes_ )
plt.xlabel ('Prediction du modele ')
plt.ylabel ('Vrai diagnostic ')
plt.title ('Matrice de confusion - SenSante ')
plt.tight_layout ()
plt.savefig ('figures/confusion_matrix.png', dpi =150)
plt.show ()
print (" Figure sauvegardee dans figures/confusion_matrix.png ")


import joblib
import os
# Creer le dossier models / s'il n'existe pas
os.makedirs (" models ", exist_ok = True )
# Serialiser le modele
joblib.dump (model , " models/model.pkl ")
# Verifier la taille du fichier
size = os.path.getsize (" models/model.pkl ")
print (f" Modele sauvegarde : models/model.pkl ")
print (f" Taille : { size/1024:.1f} Ko")


# Sauvegarder les encodeurs ( indispensables pour les nouvelles donnees )
joblib.dump ( le_sexe , " models/encoder_sexe.pkl ")
joblib.dump ( le_region , " models/encoder_region.pkl ")
# Sauvegarder la liste des features ( pour reference )
joblib.dump ( feature_cols , " models/feature_cols.pkl ")
print (" Encodeurs et metadata sauvegardés .")



# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER ( pas depuis la memoire )
model_loaded = joblib.load (" models/model.pkl ")
le_sexe_loaded = joblib.load (" models/encoder_sexe.pkl ")
le_region_loaded = joblib.load