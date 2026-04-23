"""
SenSante - Entrainement du modele ML
Lab 2 : Machine Learning avec scikit-learn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===== CREER LES DOSSIERS NECESSAIRES =====
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===== CHARGER LE DATASET =====
df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print("SENSANTE - Entrainement du modele")
print("=" * 50)

print(f"\nDataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")


# ===== PREPARER LES FEATURES ET LA CIBLE =====
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Definir les features (X) et la cible (y)
feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys',
    'toux', 'fatigue', 'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")   # (500, 8)
print(f"Cible : {y.shape}")        # (500,)


# ===== SEPARER LES DONNEES =====
# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42,     # Pour avoir les memes resultats a chaque fois
    stratify=y           # Garder les memes proportions de diagnostics
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")


# ===== ENTRAINER LE MODELE =====
model = RandomForestClassifier(
    n_estimators=100,    # 100 arbres de decision
    random_state=42      # Reproductibilite
)

model.fit(X_train, y_train)

print("\nModele entraine !")
print(f"Nombre d'arbres    : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes            : {list(model.classes_)}")


# ===== PREDICTIONS =====
y_pred = model.predict(X_test)

# Comparer les 10 premieres predictions avec la realite
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction':      y_pred[:10]
})
print(f"\n--- 10 premieres predictions ---")
print(comparison.to_string(index=False))


# ===== EVALUATION =====
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


# ===== VISUALISATION =====
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")


# ===== SAUVEGARDER LE MODELE =====
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"\nModele sauvegarde : models/model.pkl")
print(f"Taille            : {size / 1024:.1f} Ko")
print("Encodeurs et metadata sauvegardes.")


# ===== VERIFICATION : RECHARGER ET TESTER =====
# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
feature_cols_loaded = joblib.load("models/feature_cols.pkl")

# Tester avec un nouveau patient fictif
nouveau_patient = {
    'age': 35,
    'sexe': 'M',
    'temperature': 38.5,
    'tension_sys': 130,
    'toux': 1,
    'fatigue': 1,
    'maux_tete': 0,
    'region': 'Dakar'
}

# Encoder les variables categoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Construire le vecteur de features
X_nouveau = pd.DataFrame([{
    'age':            nouveau_patient['age'],
    'sexe_encoded':   sexe_enc,
    'temperature':    nouveau_patient['temperature'],
    'tension_sys':    nouveau_patient['tension_sys'],
    'toux':           nouveau_patient['toux'],
    'fatigue':        nouveau_patient['fatigue'],
    'maux_tete':      nouveau_patient['maux_tete'],
    'region_encoded': region_enc
}])

prediction = model_loaded.predict(X_nouveau)[0]
probabilites = model_loaded.predict_proba(X_nouveau)[0]

print(f"\n--- Test avec un nouveau patient ---")
print(f"Diagnostic predit : {prediction}")
print("Probabilites :")
for classe, proba in zip(model_loaded.classes_, probabilites):
    print(f"  {classe:12s} : {proba:.2%}")

print(f"\n{'=' * 50}")
print("Entrainement termine !")
print("Prochain lab : creer une API avec Flask")
print(f"{'=' * 50}")