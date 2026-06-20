# API de Reconnaissance des Boîtes de Médicaments

API de reconnaissance de boîtes de médicaments utilisant la vision par ordinateur, l'OCR et le fuzzy matching, avec support bilingue **français / arabe**.

## Réalisé par
- Khaoula El Oualid
- Oumaima Goualla

## Encadré par
- Pr. Abdelhak Mahmoudi
- Yassine Lehmiani

---

## Outils et technologies utilisées

| Outil | Rôle |
|---|---|
| **FastAPI** | Backend / API REST |
| **OpenCV** | Prétraitement des images |
| **EasyOCR** | OCR (reconnaissance de texte en français, anglais et arabe) |
| **RapidFuzz** | Fuzzy matching du texte extrait |
| **Streamlit** | Interface graphique de test |
| **Pandas** | Chargement et manipulation du dataset CSV |
| **Matplotlib** | Génération des graphiques d'évaluation |

---

## Architecture du projet

```
Medication-box-recognition-API/
│
├── api.py                  # API FastAPI (point d'entrée principal)
├── front.py                # Interface graphique Streamlit
├── evaluation.py           # Script d'évaluation du pipeline
├── pipeline.ipynb        # Notebook d'exploration du pipeline
└── dataset/
    ├── medicaments.csv         # Base de données des médicaments (FR + AR)
    ├── images/                 # Images de test
    ├── annotations.csv         # Annotations pour l'évaluation (image;nom_correct)
    ├── rapport_evaluation.csv  # Rapport généré après évaluation
    └── graphiques/             # Graphiques PNG générés après évaluation

```

---

## Processus de fonctionnement

```
Image de boîte
      │
      ▼
1. Prétraitement (OpenCV)
   ├── Redimensionnement (hauteur minimale 1200px)
   ├── Conversion en niveaux de gris
   ├── Amélioration du contraste (CLAHE)
   ├── Floutage gaussien (débruitage)
   └── Binarisation (Otsu + seuillage adaptatif)
      │
      ▼
2. OCR bilingue (EasyOCR)
   ├── Extraction du texte latin (fr + en)
   └── Extraction du texte arabe (ar)
      │
      ▼
3. Fuzzy Matching (RapidFuzz)
   ├── Score nom    (poids 70%) — comparaison FR et AR
   ├── Score dosage (poids 20%) — détection des chiffres OCR
   └── Score forme  (poids 10%) — comparaison de la forme pharmaceutique
      │
      ▼
4. Décision
   ├── Score ≥ 75 → reconnu, confiance haute
   ├── Score ≥ 60 → reconnu, confiance bonne
   ├── Score ≥ 50 → reconnu, confiance moyenne
   └── Score < 50 → rejeté (médicament non reconnu)
      │
      ▼
5. Réponse JSON (nom FR/AR, DCI FR/AR, dosage FR/AR , forme FR/AR, présentation FR/AR)
```

---

## Endpoints de l'API

### `POST /predict`
Identifie un médicament à partir d'une image de sa boîte.

**Paramètre :**
- `file` : image de la boîte (JPG, PNG, WEBP, BMP, TIFF)

**Réponse (médicament reconnu) :**
```json
{
  "reconnu": true,
  "confiance": "haute",
  "scores": {
    "nom": 92.5,
    "dosage": 80.0,
    "forme": 75.0,
    "final": 88.3
  },
  "medicament": {
    "nom_fr": "DOLAMINE",
    "nom_ar": "دولامين",
    "dci_fr": "PARACETAMOL",
    "dci_ar": "باراسيتامول",
    "dosage_fr": "500 MG",
    "dosage_ar": "500 ملغ",
    "forme_fr": "COMPRIME",
    "forme_ar": "قرص",
    "presentation_fr": "BOITE DE 16 COMPRIMES",
    "presentation_ar": "علبة 16 قرصاً"
  }
}
```

**Réponse (médicament non reconnu) :**
```json
{
  "reconnu": false,
  "confiance": "rejeté",
  "scores": { "final": 32.0, ... },
  "message": {
    "attention": "Médicament non reconnu",
    "details": ["Veuillez reprendre la photo...", "..."]
  }
}
```

### `GET /health`
Vérifie que l'API est opérationnelle.

```json
{
  "status": "ok",
  "medicaments": 100,
  "ocr_fr": "chargé",
  "ocr_ar": "chargé"
}
```

### `GET /`
Retourne les endpoints disponibles et la version de l'API.

```json
{
  "api": "Reconnaissance Médicaments",
  "version": "1.0.0",
  "endpoints": {
    "POST /predict": "Identifier un médicament depuis une image",
    "GET  /health": "Vérifier l'état de l'API",
    "GET  /docs": "Documentation interactive (Swagger)"
  }
}
```

### `GET /docs`
Documentation interactive Swagger UI (générée automatiquement par FastAPI).

---

## Dataset

Le fichier `dataset/medecaments.csv` (séparateur `;`) doit contenir les colonnes suivantes :

| Colonne | Description |
|---|---|
| `CODE` | Code unique du médicament |
| `NOM` | Nom commercial (français) |
| `الاسم` | Nom commercial (arabe) |
| `DCI1` | Dénomination commune internationale (français) |
| `معلومات الدواء` | DCI (arabe) |
| `DOSAGE1` | Dosage |
| `UNITE_DOSAGE1` | Unité de dosage (français) |
| `وحدة الجرعة` | Unité de dosage (arabe) |
| `FORME` | Forme galénique (français) |
| `صيغة` | Forme galénique (arabe) |
| `PRESENTATION` | Présentation / conditionnement (français) |
| `شكل الدواء` | Présentation (arabe) |

---

## Installation et utilisation

```bash
# Clonage du dépôt
git clone https://github.com/oumaimagoualla/Medication-box-recognition-API.git
cd Medication-box-recognition-API

# Création de l'environnement virtuel
python -m venv .venv
.venv/Scripts/Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'API
fastapi run api.py
# ou : uvicorn api:app --reload --port 8000

# Lancement de l'interface graphique de test (dans un autre terminal)
streamlit run front.py
```

L'API sera disponible sur `http://localhost:8000`.
La documentation Swagger sera accessible sur `http://localhost:8000/docs`.

---

## Évaluation du pipeline

Le script `evaluation.py` permet d'évaluer les performances du système sur un jeu de données annoté.

### Prérequis

1. L'API doit être en cours d'exécution :
   ```bash
   uvicorn api:app --port 8000
   ```

2. Créer le fichier `dataset/annotations.csv` au format suivant (séparateur `;`) :
   ```
   image;nom_correct
   dolamine.jpg;DOLAMINE
   oradexon.jpg;ORADEXON
   ```
   Les images référencées doivent se trouver dans `dataset/images/`.

3. Pour les images arabes, nommer les fichiers avec le suffixe `_ar` (ex. `dolamine_ar.jpg`) afin que le script puisse calculer les métriques FR/AR séparément.

### Lancement

```bash
python evaluation.py
```

### Métriques calculées

- **Accuracy Top-1** : pourcentage d'images correctement identifiées
- **Taux de rejet** : images dont le score final est inférieur au seuil (50)
- **Taux d'erreur** : images reconnues mais avec le mauvais médicament
- **Accuracy FR vs AR** : comparaison des performances selon la langue de l'image

### Sorties générées

| Fichier | Description |
|---|---|
| `dataset/rapport_evaluation.csv` | Résultats détaillés image par image |
| `dataset/graphiques/1_resultats_globaux.png` | Camembert des résultats (correct / rejeté / erreur) |
| `dataset/graphiques/2_fr_vs_ar.png` | Accuracy images françaises vs arabes |
| `dataset/graphiques/3_distribution_scores.png` | Distribution des scores par catégorie |
| `dataset/graphiques/4_metriques.png` | Récapitulatif accuracy / taux d'erreur / taux de rejet |

---

## Interface graphique (Streamlit)

L'interface `front.py` permet de tester l'API visuellement :

1. Déposer ou sélectionner une image de boîte de médicament
2. Cliquer sur **Identifier médicament**
3. Le résultat s'affiche avec le nom, le niveau de confiance et la fiche complète du médicament

Formats d'image acceptés : JPG, JPEG, PNG, WEBP, BMP, TIFF.

---

## Niveaux de confiance

| Score final | Confiance | Signification |
|---|---|---|
| ≥ 75 | Haute | Identification très fiable |
| 60 – 74 | Bonne | Identification probable |
| 50 – 59 | Moyenne | Identification incertaine, à vérifier |
| < 50 | Rejeté | Médicament non reconnu |
