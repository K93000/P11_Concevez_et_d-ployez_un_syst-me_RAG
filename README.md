# Assistant RAG – Événements en Seine-Saint-Denis (93)

## Présentation

Ce projet consiste à développer un assistant conversationnel basé sur une architecture
RAG (Retrieval-Augmented Generation), spécialisé dans les événements de Seine-Saint-Denis (93).

L'utilisateur peut poser des questions en langage naturel sur les événements et obtenir
une réponse basée sur les données disponibles dans la base de connaissances.

## Architecture

Le projet suit le pipeline suivant :

OpenAgenda
↓
Extraction et filtrage des données
↓
Nettoyage et structuration
↓
Chunking
↓
Embeddings Mistral
↓
Base vectorielle FAISS
↓
Retrieval
↓
Mistral Small
↓
Réponse
↓
Interface Streamlit

## Technologies utilisées

- Python
- OpenAgenda
- LangChain
- Mistral AI
- FAISS
- RAGAS
- Streamlit
- BeautifulSoup
- Pandas

## Données

Les données utilisées proviennent d'OpenAgenda.

Le pipeline de préparation permet notamment de :

- sélectionner les événements de Seine-Saint-Denis (93)
- filtrer les événements selon la période retenue
- gérer les doublons
- vérifier la présence d'un identifiant
- nettoyer les contenus HTML
- structurer les informations utiles à la recherche

Le jeu de données utilisé contient environ 3 880 événements.

## Préparation des données

Les événements sont transformés en documents structurés contenant notamment :

- titre
- ville
- lieu
- adresse
- date
- tarif / conditions
- accessibilité
- catégorie
- mots-clés
- description

Les documents sont ensuite découpés en chunks afin de faciliter la recherche
sémantique.

Configuration retenue pour le prototype :

- `chunk_size = 1200`
- `chunk_overlap = 100`

Les embeddings sont générés avec le modèle `mistral-embed`.

Les vecteurs sont ensuite stockés dans une base FAISS.

## Fonctionnement du RAG

Lorsqu'un utilisateur pose une question :

1. la question est transformée en représentation vectorielle ;
2. FAISS recherche les chunks les plus proches ;
3. les documents récupérés sont transmis au modèle Mistral ;
4. Mistral génère une réponse à partir du contexte fourni.

Le système utilise actuellement :

- `TOP_K = 5`
- modèle d'embedding : `mistral-embed`
- modèle de génération : `mistral-small-latest`

Le prompt demande au modèle de répondre uniquement à partir du contexte
récupéré et de ne pas inventer d'informations.

## Interface utilisateur

L'application est développée avec Streamlit.

Elle permet :

- de poser une question en langage naturel ;
- d'obtenir une réponse du système RAG ;
- de consulter les sources utilisées pour générer la réponse.

## Évaluation

Une évaluation a été mise en place avec RAGAS.

Le jeu d'évaluation contient 15 questions réparties en trois catégories :

- questions simples ;
- questions complexes ;
- questions hors périmètre.

L'évaluation porte notamment sur :

- la fidélité des réponses au contexte ;
- la pertinence des documents récupérés ;
- la couverture des informations nécessaires.

Les résultats détaillés de l'évaluation sont disponibles dans :

`evaluation/ragas_evaluation_report.csv`

## Installation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Créer ensuite un fichier `.env` à partir de `.env.example` et renseigner sa clé API Mistral :

```env
MISTRAL_API_KEY=votre_clé_api
```

## Utilisation

### Préparer les données

```bash
python fetch_data.py
```

### Construire la base vectorielle

```bash
python build_vector_db.py
```

### Lancer l'application

```bash
streamlit run app.py
```

### Lancer l'évaluation

```bash
python evaluate_ragas.py
```