import json
import os

from bs4 import BeautifulSoup

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    MISTRAL_API_KEY,
)


def clean_html(text):
    """Supprime les balises HTML et conserve uniquement le texte brut utile."""
    if not text:
        return ""

    return BeautifulSoup(
        str(text),
        "html.parser"
    ).get_text(" ", strip=True)


def build_vector_db():

    json_path = os.path.join(
        "data",
        "events_93_raw.json"
    )

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Le fichier {json_path} est introuvable."
        )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:
        events = json.load(f)

    print("--- [1/3] CHARGEMENT ET NETTOYAGE DES DONNÉES ---")
    print(f"✓ {len(events)} événements chargés")

    documents = []

    for event in events:

        if not isinstance(event, dict):
            continue

        # -----------------------------
        # Nettoyage HTML des champs textuels
        # -----------------------------

        title = clean_html(
            event.get("title_fr") or "Événement"
        )

        description = clean_html(
            event.get("description_fr") or ""
        )

        long_description = clean_html(
            event.get("longdescription_fr") or ""
        )

        location = clean_html(
            event.get("location_name") or ""
        )

        address = clean_html(
            event.get("location_address") or ""
        )

        conditions = clean_html(
            event.get("conditions_fr") or "Non renseigné"
        )

        accessibility = clean_html(
            event.get("accessibility_fr") or ""
        )

        # -----------------------------
        # Informations structurées
        # -----------------------------

        city = event.get(
            "location_city"
        ) or "Seine-Saint-Denis"

        dates = event.get(
            "daterange_fr"
        ) or ""

        category = event.get(
            "category"
        ) or ""

        if isinstance(category, list):
            category = ", ".join(
                map(str, category)
            )

        keywords = event.get(
            "keywords_fr"
        ) or ""

        if isinstance(keywords, list):
            keywords = ", ".join(
                map(str, keywords)
            )

        # -----------------------------
        # Construction du document RAG
        # -----------------------------

        content = f"""
Titre : {title}
Ville : {city}
Lieu : {location}
Adresse : {address}
Date : {dates}
Tarif / Conditions : {conditions}
Accessibilité : {accessibility}
Catégorie : {category}
Mots-clés : {keywords}
Description : {description}
Description détaillée : {long_description}
""".strip()

        metadata = {
            "uid": event.get("uid"),
            "title": title,
            "city": city,
            "location": location,
            "dates": dates,
            "category": category,
            "conditions": conditions,
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    print(f"✓ {len(documents)} documents structurés préparés")

    if not documents:
        raise ValueError("Aucun document à indexer.")

    # -----------------------------
    # CHUNKING
    # -----------------------------

    print("--- [2/3] DECOUPAGE (CHUNKING) ---")
    print(f"Chunk size : {CHUNK_SIZE}")
    print(f"Chunk overlap : {CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✓ {len(chunks)} chunks générés")

    # -----------------------------
    # EMBEDDINGS + FAISS
    # -----------------------------

    print("--- [3/3] CRÉATION ET SAUVEGARDE DE LA BASE FAISS ---")

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        mistral_api_key=MISTRAL_API_KEY
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"✓ Base vectorielle sauvegardée dans '{FAISS_INDEX_PATH}'")


if __name__ == "__main__":
    build_vector_db()