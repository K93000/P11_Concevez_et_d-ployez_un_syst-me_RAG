import json
import os
import sys
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Patch de compatibilité RAGAS (contourne un import interne défectueux de Ragas)
import langchain_google_vertexai
sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from langchain_mistralai import ChatMistralAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_chain import get_rag_chain

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY introuvable dans le fichier .env.")

# 1. Instanciation du juge LLM pour RAGAS (Mistral Small)
evaluator_llm = LangchainLLMWrapper(
    ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        mistral_api_key=mistral_api_key,
    )
)

# Seules les 3 métriques actives sont conservées (pas de Answer Relevancy / pas de Wrapper Embeddings)
metrics = [
    Faithfulness(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]

# 2. Chargement du jeu de test
print("\n--- [1/3] CHARGEMENT DU DATASET ---")
dataset_path = os.path.join("evaluation", "eval_dataset.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 3. Exécution du RAG
print("\n--- [2/3] GÉNÉRATION DES RÉPONSES ---")
chain, retriever = get_rag_chain()

results = []
for i, item in enumerate(eval_data, start=1):
    question = item["user_input"]
    category = item["category"]

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    results.append(
        {
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "ground_truth": item["reference"],
            "category": category,
        }
    )

# 4. Calcul et export des métriques
print("\n--- [3/3] CALCUL DES MÉTRIQUES RAGAS ---")
ragas_dataset = Dataset.from_list(results)
scores = evaluate(dataset=ragas_dataset, metrics=metrics)

df_scores = scores.to_pandas()
df_scores["category"] = [item["category"] for item in results]

metrics_cols = ["faithfulness", "context_precision", "context_recall"]

print("\n================ RÉSULTATS GLOBAUX RAGAS ================")
print(df_scores[metrics_cols].mean())

print("\n---------------- RÉSULTATS PAR CATÉGORIE ----------------")
print(df_scores.groupby("category")[metrics_cols].mean())

output_file = os.path.join("evaluation", "ragas_evaluation_report.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_scores.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✓ Rapport sauvegardé sous : {output_file}")