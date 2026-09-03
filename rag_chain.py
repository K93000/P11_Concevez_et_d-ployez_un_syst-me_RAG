import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    MISTRAL_API_KEY,
    TOP_K,
)

load_dotenv()


def format_docs(docs):
    """Formate les documents extraits de la base vectorielle."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Événement {i} ---\n{doc.page_content}\n")
    return "\n".join(formatted)


def get_rag_chain(k=TOP_K):
    """Initialise et retourne la chaîne RAG et son retriever."""
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY non trouvée dans le fichier .env")

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL, mistral_api_key=MISTRAL_API_KEY
    )
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = ChatMistralAI(
        model=LLM_MODEL, temperature=0.2, mistral_api_key=MISTRAL_API_KEY
    )

    system_prompt = (
        "Vous êtes un assistant virtuel spécialisé dans les événements de Seine-Saint-Denis (93).\n"
        "Répondez à la question de l'utilisateur de manière précise, polie et concise, "
        "en vous basant exclusivement sur les événements fournis dans le contexte.\n"
        "Si la question ne concerne pas les événements de Seine-Saint-Denis (93), "
        "indiquez que vous êtes spécialisé uniquement dans ce domaine.\n"
        "Si les données du contexte ne contiennent pas l'information demandée, "
        "indiquez clairement que vous ne disposez pas de cette information dans votre base.\n"
        "N'inventez aucun événement, lieu, tarif, date ou information.\n\n"
        "Contexte des événements disponibles :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def query_chatbot(question: str) -> str:
    """Interroge le chatbot RAG."""
    chain, _ = get_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    question_test = "Quels sont les concerts ou événements musicaux qui ont eu lieu à Aubervilliers ?"
    print(f"Question : {question_test}\n")
    response = query_chatbot(question_test)
    print(f"Réponse du Chatbot :\n{response}")