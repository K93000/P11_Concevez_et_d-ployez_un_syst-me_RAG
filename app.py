import streamlit as st
from rag_chain import get_rag_chain

st.set_page_config(
    page_title="Assistant Événements 93", page_icon="🎭", layout="centered"
)

st.title("🎭 Assistant Événements Seine-Saint-Denis (93)")
st.markdown(
    "Posez vos questions sur les événements culturels, artistiques et d'insertion qui se sont déroulés en Seine-Saint-Denis."
)


@st.cache_resource
def load_rag():
    return get_rag_chain()


chain, retriever = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Quels événements ont eu lieu à Bondy ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements en cours..."):
            response = chain.invoke(prompt)
            st.markdown(response)

            docs = retriever.invoke(prompt)
            with st.expander("Voir les sources utilisées"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Source {i} :** {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})