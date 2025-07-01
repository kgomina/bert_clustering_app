# app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model_utils import process_sentences, cluster_embeddings, reduce_dimensions, mean_cosine_distance

# 🌈 Style visuel avec CSS
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right, #dfe9f3, #ffffff);
        color: #000000;
    }

    h1, h2, h3 {
        color: #003366;
    }

    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Configuration de page
st.set_page_config(page_title="Désambiguïsation du mot 'bank'", layout="wide")

# 🧠 Titre
st.title("🔍 Analyse sémantique contextuelle du mot *'bank'*")
st.markdown("Cette application utilise **BERT** pour détecter automatiquement les différents sens du mot **'bank'** selon son contexte, puis applique du **clustering et une visualisation t-SNE**.")

# 📝 Entrée utilisateur
st.subheader("✍️ Entrez vos phrases contenant le mot *bank* :")

sentences_input = st.text_area("Chaque phrase sur une ligne :", value=
"""He went to the bank to deposit money.
She stood on the river bank and watched the water.
The bank approved her loan application.
Children played by the river bank.
He opened a new account at the bank.
They built a house near the bank of the river.
He left the bank after 10 years of work.
A boat was tied to the river bank.""")

sentences = [s.strip() for s in sentences_input.strip().split("\n") if s.strip()]
if len(sentences) < 2:
    st.warning("🚨 Veuillez saisir au moins deux phrases.")
    st.stop()

# 🔁 Traitement des données
with st.spinner("🔄 Traitement en cours avec BERT..."):
    embeddings, valid_sentences = process_sentences(sentences)
    emb_2d = reduce_dimensions(embeddings)
    labels = cluster_embeddings(embeddings)
    dist = mean_cosine_distance(embeddings)

# 📊 Affichage du DataFrame avec style
st.subheader("📋 Résultat du clustering")
df = pd.DataFrame({
    "Phrase": valid_sentences,
    "Cluster": labels
})
st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)

# 📈 Visualisation t-SNE
st.subheader("🌐 Visualisation 2D des clusters (t-SNE)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette="Set2", s=100, ax=ax)
for i, txt in enumerate(valid_sentences):
    ax.annotate(f"{i+1}", (emb_2d[i,0], emb_2d[i,1]), fontsize=9)
ax.set_title("Clusters sémantiques de 'bank'")
st.pyplot(fig)

# 📌 Statistiques rapides
st.subheader("📏 Statistique")
col1, col2 = st.columns(2)
with col1:
    st.metric("🧮 Nombre de phrases", len(valid_sentences))
with col2:
    st.metric("📐 Distance moyenne cosinus", f"{dist:.4f}")
