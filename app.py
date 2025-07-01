# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model_utils import process_sentences, cluster_embeddings, reduce_dimensions, mean_cosine_distance

st.set_page_config(page_title="Désambiguïsation du mot 'bank'", layout="wide")

st.title("Analyse sémantique contextuelle avec BERT")

st.write("Ce projet utilise BERT pour détecter automatiquement les différents sens du mot *'bank'* selon son contexte.")

# Entrée utilisateur
sentences_input = st.text_area("Saisissez les phrases (une par ligne)", value=
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
    st.warning("Veuillez saisir au moins deux phrases.")
    st.stop()

# Traitement
with st.spinner("Extraction des embeddings BERT..."):
    embeddings, valid_sentences = process_sentences(sentences)
    emb_2d = reduce_dimensions(embeddings)
    labels = cluster_embeddings(embeddings)
    dist = mean_cosine_distance(embeddings)

# Affichage du DataFrame
df = pd.DataFrame({
    "Phrase": valid_sentences,
    "Cluster": labels
})
st.subheader("Résultat du clustering")
st.dataframe(df)

# Affichage du graphique t-SNE
st.subheader("Visualisation des clusters en 2D (t-SNE)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette="Set2", s=100, ax=ax)
for i, txt in enumerate(valid_sentences):
    ax.annotate(f"{i+1}", (emb_2d[i,0], emb_2d[i,1]), fontsize=9)
ax.set_title("Clusters sémantiques de 'bank'")
st.pyplot(fig)

# Distance moyenne
st.subheader("Distance moyenne cosinus entre les contextes")
st.metric("Distance moyenne", f"{dist:.4f}")
