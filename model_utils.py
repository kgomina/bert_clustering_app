# model_utils.py
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

def extract_embedding(sentence, target="bank"):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state.squeeze(0)
    token_ids = tokenizer.convert_ids_to_tokens(input_ids[0])
    indices = [i for i, tok in enumerate(token_ids) if target in tok]
    if indices:
        vec = torch.mean(embeddings[indices], dim=0)
        return vec.cpu().numpy()
    else:
        return None

def process_sentences(sentences, target="bank"):
    embeddings = []
    valid_sentences = []
    for sent in sentences:
        vec = extract_embedding(sent, target)
        if vec is not None:
            embeddings.append(vec)
            valid_sentences.append(sent)
    return np.array(embeddings), valid_sentences

def cluster_embeddings(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    return tsne.fit_transform(embeddings)

def mean_cosine_distance(embeddings):
    dist_matrix = cosine_distances(embeddings)
    upper_triangle = dist_matrix[np.triu_indices(len(embeddings), k=1)]
    return np.mean(upper_triangle)
