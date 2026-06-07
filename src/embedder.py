# Importer les biblios nécessaires

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunker import decouper_documents
from loader import charger_documents

# Méthode pour générer les embeddings et stocker dans FAISS

def creer_vs(chunks, save_path="./stockage"):
    embeddings = HuggingFaceEmbeddings(model_name="OrdalieTech/Solon-embeddings-large-0.1")
    vecteurs_stockage = FAISS.from_documents(chunks, embeddings)
    vecteurs_stockage.save_local(save_path)
    print(f"Sauvegardé dans {save_path}!")
    return vecteurs_stockage
