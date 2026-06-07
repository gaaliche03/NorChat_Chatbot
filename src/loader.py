
# Importer les biblios nécessaires

import os
from langchain_community.document_loaders import TextLoader

# Méthode pour charger les documennts

def charger_documents(data_dir="data/"):
    documents = []
    for f in os.listdir(data_dir):
        if f.endswith(".md"):
            chargement = TextLoader(os.path.join(data_dir, f))
            documents.extend(chargement.load())
    return documents


