# Importer les biblios nécessaires

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Méthode pour charger la base FAISS

def charger_faiss(path="./stockage"):
    embeddings = HuggingFaceEmbeddings(model_name="OrdalieTech/Solon-embeddings-large-0.1")
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vs

# Méthode pour récupérer les chunks pertinents

def recuperer_chunks(question, base, k=3):
    return base.similarity_search(question, k=k)

# Test du bon fonctionnement, et la pertinence des réponses

if __name__ == "__main__":

    db = charger_faiss()
    queries = ["Quelles sont les frais du master ?", 
           "Quelle est la procédure de l'inscription pour les étudiants étrangers?", 
           "La présence en TP est-elle obligatoire ?"]
    for q in queries:
        print(f"\n\n- Query: {q}")
        resultats = recuperer_chunks(q, db)
        for i, r in enumerate(resultats):
            print(f"\n- Resultat {i+1}\n{r.page_content}")