# Importer les méthodes nécessaires

from retriever import charger_faiss
from chain import creer_chaine, poser_question

# Test du bon fonctionnement conversationnel

db = charger_faiss()
client, vs = creer_chaine(db)
historique = []
print("Avec quoi NorChat peut-il vous aider ? (tapez X pour quitter)\n")
while True:
    question = input("Vous : ")
    if question.lower() == "x":
        break
    reponse, historique = poser_question(question, client, vs, historique)
    print(f"\nNorChat : {reponse}\n")