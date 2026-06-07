# Importer les biblios nécessaires

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from src.retriever import recuperer_chunks

# Loader mon token Huggingface

load_dotenv()

# Créer une template pour le prompt qu'on donnera au LLM

PROMPT_TEMPLATE = """Tu es NorChat, un assistant intelligent, sympathique et agréable de l'Université de Rouen Normandie. Tu réponds de manière naturelle, claire et conviviale.

Règles importantes :
- Tu ne génères JAMAIS de réponse à partir de tes propres connaissances. Tu utilises UNIQUEMENT les informations du contexte fourni, rien d'autre.
- Si la question est liée à l'université mais que tu ne trouves pas la réponse dans le contexte, réponds naturellement comme "Je ne suis pas sûr de ça, mais tu peux vérifier sur www.univ-rouen.fr"
- Si la question n'est pas liée à l'université (ex: questions hors-sujet, culture générale, blagues...), réponds simplement que tu es spécialisé dans les questions sur l'Université de Rouen Normandie uniquement.
- Si la question est vague ou ambiguë, pose une question de clarification avant de répondre plutôt que de donner une réponse incomplète ou floue.
- Ne fais jamais référence au "contexte", aux "sections" ou aux "documents". Réponds comme si tu connaissais naturellement ces informations.
- Réponds toujours en français, même si la question est posée dans une autre langue.

Historique de la conversation :
{historique}

Contexte :
{context}

Question : {question}

Réponse :"""

# Méthode pour creer la chaine

def creer_chaine(vs):
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=os.getenv("HF_TOKEN")
    )
    return client, vs

# Méthode pour poser une question et obtenir une réponse

def poser_question(question, client, vs, historique=[]):

    chunks = recuperer_chunks(question, vs)
    context = "\n\n".join([c.page_content for c in chunks])
    historique_str = "\n".join([f"Vous: {h[0]}\nNorChat: {h[1]}" for h in historique])
    prompt_final = PROMPT_TEMPLATE.format(historique=historique_str, context=context, question=question)
    reponse = client.chat_completion(
        messages=[{"role": "user", "content": prompt_final}],
        max_tokens=512,
        temperature=0.3
    )
    reponse_text = reponse.choices[0].message.content
    historique.append((question, reponse_text))

    return reponse_text, historique