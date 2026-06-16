from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from src.retriever import charger_faiss, recuperer_chunks
from src.chain import creer_chaine, poser_question

from dotenv import load_dotenv
load_dotenv()

questions = [
    "Combien coûte une licence ?",
    "Existe-t-il des transports en commun pour se rendre à l'université et en revenir ?",
    "Puis-je travailler à temps partiel en tant qu'étudiant étranger ?",
    "Dois-je utiliser Parcoursup pour la licence ou le master ?",
    "Est-il nécessaire d'ouvrir un compte bancaire français en tant qu'étudiant entrant, ou puis-je utiliser mon compte étranger ?",
    "Quel est le numéro de téléphone d'Espace Monde à l'université ?",
    "Si j'obtiens 8/20 comme note finale pour une UE, est-ce acquis ?",
    "Si j'obtiens 12/20 comme note globale finale en première année de master, mais 8/20 en deuxième année, est-ce que les notes se compensent et est-ce que ma note finale pour le diplôme est de 10/20 ?",
    "Si je suis absent à un TP, combien de temps ai-je pour informer le professeur ?",
    "Le professeur nous a donné la date de l'examen une semaine à l'avance, est-ce autorisé ?",
    "Si j'ai échoué à un cours et que je suis soumis à une procédure de rattrapage, mais que ma nouvelle note est encore plus basse qu'avant, est-ce qu'ils conservent l'ancienne note élevée ?",
    "Puis-je devenir ingénieur en intelligence artificielle après un master en science des données ?",
    "Existe-t-il un laboratoire de recherche dédié au master en science des données ?",
    "Les deux années de master sont-elles enseignées en français ?",
    "Le master en science des données est-il proposé en alternance ? Si oui, quel est le rythme ?",
    "Puis-je effectuer un stage de deux mois à la fin de mon M2 en science des données ?",
    "Si je souhaite me spécialiser en robotique et automatisation, quel parcours du master en science des données est le plus adapté ?",
    "Si j'ai effectué un M1 en science des données et intelligence artificielle dans une autre université, puis-je candidater pour un M2 à l'université de Rouen-Normandie ?",
    "Puis-je faire traduire mes documents officiels en français dans mon pays d'origine ?",
    "Je suis étudiant Erasmus+, dois-je payer la CVEC ?"
]

reponses = [
    "178 euros.",
    "Oui, le Réseau Astuce est le réseau de transports en commun de la métropole et est accessible sur tous les campus.",
    "Oui, vous pouvez travailler jusqu'à 964 heures par an.",
    "Cette plateforme sert à la première candidature à la licence.",
    "L'ouverture d'un compte bancaire français est fortement conseillée et nécessaire.",
    "C'est le 02 35 14 00 26/27.",
    "Non, il faut obtenir au minimum 10/20 pour valider une UE.",
    "Non, les notes des deux années de master ne se compensent pas.",
    "Vous avez une semaine maximum pour l'informer et justifier l'absence.",
    "Non, la date doit être communiquée au minimum 15 jours à l'avance.",
    "Non, c'est la note obtenue pendant le rattrapage qui est prise en compte.",
    "Oui.",
    "Oui, il s'appelle LITIS.",
    "Non, la première année est en français et la seconde en anglais.",
    "Oui, l'alternance est possible avec ce rythme: deux semaines à l'école et deux semaines en entreprise.",
    "Non, la durée du stage en M2 est comprise entre 4 et 6 mois.",
    "Dans ce cas, le parcours le plus adapté est SIME.",
    "Oui, vous pouvez candidater via eCandidat.",
    "Non, la traduction doit impérativement être effectuée en France par un traducteur assermenté.",
    "Non, vous n'avez pas à payer dans ce cas."
]

#récupération du travail de notre chatbot

db = charger_faiss()
client, vs = creer_chaine(db)
contexts = []
answers = []

for q in questions:
    chunks = recuperer_chunks(q, vs)
    contexts.append([c.page_content for c in chunks])

    reponse, _ = poser_question(q, client, vs, historique=[])
    answers.append(reponse)

dataset = Dataset.from_dict({
    "question": questions,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": reponses
})

#juge LLM local

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=512,
    temperature=0.1,
    device_map="auto",
    trust_remote_code=True
)

llm = HuggingFacePipeline(pipeline=pipe)
ragas_llm = LangchainLLMWrapper(llm)

embeddings = HuggingFaceEmbeddings(model_name="OrdalieTech/Solon-embeddings-large-0.1")
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

resultat = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(timeout=600, max_retries=3)
)

print(resultat)