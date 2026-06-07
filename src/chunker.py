# Importer les biblios nécessaires
import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loader import charger_documents

# Méthode pour nettoyer le texte
def nettoyer_texte(texte):
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r'[^\w\s\n\.\,\:\;\?\!\-\(\)]', '', texte)
    return texte.strip()

# Méthode pour le chunking
def decouper_documents(documents, chunk_size=500, chunk_overlap=50):
    hdrs = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    md_split = MarkdownHeaderTextSplitter(headers_to_split_on=hdrs)
    rec_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        doc.page_content = nettoyer_texte(doc.page_content)
        md_chunks = md_split.split_text(doc.page_content)
        chunks.extend(rec_split.split_documents(md_chunks))
    return chunks
