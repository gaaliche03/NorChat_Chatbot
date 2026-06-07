import os
from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
from src.retriever import charger_faiss
from src.chain import creer_chaine, poser_question

st.set_page_config(
    page_title="NorChat",
    page_icon="icon.png",
    layout="centered"
)

# CSS
st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600&display=swap');

            :root {
                --violet:  #6B3FA0;
                --noir:    #1C1C2E;
                --gris-f:  #F3F4F8;
                --gris-t:  #8890A8;
                --blanc:   #FFFFFF;
                --bordure: #DDE1ED;
            }

            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif !important;
                background-color: var(--gris-f) !important;
                color: var(--noir) !important;
            }
            #MainMenu, footer, header { visibility: hidden; }

            /* zone principale*/
            .block-container {
                max-width: 780px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-top: 2rem !important;
                padding-bottom: 1rem !important;
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
            }

            /* barre en haut de urn */
            .top-bar {
                height: 3px;
                background: linear-gradient(90deg, #6B3FA0 0%, #00AEEF 35%, #E8629A 65%, #F7941D 100%);
                position: fixed;
                top: 0; left: 0; right: 0;
                z-index: 9999;
            }

            /*1page d'acceuil */
            .welcome-wrap {
                text-align: center;
                padding: 48px 0 32px;
            }
            .version-badge {
                display: inline-block;
                background: #f0eaf8;
                color: var(--violet);
                font-family: 'Outfit', sans-serif;
                font-size: 0.65rem;
                font-weight: 600;
                padding: 3px 10px;
                border-radius: 20px;
                margin-left: 8px;
                vertical-align: middle;
                letter-spacing: 0.04em;
            }
            .welcome-sub {
                font-size: 0.95rem;
                color: var(--gris-t);
                font-weight: 300;
                margin-bottom: 10px;
            }
            .urn-bars {
                display: flex;
                justify-content: center;
                gap: 6px;
                margin-bottom: 40px;
            }
            .urn-bar {
                width: 28px; height: 4px;
                border-radius: 2px;
                display: inline-block;
            }

            /*carousel: les suggestions de question */
            .carousel-box {
                background: var(--blanc);
                border: 1.5px solid #C8CEDD;
                border-radius: 12px;
                padding: 22px 30px;
                text-align: center;
                font-size: 0.9rem;
                color: #4a4a6a;
                font-style: italic;
                line-height: 1.6;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                min-height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .carousel-box strong { color: var(--violet); font-style: normal; }

            .carousel-dots {
                display: flex;
                justify-content: center;
                gap: 7px;
                margin-top: 12px;
                margin-bottom: 16px;
            }
            .cdot {
                width: 7px; height: 7px;
                border-radius: 50%;
                display: inline-block;
            }

            /*bouton de poser la question*/
            .stButton > button {
                background: var(--blanc) !important;
                border: 1.5px solid #C8CEDD !important;
                color: var(--noir) !important;
                border-radius: 10px !important;
                font-family: 'Outfit', sans-serif !important;
                font-size: 0.84rem !important;
                text-align: center !important;
                padding: 10px 14px !important;
                line-height: 1.5 !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
                transition: all 0.15s !important;
                height: auto !important;
            }
            .stButton > button:hover {
                border-color: var(--violet) !important;
                box-shadow: 0 2px 10px rgba(107,63,160,0.12) !important;
            }

            /* disclaimer: verif sur lien de urn*/
            .disclaimer {
                text-align: center;
                font-size: 0.68rem;
                color: var(--gris-t);
                margin-top: 10px;
                padding-bottom: 4px;
            }
            .disclaimer a { color: var(--violet); text-decoration: none; }

            /*input de question*/
            [data-testid="stChatInput"] > div {
                background: var(--blanc) !important;
                border: 1px solid var(--bordure) !important;
                border-radius: 10px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            }
            [data-testid="stChatInput"] > div:focus-within {
                border-color: var(--violet) !important;
                box-shadow: 0 0 0 3px rgba(107,63,160,0.1) !important;
            }
            [data-testid="stChatInput"] textarea {
                font-family: 'Outfit', sans-serif !important;
                font-size: 0.9rem !important;
                color: var(--noir) !important;
            }
            </style>
            <div class="top-bar"></div>""", unsafe_allow_html=True)

#chargement du chatbot se fait seulement la 1ere fois lors du run
# @st.cache_resource charge FAISS et le client LLM une seule fois au démarrage 

@st.cache_resource(show_spinner=False)
def charger_ressources():
    vs = charger_faiss()
    client, vs = creer_chaine(vs)
    return client, vs

if "messages"      not in st.session_state: st.session_state.messages      = []       # contient la listes de tous les msgs de la convo (role,contenu)
if "historique"    not in st.session_state: st.session_state.historique    = []     # mémoire envoyé au llm dans prompt, liste du tuples (question,reponse)
if "question_auto" not in st.session_state: st.session_state.question_auto = None #question de carossel
if "sug_idx"       not in st.session_state: st.session_state.sug_idx       = 0 #chaque sugg contient un indx pour savoir quelle question

with st.spinner("Chargement de NorChat…"):
    try:
        client, vs = charger_ressources()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

# carousel: suggestions

SUGGESTIONS = [
    "Conditions d'admission en Master Data Science",
    "Procédure d'inscription pour étudiants étrangers",
    "Quels sont les frais de scolarité ?",
    "Quelles matières sont enseignées en M1 ?",
]

#ecran d'acceuil : 1ere page vue

if not st.session_state.messages:

    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
                <div class="welcome-wrap">
                    <div class="welcome-title">
                        <img src="data:image/png;base64,{logo_b64}" style="height: 80px; margin-bottom: 10px;" />
                        <span class="version-badge">v1.0</span>
                    </div>
                    <div class="welcome-sub">L'assistant IA de l'Université de Rouen Normandie</div>
                    <div class="urn-bars">
                        <span class="urn-bar" style="background:#6B3FA0"></span>
                        <span class="urn-bar" style="background:#00AEEF"></span>
                        <span class="urn-bar" style="background:#E8629A"></span>
                        <span class="urn-bar" style="background:#F7941D"></span>
                    </div>
                </div>""", unsafe_allow_html=True)

    #carousel
    n = len(SUGGESTIONS)  #nbre de sugg
    idx = st.session_state.sug_idx #indx actuel
    q_actuelle = SUGGESTIONS[idx]

    col_l, col_c, col_r = st.columns([1, 10, 1])
    with col_l:
        if st.button("‹", key="prev_sug"):
            st.session_state.sug_idx = (idx - 1) % n
            st.rerun()
    with col_c:
        st.markdown(f"""
                    <div class="carousel-box">
                        <strong>"</strong>{q_actuelle}<strong>"</strong>
                    </div>""", unsafe_allow_html=True)
    with col_r:
        if st.button("›", key="next_sug"):
            st.session_state.sug_idx = (idx + 1) % n
            st.rerun()

    #dots
    dots_html = '<div class="carousel-dots">'
    for i in range(n):
        color = "#6B3FA0" if i == idx else "#DDE1ED"
        dots_html += f'<span class="cdot" style="background:{color}"></span>'
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)

    #bouton "Poser cette question"
    _, col_btn, _ = st.columns([2, 3, 2])
    with col_btn:
        if st.button("Poser cette question →", key="ask_sug", use_container_width=True):
            st.session_state.question_auto = q_actuelle
            st.rerun()

    # Logo université
    with open("logo_univ.jpg", "rb") as f:
        univ_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
                <div style="text-align:center; margin-top: 30px;">
                    <img src="data:image/jpeg;base64,{univ_b64}" style="height: 60px; opacity: 0.8;" />
                </div>""", unsafe_allow_html=True)

#zone de chat 

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

#input

question = st.chat_input("Posez votre question sur l'Université de Rouen Normandie…")

if st.session_state.question_auto:
    question = st.session_state.question_auto
    st.session_state.question_auto = None

#disclaimer

st.markdown("""
            <div class="disclaimer">
                NorChat peut commettre des erreurs. Il est recommandé de vérifier les informations sur
                <a href="https://www.univ-rouen.fr" target="_blank">univ-rouen.fr</a>
            </div>""", unsafe_allow_html=True)

#traiment des questions

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                reponse, st.session_state.historique = poser_question(
                    question, client, vs, st.session_state.historique
                )
            except Exception as e:
                reponse = f"Une erreur s'est produite : {e}"
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    st.rerun()