import streamlit as st
from PIL import Image
import base64
from pathlib import Path

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Portfolio | Haron MATAOUI",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FONCTION POUR ENCODER LES IMAGES ---
def img_to_bytes(img_path):
    """Convertit une image locale en base64 pour l'affichage en HTML."""
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        return None

# --- CSS PERSONNALISÉ ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        h1, h2 {
            color: #0d1b2a;
        }
        h3 {
            color: #1b263b;
        }

        /* --- BARRE LATÉRALE --- */
        [data-testid="stSidebar"] {
            background-color: #e6f0ff; /* Bleu clair ENSIIE */
            border-right: 1px solid #cbd5e1;
        }

        /* --- Photos de profil côte à côte --- */
        .profile-pics {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .profile-pics img {
            border-radius: 50%;
            width: 120px;
            height: 120px;
            object-fit: cover;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .profile-pics img:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        }

        /* --- Carte projet (compacte) --- */
        .project-card {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.06);
            transition: all 0.3s ease-in-out;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100%;
            max-height: 370px;
        }
        .project-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }
        .project-card img {
            width: 100%;
            height: 140px;
            object-fit: cover;
            border-radius: 10px 10px 0 0;
        }
        .project-content {
            padding: 1rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        .project-text h3 { 
            margin: 0 0 0.3rem 0; 
            font-size: 1.05rem;
        }
        .project-text p {
            font-size: 0.9rem;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        .project-tags {
            margin-bottom: 0.8rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .tech-tag {
            background-color: #e0e7ff;
            color: #3730a3;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .project-buttons {
            margin-top: auto;
            display: flex;
            gap: 0.4rem;
        }
        .button-link {
            background-color: #1b263b;
            color: white !important;
            padding: 0.4rem 0.8rem;
            border-radius: 0.5rem;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.85rem;
            text-align: center;
            flex-grow: 1;
        }
        .button-link.Lancer {
            background-color: #415a77;
        }
        .button-link:hover {
            opacity: 0.85;
        }

        /* --- Icônes de contact (modifié ici) --- */
        .contact-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        .contact-item svg {
            width: 16px;  /* Taille réduite */
            height: 16px; /* Taille réduite */
            fill: #415a77;
        }
        .contact-item a {
            font-size: 0.9rem; /* texte légèrement plus petit */
            color: #1b263b;
            text-decoration: none;
        }
        .contact-item a:hover {
            color: #2563eb;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    img1 = img_to_bytes("assets/photo_profil.png")
    img2 = img_to_bytes("assets/photo_profil3.jpg")
    if img1 and img2:
        st.markdown(
            f"""
            <div class="profile-pics">
                <a href="https://www.ensiie.fr" target="_blank" title="ENSIIE - Institut Mines Télécom">
                    <img src="data:image/png;base64,{img1}">
                <a href="https://www.imt.fr" target="_blank" title="IMT - Institut Mines Télécom">
                    <img src="data:image/jpg;base64,{img2}">
            </div>
            """,
            unsafe_allow_html=True
        )
    elif img1:
        st.image(Image.open("assets/photo_profil.png"), width=150)
    st.markdown("<h1 style='text-align: center;'>Haron MATAOUI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>🎓 Élève ingénieur à l’ENSIIE (IMT)</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>💡 Passionné par la Data Science & le Machine Learning</p>", unsafe_allow_html=True)
    
    st.divider()

    st.markdown("### Compétences Techniques")
    st.markdown("""
    - **Langages** : Python, R, SQL, C++
    - **Data Science** : Pandas, NumPy, Scikit-learn, PyTorch, tensorMatplotlib, Seaborn
    - **Web/Frameworks** : Streamlit, Flask, HTML, CSS
    - **Outils** : Git, Jupyter Notebook
    """)

    st.divider()

    st.markdown("### Contact")
    st.markdown("""
        <div class="contact-item">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M22 6C22 4.9 21.1 4 20 4H4C2.9 4 2 4.9 2 6V18C2 19.1 2.9 20 4 20H20C21.1 20 22 19.1 22 18V6ZM20 6L12 11L4 6H20ZM20 18H4V8L12 13L20 8V18Z"/></svg>
            <a href="mailto:haron.mataoui8@gmail.com">haron.mataoui8@gmail.com</a>
        </div>
        <div class="contact-item">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 3A2 2 0 0 1 21 5V19A2 2 0 0 1 19 21H5A2 2 0 0 1 3 19V5A2 2 0 0 1 5 3H19M18.5 18.5V13.2A3.26 3.26 0 0 0 15.24 9.94C14.39 9.94 13.4 10.46 12.92 11.24V10.13H10.13V18.5H12.92V13.57C12.92 12.8 13.54 12.17 14.31 12.17A1.4 1.4 0 0 1 15.71 13.57V18.5H18.5M6.88 8.56A1.68 1.68 0 0 0 8.56 6.88C8.56 6 8 5.44 7.21 5.44C6.42 5.44 5.86 6 5.86 6.88C5.86 7.77 6.42 8.56 6.88 8.56M8.27 18.5V10.13H5.5V18.5H8.27Z"/></svg>
            <a href="https://www.linkedin.com/in/haron-mataoui-626318289/" target="_blank">LinkedIn</a>
        </div>
        <div class="contact-item">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M2.39,12.05a10.21,10.21,0,0,1,1-.46c.33-.12.68-.22,1-.31.33-.09.66-.17,1-.23L6,10.87,7.26,6.5S7.3,6.33,7.66,6.17,8.28,6,8.28,6l.21.05.28.32.09.43s-1,3.44-1.05,3.61c-.05.17-.09.33-.12.51s0,.36,0,.54a1.64,1.64,0,0,0,.14.65,1.23,1.23,0,0,0,.43.5.91.91,0,0,0,.6.19.87.87,0,0,0,.68-.26,1.23,1.23,0,0,0,.26-.72.76.76,0,0,0,0-.23c0-.12,0-.24,0-.37s0-.28,0-.42l-.06-.43-.12-.42L7.6,9.19,8,8.08l.33-1.11L9.1,3.83s.06-.21.21-.35a.76.76,0,0,1,.48-.24.72.72,0,0,1,.51.1,1.1,1.1,0,0,1,.33.39L12,8.19l.06.43s0,.14,0,.26,0,.24,0,.34a1.23,1.23,0,0,0,.21.72.87.87,0,0,0,.66.26.91.91,0,0,0,.6-.19,1.23,1.23,0,0,0,.43-.5,1.64,1.64,0,0,0,.14-.65c0-.18,0-.36,0-.54s-.09-.34-.12-.51c-.05-.17-1.05-3.61-1.05-3.61l.09-.43.28-.32.21-.05s.58,0,1,.16,0,.33,0,.33L16.74,11l.6,2.08-1.11.33-1.12.33L14,14.15l-.42.12-.43.12-.42.06c-.14,0-.28,0-.42,0s-.26,0-.39,0-.26-.06-.39-.09a1.13,1.13,0,0,1-.35-.16,1,1,0,0,1-.26-.26.76.76,0,0,1-.09-.41.72.72,0,0,1,.06-.33l.12-.43L12.33,12,12,10.88,10.87,7.26,10.42,9l-.26.91-.21.78-.17.6-.12.45c-.09.3-.17.62-.23,1s-.12.61-.14,1.08a4.12,4.12,0,0,0,.26,1.68,3.22,3.22,0,0,0,1,1.2,3.78,3.78,0,0,0,1.59.7,4.35,4.35,0,0,0,1.81.23,5.75,5.75,0,0,0,2.1-.33,5.09,5.09,0,0,0,1.83-1,4.33,4.33,0,0,0,1.25-1.5,3.73,3.73,0,0,0,.43-2c0-.23-.05-.46-.09-.68a4.48,4.48,0,0,0-.16-.65l-.21-.57-.28-.54-2-4.22a.49.49,0,0,1-.1-.31.36.36,0,0,1,.06-.23.31.31,0,0,1,.21-.14.33.33,0,0,1,.28.06l2.1,1.25,2.44,1.47,2.83,1.68a12.83,12.83,0,0,1,1,.73,10.32,10.32,0,0,1,1.45,1.81,10.16,10.16,0,0,1,0,8.42,10.25,10.25,0,0,1-1.45,1.81,10.38,10.38,0,0,1-1.81,1.45,10.28,10.28,0,0,1-8.42,0,10.25,10.25,0,0,1-1.81-1.45,10.38,10.38,0,0,1-1.45-1.81,10.16,10.16,0,0,1,0-8.42Z"/></svg>
            <a href="https://gitlab.com/users/haron.mataoui8/projects" target="_blank">GitLab</a>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.info("🔍 Recherche stage (3-4 mois) - Mai 2025")


# --- CONTENU PRINCIPAL ---
st.title("Haron MATAOUI, Ingénieur & Data Scientist")
st.write("""
Je suis élève ingénieur à l’**ENSIIE – Institut Mines Télécom**. Passionné par la **data**, ma démarche consiste à transformer des données complexes en informations stratégiques et en solutions intelligentes.
""")
st.divider()

# --- PROJETS (inchangés sauf taille réduite) ---
projects = [
    {
        "title": "📈 Simulation Monte Carlo - Portefeuille", 
        "desc": "Analyse des rendements potentiels et performance d’un portefeuille d’investissement.", 
        "img": "assets/projet1.png", 
        "code": "https://gitlab.com/haron.mataoui8/analyse-de-portefeuille-d-investissement-par-simulation-de-monte-carlo", 
        "Lancer": "/Simulation_Monte_Carlo_-_Portefeuille_Boursier",
        "tech": ["Python", "Pandas", "NumPy", "Matplotlib"]
    },
    {
        "title": "🏠 Prédiction du statut de prêt", 
        "desc": "Prédiction du statut d’un prêt bancaire à partir de données clients.", 
        "img": "assets/projet2.png", 
        "code": "https://gitlab.com/haron.mataoui8/machine-learning-pret-bancaire", 
        "Lancer": "/Prediction_pret",
        "tech": ["Scikit-learn", "Pandas", "Streamlit"]
    },
    {
        "title": "🩺 Prédiction diabète", 
        "desc": "Analyse prédictive basée sur des données médicales pour détecter le diabète.", 
        "img": "assets/projet3.png", 
        "code": "https://gitlab.com/haron.mataoui8/machine-learning-diabete", 
        "Lancer": "/Diabetes_Classification",
        "tech": ["Machine Learning", "Seaborn", "Classification"]
    },
    {
        "title": "💡 Production d’électricité", 
        "desc": "Analyse et visualisation de la production et consommation électrique mondiale.", 
        "img": "assets/projet4.png", 
        "Lancer": "/Diabetes_Classification",
        "tech": ["Flask", "Plotly", "Analyse de données"]
    },

    {
        "title": " Analyse de portefeuille ", 
        "desc": " Permet d’explorer des données boursières historiques de plusieurs actions, de visualiser leurs prix ajustés, rendements, volatilité, corrélations et de réaliser une analyse en composantes principales (PCA). Elle intègre également un modèle simple de prévision linéaire des prix futurs.", 
        "img": "assets/pf.png", 
        "Lancer":"/Analyse_de_portefeuille",
        "tech": ["yfinance", "NumPy", "pandas", "sklearn", "PCA"]
    },

    {
        "title": "💳 Détection de Fraude Bancaire", 
        "desc": "Analyse un jeu de données de transactions par carte bancaire. Applique un rééquilibrage (SMOTE) pour gérer la rareté des fraudes et entraîne un réseau de neurones avec TensorFlow/Keras pour classer les transactions comme légitimes ou frauduleuses. Inclut une interface de prédiction en temps réel.", 
        "img": "assets/phishing.jpg", 
        "code": "http://localhost:8506/fraud_detection_tensorflow", 
        "Lancer":"/raud_detection_tensor",
        "tech": ["TensorFlow", "Keras", "Streamlit", "pandas", "scikit-learn", "imbalanced-learn"]
    }



]

st.header("Mes Projets Récents")
cols = st.columns(2)

for i, proj in enumerate(projects):
    with cols[i % 2]:
        img_encoded = img_to_bytes(proj["img"])
        image_html = f"<img src='data:image/png;base64,{img_encoded}'>" if img_encoded else "<div style='height: 140px; background:#e0e0e0; border-radius:10px 10px 0 0;'></div>"
        tags_html = "".join([f"<span class='tech-tag'>{t}</span>" for t in proj["tech"]])
        buttons_html = f"""
        <div class="project-buttons">
            <a href="{proj['code']}" target="_blank" class="button-link">Code ↗️</a>
            {f'<a href="{proj["Lancer"]}" target="_self" class="button-link Lancer">Lancer</a>' if proj["Lancer"] else ""}
        </div>
        """
        project_html = f"""
        <div class="project-card">
            {image_html}
            <div class="project-content">
                <div class="project-text">
                    <h3>{proj['title']}</h3>
                    <p>{proj['desc']}</p>
                </div>
                <div class="project-tags">{tags_html}</div>
                {buttons_html}
            </div>
        </div>
        """
        st.markdown(project_html, unsafe_allow_html=True)
        st.write("")

st.divider()
st.success("💬 N’hésitez pas à me contacter pour échanger sur mes projets ou sur une opportunité de stage !")
