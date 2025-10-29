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

# --- FONCTION UTILITAIRE ---
def img_to_bytes(img_path):
    try:
        img_bytes = Path(img_path).read_bytes()
        return base64.b64encode(img_bytes).decode()
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

/* TITRES */
h1 {
    background: linear-gradient(90deg, #1b263b, #415a77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2, h3 {
    color: #0d1b2a;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #e6f0ff;
    border-right: 1px solid #cbd5e1;
}

/* PROFILE PICS */
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

/* PROJECT CARD */
.project-card {
    background-color: #ffffff;
    border-radius: 14px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.06);
    transition: all 0.3s ease-in-out;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
}
.project-card:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.project-card img {
    width: 100%;
    height: 150px;
    object-fit: cover;
}
.project-content {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
.project-text h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.05rem;
}
.project-text p {
    font-size: 0.9rem;
    line-height: 1.3;
}
.project-tags {
    margin-top: 0.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.tech-tag {
    background-color: #e0e7ff;
    color: #3730a3;
    padding: 0.25rem 0.7rem;
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
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: white !important;
    padding: 0.4rem 0.8rem;
    border-radius: 0.5rem;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.85rem;
    text-align: center;
    flex-grow: 1;
    transition: all 0.3s;
}
.button-link:hover {
    opacity: 0.9;
    transform: translateY(-2px);
}
.button-link.Lancer {
    background: linear-gradient(135deg, #0f172a, #334155);
}

/* CONTACT */
.contact-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.contact-item svg {
    width: 16px;
    height: 16px;
    fill: #415a77;
}
.contact-item a {
    font-size: 0.9rem;
    color: #1b263b;
    text-decoration: none;
}
.contact-item a:hover {
    color: #2563eb;
    text-decoration: underline;
}

/* RECHERCHE INPUT */
input[type="text"] {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
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
                <a href="https://www.ensiie.fr" target="_blank" title="ENSIIE"><img src="data:image/png;base64,{img1}"></a>
                <a href="https://www.imt.fr" target="_blank" title="IMT"><img src="data:image/jpg;base64,{img2}"></a>
            </div>
            """, unsafe_allow_html=True
        )
    st.markdown("<h1 style='text-align: center;'>Haron MATAOUI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>🎓 Élève ingénieur à l’ENSIIE (IMT)</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'> Passionné par la Data Science & le Machine Learning</p>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("###  Mon CV")
    cv_url = "https://raw.githubusercontent.com/haron-mataoui/portfolio-haron/main/assets/CV_Haron_MATAOUI.pdf"
    st.markdown(f"""
        <div style="text-align:center;">
            <a href="{cv_url}" target="_blank" 
            style="background:#1d4ed8;color:white;padding:0.6rem 1rem;border-radius:8px;text-decoration:none;">
            📥 Télécharger le CV
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("###  Compétences")
    st.markdown("""
    - **Langages** : Python, R, SQL, C++
    - **Data Science** : Pandas, NumPy, Scikit-learn, TensorFlow
    - **Web** : Streamlit, Flask, HTML, CSS
    - **Outils** : Git, Jupyter
    """)
    st.divider()
    st.markdown("###  Contact")
    st.markdown("""
    <div class="contact-item">
        <a href="mailto:haron.mataoui8@gmail.com"> haron.mataoui8@gmail.com</a>
    </div>
    <div class="contact-item">
        <a href="https://www.linkedin.com/in/haron-mataoui-626318289/" target="_blank"> LinkedIn</a>
    </div>
    <div class="contact-item">
        <a href="https://gitlab.com/users/haron.mataoui8/projects" target="_blank"> GitLab</a>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.info(" Internship search (3-4 months) - May 2025")

# --- CONTENU PRINCIPAL ---
st.title("Haron MATAOUI — Engineer & Data Scientist")
st.write("""
I am an engineering student at **ENSIIE – Institut Mines Télécom**.  
My focus is on **data-driven decision-making** and **intelligent model design**.
""")
st.divider()

# --- PROJETS ---
projects = [
    {
        "title": "📈 Monte Carlo simulation - Portfolio",
        "desc": "Analysis of potential returns and performance of an investment portfolio.",
        "img": "assets/projet1.png",
        "code": "https://gitlab.com/haron.mataoui8/analyse-de-portefeuille-d-investissement-par-simulation-de-monte-carlo",
        "Lancer": "/Simulation_Monte_Carlo_-_Portefeuille_Boursier",
        "tech": ["Python", "Pandas", "NumPy", "Matplotlib"]
    },
    {
        "title": "🏠 Prediction of loan status",
        "desc": "Predicting the status of a bank loan based on customer data.",
        "img": "assets/projet2.png",
        "code": "https://gitlab.com/haron.mataoui8/machine-learning-pret-bancaire",
        "Lancer": "/Prediction_pret",
        "tech": ["Scikit-learn", "Pandas", "Streamlit"]
    },
    {
        "title": "🩺 Diabetes prediction",
        "desc": "Predictive analysis based on medical data to detect diabetes.",
        "img": "assets/projet3.png",
        "code": "https://gitlab.com/haron.mataoui8/machine-learning-diabete",
        "Lancer": "/Diabetes_Classification",
        "tech": ["Machine Learning", "Seaborn", "Classification"]
    },
    {
        "title": "💡 Electricity generation",
        "desc": "Visualization of global electricity production and consumption.",
        "img": "assets/projet4.png",
        "Lancer": "https://haron2003.pythonanywhere.com",
        "tech": ["Flask", "Plotly", "Data analysis"]
    },
    {
        "title": "📊 Portfolio analysis",
        "desc": "Explore historical market data, PCA and volatility analysis.",
        "img": "assets/pf.png",
        "code": "https://gitlab.com/haron.mataoui8",
        "Lancer": "/Analyse_de_portefeuille",
        "tech": ["yfinance", "NumPy", "sklearn", "Pandas"]
    },
    {
        "title": "💳 Bank Fraud Detection",
        "desc": "Fraud detection using neural networks and SMOTE balancing.",
        "img": "assets/phishing.jpg",
        "code": "https://gitlab.com/haron.mataoui8",
        "Lancer": "/fraud_detection_tensorflow",
        "tech": ["TensorFlow", "Keras", "Streamlit"]
    }
]



# --- AFFICHAGE DES PROJETS ---
cols = st.columns(2)
for i, proj in enumerate(projects):
    with cols[i % 2]:
        img_encoded = img_to_bytes(proj["img"]) if proj.get("img") else None
        image_html = (
            f"<img src='data:image/png;base64,{img_encoded}'>" 
            if img_encoded else "<div style='height:140px;background:#e0e0e0;'></div>"
        )
        tags_html = "".join(f"<span class='tech-tag'>{t}</span>" for t in proj["tech"])
        buttons_html = "<div class='project-buttons'>"
        if proj.get("code"):
            buttons_html += f'<a href="{proj["code"]}" target="_blank" class="button-link">💻 Code</a>'
        if proj.get("Lancer"):
            buttons_html += f'<a href="{proj["Lancer"]}" target="_self" class="button-link Lancer">🚀 Launch</a>'
        buttons_html += "</div>"

        st.markdown(f"""
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
        """, unsafe_allow_html=True)
        st.write("")

st.divider()
st.success("💬 Feel free to contact me for collaborations or internship opportunities!")
