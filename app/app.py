from pathlib import Path

import streamlit as st
import sys 

from components.sidebar_brand import render_sidebar_brand
from components.sidebar_status import render_sidebar_status
from components.styles import load_css
from config import APP_NAME

# ============================================================
# RUTA RAÍZ DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_sidebar_brand()
# ---------------------------------------------------------
# Cargar estilos CSS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "assets" / "styles.css"

if CSS_PATH.exists():
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


# ---------------------------------------------------------
# Definición de páginas
# ---------------------------------------------------------
inicio = st.Page(
    "pages/inicio.py",
    title="Inicio",
    icon=":material/home:",
    default=True,
)

conagua = st.Page(
    "pages/conagua.py",
    title="Datos CONAGUA",
    icon=":material/rainy:",
)

sustax = st.Page(
    "pages/sustax.py",
    title="Escenarios SSP",
    icon=":material/satellite_alt:",
)

gev = st.Page(
    "pages/gev.py",
    title="Análisis GEV",
    icon=":material/show_chart:",
)

excedencias = st.Page(
    "pages/excedencias.py",
    title="Excedencias",
    icon=":material/water:",
)

mapas = st.Page(
    "pages/mapas.py",
    title="Mapas de riesgo",
    icon=":material/map:",
)

tendencias = st.Page(
    "pages/tendencias.py",
    title="Tendencias climáticas",
    icon=":material/trending_up:",
)

ia = st.Page(
    "pages/inteligencia_artificial.py",
    title="Inteligencia Artificial",
    icon=":material/psychology:",
)

reportes = st.Page(
    "pages/reportes.py",
    title="Reportes",
    icon=":material/description:",
)

configuracion = st.Page(
    "pages/configuracion.py",
    title="Configuración",
    icon=":material/settings:",
)


# ---------------------------------------------------------
# Navegación principal
# ---------------------------------------------------------
pagina = st.navigation(
    {
        "Plataforma": [inicio],
        "Análisis hidrometeorológico": [
            conagua,
            sustax,
            gev,
            excedencias,
            mapas,
            tendencias,
        ],
        "Herramientas": [
            ia,
            reportes,
            configuracion,
        ],
    }
)

render_sidebar_status()
pagina.run() 