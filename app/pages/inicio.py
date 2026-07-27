import streamlit as st

from components.activity_panel import render_activity_panel
from components.status_panel import render_status_panel
from components.control_center import render_control_center
from components.hero import render_hero
from components.cards import module_card

from config import (
    PROJECT_DESCRIPTION,
    PROJECT_PROGRESS,
    INSTITUTION,
    PROGRAM,
    VERSION,
)


# ============================================================
# HERO PRINCIPAL
# ============================================================

render_hero()
render_status_panel()

# ============================================================
# ESTADO GENERAL DE LA PLATAFORMA
# ============================================================

st.html(
    '<div class="dcag-section-title">Estado de la plataforma</div>'
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Módulos científicos",
        value="5",
        delta="implementados",
    )

with col2:
    st.metric(
        label="Fuentes de datos",
        value="3",
        delta="CONAGUA · ERA5 · SSP",
    )

with col3:
    st.metric(
        label="Avance plataforma",
        value=f"{PROJECT_PROGRESS}%",
    )

with col4:
    st.metric(
        label="Versión",
        value=VERSION,
        delta="Desarrollo",
    )

st.progress(PROJECT_PROGRESS / 100)


# ============================================================
# MÓDULOS CIENTÍFICOS
# ============================================================

st.html(
    '<div class="dcag-section-title">Módulos científicos</div>'
)


# ------------------------------------------------------------
# FILA 1
# ------------------------------------------------------------

fila1 = st.columns(4)

with fila1[0]:
    module_card(
        icon="🌧️",
        title="Datos CONAGUA",
        description=(
            "Procesamiento y análisis de series históricas "
            "de estaciones meteorológicas."
        ),
        status="● Disponible",
        status_class="status-available",
        page="pages/conagua.py",
    )

with fila1[1]:
    module_card(
        icon="🛰️",
        title="Escenarios SSP",
        description=(
            "Análisis de escenarios climáticos futuros "
            "mediante datos Sustax."
        ),
        status="● En integración",
        status_class="status-integration",
        page="pages/sustax.py",
    )

with fila1[2]:
    module_card(
        icon="📈",
        title="Análisis GEV",
        description=(
            "Modelado estadístico de precipitaciones extremas "
            "y cálculo de periodos de retorno."
        ),
        status="● Disponible",
        status_class="status-available",
        page="pages/gev.py",
    )

with fila1[3]:
    module_card(
        icon="🌊",
        title="Excedencias",
        description=(
            "Estimación de probabilidades para eventos "
            "de precipitación extrema."
        ),
        status="● Disponible",
        status_class="status-available",
        page="pages/excedencias.py",
    )


st.write("")


# ------------------------------------------------------------
# FILA 2
# ------------------------------------------------------------

fila2 = st.columns(4)

with fila2[0]:
    module_card(
        icon="🗺️",
        title="Mapas de riesgo",
        description=(
            "Análisis espacial, interpolación y generación "
            "de cartografía de riesgo."
        ),
        status="● Disponible",
        status_class="status-available",
        page="pages/mapas.py",
    )

with fila2[1]:
    module_card(
        icon="📊",
        title="Tendencias climáticas",
        description=(
            "Evaluación temporal de precipitación histórica "
            "y escenarios climáticos futuros."
        ),
        status="● Próximo módulo",
        status_class="status-next",
        page="pages/tendencias.py",
    )

with fila2[2]:
    module_card(
        icon="🧠",
        title="Inteligencia Artificial",
        description=(
            "Modelos predictivos e interpretación automática "
            "de resultados científicos."
        ),
        status="● Planeado",
        status_class="status-planned",
        page="pages/inteligencia_artificial.py",
    )

with fila2[3]:
    module_card(
        icon="📄",
        title="Reportes",
        description=(
            "Generación automática de resultados, figuras "
            "y documentos técnicos."
        ),
        status="● Planeado",
        status_class="status-planned",
        page="pages/reportes.py",
    )

render_control_center()
render_activity_panel()

# ============================================================
# PROYECTO DOCTORAL
# ============================================================

st.html(
    '<div class="dcag-section-title">Proyecto doctoral</div>'
)

st.info(PROJECT_DESCRIPTION)

st.caption(
    f"{INSTITUTION} · {PROGRAM} · "
    f"Doctorado-DCAG {VERSION}"
)