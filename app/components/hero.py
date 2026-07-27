import streamlit as st

from config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_DESCRIPTION,
)


def render_hero():
    """Renderiza el encabezado principal de la plataforma."""

    hero_html = f"""
<section class="dcag-hero">
    <div class="dcag-hero-badge">
        🌧️ Sistema científico de apoyo a la gestión del riesgo
    </div>

    <h1>{APP_NAME}</h1>

    <h2>{APP_SUBTITLE}</h2>

    <p>{APP_DESCRIPTION}</p>

    <div class="dcag-hero-tags">
        <span>🌧️ CONAGUA</span>
        <span>🛰️ ERA5 / SSP</span>
        <span>📊 Modelado estadístico</span>
        <span>🧠 Inteligencia Artificial</span>
        <span>🌎 Estado de Colima</span>
    </div>
</section>
"""

    st.html(hero_html)