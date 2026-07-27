import streamlit as st

from config import (
    APP_NAME,
    INSTITUTION,
    PROGRAM,
    VERSION,
)


def render_sidebar_brand():
    """
    Renderiza la identidad visual principal del sidebar.
    """

    st.sidebar.html(
        f"""
        <div class="dcag-sidebar-brand">

            <div class="dcag-sidebar-logo">
                🌧️
            </div>

            <div class="dcag-sidebar-title">
                {APP_NAME}
            </div>

            <div class="dcag-sidebar-subtitle">
                Plataforma científica
            </div>

            <div class="dcag-sidebar-divider"></div>

            <div class="dcag-sidebar-project">
                Predicción de inundaciones
                <br>
                Estado de Colima
            </div>

            <div class="dcag-sidebar-institution">
                {INSTITUTION}
                <br>
                {PROGRAM}
            </div>

            <div class="dcag-sidebar-version">
                v{VERSION}
            </div>

        </div>
        """
    )