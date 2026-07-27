import pandas as pd
import streamlit as st


def render_home_map():
    """
    Muestra un mapa ligero del estado de Colima
    usando coordenadas de referencia.

    Más adelante este componente se conectará
    con estaciones CONAGUA, excedencias y riesgo.
    """

    # Puntos de referencia para visualizar el estado
    puntos = pd.DataFrame(
        {
            "nombre": [
                "Manzanillo",
                "Tecomán",
                "Colima",
                "Villa de Álvarez",
            ],
            "lat": [
                19.0522,
                18.9139,
                19.2433,
                19.2670,
            ],
            "lon": [
                -104.3158,
                -103.8742,
                -103.7241,
                -103.7377,
            ],
        }
    )

    st.html(
        """
        <div class="dcag-panel-title">
            🗺️ Cobertura de análisis — Estado de Colima
        </div>
        """
    )

    st.map(
        puntos,
        latitude="lat",
        longitude="lon",
        height=290,
        zoom=8,
    )

    st.caption(
        "Vista preliminar de cobertura territorial. "
        "Posteriormente se integrarán las estaciones CONAGUA, "
        "excedencias, escenarios SSP y zonas de riesgo."
    )