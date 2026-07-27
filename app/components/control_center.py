import streamlit as st

from components.home_map import render_home_map


def render_control_center():
    """Renderiza el centro de control científico del Home."""

    st.html(
        '<div class="dcag-section-title">Centro de control científico</div>'
    )

    col_map, col_summary = st.columns(
        [2, 1],
        gap="large",
    )

    # ========================================================
    # MAPA
    # ========================================================

    with col_map:
        with st.container(border=True):
            render_home_map()

    # ========================================================
    # RESUMEN
    # ========================================================

    with col_summary:

        st.html(
            """
            <div class="dcag-control-panel">

                <div class="dcag-panel-title">
                    Resumen del sistema
                </div>

                <div class="dcag-summary-item">
                    <span>🌧️ Estaciones</span>
                    <strong>En preparación</strong>
                </div>

                <div class="dcag-summary-item">
                    <span>📈 Modelo estadístico</span>
                    <strong>GEV</strong>
                </div>

                <div class="dcag-summary-item">
                    <span>🛰️ Fuentes climáticas</span>
                    <strong>CONAGUA · ERA5 · SSP</strong>
                </div>

                <div class="dcag-summary-item">
                    <span>🗺️ Cobertura</span>
                    <strong>Colima</strong>
                </div>

                <div class="dcag-summary-item">
                    <span>🧠 IA local</span>
                    <strong class="status-pending">
                        Pendiente
                    </strong>
                </div>

                <div class="dcag-summary-item">
                    <span>⚙️ Plataforma</span>
                    <strong class="status-online">
                        Operativa
                    </strong>
                </div>

            </div>
            """
        )