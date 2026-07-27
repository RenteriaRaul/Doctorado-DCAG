import streamlit as st


def render_status_panel():
    """Panel visual del estado general del sistema."""

    st.html(
        """
        <div class="dcag-status-grid">

            <div class="dcag-status-box">
                <div class="dcag-status-label">Estado del sistema</div>
                <div class="dcag-status-value status-online">
                    ● Operativo
                </div>
            </div>

            <div class="dcag-status-box">
                <div class="dcag-status-label">Datos CONAGUA</div>
                <div class="dcag-status-value">
                    Disponibles
                </div>
            </div>

            <div class="dcag-status-box">
                <div class="dcag-status-label">Escenarios SSP</div>
                <div class="dcag-status-value">
                    En integración
                </div>
            </div>

            <div class="dcag-status-box">
                <div class="dcag-status-label">IA local</div>
                <div class="dcag-status-value status-pending">
                    No conectada
                </div>
            </div>

        </div>
        """
    )