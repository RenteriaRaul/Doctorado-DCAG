import streamlit as st


def render_sidebar_status():
    """
    Renderiza indicadores rápidos en la parte inferior del sidebar.
    """

    st.sidebar.html(
        """
        <div class="dcag-sidebar-status">

            <div class="dcag-sidebar-status-title">
                Estado del sistema
            </div>

            <div class="dcag-sidebar-status-row">
                <span>Plataforma</span>
                <strong class="status-online">
                    ● Operativa
                </strong>
            </div>

            <div class="dcag-sidebar-status-row">
                <span>Datos</span>
                <strong>
                    Disponibles
                </strong>
            </div>

            <div class="dcag-sidebar-status-row">
                <span>IA local</span>
                <strong class="status-pending">
                    ● Pendiente
                </strong>
            </div>

        </div>
        """
    )