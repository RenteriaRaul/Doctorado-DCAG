import streamlit as st


def render_activity_panel():
    """
    Renderiza el panel de actividad y desarrollo
    de los módulos científicos.
    """

    st.html(
        '<div class="dcag-section-title">Estado de los módulos</div>'
    )

    col_activity, col_development = st.columns(
        [1.4, 1],
        gap="large"
    )

    # ========================================================
    # ACTIVIDAD / MÓDULOS
    # ========================================================

    with col_activity:

        st.html(
            """
            <div class="dcag-activity-panel">

                <div class="dcag-panel-title">
                    Actividad de la plataforma
                </div>

                <div class="dcag-activity-item">

                    <div class="dcag-activity-icon">
                        🌧️
                    </div>

                    <div class="dcag-activity-content">
                        <strong>Datos CONAGUA</strong>
                        <span>
                            Series históricas de precipitación disponibles
                            para su integración.
                        </span>
                    </div>

                    <div class="dcag-mini-status status-online">
                        Disponible
                    </div>

                </div>


                <div class="dcag-activity-item">

                    <div class="dcag-activity-icon">
                        📈
                    </div>

                    <div class="dcag-activity-content">
                        <strong>Análisis GEV</strong>
                        <span>
                            Periodos de retorno y modelado de
                            precipitaciones extremas.
                        </span>
                    </div>

                    <div class="dcag-mini-status status-online">
                        Disponible
                    </div>

                </div>


                <div class="dcag-activity-item">

                    <div class="dcag-activity-icon">
                        🌊
                    </div>

                    <div class="dcag-activity-content">
                        <strong>Excedencias</strong>
                        <span>
                            Análisis de probabilidad para umbrales
                            de precipitación extrema.
                        </span>
                    </div>

                    <div class="dcag-mini-status status-online">
                        Disponible
                    </div>

                </div>


                <div class="dcag-activity-item">

                    <div class="dcag-activity-icon">
                        🛰️
                    </div>

                    <div class="dcag-activity-content">
                        <strong>Escenarios SSP</strong>
                        <span>
                            Incorporación de proyecciones climáticas
                            futuras mediante Sustax.
                        </span>
                    </div>

                    <div class="dcag-mini-status status-development">
                        En integración
                    </div>

                </div>

            </div>
            """
        )

    # ========================================================
    # SIGUIENTES ETAPAS
    # ========================================================

    with col_development:

        st.html(
            """
            <div class="dcag-development-panel">

                <div class="dcag-panel-title">
                    Próximas etapas
                </div>

                <div class="dcag-step">

                    <div class="dcag-step-number">
                        01
                    </div>

                    <div>
                        <strong>Tendencias climáticas</strong>
                        <p>
                            Integración del análisis histórico y
                            escenarios futuros.
                        </p>
                    </div>

                </div>


                <div class="dcag-step">

                    <div class="dcag-step-number">
                        02
                    </div>

                    <div>
                        <strong>Mapas de riesgo</strong>
                        <p>
                            Integración espacial de resultados
                            hidrometeorológicos.
                        </p>
                    </div>

                </div>


                <div class="dcag-step">

                    <div class="dcag-step-number">
                        03
                    </div>

                    <div>
                        <strong>Inteligencia Artificial</strong>
                        <p>
                            Modelos predictivos e interpretación
                            automatizada.
                        </p>
                    </div>

                </div>


                <div class="dcag-step">

                    <div class="dcag-step-number">
                        04
                    </div>

                    <div>
                        <strong>Reportes científicos</strong>
                        <p>
                            Exportación automatizada de resultados,
                            tablas y figuras.
                        </p>
                    </div>

                </div>

            </div>
            """
        )