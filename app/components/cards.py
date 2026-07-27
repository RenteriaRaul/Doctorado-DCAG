import streamlit as st


def module_card(
    icon: str,
    title: str,
    description: str,
    status: str,
    status_class: str,
    page: str | None = None,
    button_text: str = "Explorar →",
):
    """
    Renderiza una tarjeta reutilizable para los módulos científicos.

    Parameters
    ----------
    icon : str
        Icono o emoji del módulo.

    title : str
        Nombre del módulo.

    description : str
        Descripción breve.

    status : str
        Estado actual del módulo.

    status_class : str
        Clase CSS asociada al estado.

    page : str | None
        Ruta interna de la página de Streamlit.

    button_text : str
        Texto mostrado en el acceso al módulo.
    """

    card_html = f"""
    <div class="dcag-card">

        <div class="dcag-card-icon">
            {icon}
        </div>

        <div class="dcag-card-title">
            {title}
        </div>

        <div class="dcag-card-description">
            {description}
        </div>

        <div class="dcag-card-status {status_class}">
            {status}
        </div>

    </div>
    """

    st.html(card_html)

    # Enlace interno de Streamlit
    if page is not None:
        st.page_link(
            page,
            label=button_text,
            icon=":material/arrow_forward:",
            use_container_width=True,
        )