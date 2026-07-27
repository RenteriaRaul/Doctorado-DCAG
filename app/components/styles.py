from pathlib import Path

import streamlit as st


def load_css():
    """
    Carga los estilos CSS generales de la plataforma Doctorado-DCAG.
    """

    css_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "styles.css"
    )

    if not css_path.exists():
        st.warning(
            f"No se encontró el archivo de estilos: {css_path}"
        )
        return

    try:
        css_content = css_path.read_text(
            encoding="utf-8"
        )

        st.html(
            f"<style>{css_content}</style>"
        )

    except Exception as error:
        st.error(
            f"No fue posible cargar los estilos CSS: {error}"
        )