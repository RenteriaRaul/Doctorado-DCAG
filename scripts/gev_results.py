import pandas as pd


# ============================================================
# COLUMNAS HISTÓRICAS QUE YA NO SE UTILIZAN EN LA INTERFAZ
# ============================================================

COLUMNAS_BOOTSTRAP_B = [
    "CI_low95_bootB",
    "CI_high95_bootB",
    "bootB_naccepted",
]


# ============================================================
# PREPARAR TABLA PARA EXPORTACIÓN
# ============================================================

def preparar_maestro_exportacion(
    maestro,
):
    """
    Genera la versión oficial descargable del resultado GEV.

    Conserva los nombres técnicos originales para facilitar
    reproducibilidad y análisis posterior, pero elimina las
    columnas históricas de Bootstrap B, ya que el proyecto
    utiliza Bootstrap robusto como método oficial de
    incertidumbre.

    Parámetros
    ----------
    maestro : pd.DataFrame
        Tabla maestra producida por el motor GEV.

    Retorna
    -------
    pd.DataFrame
        Copia de la tabla sin columnas Bootstrap B.
    """
    if not isinstance(
        maestro,
        pd.DataFrame,
    ):
        raise TypeError(
            "maestro debe ser un DataFrame."
        )

    return maestro.drop(
        columns=[
            columna
            for columna in COLUMNAS_BOOTSTRAP_B
            if columna in maestro.columns
        ],
        errors="ignore",
    ).copy()


# ============================================================
# PREPARAR TABLA PARA VISUALIZACIÓN EN STREAMLIT
# ============================================================

def preparar_maestro_visual(
    maestro,
    n_boot=None,
):
    """
    Genera una tabla amigable para visualización en Streamlit.

    La tabla interna conserva sus nombres técnicos. Esta función
    únicamente prepara una copia con encabezados comprensibles
    para el investigador.

    Si se proporciona n_boot, agrega el porcentaje de réplicas
    Bootstrap robusto aceptadas.
    """
    if not isinstance(
        maestro,
        pd.DataFrame,
    ):
        raise TypeError(
            "maestro debe ser un DataFrame."
        )

    tabla = preparar_maestro_exportacion(
        maestro
    )

    if (
        n_boot is not None
        and "bootA_naccepted" in tabla.columns
    ):
        n_boot = int(
            n_boot
        )

        if n_boot <= 0:
            raise ValueError(
                "n_boot debe ser mayor que 0."
            )

        tabla[
            "bootstrap_aceptado_pct"
        ] = (
            pd.to_numeric(
                tabla[
                    "bootA_naccepted"
                ],
                errors="coerce",
            )
            / float(
                n_boot
            )
            * 100
        ).round(
            1
        )

    columnas_preferidas = [
        "station",
        "nombre",
        "estado",
        "municipio",
        "T_years",
        "level_mm",
        "CI_low95_bootA",
        "CI_high95_bootA",
        "bootA_naccepted",
        "bootstrap_aceptado_pct",
        "gev_shape",
        "gev_loc",
        "gev_scale",
        "n_years",
        "trend_slope_mm_per_year",
        "situacion",
        "latitud",
        "longitud",
        "altitud_msnm",
        "note",
        "archivo_fuente",
    ]

    columnas_existentes = [
        columna
        for columna in columnas_preferidas
        if columna in tabla.columns
    ]

    otras_columnas = [
        columna
        for columna in tabla.columns
        if columna not in columnas_existentes
    ]

    tabla = tabla[
        columnas_existentes
        + otras_columnas
    ].copy()

    nombres_visuales = {
        "station": "Estación",
        "nombre": "Nombre",
        "estado": "Estado",
        "municipio": "Municipio",
        "T_years": "Periodo de retorno (años)",
        "level_mm": "Nivel estimado (mm)",
        "CI_low95_bootA": "IC 95% inferior (mm)",
        "CI_high95_bootA": "IC 95% superior (mm)",
        "bootA_naccepted": "Réplicas válidas",
        "bootstrap_aceptado_pct": "Bootstrap aceptado (%)",
        "gev_shape": "Parámetro de forma",
        "gev_loc": "Localización (mm)",
        "gev_scale": "Escala (mm)",
        "n_years": "Máximos anuales",
        "trend_slope_mm_per_year": (
            "Pendiente máximos anuales (mm/año)"
        ),
        "situacion": "Situación",
        "latitud": "Latitud",
        "longitud": "Longitud",
        "altitud_msnm": "Altitud (msnm)",
        "note": "Observación",
        "archivo_fuente": "Archivo fuente",
    }

    return tabla.rename(
        columns=nombres_visuales
    )