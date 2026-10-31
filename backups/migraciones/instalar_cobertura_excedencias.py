
from pathlib import Path
import shutil
import ast

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

RAIZ = Path(r"G:\My Drive\Doctorado-DCAG")

ARCHIVO = RAIZ / "app" / "pages" / "excedencias.py"

RESPALDO = RAIZ / "app" / "pages" / "excedencias_respaldo.py"

MARCADOR_INICIO = (
    "# INICIO_DIAGNOSTICO_COBERTURA_CONAGUA"
)

MARCADOR_FINAL = (
    "# FIN_DIAGNOSTICO_COBERTURA_CONAGUA"
)

# ============================================================
# BLOQUE QUE SE INCORPORARÁ A STREAMLIT
# ============================================================

BLOQUE = '''
# INICIO_DIAGNOSTICO_COBERTURA_CONAGUA

# ============================================================
# 3A. PERIODO DE REFERENCIA Y CALIDAD TEMPORAL
# ============================================================

st.markdown("### Periodo de análisis y calidad de datos")

st.caption(
    "Evalúe la disponibilidad de datos diarios de CONAGUA "
    "antes de realizar la interpolación espacial."
)

modo_periodo = st.selectbox(
    "Periodo de referencia",
    [
        "2000–2024",
        "1991–2020",
        "Personalizado",
    ],
    key="exceedance_preview_period",
)

if modo_periodo == "2000–2024":
    anio_inicio = 2000
    anio_fin = 2024

elif modo_periodo == "1991–2020":
    anio_inicio = 1991
    anio_fin = 2020

else:
    columna_inicio, columna_fin = st.columns(2)

    with columna_inicio:
        anio_inicio = st.number_input(
            "Año inicial",
            min_value=1850,
            max_value=2100,
            value=2000,
            step=1,
            key="exceedance_preview_start",
        )

    with columna_fin:
        anio_fin = st.number_input(
            "Año final",
            min_value=1850,
            max_value=2100,
            value=2024,
            step=1,
            key="exceedance_preview_end",
        )

criterio_cobertura = st.selectbox(
    "Cobertura mínima",
    options=[90, 80],
    format_func=lambda valor: f"{valor} %",
    key="exceedance_preview_coverage",
)

exigir_cobertura_anual = st.checkbox(
    "Exigir cobertura mínima en cada año",
    value=False,
    key="exceedance_preview_annual",
)

if int(anio_inicio) > int(anio_fin):

    st.error(
        "El año inicial no puede superar al año final."
    )

elif carpeta_valida:

    try:

        fecha_inicio = pd.Timestamp(
            year=int(anio_inicio),
            month=1,
            day=1,
        )

        fecha_fin = pd.Timestamp(
            year=int(anio_fin),
            month=12,
            day=31,
        )

        dias_esperados = (
            fecha_fin - fecha_inicio
        ).days + 1

        registros_cobertura = []

        for estacion in estaciones_lectura:

            metadata = estacion.get("metadata", {})

            datos = estacion.get("data")

            if not isinstance(datos, pd.DataFrame):
                continue

            if not {"date", "pp"}.issubset(datos.columns):
                continue

            serie = datos[["date", "pp"]].copy()

            serie["date"] = (
                pd.to_datetime(
                    serie["date"],
                    errors="coerce",
                )
                .dt.normalize()
            )

            serie["pp"] = pd.to_numeric(
                serie["pp"],
                errors="coerce",
            )

            serie = serie.dropna(
                subset=["date", "pp"]
            )

            serie = serie.loc[
                serie["pp"] >= 0
            ]

            # Consolidar observaciones repetidas por día
            serie = (
                serie.groupby(
                    "date",
                    as_index=False,
                )["pp"]
                .max()
            )

            serie = serie.loc[
                serie["date"].between(
                    fecha_inicio,
                    fecha_fin,
                )
            ]

            dias_validos = len(serie)

            porcentaje_cobertura = (
                100 * dias_validos / dias_esperados
            )

            cumple_global = (
                porcentaje_cobertura >= criterio_cobertura
            )

            anios_incompletos = 0

            for anio in range(
                int(anio_inicio),
                int(anio_fin) + 1,
            ):

                inicio_anual = pd.Timestamp(
                    year=anio,
                    month=1,
                    day=1,
                )

                fin_anual = pd.Timestamp(
                    year=anio,
                    month=12,
                    day=31,
                )

                esperados_anuales = (
                    fin_anual - inicio_anual
                ).days + 1

                validos_anuales = int(
                    serie["date"]
                    .between(
                        inicio_anual,
                        fin_anual,
                    )
                    .sum()
                )

                cobertura_anual = (
                    100
                    * validos_anuales
                    / esperados_anuales
                )

                if cobertura_anual < criterio_cobertura:
                    anios_incompletos += 1

            elegible = (
                cumple_global
                and (
                    not exigir_cobertura_anual
                    or anios_incompletos == 0
                )
            )

            registros_cobertura.append(
                {
                    "Estación": metadata.get("station"),
                    "Nombre": metadata.get("nombre"),
                    "Municipio": metadata.get("municipio"),
                    "Días válidos": dias_validos,
                    "Días esperados": dias_esperados,
                    "Cobertura (%)": round(
                        porcentaje_cobertura,
                        2,
                    ),
                    "Años bajo criterio": anios_incompletos,
                    "Elegible": elegible,
                    "Latitud": metadata.get("latitud"),
                    "Longitud": metadata.get("longitud"),
                }
            )

        tabla_cobertura = pd.DataFrame(
            registros_cobertura
        )

        if not tabla_cobertura.empty:

            estaciones_elegibles = int(
                tabla_cobertura["Elegible"].sum()
            )

            estaciones_excluidas = (
                len(tabla_cobertura)
                - estaciones_elegibles
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Estaciones evaluadas",
                len(tabla_cobertura),
            )

            col2.metric(
                "Estaciones elegibles",
                estaciones_elegibles,
            )

            col3.metric(
                "Estaciones excluidas",
                estaciones_excluidas,
            )

            st.dataframe(
                tabla_cobertura.sort_values(
                    ["Elegible", "Cobertura (%)"],
                    ascending=[False, False],
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Descargar diagnóstico de calidad",
                data=(
                    tabla_cobertura
                    .to_csv(index=False)
                    .encode("utf-8-sig")
                ),
                file_name=(
                    f"cobertura_{anio_inicio}_"
                    f"{anio_fin}_"
                    f"{criterio_cobertura}pct.csv"
                ),
                mime="text/csv",
                key="exceedance_preview_download",
            )

            st.warning(
                "Etapa de diagnóstico: el periodo y la "
                "cobertura seleccionados todavía no se "
                "aplican al cálculo de excedencias. "
                "Los mapas continúan utilizando el "
                "procesamiento histórico existente."
            )

        else:

            st.warning(
                "No se encontraron estaciones compatibles "
                "para evaluar la cobertura."
            )

    except Exception as error:

        st.error(
            "No fue posible evaluar la cobertura temporal."
        )

        st.exception(error)

else:

    st.info(
        "Seleccione primero una carpeta válida "
        "de estaciones CONAGUA."
    )

# FIN_DIAGNOSTICO_COBERTURA_CONAGUA
'''

# ============================================================
# INSTALACIÓN SEGURA
# ============================================================

def main():

    if not ARCHIVO.is_file():
        raise FileNotFoundError(
            f"No se encontró el módulo: {ARCHIVO}"
        )

    contenido = ARCHIVO.read_text(
        encoding="utf-8-sig"
    )

    if MARCADOR_INICIO in contenido:
        print(
            "El diagnóstico ya está instalado. "
            "No se realizaron modificaciones."
        )
        return

    # Evitar instalarlo dos veces si ya existe
    # una versión previa del mismo diagnóstico.
    if "exceedance_preview_coverage" in contenido:
        print(
            "Se detectó una versión previa del "
            "diagnóstico de cobertura. "
            "No se realizaron modificaciones."
        )
        return

    # Buscar el encabezado original de la
    # sección 4 para insertar antes el diagnóstico.
    marcador_seccion = (
        "# ============================================================\n"
        "# 4. EJECUCIÓN\n"
        "# ============================================================"
    )

    contenido_normalizado = contenido.replace(
        "\r\n",
        "\n",
    )

    if marcador_seccion not in contenido_normalizado:
        raise RuntimeError(
            "No se encontró la sección 4. "
            "El archivo original no fue modificado."
        )

    nuevo_contenido = contenido_normalizado.replace(
        marcador_seccion,
        BLOQUE.strip() + "\n\n\n"
        + marcador_seccion,
        1,
    )

    # Verificar sintaxis antes de guardar.
    ast.parse(nuevo_contenido)

    # Crear respaldo solamente si todavía no existe.
    if not RESPALDO.exists():
        shutil.copy2(
            ARCHIVO,
            RESPALDO,
        )

    ARCHIVO.write_text(
        nuevo_contenido,
        encoding="utf-8",
    )

    print("=" * 60)
    print("INSTALACIÓN COMPLETADA")
    print("=" * 60)
    print(f"Módulo actualizado: {ARCHIVO}")
    print(f"Respaldo disponible: {RESPALDO}")
    print("Sintaxis Python validada correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
