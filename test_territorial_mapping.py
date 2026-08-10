from pathlib import Path

from scripts.gis_manager import GISManager
from scripts.territorial_mapping import (
    generar_mapa_raster_territorial,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MARCO = (
    r"C:\Users\rente\Desktop\Marco_geoestadistico_2025"
)

CARPETA_RESULTADOS = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_territorial_mapping"
)

RASTER_ORIGINAL = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_mapas"
    r"\excedencia_interpolada.tif"
)

ESTADO = "Colima"
MUNICIPIO = "Manzanillo"
LOCALIDAD = "Manzanillo"

AGEB_URBANA = "0564"
AGEB_RURAL = "0032"


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def imprimir_resultado(
    etiqueta,
    resultado,
):
    """
    Imprime un resumen del producto territorial generado.
    """

    print()
    print("=" * 72)
    print(etiqueta)
    print("=" * 72)

    print()
    print(
        f"Nivel: {resultado['nivel']}"
    )

    print(
        f"Territorio: "
        f"{resultado['nombre_territorio']}"
    )

    print(
        f"Slug: "
        f"{resultado['slug_territorio']}"
    )

    print()
    print("GeoTIFF:")
    print(
        resultado["output_tif"]
    )

    print()
    print("PNG:")
    print(
        resultado["output_png"]
    )

    print()
    print("Bounds territoriales:")
    print(
        resultado["territorio"]["bounds"]
    )

    print()
    print("Bounds raster recortado:")
    print(
        resultado["raster"][
            "bounds_recortados"
        ]
    )

    print()
    print("Dimensiones raster:")
    print(
        resultado["raster"][
            "dimensiones_recortadas"
        ]
    )

    print()
    print("Rango del mapa:")

    print(
        "  mínimo:",
        resultado["mapa"][
            "valor_minimo"
        ],
    )

    print(
        "  máximo:",
        resultado["mapa"][
            "valor_maximo"
        ],
    )

    print(
        "  promedio:",
        resultado["mapa"][
            "valor_promedio"
        ],
    )


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA TERRITORIAL MAPPING")
    print("=" * 72)

    # --------------------------------------------------------
    # VALIDAR RASTER ORIGINAL
    # --------------------------------------------------------

    if not RASTER_ORIGINAL.exists():
        raise FileNotFoundError(
            "No se encontró el raster original:\n"
            f"{RASTER_ORIGINAL}\n\n"
            "Ejecute primero test_mapas.py."
        )

    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Raster original:")
    print(
        RASTER_ORIGINAL
    )

    # --------------------------------------------------------
    # INICIAR GIS MANAGER
    # --------------------------------------------------------

    gis = GISManager(
        root=RUTA_MARCO
    )

    print()
    print(
        "GIS Manager iniciado correctamente."
    )

    # ========================================================
    # CASO 1 — ESTADO
    # ========================================================

    resultado_estado = (
        generar_mapa_raster_territorial(
            gis=gis,
            input_tif=RASTER_ORIGINAL,
            output_tif=(
                CARPETA_RESULTADOS
                / "excedencia_estado_colima.tif"
            ),
            output_png=(
                CARPETA_RESULTADOS
                / "mapa_excedencia_estado_colima.png"
            ),
            nivel="estado",
            estado=ESTADO,
            territorio="principal",
            title=(
                "Probabilidad empírica de excedencia "
                "— Estado de Colima"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=False,
            figsize=(11, 9),
        )
    )

    imprimir_resultado(
        "CASO 1 — ESTADO DE COLIMA",
        resultado_estado,
    )

    # ========================================================
    # CASO 2 — MUNICIPIO
    # ========================================================

    resultado_municipio = (
        generar_mapa_raster_territorial(
            gis=gis,
            input_tif=RASTER_ORIGINAL,
            output_tif=(
                CARPETA_RESULTADOS
                / "excedencia_municipio_manzanillo.tif"
            ),
            output_png=(
                CARPETA_RESULTADOS
                / "mapa_excedencia_municipio_manzanillo.png"
            ),
            nivel="municipio",
            estado=ESTADO,
            municipio=MUNICIPIO,
            territorio="principal",
            title=(
                "Probabilidad empírica de excedencia "
                "— Municipio de Manzanillo"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=False,
            figsize=(11, 9),
        )
    )

    imprimir_resultado(
        "CASO 2 — MUNICIPIO DE MANZANILLO",
        resultado_municipio,
    )

    # ========================================================
    # CASO 3 — LOCALIDAD
    # ========================================================

    resultado_localidad = (
        generar_mapa_raster_territorial(
            gis=gis,
            input_tif=RASTER_ORIGINAL,
            output_tif=(
                CARPETA_RESULTADOS
                / "excedencia_localidad_manzanillo.tif"
            ),
            output_png=(
                CARPETA_RESULTADOS
                / "mapa_excedencia_localidad_manzanillo.png"
            ),
            nivel="localidad",
            estado=ESTADO,
            municipio=MUNICIPIO,
            localidad=LOCALIDAD,
            territorio="principal",
            title=(
                "Probabilidad empírica de excedencia "
                "— Localidad de Manzanillo"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=False,
            figsize=(11, 9),
        )
    )

    imprimir_resultado(
        "CASO 3 — LOCALIDAD DE MANZANILLO",
        resultado_localidad,
    )

    # ========================================================
    # CASO 4 — AGEB URBANA
    # ========================================================

    resultado_ageb_urbana = (
        generar_mapa_raster_territorial(
            gis=gis,
            input_tif=RASTER_ORIGINAL,
            output_tif=(
                CARPETA_RESULTADOS
                / "excedencia_ageb_urbana_0564.tif"
            ),
            output_png=(
                CARPETA_RESULTADOS
                / "mapa_excedencia_ageb_urbana_0564.png"
            ),
            nivel="ageb_urbana",
            estado=ESTADO,
            municipio=MUNICIPIO,
            localidad=LOCALIDAD,
            ageb=AGEB_URBANA,
            territorio="principal",
            title=(
                "Probabilidad empírica de excedencia "
                "— AGEB urbana 0564"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=False,
            figsize=(11, 9),
        )
    )

    imprimir_resultado(
        "CASO 4 — AGEB URBANA 0564",
        resultado_ageb_urbana,
    )

    # ========================================================
    # CASO 5 — AGEB RURAL
    # ========================================================

    resultado_ageb_rural = (
        generar_mapa_raster_territorial(
            gis=gis,
            input_tif=RASTER_ORIGINAL,
            output_tif=(
                CARPETA_RESULTADOS
                / "excedencia_ageb_rural_0032.tif"
            ),
            output_png=(
                CARPETA_RESULTADOS
                / "mapa_excedencia_ageb_rural_0032.png"
            ),
            nivel="ageb_rural",
            estado=ESTADO,
            municipio=MUNICIPIO,
            ageb=AGEB_RURAL,
            territorio="principal",
            title=(
                "Probabilidad empírica de excedencia "
                "— AGEB rural 0032"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=False,
            figsize=(11, 9),
        )
    )

    imprimir_resultado(
        "CASO 5 — AGEB RURAL 0032",
        resultado_ageb_rural,
    )

    # ========================================================
    # VALIDACIONES AUTOMÁTICAS
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIONES AUTOMÁTICAS")
    print("=" * 72)

    resultados = [
        resultado_estado,
        resultado_municipio,
        resultado_localidad,
        resultado_ageb_urbana,
        resultado_ageb_rural,
    ]

    for resultado in resultados:

        path_tif = Path(
            resultado["output_tif"]
        )

        path_png = Path(
            resultado["output_png"]
        )

        if not path_tif.exists():
            raise RuntimeError(
                f"No se generó el GeoTIFF: {path_tif}"
            )

        if not path_png.exists():
            raise RuntimeError(
                f"No se generó el PNG: {path_png}"
            )

        dimensiones = (
            resultado["raster"][
                "dimensiones_recortadas"
            ]
        )

        if (
            dimensiones[0] <= 0
            or dimensiones[1] <= 0
        ):
            raise ValueError(
                "El raster territorial tiene "
                "dimensiones inválidas."
            )

        print(
            f"OK — {resultado['nombre_territorio']}"
        )

    # --------------------------------------------------------
    # VALIDAR CLAVES AGEB
    # --------------------------------------------------------

    if (
        resultado_ageb_urbana[
            "territorio"
        ]["ageb"]
        != AGEB_URBANA
    ):
        raise ValueError(
            "La AGEB urbana seleccionada "
            "no corresponde a la clave esperada."
        )

    if (
        resultado_ageb_rural[
            "territorio"
        ]["ageb"]
        != AGEB_RURAL
    ):
        raise ValueError(
            "La AGEB rural seleccionada "
            "no corresponde a la clave esperada."
        )

    print()
    print(
        "Claves AGEB validadas correctamente."
    )

    # --------------------------------------------------------
    # VALIDAR JERARQUÍA ESTADO → MUNICIPIO → LOCALIDAD
    # --------------------------------------------------------

    dimensiones_estado = (
        resultado_estado[
            "raster"
        ][
            "dimensiones_recortadas"
        ]
    )

    dimensiones_municipio = (
        resultado_municipio[
            "raster"
        ][
            "dimensiones_recortadas"
        ]
    )

    dimensiones_localidad = (
        resultado_localidad[
            "raster"
        ][
            "dimensiones_recortadas"
        ]
    )

    area_pixeles_estado = (
        dimensiones_estado[0]
        * dimensiones_estado[1]
    )

    area_pixeles_municipio = (
        dimensiones_municipio[0]
        * dimensiones_municipio[1]
    )

    area_pixeles_localidad = (
        dimensiones_localidad[0]
        * dimensiones_localidad[1]
    )

    print()
    print(
        "Celdas aproximadas por extensión raster:"
    )

    print(
        f"Estado: {area_pixeles_estado}"
    )

    print(
        f"Municipio: {area_pixeles_municipio}"
    )

    print(
        f"Localidad: {area_pixeles_localidad}"
    )

    if not (
        area_pixeles_estado
        >= area_pixeles_municipio
        >= area_pixeles_localidad
    ):
        raise ValueError(
            "La jerarquía espacial de los recortes "
            "no es la esperada."
        )

    print()
    print(
        "Jerarquía espacial correcta:"
    )

    print(
        "Estado >= Municipio >= Localidad"
    )

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)

    print()
    print("Archivos generados en:")
    print(
        CARPETA_RESULTADOS
    )


if __name__ == "__main__":
    main()