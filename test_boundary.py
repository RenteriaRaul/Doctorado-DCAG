from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

from scripts.boundary import (
    plot_geotiff_recortado,
    recortar_geotiff_con_limite,
)
from scripts.gis_manager import GISManager


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MARCO = (
    r"C:\Users\rente\Desktop\Marco_geoestadistico_2025"
)

ESTADO = "Colima"
MUNICIPIO = "Manzanillo"

CARPETA_RESULTADOS = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_mapas"
)

GEOTIFF_ORIGINAL = (
    CARPETA_RESULTADOS
    / "excedencia_interpolada.tif"
)

GEOTIFF_ESTADO = (
    CARPETA_RESULTADOS
    / "excedencia_colima_recortada.tif"
)

PNG_ESTADO = (
    CARPETA_RESULTADOS
    / "mapa_excedencia_colima.png"
)

GEOTIFF_MUNICIPIO = (
    CARPETA_RESULTADOS
    / "excedencia_manzanillo_recortada.tif"
)

PNG_MUNICIPIO = (
    CARPETA_RESULTADOS
    / "mapa_excedencia_manzanillo.png"
)


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def validar_geotiff(path_tif, etiqueta):

    with rasterio.open(
        path_tif
    ) as src:

        print()
        print(f"VALIDACIÓN — {etiqueta}")
        print("-" * 72)

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Dimensiones: "
            f"{src.width} × {src.height}"
        )

        print(
            f"Bounds: {src.bounds}"
        )

        print(
            f"Nodata: {src.nodata}"
        )

        print(
            "Orientación norte-arriba: "
            f"{src.transform.e < 0}"
        )

        if src.transform.e >= 0:
            raise ValueError(
                f"El GeoTIFF de {etiqueta} "
                "no tiene orientación north-up."
            )


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA BOUNDARY + GIS MANAGER")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. VALIDAR RASTER ORIGINAL
    # --------------------------------------------------------

    if not GEOTIFF_ORIGINAL.exists():

        raise FileNotFoundError(
            "No se encontró el GeoTIFF original:\n"
            f"{GEOTIFF_ORIGINAL}\n\n"
            "Ejecute primero test_mapas.py."
        )

    print()
    print("GeoTIFF original:")
    print(
        GEOTIFF_ORIGINAL
    )

    # --------------------------------------------------------
    # 2. INICIAR GIS MANAGER
    # --------------------------------------------------------

    gis = GISManager(
        root=RUTA_MARCO
    )

    print()
    print(
        "GIS Manager iniciado correctamente."
    )

    # ========================================================
    # CASO 1 — ESTADO DE COLIMA
    # ========================================================

    print()
    print("=" * 72)
    print("CASO 1 — RECORTE ESTATAL")
    print("=" * 72)

    limite_estado = gis.obtener_limite_estado(
        estado=ESTADO,
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print(
        f"Estado: {ESTADO}"
    )

    print(
        f"Bounds: "
        f"{limite_estado.total_bounds}"
    )

    resultado_estado = (
        recortar_geotiff_con_limite(
            input_tif=GEOTIFF_ORIGINAL,
            output_tif=GEOTIFF_ESTADO,
            gdf_limite=limite_estado,
            crop=True,
            all_touched=False,
            overwrite=True,
        )
    )

    print()
    print("GeoTIFF estatal generado:")
    print(
        resultado_estado[
            "output_tif"
        ]
    )

    fig_estado, _, metadata_estado = (
        plot_geotiff_recortado(
            path_tif=GEOTIFF_ESTADO,
            gdf_limite=limite_estado,
            title=(
                "Probabilidad empírica de excedencia "
                f"— {ESTADO}"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=True,
            sigma_suavizado=1.2,
            interpolation_display="bilinear",
            figsize=(11, 9),
        )
    )

    fig_estado.savefig(
        PNG_ESTADO,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig_estado
    )

    print()
    print("PNG estatal generado:")
    print(
        PNG_ESTADO
    )

    validar_geotiff(
        GEOTIFF_ESTADO,
        "Estado de Colima",
    )

    # ========================================================
    # CASO 2 — MUNICIPIO DE MANZANILLO
    # ========================================================

    print()
    print("=" * 72)
    print("CASO 2 — RECORTE MUNICIPAL")
    print("=" * 72)

    limite_municipio = (
        gis.obtener_municipio(
            estado=ESTADO,
            municipio=MUNICIPIO,
            crs_destino="EPSG:4326",
            territorio="principal",
        )
    )

    print()
    print(
        f"Municipio: {MUNICIPIO}"
    )

    print(
        f"Bounds: "
        f"{limite_municipio.total_bounds}"
    )

    resultado_municipio = (
        recortar_geotiff_con_limite(
            input_tif=GEOTIFF_ORIGINAL,
            output_tif=GEOTIFF_MUNICIPIO,
            gdf_limite=limite_municipio,
            crop=True,
            all_touched=False,
            overwrite=True,
        )
    )

    print()
    print("GeoTIFF municipal generado:")
    print(
        resultado_municipio[
            "output_tif"
        ]
    )

    fig_municipio, _, metadata_municipio = (
        plot_geotiff_recortado(
            path_tif=GEOTIFF_MUNICIPIO,
            gdf_limite=limite_municipio,
            title=(
                "Probabilidad empírica de excedencia "
                f"— {MUNICIPIO}, {ESTADO}"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            suavizar_visual=True,
            sigma_suavizado=1.2,
            interpolation_display="bilinear",
            figsize=(11, 9),
        )
    )

    fig_municipio.savefig(
        PNG_MUNICIPIO,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig_municipio
    )

    print()
    print("PNG municipal generado:")
    print(
        PNG_MUNICIPIO
    )

    validar_geotiff(
        GEOTIFF_MUNICIPIO,
        "Municipio de Manzanillo",
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 72)
    print("RESUMEN DE RESULTADOS")
    print("=" * 72)

    print()
    print("Archivo original conservado:")
    print(
        GEOTIFF_ORIGINAL
    )

    print()
    print("Resultados estatales:")
    print(
        GEOTIFF_ESTADO
    )
    print(
        PNG_ESTADO
    )

    print()
    print("Resultados municipales:")
    print(
        GEOTIFF_MUNICIPIO
    )
    print(
        PNG_MUNICIPIO
    )

    print()
    print("Metadatos estatales:")
    print(
        metadata_estado
    )

    print()
    print("Metadatos municipales:")
    print(
        metadata_municipio
    )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()