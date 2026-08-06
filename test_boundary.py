from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

from scripts.boundary import (
    cargar_limite_territorial,
    plot_geotiff_recortado,
    recortar_geotiff_con_limite,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_RESULTADOS = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_mapas"
)

GEOTIFF_ORIGINAL = (
    CARPETA_RESULTADOS
    / "excedencia_interpolada.tif"
)

GEOTIFF_RECORTADO = (
    CARPETA_RESULTADOS
    / "excedencia_interpolada_recortada.tif"
)

PNG_RECORTADO = (
    CARPETA_RESULTADOS
    / "mapa_interpolado_recortado.png"
)

# CAMBIA ESTA RUTA por la ubicación real de tu límite estatal.
ARCHIVO_LIMITE = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\colima_state_boundary.kml"
)


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA DE RECORTE TERRITORIAL")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. CARGAR LÍMITE
    # --------------------------------------------------------

    limite, metadata_limite = (
        cargar_limite_territorial(
            path_boundary=ARCHIVO_LIMITE,
        )
    )

    print()
    print("LÍMITE TERRITORIAL")
    print("-" * 72)

    for clave, valor in metadata_limite.items():
        print(f"{clave}: {valor}")

    # --------------------------------------------------------
    # 2. RECORTAR GEOTIFF
    # --------------------------------------------------------

    resultado = recortar_geotiff_con_limite(
        input_tif=GEOTIFF_ORIGINAL,
        output_tif=GEOTIFF_RECORTADO,
        gdf_limite=limite,
        crop=True,
        all_touched=False,
        overwrite=True,
    )

    print()
    print("GEOTIFF RECORTADO")
    print("-" * 72)

    for clave in [
        "input_tif",
        "output_tif",
        "crs",
        "nodata",
        "bounds_originales",
        "bounds_recortados",
        "dimensiones_originales",
        "dimensiones_recortadas",
    ]:
        print(
            f"{clave}: {resultado[clave]}"
        )

    # --------------------------------------------------------
    # 3. GENERAR PNG
    # --------------------------------------------------------

    fig, _, metadata_mapa = (
        plot_geotiff_recortado(
            path_tif=GEOTIFF_RECORTADO,
            gdf_limite=limite,
            title=(
                "Probabilidad empírica de excedencia "
                "recortada al límite territorial"
            ),
            colorbar_label=(
                "Probabilidad de excedencia (%)"
            ),
            cmap="YlOrRd",
            convertir_a_porcentaje=True,
            mostrar_limite=True,
            figsize=(11, 9),
        )
    )

    fig.savefig(
        PNG_RECORTADO,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print("PNG generado:")
    print(PNG_RECORTADO)

    print()
    print("METADATOS DEL MAPA")
    print("-" * 72)

    for clave, valor in metadata_mapa.items():
        print(f"{clave}: {valor}")

    # --------------------------------------------------------
    # 4. VALIDAR RESULTADO
    # --------------------------------------------------------

    with rasterio.open(
        GEOTIFF_RECORTADO
    ) as src:

        print()
        print("VALIDACIÓN DEL GEOTIFF RECORTADO")
        print("-" * 72)

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Dimensiones: {src.width} × {src.height}"
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

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)

    print()
    print("Archivos originales conservados:")

    print(
        GEOTIFF_ORIGINAL
    )

    print()
    print("Archivos nuevos:")

    print(
        GEOTIFF_RECORTADO
    )

    print(
        PNG_RECORTADO
    )


if __name__ == "__main__":
    main()