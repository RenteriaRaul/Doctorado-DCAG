from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.gis_manager import GISManager
from scripts.interpolation import (
    interpolar_superficie_observacional,
    plot_superficie_observacional,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MARCO = (
    r"C:\Users\rente\Desktop\Marco_geoestadistico_2025"
)

ESTADO = "Colima"

CARPETA_RESULTADOS = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_mapas"
)

CSV_UNIDO = (
    CARPETA_RESULTADOS
    / "excedencias_con_coordenadas.csv"
)

PNG_CONTROL = (
    CARPETA_RESULTADOS
    / "mapa_interpolacion_observacional.png"
)


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA MOTOR INTERPOLACIÓN OBSERVACIONAL")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. CARGAR DATOS
    # --------------------------------------------------------

    if not CSV_UNIDO.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{CSV_UNIDO}\n\n"
            "Ejecute primero test_mapas.py."
        )

    df = pd.read_csv(
        CSV_UNIDO
    )

    # --------------------------------------------------------
    # 2. INTERPOLACIÓN OBSERVACIONAL
    # --------------------------------------------------------

    resultado = interpolar_superficie_observacional(
        df=df,
        col_lon="LONGITUD",
        col_lat="LATITUD",
        col_val="EXCEDENCIA_50MM",
        resolucion=300,
        eliminar_duplicados=True,
    )

    print()
    print("CONTROL DE CALIDAD")
    print("-" * 72)

    for clave, valor in resultado["calidad"].items():
        print(
            f"{clave}: {valor}"
        )

    # --------------------------------------------------------
    # 3. GIS MANAGER
    # --------------------------------------------------------

    gis = GISManager(
        root=RUTA_MARCO
    )

    limite = gis.obtener_limite_estado(
        estado=ESTADO,
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    # --------------------------------------------------------
    # 4. GENERAR MAPA
    # --------------------------------------------------------

    fig, _ = plot_superficie_observacional(
        resultado=resultado,
        col_lon="LONGITUD",
        col_lat="LATITUD",
        col_label="NOMBRE",
        title=(
            "Interpolated Exceedance Map — "
            "Método observacional"
        ),
        colorbar_label=(
            "Probability of Exceedance ≥ 50 mm"
        ),
        cmap="YlOrRd",
        levels=15,
        show_stations=True,
        show_labels=True,
        gdf_limite=limite,
        figsize=(12, 9),
    )

    # --------------------------------------------------------
    # 5. GUARDAR PNG
    # --------------------------------------------------------

    fig.savefig(
        PNG_CONTROL,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Mapa generado:\n{PNG_CONTROL}"
    )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()