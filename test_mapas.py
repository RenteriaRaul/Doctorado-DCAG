import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rasterio

from scripts.interpolation import (
    crear_malla_interpolacion,
    interpolar_superficie,
    plot_superficie_interpolada,
    preparar_datos_interpolacion,
    rellenar_nan_con_nearest,
)
from scripts.mapping import (
    crear_geodataframe_estaciones,
    plot_mapa_estetico_avanzado,
)
from scripts.raster_export import (
    exportar_desde_puntos_a_geotiff,
)


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

# Carpeta donde se generaron los resultados de excedencias
CARPETA_EXCEDENCIAS = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\smn_downloads\Estaciones_Colima"
    r"\_salidas_excedencias"
)

# Archivo con claves, nombres y coordenadas de las estaciones
ARCHIVO_COORDENADAS = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\estaciones_conagua_excel\Colima"
    r"\CoordenadasEstacionesColima.xlsx"
)

# Carpeta donde guardaremos esta prueba
CARPETA_SALIDA = Path(
    r"G:\My Drive\Doctorado-DCAG\results\test_mapas"
)

# Variable que se desea representar e interpolar
VARIABLE = "prob_excedencia"

# Columnas espaciales
COL_CLAVE = "CLAVE"
COL_LON = "LONGITUD"
COL_LAT = "LATITUD"
COL_NOMBRE = "NOMBRE"

# Configuración de interpolación
METODO = "linear"
RELLENAR_EXTERIOR = False

# Resolución de la malla
NX = 300
NY = 300

# Margen en grados porque trabajamos en EPSG:4326
MARGEN = 0.05

# CRS de los datos
CRS = "EPSG:4326"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encontrar_master_mas_reciente(carpeta):
    """
    Localiza el archivo MASTER de excedencias más reciente.
    """
    patron = str(
        carpeta / "MASTER_excedencia_*.csv"
    )

    archivos = glob.glob(
        patron
    )

    if not archivos:
        raise FileNotFoundError(
            "No se encontró ningún archivo MASTER de excedencias en: "
            f"{carpeta}"
        )

    archivo_reciente = max(
        archivos,
        key=os.path.getmtime,
    )

    return Path(
        archivo_reciente
    )


def normalizar_clave_estacion(serie):
    """
    Extrae la parte numérica de identificadores como:

    dat6001 -> 6001
    6001    -> 6001
    """
    return (
        serie
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype("Int64")
    )


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    print("=" * 72)
    print("PRUEBA DEL MOTOR ESPACIAL")
    print("=" * 72)

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. LOCALIZAR RESULTADOS DE EXCEDENCIAS
    # --------------------------------------------------------

    master_path = encontrar_master_mas_reciente(
        CARPETA_EXCEDENCIAS
    )

    print()
    print("Archivo MASTER localizado:")
    print(master_path)

    df_excedencia = pd.read_csv(
        master_path
    )

    print()
    print(
        f"Registros de excedencia: {len(df_excedencia)}"
    )

    print(
        "Columnas de excedencia:",
        list(df_excedencia.columns),
    )

    # --------------------------------------------------------
    # 2. LEER COORDENADAS
    # --------------------------------------------------------

    if not ARCHIVO_COORDENADAS.exists():
        raise FileNotFoundError(
            "No se encontró el archivo de coordenadas: "
            f"{ARCHIVO_COORDENADAS}"
        )

    df_coords = pd.read_excel(
        ARCHIVO_COORDENADAS
    )

    print()
    print(
        f"Registros de coordenadas: {len(df_coords)}"
    )

    print(
        "Columnas de coordenadas:",
        list(df_coords.columns),
    )

    columnas_coords_requeridas = [
        COL_CLAVE,
        COL_LON,
        COL_LAT,
    ]

    faltantes_coords = [
        columna
        for columna in columnas_coords_requeridas
        if columna not in df_coords.columns
    ]

    if faltantes_coords:
        raise ValueError(
            "El archivo de coordenadas no contiene las columnas: "
            f"{faltantes_coords}"
        )

    if "station" not in df_excedencia.columns:
        raise ValueError(
            "La tabla maestra no contiene la columna 'station'."
        )

    if VARIABLE not in df_excedencia.columns:
        raise ValueError(
            f"La tabla maestra no contiene la variable '{VARIABLE}'."
        )

    # --------------------------------------------------------
    # 3. NORMALIZAR CLAVES Y UNIR TABLAS
    # --------------------------------------------------------

    df_excedencia["CLAVE_UNION"] = (
        normalizar_clave_estacion(
            df_excedencia["station"]
        )
    )

    df_coords["CLAVE_UNION"] = (
        normalizar_clave_estacion(
            df_coords[COL_CLAVE]
        )
    )

    columnas_coordenadas = [
        "CLAVE_UNION",
        COL_CLAVE,
        COL_LON,
        COL_LAT,
    ]

    if COL_NOMBRE in df_coords.columns:
        columnas_coordenadas.append(
            COL_NOMBRE
        )

    df_unido = pd.merge(
        df_excedencia,
        df_coords[columnas_coordenadas],
        on="CLAVE_UNION",
        how="left",
    )

    print()
    print(
        f"Registros después de la unión: {len(df_unido)}"
    )

    sin_coordenadas = int(
        df_unido[
            [COL_LON, COL_LAT]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Estaciones sin coordenadas: {sin_coordenadas}"
    )

    csv_unido = (
        CARPETA_SALIDA
        / "excedencias_con_coordenadas.csv"
    )

    df_unido.to_csv(
        csv_unido,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Tabla unida guardada en:")
    print(csv_unido)

    # --------------------------------------------------------
    # 4. CREAR GEODATAFRAME Y MAPA PUNTUAL
    # --------------------------------------------------------

    gdf, calidad_mapa = (
        crear_geodataframe_estaciones(
            df=df_unido,
            col_lon=COL_LON,
            col_lat=COL_LAT,
            col_val=VARIABLE,
            crs=CRS,
            eliminar_duplicados=True,
        )
    )

    print()
    print("Control de calidad del mapa:")
    for clave, valor in calidad_mapa.items():
        print(f"  {clave}: {valor}")

    columna_etiqueta = (
        COL_NOMBRE
        if COL_NOMBRE in gdf.columns
        else "station"
    )

    fig_puntos, _ = (
        plot_mapa_estetico_avanzado(
            gdf=gdf,
            col_val=VARIABLE,
            col_label=columna_etiqueta,
            cmap="YlOrRd",
            title=(
                "Probabilidad empírica de excedencia "
                "de precipitación"
            ),
            colorbar_label=(
                "Probabilidad de excedencia"
            ),
            markersize=75,
            show_labels=True,
            max_labels=40,
        )
    )

    png_puntos = (
        CARPETA_SALIDA
        / "mapa_puntual_excedencias.png"
    )

    fig_puntos.savefig(
        png_puntos,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig_puntos
    )

    print()
    print("Mapa puntual guardado en:")
    print(png_puntos)

    # --------------------------------------------------------
    # 5. PREPARAR DATOS PARA INTERPOLACIÓN
    # --------------------------------------------------------

    (
        data_interp,
        points,
        values,
        calidad_interp,
    ) = preparar_datos_interpolacion(
        df=df_unido,
        col_lon=COL_LON,
        col_lat=COL_LAT,
        col_val=VARIABLE,
        crs=CRS,
        eliminar_duplicados=True,
    )

    print()
    print("Control de calidad de interpolación:")
    for clave, valor in calidad_interp.items():
        print(f"  {clave}: {valor}")

    # --------------------------------------------------------
    # 6. CREAR MALLA E INTERPOLAR
    # --------------------------------------------------------

    GX, GY, extent = (
        crear_malla_interpolacion(
            df=data_interp,
            col_lon=COL_LON,
            col_lat=COL_LAT,
            nx=NX,
            ny=NY,
            margin=MARGEN,
        )
    )

    Z = interpolar_superficie(
        points=points,
        values=values,
        GX=GX,
        GY=GY,
        method=METODO,
    )

    if RELLENAR_EXTERIOR:
        Z = rellenar_nan_con_nearest(
            points=points,
            values=values,
            GX=GX,
            GY=GY,
            Z=Z,
        )

        print()
        print(
            "Advertencia: se rellenaron valores fuera del "
            "casco convexo mediante vecino más cercano."
        )

    print()
    print("Extensión de la malla:")
    for clave, valor in extent.items():
        print(f"  {clave}: {valor}")

    # --------------------------------------------------------
    # 7. GENERAR MAPA INTERPOLADO
    # --------------------------------------------------------

    fig_interp, _ = (
        plot_superficie_interpolada(
            df=data_interp,
            GX=GX,
            GY=GY,
            Z=Z,
            title=(
                "Superficie interpolada de probabilidad "
                "de excedencia"
            ),
            cmap="YlOrRd",
            colorbar_label=(
                "Probabilidad de excedencia"
            ),
            col_lon=COL_LON,
            col_lat=COL_LAT,
            col_label=columna_etiqueta,
            show_labels=True,
            max_labels=40,
            q_low=2,
            q_high=98,
            n_levels=15,
        )
    )

    png_interpolado = (
        CARPETA_SALIDA
        / "mapa_interpolado_excedencias.png"
    )

    fig_interp.savefig(
        png_interpolado,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig_interp
    )

    print()
    print("Mapa interpolado guardado en:")
    print(png_interpolado)

    # --------------------------------------------------------
    # 8. EXPORTAR GEOTIFF
    # --------------------------------------------------------

    tif_path = (
        CARPETA_SALIDA
        / "excedencia_interpolada.tif"
    )

    resultado_tif = (
        exportar_desde_puntos_a_geotiff(
            df=df_unido,
            out_tif=str(tif_path),
            col_lon=COL_LON,
            col_lat=COL_LAT,
            col_val=VARIABLE,
            margin=MARGEN,
            nx=NX,
            ny=NY,
            method=METODO,
            fill_nearest=RELLENAR_EXTERIOR,
            nodata=-9999.0,
            crs=CRS,
            dtype="float32",
            eliminar_duplicados=True,
        )
    )

    print()
    print("GeoTIFF guardado en:")
    print(
        resultado_tif["out_tif"]
    )

    print()
    print("Control de calidad del GeoTIFF:")
    for clave, valor in (
        resultado_tif["calidad"].items()
    ):
        print(f"  {clave}: {valor}")

    # --------------------------------------------------------
    # 9. VALIDAR EL GEOTIFF ESCRITO
    # --------------------------------------------------------

    with rasterio.open(
        resultado_tif["out_tif"]
    ) as src:
        print()
        print("VALIDACIÓN DEL GEOTIFF")
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
            f"Transform: {src.transform}"
        )

        print(
            f"Nodata: {src.nodata}"
        )

        print(
            f"Orientación norte-arriba: "
            f"{src.transform.e < 0}"
        )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)

    print()
    print("Archivos generados:")

    print(
        f"1. {csv_unido}"
    )

    print(
        f"2. {png_puntos}"
    )

    print(
        f"3. {png_interpolado}"
    )

    print(
        f"4. {tif_path}"
    )


if __name__ == "__main__":
    main()