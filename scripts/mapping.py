import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point


# ============================================================
# PREPARACIÓN Y VALIDACIÓN ESPACIAL
# ============================================================

def preparar_datos_para_mapa(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val=None,
    crs="EPSG:4326",
    eliminar_duplicados=True,
):
    """
    Limpia y valida una tabla antes de crear el GeoDataFrame.

    Cuando existen estaciones con las mismas coordenadas, se conserva
    solamente la primera aparición. Los valores no se suman, promedian
    ni combinan.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla con coordenadas y, opcionalmente, una variable numérica.

    col_lon : str
        Columna de longitud o coordenada X.

    col_lat : str
        Columna de latitud o coordenada Y.

    col_val : str o None
        Variable que se utilizará para colorear las estaciones.

    crs : str
        Sistema de referencia espacial.

    eliminar_duplicados : bool
        Si True, conserva solamente la primera estación cuando existen
        coordenadas duplicadas.

    Retorna
    -------
    data : pd.DataFrame
        Tabla limpia.

    calidad : dict
        Resumen del control de calidad.
    """
    columnas_requeridas = [
        col_lon,
        col_lat,
    ]

    if col_val is not None:
        columnas_requeridas.append(
            col_val
        )

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}"
        )

    data = df.copy()

    total_original = len(data)

    for columna in columnas_requeridas:
        data[columna] = pd.to_numeric(
            data[columna],
            errors="coerce",
        )

    registros_no_numericos = int(
        data[columnas_requeridas]
        .isna()
        .any(axis=1)
        .sum()
    )

    data = data.dropna(
        subset=columnas_requeridas
    ).copy()

    registros_fuera_rango = 0

    if str(crs).upper() == "EPSG:4326":
        mascara_valida = (
            data[col_lon].between(-180, 180)
            & data[col_lat].between(-90, 90)
        )

        registros_fuera_rango = int(
            (~mascara_valida).sum()
        )

        data = data[
            mascara_valida
        ].copy()

    duplicados_detectados = int(
        data.duplicated(
            subset=[
                col_lon,
                col_lat,
            ],
            keep=False,
        ).sum()
    )

    duplicados_eliminados = 0

    if eliminar_duplicados:
        total_antes = len(data)

        data = data.drop_duplicates(
            subset=[
                col_lon,
                col_lat,
            ],
            keep="first",
        ).copy()

        duplicados_eliminados = (
            total_antes - len(data)
        )

    if data.empty:
        raise ValueError(
            "No quedaron estaciones válidas después de limpiar "
            "las coordenadas y la variable seleccionada."
        )

    calidad = {
        "total_registros_originales": total_original,
        "registros_no_numericos": registros_no_numericos,
        "registros_fuera_rango": registros_fuera_rango,
        "duplicados_detectados": duplicados_detectados,
        "duplicados_eliminados": duplicados_eliminados,
        "total_estaciones_validas": len(data),
    }

    return data, calidad


# ============================================================
# CREACIÓN DEL GEODATAFRAME
# ============================================================

def crear_geodataframe_estaciones(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val=None,
    crs="EPSG:4326",
    eliminar_duplicados=True,
):
    """
    Convierte un DataFrame en GeoDataFrame de estaciones.

    Retorna
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame con geometrías puntuales.

    calidad : dict
        Resumen de limpieza y validación.
    """
    data, calidad = preparar_datos_para_mapa(
        df=df,
        col_lon=col_lon,
        col_lat=col_lat,
        col_val=col_val,
        crs=crs,
        eliminar_duplicados=eliminar_duplicados,
    )

    geometry = [
        Point(x, y)
        for x, y in zip(
            data[col_lon],
            data[col_lat],
        )
    ]

    gdf = gpd.GeoDataFrame(
        data.copy(),
        geometry=geometry,
        crs=crs,
    )

    return gdf, calidad


# ============================================================
# VALIDACIÓN DE LÍMITE TERRITORIAL
# ============================================================

def preparar_limite_territorial(
    gdf_limite,
    crs_destino,
):
    """
    Valida y reproyecta opcionalmente un límite territorial.

    Parámetros
    ----------
    gdf_limite : geopandas.GeoDataFrame o None
        Polígono territorial.

    crs_destino : str o CRS
        CRS que utilizará el mapa.

    Retorna
    -------
    geopandas.GeoDataFrame o None
    """
    if gdf_limite is None:
        return None

    if not isinstance(
        gdf_limite,
        gpd.GeoDataFrame,
    ):
        raise TypeError(
            "gdf_limite debe ser un GeoDataFrame."
        )

    if gdf_limite.empty:
        raise ValueError(
            "El límite territorial está vacío."
        )

    limite = gdf_limite.copy()

    limite = limite[
        limite.geometry.notna()
    ].copy()

    limite = limite[
        ~limite.geometry.is_empty
    ].copy()

    if limite.empty:
        raise ValueError(
            "El límite territorial no contiene geometrías válidas."
        )

    if limite.crs is None:
        raise ValueError(
            "El límite territorial no tiene un CRS definido."
        )

    if crs_destino is not None:
        limite = limite.to_crs(
            crs_destino
        )

    return limite


# ============================================================
# MAPA PUNTUAL DE ESTACIONES
# ============================================================

def plot_mapa_estaciones(
    gdf,
    col_val,
    cmap="YlOrRd",
    title="Mapa de estaciones",
    colorbar_label="Valor",
    markersize=80,
    edgecolor="black",
    figsize=(10, 8),
    legend=True,
    gdf_limite=None,
    limite_edgecolor="black",
    limite_facecolor="none",
    limite_linewidth=1.2,
    ax=None,
):
    """
    Genera un mapa puntual de estaciones coloreadas por una variable.

    Parámetros
    ----------
    gdf : geopandas.GeoDataFrame
        Estaciones.

    col_val : str
        Variable numérica utilizada para colorear los puntos.

    gdf_limite : geopandas.GeoDataFrame o None
        Límite territorial opcional.

    ax : matplotlib.axes.Axes o None
        Eje existente. Si es None, se crea una figura nueva.

    Retorna
    -------
    fig, ax
    """
    if not isinstance(
        gdf,
        gpd.GeoDataFrame,
    ):
        raise TypeError(
            "gdf debe ser un GeoDataFrame."
        )

    if gdf.empty:
        raise ValueError(
            "El GeoDataFrame de estaciones está vacío."
        )

    if col_val not in gdf.columns:
        raise ValueError(
            f"No se encontró la variable '{col_val}'."
        )

    data = gdf.copy()

    data[col_val] = pd.to_numeric(
        data[col_val],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            col_val,
            "geometry",
        ]
    ).copy()

    if data.empty:
        raise ValueError(
            f"No existen valores válidos para '{col_val}'."
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize
        )
    else:
        fig = ax.figure

    limite = preparar_limite_territorial(
        gdf_limite=gdf_limite,
        crs_destino=data.crs,
    )

    if limite is not None:
        limite.plot(
            ax=ax,
            facecolor=limite_facecolor,
            edgecolor=limite_edgecolor,
            linewidth=limite_linewidth,
            zorder=1,
        )

    data.plot(
        ax=ax,
        column=col_val,
        cmap=cmap,
        legend=legend,
        edgecolor=edgecolor,
        markersize=markersize,
        zorder=3,
        legend_kwds={
            "label": colorbar_label,
            "shrink": 0.75,
        } if legend else None,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Longitud / Coordenada X"
    )

    ax.set_ylabel(
        "Latitud / Coordenada Y"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# ETIQUETAS DE ESTACIONES
# ============================================================

def agregar_etiquetas_estaciones(
    ax,
    gdf,
    col_label="station",
    fontsize=8,
    color="black",
    dx=0.0,
    dy=0.0,
    bbox_labels=False,
    max_labels=None,
):
    """
    Agrega etiquetas a un mapa existente.

    Parámetros
    ----------
    max_labels : int o None
        Número máximo de etiquetas que se mostrarán. Si es None,
        se muestran todas.
    """
    if col_label not in gdf.columns:
        raise ValueError(
            f"No se encontró la columna de etiquetas "
            f"'{col_label}'."
        )

    data = gdf[
        gdf.geometry.notna()
    ].copy()

    if max_labels is not None:
        max_labels = int(
            max_labels
        )

        if max_labels < 0:
            raise ValueError(
                "max_labels no puede ser negativo."
            )

        data = data.head(
            max_labels
        )

    for _, row in data.iterrows():
        text_kwargs = {}

        if bbox_labels:
            text_kwargs["bbox"] = {
                "facecolor": "white",
                "edgecolor": "gray",
                "boxstyle": "round,pad=0.2",
                "alpha": 0.65,
            }

        ax.text(
            row.geometry.x + dx,
            row.geometry.y + dy,
            str(
                row[col_label]
            ),
            fontsize=fontsize,
            ha="center",
            va="center",
            color=color,
            zorder=4,
            **text_kwargs,
        )

    return ax


# ============================================================
# MAPA COMPLETO CON ETIQUETAS
# ============================================================

def plot_mapa_con_etiquetas(
    gdf,
    col_val,
    col_label="station",
    cmap="YlOrRd",
    title="Mapa de estaciones",
    colorbar_label="Valor",
    markersize=80,
    figsize=(12, 9),
    gdf_limite=None,
    show_labels=True,
    fontsize=8,
    label_color="black",
    dx=0.0,
    dy=0.0,
    bbox_labels=True,
    max_labels=None,
):
    """
    Genera un mapa de estaciones con etiquetas opcionales.
    """
    fig, ax = plot_mapa_estaciones(
        gdf=gdf,
        col_val=col_val,
        cmap=cmap,
        title=title,
        colorbar_label=colorbar_label,
        markersize=markersize,
        figsize=figsize,
        gdf_limite=gdf_limite,
    )

    if show_labels:
        agregar_etiquetas_estaciones(
            ax=ax,
            gdf=gdf,
            col_label=col_label,
            fontsize=fontsize,
            color=label_color,
            dx=dx,
            dy=dy,
            bbox_labels=bbox_labels,
            max_labels=max_labels,
        )

    fig.tight_layout()

    return fig, ax


# ============================================================
# MAPA ESTÉTICO PARA TESIS O ARTÍCULO
# ============================================================

def plot_mapa_estetico_avanzado(
    gdf,
    col_val,
    col_label="station",
    cmap="YlOrRd",
    title="Mapa espacial de estaciones",
    colorbar_label="Valor",
    markersize=60,
    figsize=(14, 12),
    gdf_limite=None,
    show_labels=True,
    dx=0.01,
    dy=0.0,
    max_labels=None,
):
    """
    Genera un mapa con estilo más cuidado para tesis o artículo.
    """
    fig, ax = plt.subplots(
        figsize=figsize
    )

    limite = preparar_limite_territorial(
        gdf_limite=gdf_limite,
        crs_destino=gdf.crs,
    )

    if limite is not None:
        limite.plot(
            ax=ax,
            facecolor="#F5F5F5",
            edgecolor="#343434",
            linewidth=1.3,
            zorder=1,
        )

    data = gdf.copy()

    if col_val not in data.columns:
        raise ValueError(
            f"No se encontró la variable '{col_val}'."
        )

    data[col_val] = pd.to_numeric(
        data[col_val],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            col_val,
            "geometry",
        ]
    ).copy()

    if data.empty:
        raise ValueError(
            f"No existen valores válidos para '{col_val}'."
        )

    data.plot(
        ax=ax,
        column=col_val,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        markersize=markersize,
        zorder=3,
        legend_kwds={
            "label": colorbar_label,
            "shrink": 0.75,
        },
    )

    if show_labels:
        agregar_etiquetas_estaciones(
            ax=ax,
            gdf=data,
            col_label=col_label,
            fontsize=7,
            color="black",
            dx=dx,
            dy=dy,
            bbox_labels=True,
            max_labels=max_labels,
        )

    ax.set_title(
        title,
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Longitud / Coordenada X"
    )

    ax.set_ylabel(
        "Latitud / Coordenada Y"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    fig.tight_layout()

    return fig, ax