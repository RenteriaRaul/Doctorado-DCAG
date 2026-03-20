import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def crear_geodataframe_estaciones(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    crs="EPSG:4326",
):
    """
    Convierte un DataFrame en GeoDataFrame con geometría de puntos.

    Retorna
    -------
    gdf : GeoDataFrame
    """
    geometry = [Point(xy) for xy in zip(df[col_lon], df[col_lat])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    return gdf


def plot_mapa_estaciones_excedencia(
    gdf,
    col_val="EXCEDENCIA_50MM",
    cmap="YlOrRd",
    title="Mapa de Excedencia",
    markersize=80,
    edgecolor="black",
    figsize=(10, 8),
    legend=True,
):
    """
    Mapa base de estaciones coloreadas por excedencia.
    """
    fig, ax = plt.subplots(figsize=figsize)

    gdf.dropna(subset=[col_val]).plot(
        ax=ax,
        column=col_val,
        cmap=cmap,
        legend=legend,
        edgecolor=edgecolor,
        markersize=markersize,
    )

    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True)

    return fig, ax


def agregar_etiquetas_estaciones(
    ax,
    gdf,
    col_label="CLAVE",
    fontsize=8,
    color="black",
    dx=0.0,
    dy=0.0,
):
    """
    Agrega etiquetas a un mapa existente.
    """
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf[col_label]):
        ax.text(
            x + dx,
            y + dy,
            str(label),
            fontsize=fontsize,
            ha="center",
            va="center",
            color=color,
        )


def plot_mapa_con_etiquetas(
    gdf,
    col_val="EXCEDENCIA_50MM",
    col_label="CLAVE",
    cmap="YlOrRd",
    title="Mapa de Excedencia con Etiquetas",
    markersize=80,
    figsize=(12, 9),
):
    """
    Mapa completo con etiquetas.
    """
    fig, ax = plot_mapa_estaciones_excedencia(
        gdf,
        col_val=col_val,
        cmap=cmap,
        title=title,
        markersize=markersize,
        figsize=figsize,
    )

    agregar_etiquetas_estaciones(ax, gdf, col_label=col_label)

    return fig, ax


def plot_mapa_estetico_avanzado(
    gdf,
    col_val="EXCEDENCIA_50MM",
    col_label="NOMBRE",
    cmap="YlOrRd",
    title="Mapa de Excedencia ≥ 50 mm - Colima",
    markersize=60,
    figsize=(14, 12),
    dx=0.01,
    dy=0.0,
):
    """
    Versión más cuidada (tipo tesis/artículo).
    """
    fig, ax = plt.subplots(figsize=figsize)

    gdf.dropna(subset=[col_val]).plot(
        ax=ax,
        column=col_val,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        markersize=markersize,
    )

    for _, row in gdf.iterrows():
        ax.text(
            row.geometry.x + dx,
            row.geometry.y + dy,
            str(row[col_label]),
            fontsize=7,
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                boxstyle="round,pad=0.2",
                alpha=0.6,
            ),
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True)

    fig.tight_layout()
    return fig, ax
