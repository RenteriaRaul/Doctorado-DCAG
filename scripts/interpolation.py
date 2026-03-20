import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def crear_malla_interpolacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    nx=200,
    ny=200,
    margin=0.0,
):
    """
    Crea una malla regular para interpolación espacial.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas.
    col_lon : str
        Columna de longitud.
    col_lat : str
        Columna de latitud.
    nx : int
        Número de nodos en X.
    ny : int
        Número de nodos en Y.
    margin : float
        Margen adicional alrededor del extent.

    Retorna
    -------
    GX, GY : np.ndarray
        Malla regular tipo meshgrid.
    extent : dict
        Límites espaciales usados.
    """
    lon_min = float(df[col_lon].min()) - margin
    lon_max = float(df[col_lon].max()) + margin
    lat_min = float(df[col_lat].min()) - margin
    lat_max = float(df[col_lat].max()) + margin

    gx = np.linspace(lon_min, lon_max, nx)
    gy = np.linspace(lat_min, lat_max, ny)

    GX, GY = np.meshgrid(gx, gy, indexing="xy")

    extent = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "nx": nx,
        "ny": ny,
    }

    return GX, GY, extent


def interpolar_superficie(
    points,
    values,
    GX,
    GY,
    method="linear",
):
    """
    Interpola una superficie usando scipy.griddata.

    Parámetros
    ----------
    points : np.ndarray
        Coordenadas de puntos (n, 2).
    values : np.ndarray
        Valores en los puntos.
    GX, GY : np.ndarray
        Malla de interpolación.
    method : str
        Método: 'linear', 'cubic' o 'nearest'.

    Retorna
    -------
    Z : np.ndarray
        Superficie interpolada.
    """
    Z = griddata(points, values, (GX, GY), method=method)
    return Z


def rellenar_nan_con_nearest(
    points,
    values,
    GX,
    GY,
    Z,
):
    """
    Rellena NaN de una superficie interpolada usando nearest.

    Parámetros
    ----------
    points : np.ndarray
        Coordenadas de estaciones.
    values : np.ndarray
        Valores observados.
    GX, GY : np.ndarray
        Malla de interpolación.
    Z : np.ndarray
        Superficie a rellenar.

    Retorna
    -------
    Z_filled : np.ndarray
        Superficie con NaN rellenados.
    """
    Z_near = griddata(points, values, (GX, GY), method="nearest")
    Z_filled = np.where(np.isnan(Z), Z_near, Z)
    return Z_filled


def calcular_niveles_robustos(
    superficies,
    q_low=2,
    q_high=98,
    n_levels=15,
):
    """
    Calcula niveles robustos comunes para contourf a partir de una o varias superficies.

    Parámetros
    ----------
    superficies : list[np.ndarray] o np.ndarray
        Una o varias superficies.
    q_low : float
        Percentil inferior.
    q_high : float
        Percentil superior.
    n_levels : int
        Número de niveles.

    Retorna
    -------
    levels : np.ndarray
        Niveles comunes.
    vmin : float
        Valor mínimo robusto.
    vmax : float
        Valor máximo robusto.
    """
    if isinstance(superficies, np.ndarray):
        superficies = [superficies]

    arr = np.concatenate([s[np.isfinite(s)].ravel() for s in superficies if np.any(np.isfinite(s))])

    if arr.size == 0:
        raise ValueError("No hay valores finitos para calcular niveles.")

    vmin = np.nanpercentile(arr, q_low)
    vmax = np.nanpercentile(arr, q_high)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    levels = np.linspace(vmin, vmax, n_levels)
    return levels, vmin, vmax


def comparar_interpolaciones(
    df,
    points,
    values,
    GX,
    GY,
    fill_outside=False,
    cmap="YlOrRd",
    title="Comparativo de Interpolación",
    col_lon="LONGITUD",
    col_lat="LATITUD",
):
    """
    Genera comparación visual entre interpolación lineal y cúbica.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas.
    points : np.ndarray
        Coordenadas de estaciones.
    values : np.ndarray
        Valores de las estaciones.
    GX, GY : np.ndarray
        Malla común.
    fill_outside : bool
        Si True, rellena NaN fuera del casco convexo con nearest.
    cmap : str
        Mapa de color.
    title : str
        Título general de la figura.
    col_lon : str
        Columna de longitud.
    col_lat : str
        Columna de latitud.

    Retorna
    -------
    fig, axes, resultados : tuple
        Figura, ejes y diccionario con superficies generadas.
    """
    Z_lin = interpolar_superficie(points, values, GX, GY, method="linear")
    Z_cubic = interpolar_superficie(points, values, GX, GY, method="cubic")

    if fill_outside:
        Z_lin = rellenar_nan_con_nearest(points, values, GX, GY, Z_lin)
        Z_cubic = rellenar_nan_con_nearest(points, values, GX, GY, Z_cubic)

    levels, _, _ = calcular_niveles_robustos([Z_lin, Z_cubic], q_low=2, q_high=98, n_levels=15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    im0 = axes[0].contourf(GX, GY, Z_lin, levels=levels, cmap=cmap)
    axes[0].scatter(df[col_lon], df[col_lat], c="black", s=25, edgecolors="white")
    axes[0].set_title("Linear (griddata)")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    im1 = axes[1].contourf(GX, GY, Z_cubic, levels=levels, cmap=cmap)
    axes[1].scatter(df[col_lon], df[col_lat], c="black", s=25, edgecolors="white")
    axes[1].set_title("Cubic (griddata)")
    axes[1].set_xlabel("Longitud")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Valor interpolado")

    fig.suptitle(title, fontsize=14)

    resultados = {
        "Z_linear": Z_lin,
        "Z_cubic": Z_cubic,
        "levels": levels,
    }

    return fig, axes, resultados


def plot_superficie_interpolada(
    df,
    GX,
    GY,
    Z,
    title="Mapa interpolado",
    cmap="YlOrRd",
    colorbar_label="Valor",
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_label="NOMBRE",
    show_labels=True,
    label_dx=0.01,
    label_dy=0.0,
    xlim=None,
    ylim=None,
    figsize=(14, 12),
    bbox_labels=True,
):
    """
    Genera un mapa interpolado individual con estaciones y etiquetas.

    Retorna
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    levels, _, _ = calcular_niveles_robustos(Z, q_low=2, q_high=98, n_levels=15)
    contour = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap)

    ax.scatter(
        df[col_lon],
        df[col_lat],
        c="black",
        s=30,
        edgecolors="white",
        label="Estaciones",
    )

    if show_labels and col_label in df.columns:
        for _, row in df.iterrows():
            text_kwargs = {}
            if bbox_labels:
                text_kwargs["bbox"] = dict(
                    facecolor="white",
                    edgecolor="gray",
                    boxstyle="round,pad=0.2",
                    alpha=0.6,
                )

            ax.text(
                row[col_lon] + label_dx,
                row[col_lat] + label_dy,
                str(row[col_label]),
                fontsize=7,
                **text_kwargs,
            )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend()
    ax.grid(True)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    return fig, ax
