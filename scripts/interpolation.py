import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, QhullError


# ============================================================
# PREPARACIÓN Y VALIDACIÓN DE DATOS
# ============================================================

def preparar_datos_interpolacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="prob_excedencia",
    crs="EPSG:4326",
    eliminar_duplicados=True,
):
    """
    Limpia y valida los datos antes de realizar la interpolación.

    Cuando existen estaciones con las mismas coordenadas,
    se conserva únicamente la primera aparición. Los valores
    no se suman, promedian ni combinan.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla con coordenadas y variable numérica.

    col_lon : str
        Columna de longitud o coordenada X.

    col_lat : str
        Columna de latitud o coordenada Y.

    col_val : str
        Variable numérica que será interpolada.

    crs : str
        Sistema de referencia espacial.

    eliminar_duplicados : bool
        Si True, conserva solamente la primera estación
        cuando existen coordenadas duplicadas.

    Retorna
    -------
    data : pd.DataFrame
        Datos limpios.

    points : np.ndarray
        Coordenadas con dimensiones (n, 2).

    values : np.ndarray
        Valores asociados a las estaciones.

    calidad : dict
        Resumen del proceso de limpieza.
    """
    columnas_requeridas = [
        col_lon,
        col_lat,
        col_val,
    ]

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

        # Se conserva únicamente la primera estación.
        # No se realiza suma, media, máximo ni mediana.
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
            "las coordenadas y los valores."
        )

    points = data[
        [col_lon, col_lat]
    ].to_numpy(
        dtype=float
    )

    values = data[
        col_val
    ].to_numpy(
        dtype=float
    )

    calidad = {
        "total_registros_originales": total_original,
        "registros_no_numericos": registros_no_numericos,
        "registros_fuera_rango": registros_fuera_rango,
        "duplicados_detectados": duplicados_detectados,
        "duplicados_eliminados": duplicados_eliminados,
        "total_estaciones_validas": len(data),
    }

    return (
        data,
        points,
        values,
        calidad,
    )


# ============================================================
# VALIDACIÓN DE DISTRIBUCIÓN ESPACIAL
# ============================================================

def validar_puntos_interpolacion(
    points,
    method="linear",
):
    """
    Verifica que existan suficientes puntos y que la distribución
    espacial sea adecuada para el método seleccionado.

    Parámetros
    ----------
    points : np.ndarray
        Coordenadas con dimensiones (n, 2).

    method : str
        Método de interpolación:
        - linear
        - cubic
        - nearest

    Retorna
    -------
    dict
        Información sobre la geometría de los puntos.
    """
    metodos_validos = {
        "linear",
        "cubic",
        "nearest",
    }

    if method not in metodos_validos:
        raise ValueError(
            f"Método no válido: {method}. "
            f"Use uno de: {sorted(metodos_validos)}"
        )

    points = np.asarray(
        points,
        dtype=float,
    )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "points debe tener dimensiones (n, 2)."
        )

    n_points = len(points)

    if n_points == 0:
        raise ValueError(
            "No existen estaciones válidas para interpolar."
        )

    if method == "nearest":
        return {
            "n_points": n_points,
            "convex_hull_area": np.nan,
            "colineales": False,
        }

    if n_points < 3:
        raise ValueError(
            f"La interpolación {method} requiere al menos "
            "tres estaciones válidas."
        )

    try:
        hull = ConvexHull(
            points
        )

        hull_area = float(
            hull.volume
        )

    except QhullError as error:
        raise ValueError(
            "Las estaciones parecen estar alineadas o presentan "
            "una distribución espacial insuficiente para realizar "
            f"la interpolación {method}. Detalle: {error}"
        ) from error

    if np.isclose(
        hull_area,
        0.0,
    ):
        raise ValueError(
            "Las estaciones son colineales o cubren un área "
            "prácticamente nula."
        )

    return {
        "n_points": n_points,
        "convex_hull_area": hull_area,
        "colineales": False,
    }


# ============================================================
# CREACIÓN DE MALLA
# ============================================================

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

    El margen se expresa en las unidades del sistema de
    coordenadas utilizado. En EPSG:4326 corresponde a grados.

    Retorna
    -------
    GX, GY : np.ndarray
        Malla regular.

    extent : dict
        Límites espaciales utilizados.
    """
    if col_lon not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{col_lon}'."
        )

    if col_lat not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{col_lat}'."
        )

    nx = int(
        nx
    )

    ny = int(
        ny
    )

    margin = float(
        margin
    )

    if nx < 2 or ny < 2:
        raise ValueError(
            "nx y ny deben ser mayores o iguales a 2."
        )

    if margin < 0:
        raise ValueError(
            "El margen espacial no puede ser negativo."
        )

    coordenadas = df[
        [col_lon, col_lat]
    ].copy()

    coordenadas[col_lon] = pd.to_numeric(
        coordenadas[col_lon],
        errors="coerce",
    )

    coordenadas[col_lat] = pd.to_numeric(
        coordenadas[col_lat],
        errors="coerce",
    )

    coordenadas = coordenadas.dropna(
        subset=[
            col_lon,
            col_lat,
        ]
    )

    if coordenadas.empty:
        raise ValueError(
            "No existen coordenadas válidas para crear la malla."
        )

    lon_min = float(
        coordenadas[col_lon].min()
    ) - margin

    lon_max = float(
        coordenadas[col_lon].max()
    ) + margin

    lat_min = float(
        coordenadas[col_lat].min()
    ) - margin

    lat_max = float(
        coordenadas[col_lat].max()
    ) + margin

    if np.isclose(
        lon_min,
        lon_max,
    ):
        raise ValueError(
            "Todas las estaciones tienen la misma coordenada X."
        )

    if np.isclose(
        lat_min,
        lat_max,
    ):
        raise ValueError(
            "Todas las estaciones tienen la misma coordenada Y."
        )

    gx = np.linspace(
        lon_min,
        lon_max,
        nx,
    )

    gy = np.linspace(
        lat_min,
        lat_max,
        ny,
    )

    GX, GY = np.meshgrid(
        gx,
        gy,
        indexing="xy",
    )

    extent = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "nx": nx,
        "ny": ny,
        "margin": margin,
    }

    return (
        GX,
        GY,
        extent,
    )


# ============================================================
# INTERPOLACIÓN
# ============================================================

def interpolar_superficie(
    points,
    values,
    GX,
    GY,
    method="linear",
):
    """
    Interpola una superficie usando scipy.griddata.

    Métodos permitidos:
    - linear
    - cubic
    - nearest
    """
    points = np.asarray(
        points,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values debe ser un vector unidimensional."
        )

    if len(points) != len(values):
        raise ValueError(
            "El número de coordenadas no coincide con "
            "el número de valores."
        )

    validar_puntos_interpolacion(
        points=points,
        method=method,
    )

    try:
        Z = griddata(
            points,
            values,
            (GX, GY),
            method=method,
        )

    except Exception as error:
        raise ValueError(
            f"No fue posible realizar la interpolación {method}. "
            f"Detalle: {error}"
        ) from error

    if Z is None:
        raise ValueError(
            "La interpolación no produjo una superficie."
        )

    if not np.any(
        np.isfinite(Z)
    ):
        raise ValueError(
            "La superficie interpolada no contiene valores finitos."
        )

    return Z

# ============================================================
# MAPA DE INTERPOLACIÓN OBSERVACIONAL
# ============================================================

def interpolar_superficie_observacional(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="EXCEDENCIA_50MM",
    resolucion=300,
    eliminar_duplicados=True,
):
    """
    Reproduce la metodología de interpolación utilizada
    originalmente en Google Colab para los mapas de excedencia.

    Características
    ---------------
    - interpolación scipy.griddata(method="linear");
    - malla regular de 300 x 300 por defecto;
    - sin margen adicional;
    - sin extrapolación nearest;
    - sin suavizado gaussiano;
    - sin modificación de los valores observados;
    - conserva solamente la primera estación cuando existen
      coordenadas duplicadas.

    La superficie resultante únicamente contiene valores dentro
    del dominio espacial soportado por las estaciones
    (casco convexo).

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla con coordenadas y variable a interpolar.

    col_lon : str
        Columna de longitud.

    col_lat : str
        Columna de latitud.

    col_val : str
        Variable numérica a interpolar.

    resolucion : int
        Número de nodos por eje de la malla.

    eliminar_duplicados : bool
        Si True, conserva únicamente la primera estación cuando
        existen coordenadas idénticas.

    Retorna
    -------
    resultado : dict
        Diccionario con:
        - data
        - points
        - values
        - grid_x
        - grid_y
        - grid_z
        - extent
        - calidad
    """

    columnas_requeridas = [
        col_lon,
        col_lat,
        col_val,
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}"
        )

    if resolucion < 2:
        raise ValueError(
            "La resolución debe ser mayor o igual a 2."
        )

    data = df.copy()

    total_original = len(data)

    # --------------------------------------------------------
    # CONVERSIÓN NUMÉRICA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DUPLICADOS
    # --------------------------------------------------------

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

    if len(data) < 3:
        raise ValueError(
            "Se requieren al menos tres estaciones válidas "
            "para realizar la interpolación lineal."
        )

    # --------------------------------------------------------
    # PUNTOS Y VALORES
    # --------------------------------------------------------

    points = data[
        [
            col_lon,
            col_lat,
        ]
    ].to_numpy(
        dtype=float
    )

    values = data[
        col_val
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # VALIDAR DISTRIBUCIÓN ESPACIAL
    # --------------------------------------------------------

    validar_puntos_interpolacion(
        points=points,
        method="linear",
    )

    # --------------------------------------------------------
    # MALLA ORIGINAL TIPO COLAB
    # --------------------------------------------------------
    #
    # Importante:
    # - no utiliza margen;
    # - utiliza exactamente min/max de las estaciones.
    # --------------------------------------------------------

    lon_min = float(
        data[col_lon].min()
    )

    lon_max = float(
        data[col_lon].max()
    )

    lat_min = float(
        data[col_lat].min()
    )

    lat_max = float(
        data[col_lat].max()
    )

    grid_x, grid_y = np.mgrid[
        lon_min:lon_max:complex(resolucion),
        lat_min:lat_max:complex(resolucion),
    ]

    # --------------------------------------------------------
    # INTERPOLACIÓN LINEAL
    # --------------------------------------------------------

    try:

        grid_z = griddata(
            points,
            values,
            (
                grid_x,
                grid_y,
            ),
            method="linear",
        )

    except Exception as error:

        raise ValueError(
            "No fue posible realizar la interpolación lineal. "
            f"Detalle: {error}"
        ) from error

    if grid_z is None:
        raise ValueError(
            "La interpolación no produjo una superficie."
        )

    if not np.any(
        np.isfinite(grid_z)
    ):
        raise ValueError(
            "La superficie interpolada no contiene valores válidos."
        )

    # --------------------------------------------------------
    # METADATOS
    # --------------------------------------------------------

    extent = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "nx": resolucion,
        "ny": resolucion,
        "margin": 0.0,
        "method": "linear",
    }

    calidad = {
        "total_registros_originales": total_original,
        "registros_no_numericos": registros_no_numericos,
        "duplicados_detectados": duplicados_detectados,
        "duplicados_eliminados": duplicados_eliminados,
        "total_estaciones_validas": len(data),
        "valor_minimo_observado": float(
            np.nanmin(values)
        ),
        "valor_maximo_observado": float(
            np.nanmax(values)
        ),
        "valor_minimo_interpolado": float(
            np.nanmin(grid_z)
        ),
        "valor_maximo_interpolado": float(
            np.nanmax(grid_z)
        ),
    }

    return {
        "data": data,
        "points": points,
        "values": values,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "extent": extent,
        "calidad": calidad,
    }

# ============================================================
# RELLENO EXTERIOR
# ============================================================

def rellenar_nan_con_nearest(
    points,
    values,
    GX,
    GY,
    Z,
):
    """
    Rellena los valores NaN mediante vecino más cercano.

    Advertencia
    -----------
    Este procedimiento extrapola fuera del casco convexo de las
    estaciones. El resultado debe interpretarse con cautela.
    """
    points = np.asarray(
        points,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    Z = np.asarray(
        Z,
        dtype=float,
    )

    Z_near = griddata(
        points,
        values,
        (GX, GY),
        method="nearest",
    )

    Z_filled = np.where(
        np.isnan(Z),
        Z_near,
        Z,
    )

    return Z_filled

# ============================================================
# MAPA OBSERVACIONAL TIPO COLAB
# ============================================================

def plot_superficie_observacional(
    resultado,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_label="NOMBRE",
    title="Interpolated Exceedance Map",
    colorbar_label="Probability of Exceedance ≥ 50 mm",
    cmap="YlOrRd",
    levels=15,
    show_stations=True,
    show_labels=True,
    label_dx=0.01,
    label_dy=0.01,
    gdf_limite=None,
    figsize=(12, 9),
):
    """
    Visualiza una superficie generada mediante
    interpolar_superficie_observacional().

    Reproduce el estilo metodológico original:
    contourf con número fijo de niveles.

    El límite territorial es únicamente una referencia visual.
    No modifica la interpolación.
    """

    data = resultado["data"]
    grid_x = resultado["grid_x"]
    grid_y = resultado["grid_y"]
    grid_z = resultado["grid_z"]

    fig, ax = plt.subplots(
        figsize=figsize
    )

    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=levels,
        cmap=cmap,
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
    )

    cbar.set_label(
        colorbar_label
    )

    # --------------------------------------------------------
    # ESTACIONES
    # --------------------------------------------------------

    if show_stations:

        ax.scatter(
            data[col_lon],
            data[col_lat],
            c="black",
            s=30,
            edgecolors="white",
            label="Stations",
            zorder=4,
        )

    # --------------------------------------------------------
    # ETIQUETAS
    # --------------------------------------------------------

    if (
        show_labels
        and col_label in data.columns
    ):

        for _, row in data.iterrows():

            ax.text(
                row[col_lon] + label_dx,
                row[col_lat] + label_dy,
                str(
                    row[col_label]
                ),
                fontsize=7,
                color="black",
                zorder=5,
            )

    # --------------------------------------------------------
    # LÍMITE TERRITORIAL
    # --------------------------------------------------------

    if gdf_limite is not None:

        limite = gdf_limite.copy()

        if limite.crs is not None:
            limite = limite.to_crs(
                "EPSG:4326"
            )

        limite.boundary.plot(
            ax=ax,
            color="black",
            linewidth=1.3,
            zorder=6,
        )

    # --------------------------------------------------------
    # FORMATO
    # --------------------------------------------------------

    ax.set_title(
        title,
        loc="left",
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    if show_stations:
        ax.legend()

    ax.grid(
        True
    )

    fig.tight_layout()

    return fig, ax

# ============================================================
# CÁLCULO DE NIVELES
# ============================================================

def calcular_niveles_robustos(
    superficies,
    q_low=2,
    q_high=98,
    n_levels=15,
):
    """
    Calcula niveles robustos para contourf utilizando percentiles.

    Parámetros
    ----------
    superficies : np.ndarray o list[np.ndarray]
        Una o varias superficies.

    q_low : float
        Percentil inferior.

    q_high : float
        Percentil superior.

    n_levels : int
        Número de niveles.

    Retorna
    -------
    levels, vmin, vmax
    """
    q_low = float(
        q_low
    )

    q_high = float(
        q_high
    )

    n_levels = int(
        n_levels
    )

    if not 0 <= q_low < q_high <= 100:
        raise ValueError(
            "Los percentiles deben cumplir: "
            "0 <= q_low < q_high <= 100."
        )

    if n_levels < 2:
        raise ValueError(
            "n_levels debe ser mayor o igual a 2."
        )

    if isinstance(
        superficies,
        np.ndarray,
    ):
        superficies = [
            superficies
        ]

    valores_validos = []

    for superficie in superficies:
        superficie = np.asarray(
            superficie,
            dtype=float,
        )

        finitos = superficie[
            np.isfinite(
                superficie
            )
        ]

        if finitos.size > 0:
            valores_validos.append(
                finitos.ravel()
            )

    if not valores_validos:
        raise ValueError(
            "No hay valores finitos para calcular niveles."
        )

    arr = np.concatenate(
        valores_validos
    )

    vmin = float(
        np.nanpercentile(
            arr,
            q_low,
        )
    )

    vmax = float(
        np.nanpercentile(
            arr,
            q_high,
        )
    )

    if np.isclose(
        vmin,
        vmax,
    ):
        escala = max(
            abs(vmin),
            1.0,
        )

        vmax = (
            vmin + escala * 1e-6
        )

    levels = np.linspace(
        vmin,
        vmax,
        n_levels,
    )

    return (
        levels,
        vmin,
        vmax,
    )


# ============================================================
# COMPARACIÓN DE INTERPOLACIONES
# ============================================================

def comparar_interpolaciones(
    df,
    points,
    values,
    GX,
    GY,
    fill_outside=False,
    cmap="YlOrRd",
    title="Comparativo de interpolación",
    col_lon="LONGITUD",
    col_lat="LATITUD",
    colorbar_label="Valor interpolado",
    q_low=2,
    q_high=98,
    n_levels=15,
):
    """
    Compara interpolación lineal y cúbica.

    Si fill_outside=True, los valores fuera del casco convexo
    se rellenan mediante vecino más cercano.
    """
    Z_linear = interpolar_superficie(
        points=points,
        values=values,
        GX=GX,
        GY=GY,
        method="linear",
    )

    Z_cubic = interpolar_superficie(
        points=points,
        values=values,
        GX=GX,
        GY=GY,
        method="cubic",
    )

    if fill_outside:
        Z_linear = rellenar_nan_con_nearest(
            points=points,
            values=values,
            GX=GX,
            GY=GY,
            Z=Z_linear,
        )

        Z_cubic = rellenar_nan_con_nearest(
            points=points,
            values=values,
            GX=GX,
            GY=GY,
            Z=Z_cubic,
        )

    levels, vmin, vmax = (
        calcular_niveles_robustos(
            superficies=[
                Z_linear,
                Z_cubic,
            ],
            q_low=q_low,
            q_high=q_high,
            n_levels=n_levels,
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        constrained_layout=True,
    )

    im_linear = axes[0].contourf(
        GX,
        GY,
        Z_linear,
        levels=levels,
        cmap=cmap,
    )

    axes[0].scatter(
        df[col_lon],
        df[col_lat],
        c="black",
        s=25,
        edgecolors="white",
        zorder=3,
    )

    axes[0].set_title(
        "Lineal — griddata"
    )

    axes[0].set_xlabel(
        "Longitud / Coordenada X"
    )

    axes[0].set_ylabel(
        "Latitud / Coordenada Y"
    )

    axes[0].grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    axes[1].contourf(
        GX,
        GY,
        Z_cubic,
        levels=levels,
        cmap=cmap,
    )

    axes[1].scatter(
        df[col_lon],
        df[col_lat],
        c="black",
        s=25,
        edgecolors="white",
        zorder=3,
    )

    axes[1].set_title(
        "Cúbica — griddata"
    )

    axes[1].set_xlabel(
        "Longitud / Coordenada X"
    )

    axes[1].set_ylabel(
        "Latitud / Coordenada Y"
    )

    axes[1].grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    cbar = fig.colorbar(
        im_linear,
        ax=axes.ravel().tolist(),
        shrink=0.9,
        pad=0.02,
    )

    cbar.set_label(
        colorbar_label
    )

    fig.suptitle(
        title,
        fontsize=14,
    )

    resultados = {
        "Z_linear": Z_linear,
        "Z_cubic": Z_cubic,
        "levels": levels,
        "vmin": vmin,
        "vmax": vmax,
        "fill_outside": fill_outside,
    }

    return (
        fig,
        axes,
        resultados,
    )


# ============================================================
# MAPA DE SUPERFICIE INTERPOLADA
# ============================================================

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
    col_label="station",
    show_labels=True,
    label_dx=0.01,
    label_dy=0.0,
    xlim=None,
    ylim=None,
    figsize=(14, 12),
    bbox_labels=True,
    max_labels=None,
    q_low=2,
    q_high=98,
    n_levels=15,
    gdf_limite=None,
):
    """
    Genera un mapa interpolado con estaciones, etiquetas y
    límite territorial opcional.
    """
    if col_lon not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{col_lon}'."
        )

    if col_lat not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{col_lat}'."
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    levels, _, _ = calcular_niveles_robustos(
        superficies=Z,
        q_low=q_low,
        q_high=q_high,
        n_levels=n_levels,
    )

    contour = ax.contourf(
        GX,
        GY,
        Z,
        levels=levels,
        cmap=cmap,
        zorder=1,
    )

    if gdf_limite is not None:
        try:
            gdf_limite.boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.2,
                zorder=3,
            )
        except Exception as error:
            raise ValueError(
                "No fue posible dibujar el límite territorial. "
                f"Detalle: {error}"
            ) from error

    ax.scatter(
        df[col_lon],
        df[col_lat],
        c="black",
        s=30,
        edgecolors="white",
        label="Estaciones",
        zorder=4,
    )

    if (
        show_labels
        and col_label in df.columns
    ):
        data_labels = df.copy()

        if max_labels is not None:
            max_labels = int(
                max_labels
            )

            if max_labels < 0:
                raise ValueError(
                    "max_labels no puede ser negativo."
                )

            data_labels = data_labels.head(
                max_labels
            )

        for _, row in data_labels.iterrows():
            text_kwargs = {}

            if bbox_labels:
                text_kwargs["bbox"] = {
                    "facecolor": "white",
                    "edgecolor": "gray",
                    "boxstyle": "round,pad=0.2",
                    "alpha": 0.65,
                }

            ax.text(
                row[col_lon] + label_dx,
                row[col_lat] + label_dy,
                str(
                    row[col_label]
                ),
                fontsize=7,
                zorder=5,
                **text_kwargs,
            )

    cbar = fig.colorbar(
        contour,
        ax=ax,
    )

    cbar.set_label(
        colorbar_label
    )

    ax.set_title(
        title,
        fontsize=14,
    )

    ax.set_xlabel(
        "Longitud / Coordenada X"
    )

    ax.set_ylabel(
        "Latitud / Coordenada Y"
    )

    ax.legend()

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    if xlim is not None:
        ax.set_xlim(
            *xlim
        )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    fig.tight_layout()

    return (
        fig,
        ax,
    )