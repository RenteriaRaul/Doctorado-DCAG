import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import bootstrap


def station_ci(
    p,
    n_trials=200,
    n_boot=1000,
    conf_level=0.95,
    rng=None,
    method="BCa",
):
    """
    Calcula una estimación de proporción y su intervalo de confianza
    mediante ensayos Bernoulli + bootstrap.

    Parámetros
    ----------
    p : float
        Probabilidad observada de excedencia en escala 0-1.
    n_trials : int
        Número de ensayos Bernoulli simulados.
    n_boot : int
        Número de remuestreos bootstrap.
    conf_level : float
        Nivel de confianza, por ejemplo 0.95.
    rng : numpy.random.Generator o None
        Generador aleatorio.
    method : str
        Método de scipy.stats.bootstrap, por ejemplo "BCa".

    Retorna
    -------
    mean_est : float
        Estimación central de la proporción.
    low : float
        Límite inferior del IC.
    high : float
        Límite superior del IC.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = float(p)

    if not np.isfinite(p):
        raise ValueError("La probabilidad p debe ser finita.")
    if p < 0 or p > 1:
        raise ValueError("La probabilidad p debe estar en el rango [0, 1].")
    if n_trials <= 1:
        raise ValueError("n_trials debe ser mayor que 1.")
    if n_boot <= 0:
        raise ValueError("n_boot debe ser mayor que 0.")
    if conf_level <= 0 or conf_level >= 1:
        raise ValueError("conf_level debe estar entre 0 y 1.")

    sample = rng.binomial(1, p, size=n_trials)

    res = bootstrap(
        (sample,),
        np.mean,
        confidence_level=conf_level,
        n_resamples=n_boot,
        method=method,
        random_state=rng,
    )

    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    mean_est = float(sample.mean())

    return mean_est, low, high


def calcular_ic_excedencia_estaciones(
    df,
    col_p="EXCEDENCIA_50MM",
    n_trials=200,
    n_boot=1000,
    conf_level=0.95,
    rng=None,
    output_scale="percent",
):
    """
    Calcula estimación central e intervalo de confianza por estación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna de probabilidad.
    col_p : str
        Columna de probabilidad base, en escala 0-1.
    n_trials : int
        Número de ensayos Bernoulli por estación.
    n_boot : int
        Número de remuestreos bootstrap.
    conf_level : float
        Nivel de confianza.
    rng : numpy.random.Generator o None
        Generador aleatorio.
    output_scale : str
        "percent" para devolver 0-100,
        "probability" para devolver 0-1.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con columnas:
        P_MEAN, P_LOW, P_HIGH, P_WIDTH
    """
    if rng is None:
        rng = np.random.default_rng()

    if col_p not in df.columns:
        raise ValueError(f"El DataFrame no contiene la columna '{col_p}'.")

    df_out = df.copy()
    df_out[col_p] = pd.to_numeric(df_out[col_p], errors="coerce")
    df_out = df_out.dropna(subset=[col_p]).copy()

    est_means = []
    est_low = []
    est_high = []

    for p in df_out[col_p].astype(float).values:
        m, lo, hi = station_ci(
            p=p,
            n_trials=n_trials,
            n_boot=n_boot,
            conf_level=conf_level,
            rng=rng,
        )
        est_means.append(m)
        est_low.append(lo)
        est_high.append(hi)

    df_out["P_MEAN"] = np.asarray(est_means, dtype=float)
    df_out["P_LOW"] = np.asarray(est_low, dtype=float)
    df_out["P_HIGH"] = np.asarray(est_high, dtype=float)
    df_out["P_WIDTH"] = df_out["P_HIGH"] - df_out["P_LOW"]

    if output_scale == "percent":
        for c in ["P_MEAN", "P_LOW", "P_HIGH", "P_WIDTH"]:
            df_out[c] = df_out[c] * 100.0
    elif output_scale == "probability":
        pass
    else:
        raise ValueError("output_scale debe ser 'percent' o 'probability'.")

    return df_out


def interpolar_superficies_incertidumbre(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_mean="P_MEAN",
    col_low="P_LOW",
    col_high="P_HIGH",
    nx=250,
    ny=250,
    method="cubic",
    fill_nearest=True,
):
    """
    Interpola superficies de estimación central e incertidumbre.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas y columnas P_MEAN, P_LOW, P_HIGH.
    col_lon, col_lat : str
        Columnas de coordenadas.
    col_mean, col_low, col_high : str
        Columnas a interpolar.
    nx, ny : int
        Resolución de la malla.
    method : str
        Método de griddata.
    fill_nearest : bool
        Si True, rellena NaN con nearest.

    Retorna
    -------
    dict
        Diccionario con GX, GY, Z_mean, Z_low, Z_high, Z_width, points
    """
    cols_req = [col_lon, col_lat, col_mean, col_low, col_high]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    data = df.dropna(subset=cols_req).copy()

    points = data[[col_lon, col_lat]].values
    lon_min, lon_max = data[col_lon].min(), data[col_lon].max()
    lat_min, lat_max = data[col_lat].min(), data[col_lat].max()

    GX, GY = np.mgrid[
        lon_min:lon_max:complex(nx),
        lat_min:lat_max:complex(ny),
    ]

    Z_mean = griddata(points, data[col_mean].values, (GX, GY), method=method)
    Z_low = griddata(points, data[col_low].values, (GX, GY), method=method)
    Z_high = griddata(points, data[col_high].values, (GX, GY), method=method)

    if fill_nearest:
        Z_near_mean = griddata(points, data[col_mean].values, (GX, GY), method="nearest")
        Z_near_low = griddata(points, data[col_low].values, (GX, GY), method="nearest")
        Z_near_high = griddata(points, data[col_high].values, (GX, GY), method="nearest")

        Z_mean = np.where(np.isnan(Z_mean), Z_near_mean, Z_mean)
        Z_low = np.where(np.isnan(Z_low), Z_near_low, Z_low)
        Z_high = np.where(np.isnan(Z_high), Z_near_high, Z_high)

    Z_width = Z_high - Z_low

    return {
        "GX": GX,
        "GY": GY,
        "Z_mean": Z_mean,
        "Z_low": Z_low,
        "Z_high": Z_high,
        "Z_width": Z_width,
        "points": points,
        "data": data,
    }


def plot_incertidumbre_excedencia(
    df,
    GX,
    GY,
    Z_mean,
    Z_width,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    conf_level=0.95,
    cmap_mean="YlOrRd",
    cmap_width="PuBuGn",
):
    """
    Grafica dos paneles:
    - estimación central
    - ancho del intervalo de confianza

    Retorna
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    levels_m = np.linspace(np.nanpercentile(Z_mean, 2), np.nanpercentile(Z_mean, 98), 15)
    im0 = axes[0].contourf(GX, GY, Z_mean, levels=levels_m, cmap=cmap_mean)
    axes[0].scatter(df[col_lon], df[col_lat], c="black", s=25, edgecolors="white", label="Estaciones")
    axes[0].set_title("Excedencia ≥ 50 mm — Estimación central (%)")
    axes[0].set_xlabel("Longitud (°)")
    axes[0].set_ylabel("Latitud (°)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.02)
    cb0.set_label("Probabilidad (%)")

    levels_w = np.linspace(np.nanpercentile(Z_width, 2), np.nanpercentile(Z_width, 98), 15)
    im1 = axes[1].contourf(GX, GY, Z_width, levels=levels_w, cmap=cmap_width)
    axes[1].scatter(df[col_lon], df[col_lat], c="black", s=25, edgecolors="white")
    axes[1].set_title(f"Incertidumbre (ancho del IC {int(conf_level * 100)}%) — puntos porcentuales")
    axes[1].set_xlabel("Longitud (°)")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.02)
    cb1.set_label("pp (puntos porcentuales)")

    return fig, axes
