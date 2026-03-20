import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev

from scripts.bootstrap_utils import bootstrap_robusto, bootstrap_parametrico


def cargar_y_limpiar_datos(path_csv, col_fecha="date", col_pp="pp"):
    """
    Carga un CSV de precipitación y aplica limpieza básica:
    - parseo de fechas
    - conversión a numérico
    - eliminación de nulos
    - eliminación de valores negativos
    - consolidación de duplicados por fecha usando el máximo diario
    """
    df = pd.read_csv(path_csv, parse_dates=[col_fecha])

    if col_fecha not in df.columns or col_pp not in df.columns:
        raise ValueError(f"El archivo {path_csv} no contiene las columnas requeridas: {col_fecha}, {col_pp}")

    df[col_pp] = pd.to_numeric(df[col_pp], errors="coerce")
    df = df.dropna(subset=[col_fecha, col_pp])
    df = df[df[col_pp] >= 0]

    df = (
        df.sort_values(col_fecha)
        .groupby(col_fecha, as_index=False)[col_pp]
        .max()
    )

    return df


def extraer_maximos_anuales(df, col_fecha="date", col_pp="pp"):
    """
    Extrae la serie de máximos anuales a partir de un DataFrame diario.
    """
    df = df.copy()
    df["year"] = df[col_fecha].dt.year
    max_ann = df.groupby("year")[col_pp].max().dropna()
    return max_ann


def calcular_tendencia_lineal(valores):
    """
    Calcula una pendiente lineal simple sobre una serie unidimensional.
    """
    valores = np.asarray(valores, dtype=float)

    if len(valores) < 2:
        return np.nan

    x = np.arange(len(valores))
    slope = np.polyfit(x, valores, 1)[0]
    return slope


def ajustar_gev(maximos_anuales):
    """
    Ajusta una distribución GEV por máxima verosimilitud.
    """
    datos = np.asarray(maximos_anuales, dtype=float)

    if len(datos) == 0:
        raise ValueError("No hay datos para ajustar la distribución GEV.")

    c, loc, scale = gev.fit(datos)

    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        raise ValueError("El ajuste GEV produjo parámetros inválidos.")

    return c, loc, scale


def calcular_niveles_retorno(c, loc, scale, niveles_retorno):
    """
    Calcula niveles de retorno a partir de parámetros GEV.
    """
    niveles_retorno = np.asarray(niveles_retorno, dtype=float)
    niveles = gev.ppf(1 - 1 / niveles_retorno, c, loc=loc, scale=scale)
    return niveles


def construir_tabla_resultados(
    station,
    niveles_retorno,
    niveles_puntuales,
    low_a,
    high_a,
    nacc_a,
    low_b,
    high_b,
    nacc_b,
    c,
    loc,
    scale,
    n_years,
    slope,
    note,
):
    """
    Construye la tabla final de resultados por estación.
    """
    tabla = pd.DataFrame({
        "station": station,
        "T_years": niveles_retorno,
        "level_mm": niveles_puntuales,
        "CI_low95_bootA": low_a,
        "CI_high95_bootA": high_a,
        "bootA_naccepted": np.repeat(nacc_a, len(niveles_retorno)),
        "CI_low95_bootB": low_b,
        "CI_high95_bootB": high_b,
        "bootB_naccepted": np.repeat(nacc_b, len(niveles_retorno)),
        "gev_shape": np.repeat(c, len(niveles_retorno)),
        "gev_loc": np.repeat(loc, len(niveles_retorno)),
        "gev_scale": np.repeat(scale, len(niveles_retorno)),
        "n_years": np.repeat(n_years, len(niveles_retorno)),
        "trend_slope_mm_per_year": np.repeat(slope, len(niveles_retorno)),
        "note": np.repeat(note, len(niveles_retorno)),
    })

    return tabla


def guardar_grafico_estacion(
    station,
    niveles_retorno,
    niveles_puntuales,
    low_a,
    high_a,
    n_years,
    slope,
    c,
    png_out,
    plot_max_t=100,
):
    """
    Genera y guarda el gráfico de niveles de retorno con IC robusto.
    """
    plt.figure(figsize=(8, 5))

    yerr = np.vstack([
        niveles_puntuales - low_a,
        high_a - niveles_puntuales
    ])

    plt.errorbar(
        niveles_retorno,
        niveles_puntuales,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        label="GEV (puntual) + IC 95% Boot A"
    )

    plt.xscale("log")
    plt.xlim(niveles_retorno.min() * 0.9, max(plot_max_t, niveles_retorno.max()) * 1.1)
    plt.xlabel("Return period (years)")
    plt.ylabel("Precipitation (mm)")

    subt = f"(n={n_years}, slope≈{slope:.2f} mm/año, shape={c:.3f})"
    plt.title(f"Return Levels GEV – {station}\n{subt}")

    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=150)
    plt.close()


def procesar_estacion(
    path_csv,
    col_fecha="date",
    col_pp="pp",
    n_min_anios=10,
    niveles_retorno=None,
    n_boot=500,
    alpha=0.05,
    rng=None,
    usar_boot_parametrico=True,
    plot_max_t=100,
    dir_out=".",
    fecha_tag=None,
):
    """
    Procesa una estación completa:
    - carga y limpieza
    - máximos anuales
    - ajuste GEV
    - niveles de retorno
    - IC robustos y opcionalmente paramétricos
    - exportación de gráfico y CSV

    Retorna
    -------
    tabla : pd.DataFrame o None
    meta : dict
    """
    if niveles_retorno is None:
        niveles_retorno = np.array([2, 5, 10, 25, 50, 100], dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    if fecha_tag is None:
        fecha_tag = "sin_fecha"

    base = os.path.basename(path_csv)
    station = os.path.splitext(base)[0]

    try:
        df = cargar_y_limpiar_datos(path_csv, col_fecha=col_fecha, col_pp=col_pp)

        max_ann = extraer_maximos_anuales(df, col_fecha=col_fecha, col_pp=col_pp).values
        n_years = len(max_ann)

        note = ""
        if n_years < n_min_anios:
            note = f"Advertencia: solo {n_years} años (<{n_min_anios}). Incertidumbre alta."

        slope = calcular_tendencia_lineal(max_ann)

        c, loc, scale = ajustar_gev(max_ann)
        niveles_puntuales = calcular_niveles_retorno(c, loc, scale, niveles_retorno)

        low_a, high_a, nacc_a = bootstrap_robusto(
            datos=max_ann,
            niveles_retorno=niveles_retorno,
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
            shape_bounds=(-0.35, 0.35),
            max_rel_factor=10.0,
        )

        if usar_boot_parametrico:
            low_b, high_b, nacc_b = bootstrap_parametrico(
                c=c,
                loc=loc,
                scale=scale,
                niveles_retorno=niveles_retorno,
                n_muestra=n_years,
                n_boot=n_boot,
                alpha=alpha,
                rng=rng,
            )
        else:
            low_b = np.full_like(niveles_retorno, np.nan, dtype=float)
            high_b = np.full_like(niveles_retorno, np.nan, dtype=float)
            nacc_b = 0

        tabla = construir_tabla_resultados(
            station=station,
            niveles_retorno=niveles_retorno,
            niveles_puntuales=niveles_puntuales,
            low_a=low_a,
            high_a=high_a,
            nacc_a=nacc_a,
            low_b=low_b,
            high_b=high_b,
            nacc_b=nacc_b,
            c=c,
            loc=loc,
            scale=scale,
            n_years=n_years,
            slope=slope,
            note=note,
        )

        png_out = os.path.join(dir_out, f"{station}_return_levels_ROBUST_{fecha_tag}.png")
        csv_out = os.path.join(dir_out, f"{station}_return_levels_ROBUST_{fecha_tag}.csv")

        guardar_grafico_estacion(
            station=station,
            niveles_retorno=niveles_retorno,
            niveles_puntuales=niveles_puntuales,
            low_a=low_a,
            high_a=high_a,
            n_years=n_years,
            slope=slope,
            c=c,
            png_out=png_out,
            plot_max_t=plot_max_t,
        )

        tabla.to_csv(csv_out, index=False)

        meta = {
            "station": station,
            "path": path_csv,
            "png": png_out,
            "csv": csv_out,
            "n_years": n_years,
            "shape": c,
            "loc": loc,
            "scale": scale,
            "trend_slope": slope,
            "bootA_naccepted": nacc_a,
            "bootB_naccepted": nacc_b,
            "note": note,
        }

        return tabla, meta

    except Exception as e:
        return None, {
            "station": station,
            "error": str(e),
            "path": path_csv,
        }
