import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev

from scripts.bootstrap_utils import (
    bootstrap_parametrico,
    bootstrap_robusto,
)


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================

def limpiar_datos_dataframe(
    df,
    col_fecha="date",
    col_pp="pp",
):
    """
    Limpia una serie diaria de precipitación ya cargada en memoria.

    Se utiliza tanto para archivos CSV preparados como para los
    DataFrame generados por conagua_reader.py.

    Procedimiento:
    - valida columnas;
    - convierte fecha y precipitación;
    - elimina fechas/precipitaciones nulas;
    - elimina precipitaciones negativas;
    - consolida duplicados por fecha usando el máximo diario.

    Retorna
    -------
    pd.DataFrame
        Serie diaria limpia con las columnas indicadas.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Los datos de entrada deben ser un DataFrame."
        )

    if col_fecha not in df.columns:
        raise ValueError(
            f"El DataFrame no contiene la columna de fecha "
            f"'{col_fecha}'."
        )

    if col_pp not in df.columns:
        raise ValueError(
            f"El DataFrame no contiene la columna de "
            f"precipitación '{col_pp}'."
        )

    df = df.copy()

    df[col_fecha] = pd.to_datetime(
        df[col_fecha],
        errors="coerce",
    )

    df[col_pp] = pd.to_numeric(
        df[col_pp],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            col_fecha,
            col_pp,
        ]
    ).copy()

    df = df[
        df[col_pp] >= 0
    ].copy()

    df = (
        df
        .sort_values(col_fecha)
        .groupby(
            col_fecha,
            as_index=False,
        )[col_pp]
        .max()
    )

    return df


def cargar_y_limpiar_datos(
    path_csv,
    col_fecha="date",
    col_pp="pp",
):
    """
    Carga un CSV de precipitación y aplica la misma limpieza
    utilizada por el flujo interno basado en DataFrame.

    Esta función se conserva para compatibilidad con el flujo
    histórico de archivos dat*.csv.
    """
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo: {path_csv}"
        )

    df = pd.read_csv(
        path_csv
    )

    return limpiar_datos_dataframe(
        df=df,
        col_fecha=col_fecha,
        col_pp=col_pp,
    )


# ============================================================
# MÁXIMOS ANUALES
# ============================================================

def extraer_maximos_anuales(
    df,
    col_fecha="date",
    col_pp="pp",
):
    """
    Extrae la serie de máximos anuales a partir de un DataFrame
    diario previamente limpio.
    """
    df = df.copy()

    if col_fecha not in df.columns:
        raise ValueError(
            f"No existe la columna de fecha '{col_fecha}'."
        )

    if col_pp not in df.columns:
        raise ValueError(
            f"No existe la columna de precipitación '{col_pp}'."
        )

    df["year"] = (
        df[col_fecha]
        .dt.year
    )

    max_ann = (
        df
        .groupby("year")[col_pp]
        .max()
        .dropna()
    )

    return max_ann


# ============================================================
# TENDENCIA
# ============================================================

def calcular_tendencia_lineal(
    valores,
):
    """
    Calcula una pendiente lineal simple sobre una serie
    unidimensional.
    """
    valores = np.asarray(
        valores,
        dtype=float,
    )

    if len(valores) < 2:
        return np.nan

    x = np.arange(
        len(valores)
    )

    slope = np.polyfit(
        x,
        valores,
        1,
    )[0]

    return slope


# ============================================================
# AJUSTE GEV
# ============================================================

def ajustar_gev(
    maximos_anuales,
):
    """
    Ajusta una distribución GEV por máxima verosimilitud.
    """
    datos = np.asarray(
        maximos_anuales,
        dtype=float,
    )

    datos = datos[
        np.isfinite(datos)
    ]

    if len(datos) == 0:
        raise ValueError(
            "No hay datos para ajustar la distribución GEV."
        )

    c, loc, scale = gev.fit(
        datos
    )

    if (
        not np.isfinite(
            [
                c,
                loc,
                scale,
            ]
        ).all()
        or scale <= 0
    ):
        raise ValueError(
            "El ajuste GEV produjo parámetros inválidos."
        )

    return (
        c,
        loc,
        scale,
    )


# ============================================================
# NIVELES DE RETORNO
# ============================================================

def calcular_niveles_retorno(
    c,
    loc,
    scale,
    niveles_retorno,
):
    """
    Calcula niveles de retorno a partir de parámetros GEV.
    """
    niveles_retorno = np.asarray(
        niveles_retorno,
        dtype=float,
    )

    if np.any(
        niveles_retorno <= 1
    ):
        raise ValueError(
            "Todos los periodos de retorno deben ser "
            "mayores que 1."
        )

    niveles = gev.ppf(
        1 - 1 / niveles_retorno,
        c,
        loc=loc,
        scale=scale,
    )

    return niveles


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

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
    metadata=None,
):
    """
    Construye la tabla final de resultados por estación.

    metadata es opcional y permite incorporar directamente
    información proveniente de los archivos originales CONAGUA.
    """
    tabla = pd.DataFrame(
        {
            "station": station,
            "T_years": niveles_retorno,
            "level_mm": niveles_puntuales,
            "CI_low95_bootA": low_a,
            "CI_high95_bootA": high_a,
            "bootA_naccepted": np.repeat(
                nacc_a,
                len(niveles_retorno),
            ),
            "CI_low95_bootB": low_b,
            "CI_high95_bootB": high_b,
            "bootB_naccepted": np.repeat(
                nacc_b,
                len(niveles_retorno),
            ),
            "gev_shape": np.repeat(
                c,
                len(niveles_retorno),
            ),
            "gev_loc": np.repeat(
                loc,
                len(niveles_retorno),
            ),
            "gev_scale": np.repeat(
                scale,
                len(niveles_retorno),
            ),
            "n_years": np.repeat(
                n_years,
                len(niveles_retorno),
            ),
            "trend_slope_mm_per_year": np.repeat(
                slope,
                len(niveles_retorno),
            ),
            "note": np.repeat(
                note,
                len(niveles_retorno),
            ),
        }
    )

    if metadata:
        columnas_metadata = {
            "nombre": metadata.get("nombre"),
            "estado": metadata.get("estado"),
            "municipio": metadata.get("municipio"),
            "situacion": metadata.get("situacion"),
            "latitud": metadata.get("latitud"),
            "longitud": metadata.get("longitud"),
            "altitud_msnm": metadata.get("altitud_msnm"),
            "archivo_fuente": metadata.get("archivo"),
        }

        for columna, valor in columnas_metadata.items():
            tabla[columna] = np.repeat(
                valor,
                len(tabla),
            )

    return tabla


# ============================================================
# GRÁFICA
# ============================================================

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
    station_label=None,
):
    """
    Genera y guarda el gráfico de niveles de retorno con
    intervalo de confianza robusto.

    Retorna
    -------
    advertencia : str
        Mensaje cuando algún intervalo bootstrap no contiene
        al valor puntual.
    """
    niveles_retorno = np.asarray(
        niveles_retorno,
        dtype=float,
    )

    niveles_puntuales = np.asarray(
        niveles_puntuales,
        dtype=float,
    )

    low_a = np.asarray(
        low_a,
        dtype=float,
    )

    high_a = np.asarray(
        high_a,
        dtype=float,
    )

    if not (
        len(niveles_retorno)
        == len(niveles_puntuales)
        == len(low_a)
        == len(high_a)
    ):
        raise ValueError(
            "Los vectores utilizados para generar la gráfica "
            "no tienen la misma longitud."
        )

    error_inferior_original = (
        niveles_puntuales
        - low_a
    )

    error_superior_original = (
        high_a
        - niveles_puntuales
    )

    mascara_inconsistente = (
        (error_inferior_original < 0)
        | (error_superior_original < 0)
    )

    advertencia = ""

    if np.any(
        mascara_inconsistente
    ):
        periodos_afectados = (
            niveles_retorno[
                mascara_inconsistente
            ]
        )

        periodos_texto = ", ".join(
            f"{periodo:g}"
            for periodo
            in periodos_afectados
        )

        advertencia = (
            "Advertencia gráfica: el intervalo bootstrap "
            "no contiene el nivel puntual para los periodos "
            f"de retorno: {periodos_texto} años."
        )

    error_inferior = np.maximum(
        0.0,
        error_inferior_original,
    )

    error_superior = np.maximum(
        0.0,
        error_superior_original,
    )

    error_inferior = np.nan_to_num(
        error_inferior,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    error_superior = np.nan_to_num(
        error_superior,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    yerr = np.vstack(
        [
            error_inferior,
            error_superior,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.errorbar(
        niveles_retorno,
        niveles_puntuales,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        label=(
            "GEV puntual + IC 95% Bootstrap robusto"
        ),
    )

    ax.set_xscale(
        "log"
    )

    limite_inferior = (
        niveles_retorno.min()
        * 0.9
    )

    limite_superior = (
        max(
            plot_max_t,
            niveles_retorno.max(),
        )
        * 1.1
    )

    ax.set_xlim(
        limite_inferior,
        limite_superior,
    )

    ax.set_xlabel(
        "Periodo de retorno (años)"
    )

    ax.set_ylabel(
        "Precipitación (mm)"
    )

    subtitulo = (
        f"n={n_years}, "
        f"pendiente≈{slope:.2f} mm/año, "
        f"shape={c:.3f}"
    )

    etiqueta = (
        station_label
        if station_label
        else station
    )

    ax.set_title(
        f"Niveles de retorno GEV – {etiqueta}\n"
        f"({subtitulo})"
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.6,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        png_out,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return advertencia


# ============================================================
# MOTOR INTERNO COMÚN
# ============================================================

def _procesar_dataframe_gev(
    df,
    station,
    source_path=None,
    metadata=None,
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
    Motor común para procesar una estación GEV desde un
    DataFrame ya cargado.
    """
    if niveles_retorno is None:
        niveles_retorno = np.array(
            [
                2,
                5,
                10,
                25,
                50,
                100,
            ],
            dtype=float,
        )

    niveles_retorno = np.asarray(
        niveles_retorno,
        dtype=float,
    )

    if rng is None:
        rng = np.random.default_rng()

    if fecha_tag is None:
        fecha_tag = "sin_fecha"

    df_clean = limpiar_datos_dataframe(
        df=df,
        col_fecha=col_fecha,
        col_pp=col_pp,
    )

    max_ann = extraer_maximos_anuales(
        df_clean,
        col_fecha=col_fecha,
        col_pp=col_pp,
    )

    max_ann_values = max_ann.to_numpy(
        dtype=float
    )

    n_years = len(
        max_ann_values
    )

    if n_years == 0:
        raise ValueError(
            "No se obtuvieron máximos anuales para la estación."
        )

    note = ""

    if n_years < n_min_anios:
        note = (
            f"Advertencia: solo {n_years} años "
            f"(<{n_min_anios}). Incertidumbre alta."
        )

    slope = calcular_tendencia_lineal(
        max_ann_values
    )

    c, loc, scale = ajustar_gev(
        max_ann_values
    )

    niveles_puntuales = calcular_niveles_retorno(
        c=c,
        loc=loc,
        scale=scale,
        niveles_retorno=niveles_retorno,
    )

    low_a, high_a, nacc_a = bootstrap_robusto(
        datos=max_ann_values,
        niveles_retorno=niveles_retorno,
        n_boot=n_boot,
        alpha=alpha,
        rng=rng,
        shape_bounds=(
            -0.35,
            0.35,
        ),
        max_rel_factor=10.0,
    )

    if usar_boot_parametrico:
        low_b, high_b, nacc_b = (
            bootstrap_parametrico(
                c=c,
                loc=loc,
                scale=scale,
                niveles_retorno=niveles_retorno,
                n_muestra=n_years,
                n_boot=n_boot,
                alpha=alpha,
                rng=rng,
            )
        )
    else:
        low_b = np.full_like(
            niveles_retorno,
            np.nan,
            dtype=float,
        )

        high_b = np.full_like(
            niveles_retorno,
            np.nan,
            dtype=float,
        )

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
        metadata=metadata,
    )

    os.makedirs(
        dir_out,
        exist_ok=True,
    )

    png_out = os.path.join(
        dir_out,
        (
            f"{station}_return_levels_"
            f"ROBUST_{fecha_tag}.png"
        ),
    )

    csv_out = os.path.join(
        dir_out,
        (
            f"{station}_return_levels_"
            f"ROBUST_{fecha_tag}.csv"
        ),
    )

    tabla.to_csv(
        csv_out,
        index=False,
    )

    plot_created = False
    plot_warning = ""

    station_label = station

    if metadata:
        nombre = metadata.get(
            "nombre"
        )

        if nombre:
            station_label = (
                f"{station} — {nombre}"
            )

    try:
        plot_warning = guardar_grafico_estacion(
            station=station,
            station_label=station_label,
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

        plot_created = True

    except Exception as plot_error:
        plot_warning = (
            "No fue posible generar la gráfica: "
            f"{plot_error}"
        )

        png_out = None

    meta = {
        "station": station,
        "path": source_path,
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
        "plot_created": plot_created,
        "plot_warning": plot_warning,
    }

    if metadata:
        meta.update(
            {
                "nombre": metadata.get(
                    "nombre"
                ),
                "estado": metadata.get(
                    "estado"
                ),
                "municipio": metadata.get(
                    "municipio"
                ),
                "latitud": metadata.get(
                    "latitud"
                ),
                "longitud": metadata.get(
                    "longitud"
                ),
                "altitud_msnm": metadata.get(
                    "altitud_msnm"
                ),
            }
        )

    return (
        tabla,
        meta,
    )


# ============================================================
# PROCESAR ESTACIÓN CSV
# ============================================================

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
    Procesa una estación desde un CSV preparado.

    Se conserva para compatibilidad con el flujo histórico.
    """
    base = os.path.basename(
        path_csv
    )

    station = os.path.splitext(
        base
    )[0]

    try:
        df = pd.read_csv(
            path_csv
        )

        return _procesar_dataframe_gev(
            df=df,
            station=station,
            source_path=path_csv,
            metadata=None,
            col_fecha=col_fecha,
            col_pp=col_pp,
            n_min_anios=n_min_anios,
            niveles_retorno=niveles_retorno,
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
            usar_boot_parametrico=(
                usar_boot_parametrico
            ),
            plot_max_t=plot_max_t,
            dir_out=dir_out,
            fecha_tag=fecha_tag,
        )

    except Exception as error:
        return None, {
            "station": station,
            "error": str(
                error
            ),
            "path": path_csv,
        }


# ============================================================
# PROCESAR ESTACIÓN ORIGINAL CONAGUA
# ============================================================

def procesar_estacion_conagua(
    estacion,
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
    Procesa una estación original CONAGUA previamente leída
    mediante scripts.conagua_reader.leer_estacion_conagua().

    Parámetros
    ----------
    estacion : dict
        Estructura:
        {
            "metadata": {...},
            "data": DataFrame,
            "calidad": {...},
        }

    Retorna
    -------
    tabla : pd.DataFrame o None
    meta : dict
    """
    if not isinstance(
        estacion,
        dict,
    ):
        raise TypeError(
            "La estación debe ser la estructura generada "
            "por conagua_reader.py."
        )

    metadata = estacion.get(
        "metadata"
    )

    data = estacion.get(
        "data"
    )

    if metadata is None:
        raise ValueError(
            "La estación no contiene metadatos."
        )

    if data is None:
        raise ValueError(
            "La estación no contiene datos climáticos."
        )

    station = str(
        metadata.get(
            "station",
            "",
        )
    ).strip()

    if not station:
        raise ValueError(
            "No se encontró la clave de estación."
        )

    source_path = metadata.get(
        "archivo"
    )

    try:
        return _procesar_dataframe_gev(
            df=data,
            station=station,
            source_path=source_path,
            metadata=metadata,
            col_fecha="date",
            col_pp="pp",
            n_min_anios=n_min_anios,
            niveles_retorno=niveles_retorno,
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
            usar_boot_parametrico=(
                usar_boot_parametrico
            ),
            plot_max_t=plot_max_t,
            dir_out=dir_out,
            fecha_tag=fecha_tag,
        )

    except Exception as error:
        return None, {
            "station": station,
            "nombre": metadata.get(
                "nombre"
            ),
            "estado": metadata.get(
                "estado"
            ),
            "municipio": metadata.get(
                "municipio"
            ),
            "error": str(
                error
            ),
            "path": source_path,
        }