import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.conagua_reader import (
    leer_lote_conagua,
)
from scripts.station_analysis import (
    procesar_estacion,
    procesar_estacion_conagua,
)


# ============================================================
# UTILIDADES INTERNAS
# ============================================================

def _niveles_retorno_default():
    """
    Periodos de retorno predeterminados del proyecto.
    """
    return np.array(
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


def _validar_niveles_retorno(
    niveles_retorno,
):
    """
    Valida y normaliza los periodos de retorno.
    """
    if niveles_retorno is None:
        return _niveles_retorno_default()

    niveles = np.asarray(
        niveles_retorno,
        dtype=float,
    )

    if niveles.ndim != 1:
        raise ValueError(
            "Los periodos de retorno deben ser un arreglo "
            "unidimensional."
        )

    if len(niveles) == 0:
        raise ValueError(
            "Debe indicar al menos un periodo de retorno."
        )

    if not np.isfinite(
        niveles
    ).all():
        raise ValueError(
            "Los periodos de retorno deben ser finitos."
        )

    if np.any(
        niveles <= 1
    ):
        raise ValueError(
            "Todos los periodos de retorno deben ser "
            "mayores que 1."
        )

    return np.unique(
        niveles
    )


def _validar_parametros_batch(
    n_min_anios,
    n_boot,
    alpha,
):
    """
    Valida los parámetros generales del procesamiento batch.
    """
    n_min_anios = int(
        n_min_anios
    )

    n_boot = int(
        n_boot
    )

    alpha = float(
        alpha
    )

    if n_min_anios < 2:
        raise ValueError(
            "n_min_anios debe ser al menos 2."
        )

    if n_boot <= 0:
        raise ValueError(
            "n_boot debe ser mayor que 0."
        )

    if not (
        0 < alpha < 1
    ):
        raise ValueError(
            "alpha debe estar entre 0 y 1."
        )

    return (
        n_min_anios,
        n_boot,
        alpha,
    )


def _guardar_maestro_y_log(
    tablas,
    metas,
    dir_out,
    fecha_tag,
    prefijo_master,
    prefijo_log,
):
    """
    Consolida y exporta la tabla maestra y el log.

    Retorna
    -------
    maestro : pd.DataFrame o None
    log_df : pd.DataFrame
    out_master : str o None
    out_log : str
    """
    os.makedirs(
        dir_out,
        exist_ok=True,
    )

    if tablas:

        maestro = pd.concat(
            tablas,
            ignore_index=True,
        )

        out_master = os.path.join(
            dir_out,
            (
                f"{prefijo_master}_"
                f"{fecha_tag}.csv"
            ),
        )

        maestro.to_csv(
            out_master,
            index=False,
        )

    else:

        maestro = None
        out_master = None

    log_df = pd.DataFrame(
        metas
    )

    out_log = os.path.join(
        dir_out,
        (
            f"{prefijo_log}_"
            f"{fecha_tag}.csv"
        ),
    )

    log_df.to_csv(
        out_log,
        index=False,
    )

    return (
        maestro,
        log_df,
        out_master,
        out_log,
    )


# ============================================================
# BATCH HISTÓRICO DE CSV
# ============================================================

def ejecutar_proceso_batch(
    dir_in,
    patron="dat*.csv",
    col_fecha="date",
    col_pp="pp",
    n_min_anios=10,
    niveles_retorno=None,
    n_boot=500,
    alpha=0.05,
    seed=42,
    usar_boot_parametrico=True,
    plot_max_t=100,
    nombre_salida_dir="_salidas_return_levels_robusto",
):
    """
    Ejecuta el procesamiento batch de múltiples estaciones CSV.

    Esta función se conserva para compatibilidad con el flujo
    histórico basado en archivos dat*.csv.
    """
    niveles_retorno = _validar_niveles_retorno(
        niveles_retorno
    )

    (
        n_min_anios,
        n_boot,
        alpha,
    ) = _validar_parametros_batch(
        n_min_anios=n_min_anios,
        n_boot=n_boot,
        alpha=alpha,
    )

    if not os.path.isdir(
        dir_in
    ):
        raise NotADirectoryError(
            "La ruta indicada no corresponde a una "
            f"carpeta válida: {dir_in}"
        )

    dir_out = os.path.join(
        dir_in,
        nombre_salida_dir,
    )

    os.makedirs(
        dir_out,
        exist_ok=True,
    )

    fecha_tag = datetime.now().strftime(
        "%Y%m%d_%H%M"
    )

    rng = np.random.default_rng(
        int(seed)
    )

    archivos = sorted(
        glob.glob(
            os.path.join(
                dir_in,
                patron,
            )
        )
    )

    print(
        f"Archivos encontrados: {len(archivos)}"
    )

    tablas = []
    metas = []

    for path_csv in archivos:

        tabla, meta = procesar_estacion(
            path_csv=path_csv,
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

        metas.append(
            meta
        )

        if tabla is not None:
            tablas.append(
                tabla
            )

    (
        maestro,
        log_df,
        out_master,
        out_log,
    ) = _guardar_maestro_y_log(
        tablas=tablas,
        metas=metas,
        dir_out=dir_out,
        fecha_tag=fecha_tag,
        prefijo_master=(
            "MASTER_return_levels_GEV_ROBUST"
        ),
        prefijo_log=(
            "log_proceso_ROBUST"
        ),
    )

    if maestro is not None:

        print(
            f"\n>>> CSV maestro:\n{out_master}"
        )

        print(
            "\nResults:"
        )

        cols_fmt = [
            "level_mm",
            "CI_low95_bootA",
            "CI_high95_bootA",
            "CI_low95_bootB",
            "CI_high95_bootB",
        ]

        fmts = {
            columna: "{:.2f}".format
            for columna in cols_fmt
            if columna in maestro.columns
        }

        print(
            maestro
            .head(12)
            .to_string(
                index=False,
                formatters=fmts,
            )
        )

    else:

        print(
            "No se generaron tablas "
            "(revisa errores en LOG)."
        )

    print(
        f"\nLog del proceso:\n{out_log}"
    )

    return (
        maestro,
        log_df,
        out_master,
        out_log,
    )


# ============================================================
# BATCH DE ARCHIVOS ORIGINALES CONAGUA
# ============================================================

def ejecutar_proceso_batch_conagua(
    dir_in,
    patron="*.xlsx",
    n_min_anios=10,
    niveles_retorno=None,
    n_boot=500,
    alpha=0.05,
    seed=42,
    usar_boot_parametrico=True,
    plot_max_t=100,
    nombre_salida_dir="_salidas_return_levels_gev_conagua",
):
    """
    Ejecuta el procesamiento GEV por lote directamente sobre
    archivos originales de estaciones CONAGUA.

    Flujo
    -----
    Excel original CONAGUA
        -> conagua_reader.py
        -> serie diaria normalizada (date, pp)
        -> máximos anuales
        -> ajuste GEV
        -> niveles de retorno
        -> Bootstrap robusto
        -> Bootstrap paramétrico opcional
        -> CSV/PNG por estación
        -> tabla maestra y log

    No requiere:
    - archivos dat*.csv intermedios;
    - archivo auxiliar de coordenadas.

    Parámetros
    ----------
    dir_in : str o Path
        Carpeta con archivos originales CONAGUA.

    patron : str
        Patrón de búsqueda, normalmente "*.xlsx".

    n_min_anios : int
        Mínimo recomendado de máximos anuales. Las estaciones
        con menos años no se eliminan automáticamente: quedan
        marcadas con una advertencia.

    niveles_retorno : array-like o None
        Periodos de retorno. Por defecto:
        [2, 5, 10, 25, 50, 100].

    n_boot : int
        Número de réplicas bootstrap.

    alpha : float
        Nivel de significancia. 0.05 corresponde a IC 95%.

    seed : int
        Semilla reproducible.

    usar_boot_parametrico : bool
        Si True, también calcula Bootstrap B paramétrico.

    plot_max_t : int o float
        Máximo periodo de retorno mostrado en la gráfica.

    nombre_salida_dir : str
        Carpeta de resultados creada dentro de dir_in.

    Retorna
    -------
    maestro : pd.DataFrame o None
        Tabla maestra de niveles de retorno.

    log_df : pd.DataFrame
        Log integrado de lectura y procesamiento.

    out_master : str o None
        Ruta al CSV maestro.

    out_log : str
        Ruta al log.

    metadata_df : pd.DataFrame
        Metadatos de las estaciones CONAGUA leídas.
    """
    niveles_retorno = _validar_niveles_retorno(
        niveles_retorno
    )

    (
        n_min_anios,
        n_boot,
        alpha,
    ) = _validar_parametros_batch(
        n_min_anios=n_min_anios,
        n_boot=n_boot,
        alpha=alpha,
    )

    dir_in = str(
        dir_in
    )

    if not os.path.isdir(
        dir_in
    ):
        raise NotADirectoryError(
            "La ruta indicada no corresponde a una "
            f"carpeta válida: {dir_in}"
        )

    dir_out = os.path.join(
        dir_in,
        nombre_salida_dir,
    )

    os.makedirs(
        dir_out,
        exist_ok=True,
    )

    fecha_tag = datetime.now().strftime(
        "%Y%m%d_%H%M"
    )

    rng = np.random.default_rng(
        int(seed)
    )

    # --------------------------------------------------------
    # LEER ARCHIVOS ORIGINALES CONAGUA
    # --------------------------------------------------------

    (
        estaciones,
        metadata_df,
        log_lectura,
    ) = leer_lote_conagua(
        carpeta=dir_in,
        patron=patron,
    )

    print(
        f"Estaciones CONAGUA compatibles: "
        f"{len(estaciones)}"
    )

    tablas = []
    metas = []

    # --------------------------------------------------------
    # PROCESAR CADA ESTACIÓN
    # --------------------------------------------------------

    for estacion in estaciones:

        tabla, meta = procesar_estacion_conagua(
            estacion=estacion,
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

        if "error" in meta:

            meta["status"] = "error"

        else:

            meta["status"] = "ok"

        metas.append(
            meta
        )

        if tabla is not None:
            tablas.append(
                tabla
            )

    # --------------------------------------------------------
    # AÑADIR AL LOG ARCHIVOS NO COMPATIBLES
    # --------------------------------------------------------

    if (
        log_lectura is not None
        and not log_lectura.empty
    ):

        no_compatibles = log_lectura[
            log_lectura[
                "status"
            ] != "ok"
        ].copy()

        for fila in no_compatibles.itertuples():

            metas.append(
                {
                    "station": getattr(
                        fila,
                        "station",
                        None,
                    ),
                    "path": getattr(
                        fila,
                        "archivo",
                        None,
                    ),
                    "status": "incompatible",
                    "error": getattr(
                        fila,
                        "mensaje",
                        (
                            "Archivo no compatible con "
                            "el formato CONAGUA."
                        ),
                    ),
                    "plot_created": False,
                }
            )

    # --------------------------------------------------------
    # CONSOLIDAR Y EXPORTAR
    # --------------------------------------------------------

    (
        maestro,
        log_df,
        out_master,
        out_log,
    ) = _guardar_maestro_y_log(
        tablas=tablas,
        metas=metas,
        dir_out=dir_out,
        fecha_tag=fecha_tag,
        prefijo_master=(
            "MASTER_return_levels_GEV_CONAGUA_ROBUST"
        ),
        prefijo_log=(
            "log_proceso_GEV_CONAGUA_ROBUST"
        ),
    )

    # --------------------------------------------------------
    # SALIDA DE CONTROL
    # --------------------------------------------------------

    if maestro is not None:

        estaciones_exitosas = (
            maestro[
                "station"
            ]
            .astype(str)
            .nunique()
        )

        print(
            f"\nEstaciones con resultados GEV: "
            f"{estaciones_exitosas}"
        )

        print(
            f"\n>>> CSV maestro:\n"
            f"{out_master}"
        )

        print(
            "\nPrimeros resultados:"
        )

        columnas_control = [
            columna
            for columna in [
                "station",
                "nombre",
                "T_years",
                "level_mm",
                "CI_low95_bootA",
                "CI_high95_bootA",
                "n_years",
                "gev_shape",
                "gev_loc",
                "gev_scale",
            ]
            if columna in maestro.columns
        ]

        control = (
            maestro[
                columnas_control
            ]
            .head(12)
            .copy()
        )

        columnas_numericas = [
            columna
            for columna in [
                "level_mm",
                "CI_low95_bootA",
                "CI_high95_bootA",
                "gev_shape",
                "gev_loc",
                "gev_scale",
            ]
            if columna in control.columns
        ]

        for columna in columnas_numericas:

            control[
                columna
            ] = pd.to_numeric(
                control[
                    columna
                ],
                errors="coerce",
            ).round(
                4
            )

        print(
            control.to_string(
                index=False
            )
        )

    else:

        print(
            "No se generaron resultados GEV. "
            "Revise el log."
        )

    print(
        f"\nLog del proceso:\n"
        f"{out_log}"
    )

    return (
        maestro,
        log_df,
        out_master,
        out_log,
        metadata_df,
    )