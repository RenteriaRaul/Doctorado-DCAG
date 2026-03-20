import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from scripts.station_analysis import procesar_estacion


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
    Ejecuta el procesamiento batch de múltiples estaciones CSV:
    - busca archivos
    - procesa cada estación
    - exporta CSV maestro
    - exporta log del proceso

    Parámetros
    ----------
    dir_in : str
        Carpeta donde están los CSV de entrada.
    patron : str
        Patrón de búsqueda, por ejemplo 'dat*.csv'.
    col_fecha : str
        Nombre de la columna de fecha.
    col_pp : str
        Nombre de la columna de precipitación.
    n_min_anios : int
        Mínimo de años recomendado para GEV.
    niveles_retorno : array-like o None
        Periodos de retorno.
    n_boot : int
        Réplicas bootstrap.
    alpha : float
        Nivel de significancia.
    seed : int
        Semilla reproducible.
    usar_boot_parametrico : bool
        Si True, también calcula IC paramétricos.
    plot_max_t : int o float
        Máximo T mostrado en gráficos.
    nombre_salida_dir : str
        Nombre de la carpeta de salida dentro de dir_in.

    Retorna
    -------
    maestro : pd.DataFrame o None
    log_df : pd.DataFrame
    out_master : str o None
    out_log : str
    """
    if niveles_retorno is None:
        niveles_retorno = np.array([2, 5, 10, 25, 50, 100], dtype=float)

    dir_out = os.path.join(dir_in, nombre_salida_dir)
    os.makedirs(dir_out, exist_ok=True)

    fecha_tag = datetime.now().strftime("%Y%m%d_%H%M")
    rng = np.random.default_rng(seed)

    archivos = sorted(glob.glob(os.path.join(dir_in, patron)))
    print(f"Archivos encontrados: {len(archivos)}")

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
            usar_boot_parametrico=usar_boot_parametrico,
            plot_max_t=plot_max_t,
            dir_out=dir_out,
            fecha_tag=fecha_tag,
        )

        metas.append(meta)

        if tabla is not None:
            tablas.append(tabla)

    if tablas:
        maestro = pd.concat(tablas, ignore_index=True)
        out_master = os.path.join(dir_out, f"MASTER_return_levels_GEV_ROBUST_{fecha_tag}.csv")
        maestro.to_csv(out_master, index=False)

        print(f"\n>>> CSV maestro:\n{out_master}")
        print("\nResults:")

        cols_fmt = [
            "level_mm",
            "CI_low95_bootA",
            "CI_high95_bootA",
            "CI_low95_bootB",
            "CI_high95_bootB",
        ]
        fmts = {c: "{:.2f}".format for c in cols_fmt if c in maestro.columns}
        print(maestro.head(12).to_string(index=False, formatters=fmts))
    else:
        maestro = None
        out_master = None
        print("No se generaron tablas (revisa errores en LOG).")

    log_df = pd.DataFrame(metas)
    out_log = os.path.join(dir_out, f"log_proceso_ROBUST_{fecha_tag}.csv")
    log_df.to_csv(out_log, index=False)

    print(f"\nLog del proceso:\n{out_log}")

    return maestro, log_df, out_master, out_log
