import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(valor):
    """Convierte tipos comunes de numpy/pandas a valores serializables."""
    if isinstance(valor, np.integer):
        return int(valor)

    if isinstance(valor, np.floating):
        return float(valor)

    if isinstance(valor, np.ndarray):
        return valor.tolist()

    if isinstance(valor, Path):
        return str(valor)

    if isinstance(valor, (pd.Timestamp, datetime)):
        return valor.isoformat()

    raise TypeError(
        f"Tipo no serializable en JSON: {type(valor)}"
    )


def guardar_ejecucion_gev(
    maestro,
    log_df,
    metadata_df,
    carpeta_salida,
    carpeta_fuente,
    patron,
    n_boot,
    alpha,
    seed,
    n_min_anios,
    niveles_retorno,
):
    """
    Guarda de forma persistente una ejecución GEV.

    Estructura creada
    -----------------
    carpeta_salida/
        latest_run.json
        runs/
            YYYYMMDD_HHMMSS/
                MASTER_GEV_CONAGUA_Bootstrap_robusto.csv
                log_GEV_CONAGUA.csv
                metadata_estaciones_CONAGUA.csv
                run_metadata.json
    """
    carpeta_salida = Path(
        carpeta_salida
    )

    runs_dir = (
        carpeta_salida
        / "runs"
    )

    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fecha = datetime.now()

    run_id = fecha.strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        runs_dir
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_path = (
        run_dir
        / "MASTER_GEV_CONAGUA_Bootstrap_robusto.csv"
    )

    log_path = (
        run_dir
        / "log_GEV_CONAGUA.csv"
    )

    metadata_path = (
        run_dir
        / "metadata_estaciones_CONAGUA.csv"
    )

    config_path = (
        run_dir
        / "run_metadata.json"
    )

    maestro.to_csv(
        master_path,
        index=False,
        encoding="utf-8-sig",
    )

    if (
        log_df is not None
        and not log_df.empty
    ):
        log_df.to_csv(
            log_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(
            log_path,
            index=False,
            encoding="utf-8-sig",
        )

    if (
        metadata_df is not None
        and not metadata_df.empty
    ):
        metadata_df.to_csv(
            metadata_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(
            metadata_path,
            index=False,
            encoding="utf-8-sig",
        )

    estaciones = int(
        maestro["station"]
        .astype(str)
        .nunique()
    )

    config = {
        "run_id": run_id,
        "fecha_ejecucion": fecha.isoformat(),
        "carpeta_fuente": str(
            carpeta_fuente
        ),
        "patron": patron,
        "estaciones_procesadas": estaciones,
        "n_boot": int(
            n_boot
        ),
        "alpha": float(
            alpha
        ),
        "confianza": float(
            1.0 - alpha
        ),
        "seed": int(
            seed
        ),
        "n_min_anios": int(
            n_min_anios
        ),
        "niveles_retorno": [
            float(valor)
            for valor
            in niveles_retorno
        ],
        "metodo_incertidumbre": (
            "Bootstrap robusto"
        ),
        "master_path": str(
            master_path
        ),
        "log_path": str(
            log_path
        ),
        "metadata_path": str(
            metadata_path
        ),
        "run_dir": str(
            run_dir
        ),
    }

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    latest_path = (
        carpeta_salida
        / "latest_run.json"
    )

    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_path": str(
                    config_path
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return config


def listar_ejecuciones_gev(
    carpeta_salida,
):
    """
    Lista las ejecuciones GEV guardadas, de la más reciente
    a la más antigua.
    """
    carpeta_salida = Path(
        carpeta_salida
    )

    runs_dir = (
        carpeta_salida
        / "runs"
    )

    if not runs_dir.exists():
        return []

    ejecuciones = []

    for config_path in runs_dir.glob(
        "*/run_metadata.json"
    ):
        try:
            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

            config[
                "_config_path"
            ] = str(
                config_path
            )

            ejecuciones.append(
                config
            )

        except Exception:
            continue

    ejecuciones.sort(
        key=lambda item: item.get(
            "fecha_ejecucion",
            "",
        ),
        reverse=True,
    )

    return ejecuciones


def cargar_ejecucion_gev(
    config,
):
    """
    Carga una ejecución GEV guardada y devuelve la misma
    estructura principal usada por st.session_state.
    """
    master_path = Path(
        config[
            "master_path"
        ]
    )

    log_path = Path(
        config[
            "log_path"
        ]
    )

    metadata_path = Path(
        config[
            "metadata_path"
        ]
    )

    if not master_path.exists():
        raise FileNotFoundError(
            "No se encontró la tabla maestra de la "
            "ejecución GEV seleccionada."
        )

    maestro = pd.read_csv(
        master_path
    )

    if log_path.exists():
        try:
            log_df = pd.read_csv(
                log_path
            )
        except pd.errors.EmptyDataError:
            log_df = pd.DataFrame()
    else:
        log_df = pd.DataFrame()

    if metadata_path.exists():
        try:
            metadata_df = pd.read_csv(
                metadata_path
            )
        except pd.errors.EmptyDataError:
            metadata_df = pd.DataFrame()
    else:
        metadata_df = pd.DataFrame()

    return {
        "maestro": maestro,
        "log_df": log_df,
        "out_master": str(
            master_path
        ),
        "out_log": str(
            log_path
        ),
        "metadata_df": metadata_df,
        "dir_in": config.get(
            "carpeta_fuente"
        ),
        "n_archivos": int(
            config.get(
                "estaciones_procesadas",
                maestro[
                    "station"
                ].nunique(),
            )
        ),
        "n_boot": int(
            config.get(
                "n_boot",
                500,
            )
        ),
        "alpha": float(
            config.get(
                "alpha",
                0.05,
            )
        ),
        "niveles_retorno": (
            config.get(
                "niveles_retorno",
                []
            )
        ),
        "n_min_anios": int(
            config.get(
                "n_min_anios",
                10,
            )
        ),
        "seed": int(
            config.get(
                "seed",
                42,
            )
        ),
        "run_id": config.get(
            "run_id"
        ),
        "fecha_ejecucion": config.get(
            "fecha_ejecucion"
        ),
        "persistent": True,
    }


def formato_fecha_ejecucion(
    fecha_iso,
):
    """
    Formatea una fecha ISO para mostrarla al usuario.
    """
    if not fecha_iso:
        return "Fecha no disponible"

    try:
        fecha = datetime.fromisoformat(
            fecha_iso
        )

        return fecha.strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    except Exception:
        return str(
            fecha_iso
        )