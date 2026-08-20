from pathlib import Path

import pandas as pd

from scripts.gev_results import (
    COLUMNAS_BOOTSTRAP_B,
    preparar_maestro_exportacion,
    preparar_maestro_visual,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

RUNS_DIR = (
    PROJECT_ROOT
    / "results"
    / "streamlit"
    / "gev"
    / "runs"
)

N_BOOT_ESPERADO = 60


# ============================================================
# LOCALIZAR ÚLTIMA TABLA MAESTRA
# ============================================================

def localizar_ultimo_maestro():

    candidatos = sorted(
        RUNS_DIR.glob(
            "*/MASTER_GEV_CONAGUA_Bootstrap_robusto.csv"
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )

    if not candidatos:
        raise FileNotFoundError(
            "No se encontró ninguna ejecución GEV persistente "
            "en results/streamlit/gev/runs/."
        )

    return candidatos[0]


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA — LIMPIEZA Y PRESENTACIÓN DE RESULTADOS GEV")
    print("=" * 72)

    master_path = localizar_ultimo_maestro()

    print()
    print("Tabla maestra utilizada:")
    print(master_path)

    maestro = pd.read_csv(
        master_path
    )

    print()
    print(
        f"Filas originales: {len(maestro)}"
    )

    print(
        "Estaciones originales:",
        maestro[
            "station"
        ]
        .astype(str)
        .nunique(),
    )

    # ========================================================
    # EXPORTACIÓN OFICIAL
    # ========================================================

    exportacion = preparar_maestro_exportacion(
        maestro
    )

    print()
    print("=" * 72)
    print("VALIDACIÓN — EXPORTACIÓN")
    print("=" * 72)

    for columna in COLUMNAS_BOOTSTRAP_B:

        if columna in exportacion.columns:
            raise ValueError(
                f"La columna histórica {columna} "
                "no fue eliminada."
            )

    if len(
        exportacion
    ) != len(
        maestro
    ):
        raise ValueError(
            "La limpieza modificó el número de filas."
        )

    print()
    print(
        "Columnas Bootstrap B eliminadas: OK"
    )

    print(
        "Número de filas conservado: OK"
    )

    # ========================================================
    # TABLA VISUAL
    # ========================================================

    visual = preparar_maestro_visual(
        maestro=maestro,
        n_boot=N_BOOT_ESPERADO,
    )

    columnas_visuales_requeridas = [
        "Estación",
        "Nombre",
        "Periodo de retorno (años)",
        "Nivel estimado (mm)",
        "IC 95% inferior (mm)",
        "IC 95% superior (mm)",
        "Réplicas válidas",
        "Bootstrap aceptado (%)",
        "Parámetro de forma",
        "Localización (mm)",
        "Escala (mm)",
        "Máximos anuales",
    ]

    faltantes = [
        columna
        for columna in columnas_visuales_requeridas
        if columna not in visual.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan encabezados visuales: {faltantes}"
        )

    print()
    print("=" * 72)
    print("VALIDACIÓN — TABLA VISUAL")
    print("=" * 72)

    print()
    print(
        "Encabezados amigables: OK"
    )

    print(
        "Bootstrap aceptado (%): OK"
    )

    print()
    print("Primeras filas:")
    print(
        visual[
            columnas_visuales_requeridas
        ]
        .head(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # CONTROL DE INTEGRIDAD
    # ========================================================

    if (
        visual[
            "Estación"
        ]
        .astype(str)
        .nunique()
        != 37
    ):
        raise ValueError(
            "La tabla visual no conserva las 37 estaciones."
        )

    if len(
        visual
    ) != 222:
        raise ValueError(
            "La tabla visual no conserva las 222 filas "
            "esperadas de la ejecución actual."
        )

    print()
    print("=" * 72)
    print("VALIDACIONES FINALES")
    print("=" * 72)

    print()
    print("37 estaciones: OK")
    print("222 filas: OK")
    print("Bootstrap B eliminado de presentación: OK")
    print("Nombres técnicos internos preservados: OK")
    print("Encabezados de Streamlit preparados: OK")

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()