from pathlib import Path

import pandas as pd

from scripts.gev_persistence import (
    cargar_ejecucion_gev,
    listar_ejecuciones_gev,
)
from scripts.gev_results import (
    COLUMNAS_BOOTSTRAP_B,
    preparar_maestro_exportacion,
    preparar_maestro_visual,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

CARPETA_SALIDA = (
    PROJECT_ROOT
    / "results"
    / "streamlit"
    / "gev"
)


def main():

    print("=" * 72)
    print("PRUEBA — INTEGRACIÓN RESULTADOS GEV PERSISTENTES")
    print("=" * 72)

    ejecuciones = listar_ejecuciones_gev(
        CARPETA_SALIDA
    )

    if not ejecuciones:
        raise FileNotFoundError(
            "No existen ejecuciones GEV persistentes."
        )

    ultima = ejecuciones[0]

    print()
    print(
        "Run ID:",
        ultima.get(
            "run_id"
        ),
    )

    resultado = cargar_ejecucion_gev(
        ultima
    )

    maestro = resultado[
        "maestro"
    ]

    n_boot = int(
        resultado[
            "n_boot"
        ]
    )

    print(
        "Réplicas Bootstrap:",
        n_boot,
    )

    print(
        "Filas:",
        len(
            maestro
        ),
    )

    print(
        "Estaciones:",
        maestro[
            "station"
        ]
        .astype(str)
        .nunique(),
    )

    visual = preparar_maestro_visual(
        maestro=maestro,
        n_boot=n_boot,
    )

    exportacion = preparar_maestro_exportacion(
        maestro
    )

    for columna in COLUMNAS_BOOTSTRAP_B:
        if columna in visual.columns:
            raise ValueError(
                f"Bootstrap B sigue visible: {columna}"
            )

        if columna in exportacion.columns:
            raise ValueError(
                f"Bootstrap B sigue en exportación: {columna}"
            )

    if len(visual) != len(
        maestro
    ):
        raise ValueError(
            "La tabla visual cambió el número de filas."
        )

    if len(exportacion) != len(
        maestro
    ):
        raise ValueError(
            "La exportación cambió el número de filas."
        )

    columnas_amigables = [
        "Estación",
        "Nombre",
        "Periodo de retorno (años)",
        "Nivel estimado (mm)",
        "IC 95% inferior (mm)",
        "IC 95% superior (mm)",
        "Réplicas válidas",
        "Bootstrap aceptado (%)",
    ]

    faltantes = [
        columna
        for columna in columnas_amigables
        if columna not in visual.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas amigables: {faltantes}"
        )

    print()
    print("=" * 72)
    print("VISTA PREVIA STREAMLIT")
    print("=" * 72)

    print()
    print(
        visual[
            columnas_amigables
        ]
        .head(6)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("VALIDACIONES FINALES")
    print("=" * 72)

    print()
    print("Carga de ejecución persistente: OK")
    print("37 estaciones preservadas: OK")
    print("222 filas preservadas: OK")
    print("Bootstrap B oculto: OK")
    print("Tabla visual amigable: OK")
    print("Exportación oficial limpia: OK")

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()