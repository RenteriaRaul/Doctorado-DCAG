
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import pandas as pd

from scripts.sustax.projection import build_climate_change_projection


BASE = Path("results/streamlit/sustax/BY_SCENARIO")
PRUEBAS = Path("results/tests/sustax_validacion_general")

DIAGNOSTICO = PRUEBAS / "diagnostico_series.csv"
INDEPENDIENTES = PRUEBAS / "indicadores_independientes.csv"
REPORTE = PRUEBAS / "comparacion_motor_projection.csv"

ESCENARIOS = [
    "SSP119", "SSP126", "SSP245", "SSP370",
    "SSP434", "SSP460", "SSP585"
]

INDICADORES = [
    "p95_daily_mm",
    "p99_daily_mm",
    "rx1day_mm",
    "r50mm_days_per_year",
]

TOLERANCIA = 1e-8


def comparar_numero(observado, esperado):
    if pd.isna(observado) and pd.isna(esperado):
        return True, np.nan

    if pd.isna(observado) or pd.isna(esperado):
        return False, np.nan

    diferencia = abs(float(observado) - float(esperado))

    return (
        bool(np.isclose(
            observado, esperado,
            rtol=0,
            atol=TOLERANCIA
        )),
        diferencia
    )


def main():
    for ruta in [BASE, DIAGNOSTICO, INDEPENDIENTES]:
        if not ruta.exists():
            raise FileNotFoundError(ruta)

    diagnostico = pd.read_csv(DIAGNOSTICO)
    independiente = pd.read_csv(INDEPENDIENTES)

    completas = diagnostico.loc[
        diagnostico["estado"] == "VALIDADO_INDEPENDIENTE",
        ["punto", "escenario"]
    ].copy()

    if len(completas) != 63:
        raise AssertionError(
            f"Se esperaban 63 series completas: {len(completas)}"
        )

    if len(independiente) != 252:
        raise AssertionError(
            f"Se esperaban 252 indicadores: {len(independiente)}"
        )

    if independiente.duplicated(
        ["punto", "escenario", "indicador"]
    ).any():
        raise AssertionError(
            "Hay indicadores independientes duplicados"
        )

    print("Ejecutando motor científico projection.py...")

    with TemporaryDirectory(
        prefix="sustax_comparacion_"
    ) as temporal:

        motor, _, _ = build_climate_change_projection(
            byscen_total_folder=str(BASE),
            out_folder=temporal,
            ssp_list=ESCENARIOS,
            historical_period=(1981, 2010),
            future_period=(2026, 2055),
            threshold_mm=50.0,
            require_complete_period=True,
        )

    if motor.duplicated(
        ["sustax_total", "scenario"]
    ).any():
        raise AssertionError(
            "El motor produjo puntos y escenarios duplicados"
        )

    motor = motor.set_index(
        ["sustax_total", "scenario"]
    )

    registros = []

    for fila in independiente.itertuples(index=False):
        punto = fila.punto
        escenario = fila.escenario
        indicador = fila.indicador

        clave = (punto, escenario)

        if clave not in motor.index:
            raise AssertionError(
                f"El motor no devolvió {clave}"
            )

        resultado = motor.loc[clave]

        columnas = {
            "historico": f"hist_{indicador}",
            "futuro": f"future_{indicador}",
            "cambio_absoluto": f"delta_{indicador}",
            "cambio_porcentual": f"pct_change_{indicador}",
        }

        for magnitud, columna in columnas.items():
            if columna not in motor.columns:
                raise AssertionError(
                    f"Falta columna del motor: {columna}"
                )

            esperado = getattr(fila, magnitud)
            obtenido = resultado[columna]

            coincide, diferencia = comparar_numero(
                obtenido, esperado
            )

            registros.append({
                "punto": punto,
                "escenario": escenario,
                "indicador": indicador,
                "magnitud": magnitud,
                "valor_independiente": esperado,
                "valor_motor": obtenido,
                "diferencia_absoluta": diferencia,
                "coincide": coincide,
            })

    comparacion = pd.DataFrame(registros)

    # Verificar que los 12 puntos incompletos
    # no generaron cambios bajo el modo estricto.
    incompletas = diagnostico.loc[
        diagnostico["estado"] == "COBERTURA_INSUFICIENTE"
    ]

    errores_cobertura = []

    for fila in incompletas.itertuples(index=False):
        clave = (fila.punto, fila.escenario)

        if clave not in motor.index:
            continue

        resultado = motor.loc[clave]

        if bool(resultado["change_computed"]):
            errores_cobertura.append(clave)

    PRUEBAS.mkdir(parents=True, exist_ok=True)
    comparacion.to_csv(REPORTE, index=False)

    diferencias = comparacion.loc[
        ~comparacion["coincide"]
    ]

    print("\nCOMPARACIÓN AUTOMATIZADA SUSTAX")
    print("=" * 60)
    print("Series completas:", len(completas))
    print("Indicadores:", len(independiente))
    print("Comparaciones numéricas:", len(comparacion))
    print("Coincidencias:", int(comparacion["coincide"].sum()))
    print("Diferencias:", len(diferencias))
    print("Incompletas con cambios calculados:",
          len(errores_cobertura))

    print("\nCOMPARACIONES POR ESCENARIO")
    resumen = comparacion.groupby("escenario").agg(
        comparaciones=("coincide", "size"),
        coincidencias=("coincide", "sum")
    )
    print(resumen.to_string())

    print("\nReporte:", REPORTE.resolve())

    if not diferencias.empty:
        print("\nPRIMERAS DIFERENCIAS")
        print(diferencias.head(15).to_string(index=False))

    if errores_cobertura:
        print("\nERRORES DE COBERTURA")
        print(errores_cobertura)

    if len(comparacion) != 1008:
        raise AssertionError(
            "Se esperaban 1008 comparaciones numéricas"
        )

    if not diferencias.empty or errores_cobertura:
        print("\nRESULTADO: REVISIÓN NECESARIA")
        sys.exit(1)

    print("\nRESULTADO: COMPARACIÓN SATISFACTORIA")


if __name__ == "__main__":
    main()
