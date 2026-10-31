from pathlib import Path
import shutil

import pandas as pd

from scripts.sustax.merge import build_by_scenario_total


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_LONG = Path(
    r"G:\My Drive\Doctorado-DCAG\results\tests\sustax_merge"
)

CARPETA_SALIDA = Path(
    r"G:\My Drive\Doctorado-DCAG\results\tests\sustax_by_scenario"
)


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 80)
    print("PRUEBA — SEPARACIÓN DE ESCENARIOS SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Verificar entrada
    # --------------------------------------------------------

    if not CARPETA_LONG.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta LONG:\n{CARPETA_LONG}"
        )

    archivos_long = sorted(
        CARPETA_LONG.glob("*Total_LONG_mm.csv")
    )

    print("\n" + "-" * 80)
    print("1. ARCHIVOS LONG DE ENTRADA")
    print("-" * 80)

    print(f"Archivos LONG encontrados: {len(archivos_long)}")

    if len(archivos_long) != 14:
        print(
            "ADVERTENCIA: se esperaban 14 archivos LONG "
            "para esta prueba."
        )

    for archivo in archivos_long:
        print(f"  - {archivo.name}")

    if not archivos_long:
        raise RuntimeError(
            "No existen archivos LONG para procesar."
        )

    # --------------------------------------------------------
    # 2. Preparar salida limpia
    # --------------------------------------------------------

    if CARPETA_SALIDA.exists():
        shutil.rmtree(CARPETA_SALIDA)

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3. Ejecutar separación por escenarios
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. EJECUCIÓN")
    print("-" * 80)

    summary_df, summary_path = build_by_scenario_total(
        long_total_dir=str(CARPETA_LONG),
        out_dir=str(CARPETA_SALIDA),
    )

    print("Separación por escenarios ejecutada.")
    print(f"Resumen generado: {summary_path}")

    # --------------------------------------------------------
    # 4. Productos
    # --------------------------------------------------------

    archivos_csv = sorted(
        CARPETA_SALIDA.glob("*.csv")
    )

    archivos_escenario = [
        fp
        for fp in archivos_csv
        if fp.name != "SUSTAX_TOTAL_BYSCEN_summary.csv"
    ]

    print("\n" + "-" * 80)
    print("3. PRODUCTOS GENERADOS")
    print("-" * 80)

    print(
        f"Archivos de escenario: {len(archivos_escenario)}"
    )

    print(
        f"CSV totales incluyendo resumen: {len(archivos_csv)}"
    )

    for archivo in archivos_escenario:
        print(f"  - {archivo.name}")

    # --------------------------------------------------------
    # 5. Revisar resumen
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. RESUMEN POR PUNTO")
    print("-" * 80)

    if summary_df.empty:
        raise RuntimeError(
            "El resumen BY_SCENARIO está vacío."
        )

    columnas_resumen = [
        c
        for c in [
            "sustax_total",
            "lat",
            "lon",
            "n_scenarios_generated",
        ]
        if c in summary_df.columns
    ]

    print(
        summary_df[columnas_resumen].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 6. Inspección individual
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. INSPECCIÓN DE SERIES")
    print("-" * 80)

    registros = []
    escenarios_detectados = set()

    for archivo in archivos_escenario:

        df = pd.read_csv(archivo)

        nombre = archivo.name

        escenario = None

        for sc in [
            "ERA5",
            "SSP119",
            "SSP126",
            "SSP245",
            "SSP370",
            "SSP434",
            "SSP460",
            "SSP585",
        ]:
            if f"__{sc}__" in nombre:
                escenario = sc
                break

        if escenario:
            escenarios_detectados.add(escenario)

        if "date" in df.columns:
            fechas = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

            fecha_inicio = fechas.min()
            fecha_fin = fechas.max()

        else:
            fecha_inicio = None
            fecha_fin = None

        columnas_precipitacion = [
            c
            for c in df.columns
            if "Precipitation" in str(c)
        ]

        if columnas_precipitacion:

            valores = pd.to_numeric(
                df[columnas_precipitacion[0]],
                errors="coerce",
            )

            minimo = valores.min()
            media = valores.mean()
            maximo = valores.max()
            negativos = int((valores < 0).sum())

        else:

            minimo = None
            media = None
            maximo = None
            negativos = None

        registros.append(
            {
                "archivo": nombre,
                "escenario": escenario,
                "filas": len(df),
                "inicio": (
                    fecha_inicio.date()
                    if pd.notna(fecha_inicio)
                    else None
                ),
                "fin": (
                    fecha_fin.date()
                    if pd.notna(fecha_fin)
                    else None
                ),
                "min_mm": minimo,
                "media_mm": media,
                "max_mm": maximo,
                "negativos": negativos,
            }
        )

    inspeccion_df = pd.DataFrame(registros)

    print(
        inspeccion_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. Conteo por escenario
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. CONTEO POR ESCENARIO")
    print("-" * 80)

    conteo = (
        inspeccion_df["escenario"]
        .value_counts()
        .sort_index()
    )

    for escenario, cantidad in conteo.items():
        print(
            f"{escenario:<10} {cantidad:>3} archivo(s)"
        )

    # --------------------------------------------------------
    # 8. Validaciones
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("7. VALIDACIÓN")
    print("-" * 80)

    errores = []

    if len(summary_df) != 14:
        errores.append(
            f"El resumen contiene {len(summary_df)} puntos; "
            "se esperaban 14."
        )

    if len(archivos_escenario) != 38:
        errores.append(
            f"Se generaron {len(archivos_escenario)} archivos "
            "de escenario; se esperaban 38."
        )

    esperado = {
        "ERA5": 14,
        "SSP245": 12,
        "SSP370": 12,
    }

    conteo_dict = conteo.to_dict()

    for escenario, cantidad_esperada in esperado.items():

        cantidad_real = int(
            conteo_dict.get(
                escenario,
                0,
            )
        )

        if cantidad_real != cantidad_esperada:
            errores.append(
                f"{escenario}: {cantidad_real} archivos; "
                f"se esperaban {cantidad_esperada}."
            )

    negativos_totales = (
        inspeccion_df["negativos"]
        .fillna(0)
        .sum()
    )

    if negativos_totales > 0:
        errores.append(
            f"Se encontraron {int(negativos_totales)} "
            "valores negativos."
        )

    if errores:

        print("SE DETECTARON OBSERVACIONES:")

        for error in errores:
            print(f"  - {error}")

    else:

        print("OK: 14 puntos Sustax conservados.")
        print("OK: 38 series por escenario generadas.")
        print("OK: ERA5 presente en 14 puntos.")
        print("OK: SSP245 presente en 12 puntos.")
        print("OK: SSP370 presente en 12 puntos.")
        print("OK: no se detectaron valores negativos.")

    print("\nEscenarios detectados:")
    print(
        ", ".join(
            sorted(escenarios_detectados)
        )
    )

    print("\n" + "=" * 80)

    if errores:
        print("PRUEBA FINALIZADA CON OBSERVACIONES")
    else:
        print("PRUEBA SUPERADA CORRECTAMENTE")

    print("=" * 80)


if __name__ == "__main__":
    main()