from pathlib import Path

import pandas as pd

from scripts.sustax.inventory import (
    list_sustax_csv_files,
    build_sustax_inventory,
    build_group_summary,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\Datos Sustax\Sustax Original"
)


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 80)
    print("PRUEBA — CARPETA COMPLETA DE DESCARGAS ORIGINALES SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Verificar carpeta
    # --------------------------------------------------------

    if not CARPETA.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta:\n{CARPETA}"
        )

    if not CARPETA.is_dir():
        raise NotADirectoryError(
            f"La ruta no corresponde a una carpeta:\n{CARPETA}"
        )

    print(f"\nCarpeta:\n{CARPETA}")

    # --------------------------------------------------------
    # 2. Contar todos los CSV físicos
    # --------------------------------------------------------

    csv_totales = sorted(CARPETA.glob("*.csv"))

    print("\n" + "-" * 80)
    print("1. ARCHIVOS CSV")
    print("-" * 80)

    print(f"CSV encontrados físicamente: {len(csv_totales)}")

    # --------------------------------------------------------
    # 3. Detectar exportaciones originales Sustax
    # --------------------------------------------------------

    archivos_sustax = list_sustax_csv_files(
        str(CARPETA)
    )

    print(
        f"Exportaciones originales Sustax detectadas: "
        f"{len(archivos_sustax)}"
    )

    # --------------------------------------------------------
    # 4. Construir inventario
    # --------------------------------------------------------

    inventory_df, bad_df = build_sustax_inventory(
        str(CARPETA)
    )

    print("\n" + "-" * 80)
    print("2. LECTURA DEL INVENTARIO")
    print("-" * 80)

    print(
        f"Archivos leídos correctamente: "
        f"{len(inventory_df)}"
    )

    print(
        f"Archivos con error de lectura: "
        f"{len(bad_df)}"
    )

    # --------------------------------------------------------
    # 5. Resumen por puntos
    # --------------------------------------------------------

    group_df = build_group_summary(
        inventory_df
    )

    print("\n" + "-" * 80)
    print("3. PUNTOS CLIMÁTICOS")
    print("-" * 80)

    print(
        f"Puntos/grupos detectados: "
        f"{len(group_df)}"
    )

    if not inventory_df.empty:

        coords_validas = (
            pd.to_numeric(
                inventory_df["lat"],
                errors="coerce",
            ).notna()
            &
            pd.to_numeric(
                inventory_df["lon"],
                errors="coerce",
            ).notna()
        )

        print(
            f"Archivos con coordenadas válidas: "
            f"{int(coords_validas.sum())}"
            f"/{len(inventory_df)}"
        )

    # --------------------------------------------------------
    # 6. Escenarios
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. ESCENARIOS DETECTADOS")
    print("-" * 80)

    escenarios = set()

    if (
        not inventory_df.empty
        and "scenarios_detected" in inventory_df.columns
    ):

        for valor in (
            inventory_df["scenarios_detected"]
            .dropna()
            .astype(str)
        ):

            for escenario in valor.split(","):

                escenario = escenario.strip()

                if escenario:
                    escenarios.add(escenario)

    orden = [
        "ERA5",
        "SSP119",
        "SSP126",
        "SSP245",
        "SSP370",
        "SSP434",
        "SSP460",
        "SSP585",
    ]

    escenarios_ordenados = [
        x for x in orden
        if x in escenarios
    ]

    escenarios_otros = sorted(
        escenarios.difference(orden)
    )

    escenarios_finales = (
        escenarios_ordenados
        + escenarios_otros
    )

    if escenarios_finales:

        for escenario in escenarios_finales:

            n = inventory_df[
                inventory_df[
                    "scenarios_detected"
                ]
                .fillna("")
                .str.contains(
                    escenario,
                    regex=False,
                )
            ].shape[0]

            print(
                f"{escenario:<10} "
                f"{n:>4} archivo(s)"
            )

    else:

        print(
            "No se detectaron escenarios."
        )

    # --------------------------------------------------------
    # 7. Cobertura temporal
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. COBERTURA TEMPORAL")
    print("-" * 80)

    if (
        not inventory_df.empty
        and "date_start" in inventory_df.columns
        and "date_end" in inventory_df.columns
    ):

        inicio = pd.to_datetime(
            inventory_df["date_start"],
            errors="coerce",
        )

        fin = pd.to_datetime(
            inventory_df["date_end"],
            errors="coerce",
        )

        if inicio.notna().any():

            print(
                "Fecha inicial más antigua: "
                f"{inicio.min().date()}"
            )

        if fin.notna().any():

            print(
                "Fecha final más reciente:  "
                f"{fin.max().date()}"
            )

        periodos = (
            pd.DataFrame(
                {
                    "inicio": inicio.dt.date,
                    "fin": fin.dt.date,
                }
            )
            .value_counts()
            .reset_index(name="archivos")
        )

        print("\nPeriodos encontrados:")

        print(
            periodos.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 8. Resumen por punto
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. RESUMEN POR PUNTO")
    print("-" * 80)

    if not group_df.empty:

        columnas = [
            c
            for c in [
                "base_name",
                "lat",
                "lon",
                "n_files",
            ]
            if c in group_df.columns
        ]

        print(
            group_df[
                columnas
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 9. Errores
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("7. DIAGNÓSTICO")
    print("-" * 80)

    if bad_df.empty:

        print(
            "No se detectaron errores de lectura."
        )

    else:

        print(
            bad_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 10. Evaluación final
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    if len(inventory_df) == 0:

        print(
            "RESULTADO: NO SE DETECTARON ARCHIVOS "
            "SUSTAX COMPATIBLES"
        )

    elif not bad_df.empty:

        print(
            "RESULTADO: LECTURA PARCIAL — "
            "REVISAR ARCHIVOS CON ERROR"
        )

    else:

        print(
            "RESULTADO: CARPETA SUSTAX "
            "LEÍDA CORRECTAMENTE"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()