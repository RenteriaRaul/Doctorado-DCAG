from pathlib import Path
import shutil

import pandas as pd

from scripts.sustax.inventory import build_sustax_inventory
from scripts.sustax.loader import load_sustax_file_safe
from scripts.sustax.merge import build_long_total_from_inventory


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_ORIGINAL = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\Datos Sustax\Sustax Original"
)

# Salida temporal de la prueba.
# No escribiremos nada dentro de la carpeta original.
CARPETA_PRUEBA = Path(
    r"G:\My Drive\Doctorado-DCAG\results\tests\sustax_merge"
)


# ============================================================
# HELPERS
# ============================================================

def extraer_variable(columna: str) -> str:
    """
    Obtiene la variable a partir de columnas como:
    tp-Total Precipitation [ERA5]
    pr-Total Precipitation [SSP245]
    """
    texto = str(columna).strip()

    if "[" in texto:
        texto = texto.split("[", 1)[0].strip()

    return texto


def es_precipitacion(variable: str) -> bool:
    """
    Identifica únicamente las variables de precipitación
    que esperamos en esta prueba.
    """
    variable = str(variable).strip().lower()

    return (
        variable.startswith("tp-")
        or variable.startswith("pr-")
        or variable == "tp"
        or variable == "pr"
    )


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 80)
    print("PRUEBA — MERGE DE DESCARGAS ORIGINALES SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Verificar carpeta original
    # --------------------------------------------------------

    if not CARPETA_ORIGINAL.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{CARPETA_ORIGINAL}"
        )

    print(f"\nCarpeta original:\n{CARPETA_ORIGINAL}")

    # --------------------------------------------------------
    # 2. Inventario
    # --------------------------------------------------------

    inventory_df, bad_df = build_sustax_inventory(
        str(CARPETA_ORIGINAL)
    )

    print("\n" + "-" * 80)
    print("1. INVENTARIO")
    print("-" * 80)

    print(f"Archivos compatibles: {len(inventory_df)}")
    print(f"Archivos con error:   {len(bad_df)}")

    if inventory_df.empty:
        raise RuntimeError(
            "No existen archivos Sustax compatibles."
        )

    if not bad_df.empty:
        print("\nArchivos con error:")
        print(bad_df.to_string(index=False))

    # --------------------------------------------------------
    # 3. Inspección de variables reales
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. VARIABLES DETECTADAS EN LOS ORIGINALES")
    print("-" * 80)

    variables = set()
    escenarios = set()
    unidades = {}

    for _, row in inventory_df.iterrows():

        archivo = Path(row["path"])

        df, _, metadata = load_sustax_file_safe(
            str(archivo),
            return_pandas_df=True,
            return_metadata=True,
        )

        for columna in df.columns:

            variable = extraer_variable(columna)
            variables.add(variable)

            if "[" in str(columna) and "]" in str(columna):
                escenario = (
                    str(columna)
                    .split("[", 1)[1]
                    .split("]", 1)[0]
                    .strip()
                )

                escenarios.add(escenario)

        for clave, unidad in metadata.get(
            "units",
            {}
        ).items():

            unidades[clave] = unidad

    print("\nVariables:")

    for variable in sorted(variables):
        print(f"  - {variable}")

    print("\nEscenarios:")

    for escenario in sorted(escenarios):
        print(f"  - {escenario}")

    print("\nUnidades declaradas:")

    for clave, unidad in sorted(unidades.items()):
        print(f"  - {clave}: {unidad}")

    # --------------------------------------------------------
    # 4. Control científico antes del merge
    # --------------------------------------------------------

    variables_no_precipitacion = [
        variable
        for variable in variables
        if not es_precipitacion(variable)
    ]

    print("\n" + "-" * 80)
    print("3. CONTROL DE VARIABLES")
    print("-" * 80)

    if variables_no_precipitacion:

        print(
            "SE DETECTARON VARIABLES QUE NO SON "
            "PRECIPITACIÓN:"
        )

        for variable in variables_no_precipitacion:
            print(f"  - {variable}")

        print(
            "\nLa prueba se detiene antes de ejecutar "
            "merge.py para evitar aplicar ×1000 a una "
            "variable cuya unidad debe revisarse."
        )

        return

    print(
        "OK: todas las series climáticas detectadas "
        "corresponden a precipitación."
    )

    # --------------------------------------------------------
    # 5. Preparar salida limpia de prueba
    # --------------------------------------------------------

    if CARPETA_PRUEBA.exists():
        shutil.rmtree(CARPETA_PRUEBA)

    CARPETA_PRUEBA.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "-" * 80)
    print("4. EJECUCIÓN DE MERGE")
    print("-" * 80)

    print(
        f"Salida temporal:\n{CARPETA_PRUEBA}"
    )

    # --------------------------------------------------------
    # 6. Ejecutar merge actual
    # --------------------------------------------------------

    resultado = build_long_total_from_inventory(
        inventory_df=inventory_df,
        out_dir=str(CARPETA_PRUEBA),
    )

    print("\nMerge ejecutado.")

    if resultado is not None:
        print(
            f"Tipo de resultado devuelto: "
            f"{type(resultado).__name__}"
        )

    # --------------------------------------------------------
    # 7. Revisar productos creados
    # --------------------------------------------------------

    archivos_salida = sorted(
        CARPETA_PRUEBA.rglob("*.csv")
    )

    print("\n" + "-" * 80)
    print("5. PRODUCTOS GENERADOS")
    print("-" * 80)

    print(
        f"CSV generados: {len(archivos_salida)}"
    )

    if not archivos_salida:

        raise RuntimeError(
            "merge.py no generó archivos CSV."
        )

    for archivo in archivos_salida:
        print(
            f"  - {archivo.relative_to(CARPETA_PRUEBA)}"
        )

    # --------------------------------------------------------
    # 8. Inspeccionar contenido
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. INSPECCIÓN DE LOS ARCHIVOS LONG")
    print("-" * 80)

    total_filas = 0
    escenarios_salida = set()
    variables_salida = set()
    valores_negativos = 0

    resumen = []

    for archivo in archivos_salida:

        df = pd.read_csv(archivo)

        total_filas += len(df)

        if "scenario" in df.columns:
            escenarios_salida.update(
                df["scenario"]
                .dropna()
                .astype(str)
                .unique()
            )

        if "variable" in df.columns:
            variables_salida.update(
                df["variable"]
                .dropna()
                .astype(str)
                .unique()
            )

        if "value_mm" in df.columns:

            valores = pd.to_numeric(
                df["value_mm"],
                errors="coerce",
            )

            negativos = int(
                (valores < 0).sum()
            )

            valores_negativos += negativos

            minimo = valores.min()
            promedio = valores.mean()
            maximo = valores.max()

        else:

            negativos = None
            minimo = None
            promedio = None
            maximo = None

        resumen.append(
            {
                "archivo": archivo.name,
                "filas": len(df),
                "variables": (
                    ", ".join(
                        sorted(
                            df["variable"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    )
                    if "variable" in df.columns
                    else "N/D"
                ),
                "escenarios": (
                    ", ".join(
                        sorted(
                            df["scenario"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    )
                    if "scenario" in df.columns
                    else "N/D"
                ),
                "min_mm": minimo,
                "media_mm": promedio,
                "max_mm": maximo,
                "negativos": negativos,
            }
        )

    resumen_df = pd.DataFrame(resumen)

    print(
        resumen_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Resumen global
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("7. RESUMEN GLOBAL")
    print("-" * 80)

    print(f"Archivos generados: {len(archivos_salida)}")
    print(f"Filas totales:      {total_filas:,}")

    print(
        "Variables:          "
        + ", ".join(
            sorted(variables_salida)
        )
    )

    print(
        "Escenarios:         "
        + ", ".join(
            sorted(escenarios_salida)
        )
    )

    print(
        f"Valores negativos:  "
        f"{valores_negativos}"
    )

    # --------------------------------------------------------
    # 10. Validaciones finales
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("8. VALIDACIÓN")
    print("-" * 80)

    if valores_negativos > 0:

        print(
            "ADVERTENCIA: existen valores negativos "
            "de precipitación."
        )

    else:

        print(
            "OK: no se detectaron valores negativos."
        )

    if variables_salida:

        no_precip_salida = [
            variable
            for variable in variables_salida
            if not es_precipitacion(variable)
        ]

        if no_precip_salida:

            print(
                "ADVERTENCIA: el merge contiene variables "
                "que no corresponden a precipitación:"
            )

            for variable in no_precip_salida:
                print(f"  - {variable}")

        else:

            print(
                "OK: el merge contiene únicamente "
                "variables de precipitación."
            )

    print("\n" + "=" * 80)
    print("PRUEBA DE MERGE FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()