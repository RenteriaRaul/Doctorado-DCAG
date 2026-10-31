import pandas as pd

# ============================================================
# RUTAS
# ============================================================

archivo_actual = r"G:\My Drive\Doctorado-DCAG\CONAGUA\estaciones_descargadas\COLIMA\6001_ARMERIA.xlsx"

archivo_anterior = r"G:\My Drive\Doctorado\Probabilidad\Precipitación\estaciones_conagua_excel\Colima\6001_ARMERIA.xlsx"


# ============================================================
# 1. LEER DATOS
# ============================================================

actual = pd.read_excel(
    archivo_actual,
    sheet_name="Datos Clima"
)

anterior = pd.read_excel(
    archivo_anterior,
    sheet_name="Datos Clima"
)


# ============================================================
# 2. NORMALIZAR FECHA
# ============================================================

actual["FECHA"] = pd.to_datetime(
    actual["FECHA"],
    errors="coerce"
)

anterior["FECHA"] = pd.to_datetime(
    anterior["FECHA"],
    errors="coerce"
)


# ============================================================
# 3. CONVERTIR PRECIPITACIÓN A NUMÉRICO
# ============================================================
# Esto convierte valores como "NULO" en NaN.

actual["PRECIP"] = pd.to_numeric(
    actual["PRECIP"],
    errors="coerce"
)

anterior["PRECIP"] = pd.to_numeric(
    anterior["PRECIP"],
    errors="coerce"
)


# ============================================================
# 4. RESUMEN GENERAL
# ============================================================

print("=" * 70)
print("COMPARACIÓN DE PRECIPITACIÓN")
print("=" * 70)

print("\nDESCARGA ACTUAL")
print("Registros:", len(actual))
print("Fecha inicial:", actual["FECHA"].min())
print("Fecha final:", actual["FECHA"].max())
print("Valores PRECIP válidos:", actual["PRECIP"].notna().sum())
print("Valores PRECIP faltantes:", actual["PRECIP"].isna().sum())

print("\nDESCARGA ANTERIOR")
print("Registros:", len(anterior))
print("Fecha inicial:", anterior["FECHA"].min())
print("Fecha final:", anterior["FECHA"].max())
print("Valores PRECIP válidos:", anterior["PRECIP"].notna().sum())
print("Valores PRECIP faltantes:", anterior["PRECIP"].isna().sum())


# ============================================================
# 5. COMPARAR FECHAS
# ============================================================

fechas_actual = set(
    actual["FECHA"].dropna()
)

fechas_anterior = set(
    anterior["FECHA"].dropna()
)

solo_actual = sorted(
    fechas_actual - fechas_anterior
)

solo_anterior = sorted(
    fechas_anterior - fechas_actual
)

print("\n" + "=" * 70)
print("COMPARACIÓN DE FECHAS")
print("=" * 70)

print(
    "Fechas solamente en descarga actual:",
    len(solo_actual)
)

print(
    "Fechas solamente en descarga anterior:",
    len(solo_anterior)
)


# ============================================================
# 6. UNIR POR FECHA
# ============================================================

comparacion = pd.merge(
    anterior[["FECHA", "PRECIP"]],
    actual[["FECHA", "PRECIP"]],
    on="FECHA",
    how="inner",
    suffixes=("_ANTERIOR", "_ACTUAL")
)


# ============================================================
# 7. COMPARAR PRECIPITACIÓN
# ============================================================

# Ambos faltantes se consideran iguales
ambos_nan = (
    comparacion["PRECIP_ANTERIOR"].isna()
    &
    comparacion["PRECIP_ACTUAL"].isna()
)

# Diferencia real
comparacion["DIFERENTE"] = (
    comparacion["PRECIP_ANTERIOR"]
    !=
    comparacion["PRECIP_ACTUAL"]
) & ~ambos_nan


diferencias = comparacion[
    comparacion["DIFERENTE"]
].copy()


print("\n" + "=" * 70)
print("RESULTADO DE LA COMPARACIÓN")
print("=" * 70)

print(
    "Fechas compartidas:",
    len(comparacion)
)

print(
    "Fechas con diferente precipitación:",
    len(diferencias)
)


# ============================================================
# 8. MOSTRAR DIFERENCIAS
# ============================================================

if len(diferencias) > 0:

    print("\nPrimeras diferencias encontradas:")

    print(
        diferencias[
            [
                "FECHA",
                "PRECIP_ANTERIOR",
                "PRECIP_ACTUAL"
            ]
        ].head(30).to_string(index=False)
    )

else:

    print(
        "\nLos valores de precipitación coinciden "
        "en todas las fechas compartidas."
    )


# ============================================================
# 9. GUARDAR SOLO SI EXISTEN DIFERENCIAS
# ============================================================

if len(diferencias) > 0:

    diferencias.to_csv(
        "diferencias_precipitacion_6001.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nSe generó:"
        " diferencias_precipitacion_6001.csv"
    )