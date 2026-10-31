import requests
import pandas as pd
from io import StringIO

# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_DIARIOS = (
    "https://smn.conagua.gob.mx/"
    "tools/RESOURCES/Normales_Climatologicas/"
    "Diarios/col/dia06001.txt"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36"
    )
}

COLUMNAS = [
    "FECHA",
    "PRECIP",
    "EVAP",
    "TMAX",
    "TMIN"
]


# ============================================================
# 1. DESCARGAR ARCHIVO
# ============================================================

print("=" * 70)
print("PRUEBA DE DESCARGA DIARIA - CONAGUA")
print("ESTACIÓN 6001 - ARMERÍA")
print("=" * 70)

response = requests.get(
    URL_DIARIOS,
    headers=HEADERS,
    timeout=60
)

print("\nHTTP:", response.status_code)

response.raise_for_status()

texto = response.text

print(
    "Tamaño del archivo:",
    f"{len(texto):,}",
    "caracteres"
)


# ============================================================
# 2. MOSTRAR PRIMERAS LÍNEAS
# ============================================================

lineas = texto.splitlines()

print("\n" + "=" * 70)
print("PRIMERAS 30 LÍNEAS DEL ARCHIVO")
print("=" * 70)

for i, linea in enumerate(
    lineas[:30],
    start=1
):
    print(f"{i:02d}: {linea}")


# ============================================================
# 3. LOCALIZAR ENCABEZADO FECHA
# ============================================================

inicio_datos = None
encabezados = {}

for i, linea in enumerate(lineas):

    linea_limpia = linea.strip()

    if "FECHA" in linea_limpia.upper():

        inicio_datos = i

        break

    if ":" in linea_limpia:

        clave_info, valor = (
            linea_limpia.split(":", 1)
        )

        encabezados[
            clave_info.strip()
        ] = valor.strip()


if inicio_datos is None:

    raise ValueError(
        "No se encontró el encabezado FECHA."
    )


print("\nEncabezado FECHA encontrado en línea:")
print(inicio_datos + 1)


# ============================================================
# 4. MOSTRAR METADATOS
# ============================================================

print("\n" + "=" * 70)
print("METADATOS DETECTADOS")
print("=" * 70)

for clave, valor in encabezados.items():

    print(
        f"{clave}: {valor}"
    )


# ============================================================
# 5. EXTRAER DATOS CLIMÁTICOS
# ============================================================

datos_clima = [

    linea.strip()

    for linea in lineas[
        inicio_datos + 2:
    ]

    if linea.strip()

]


print("\nNúmero de líneas climáticas:")
print(len(datos_clima))


# ============================================================
# 6. CONVERTIR A DATAFRAME
# ============================================================

df = pd.read_csv(
    StringIO("\n".join(datos_clima)),
    sep=r"\s+",
    names=COLUMNAS,
    engine="python"
)

print("\nFilas leídas antes de limpieza:")
print(len(df))


# ============================================================
# 7. CONVERTIR FECHA CORRECTAMENTE
# ============================================================

df["FECHA"] = pd.to_datetime(
    df["FECHA"],
    format="%Y-%m-%d",
    errors="coerce"
)


# ============================================================
# 8. CONVERTIR VARIABLES NUMÉRICAS
# ============================================================

for columna in [
    "PRECIP",
    "EVAP",
    "TMAX",
    "TMIN"
]:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )


# ============================================================
# 9. DIAGNÓSTICO DE FECHAS NO VÁLIDAS
# ============================================================

fechas_invalidas = df["FECHA"].isna().sum()

print("\nFechas no válidas:")
print(fechas_invalidas)


# ============================================================
# 10. ELIMINAR SOLO FILAS SIN FECHA
# ============================================================

df = df.dropna(
    subset=["FECHA"]
).copy()


# ============================================================
# 11. ORDENAR
# ============================================================

df = df.sort_values(
    "FECHA"
).reset_index(drop=True)


# ============================================================
# 12. REVISAR DUPLICADOS
# ============================================================

duplicados = df.duplicated(
    subset=["FECHA"],
    keep=False
)

print("\nRegistros con fecha duplicada:")
print(duplicados.sum())

if duplicados.any():
    print(
        df.loc[
            duplicados,
            ["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"]
        ].head(20)
    )


# ============================================================
# 13. MOSTRAR LOS PRIMEROS REGISTROS
# ============================================================

print("\n" + "=" * 70)
print("DATOS PROCESADOS")
print("=" * 70)

print(df.head(10))


# ============================================================
# 14. COMPROBAR MANUALMENTE LAS PRIMERAS FECHAS
# ============================================================

print("\nComprobación 1949-01-01 a 1949-01-05:")

print(
    df[
        (df["FECHA"] >= "1949-01-01") &
        (df["FECHA"] <= "1949-01-05")
    ]
)


# ============================================================
# 15. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN CORREGIDO")
print("=" * 70)

print("Número de registros:", len(df))
print("Fecha inicial:", df["FECHA"].min())
print("Fecha final:", df["FECHA"].max())

print(
    "Máxima precipitación:",
    df["PRECIP"].max(),
    "mm"
)

print(
    "Días con precipitación >= 50 mm:",
    (df["PRECIP"] >= 50).sum()
)

print("\nValores faltantes:")
print(
    df[
        ["PRECIP", "EVAP", "TMAX", "TMIN"]
    ].isna().sum()
)