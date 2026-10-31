import requests
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_GEOJSON = (
    "https://smn.conagua.gob.mx/"
    "tools/GUI/estaciones-climatologicas/"
    "data/estaciones_climatologicas.geojson"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36"
    )
}


# ============================================================
# 1. DESCARGAR GEOJSON
# ============================================================

print("=" * 70)
print("DESCARGA DE ESTACIONES SMN-CONAGUA")
print("=" * 70)

response = requests.get(
    URL_GEOJSON,
    headers=HEADERS,
    timeout=60
)

print("HTTP:", response.status_code)

response.raise_for_status()

datos = response.json()


# ============================================================
# 2. VERIFICAR ESTRUCTURA
# ============================================================

print("\nTipo de archivo:")
print(datos.get("type"))

features = datos.get("features", [])

print("\nNúmero total de estaciones:")
print(len(features))

print("\nFecha/semana de generación:")
print(datos.get("generatedAt"))


# ============================================================
# 3. REVISAR UNA ESTACIÓN
# ============================================================

if features:

    primera = features[0]

    print("\n" + "=" * 70)
    print("EJEMPLO DE UNA ESTACIÓN")
    print("=" * 70)

    print("\nPropiedades disponibles:")

    for campo in primera.get("properties", {}).keys():
        print(" -", campo)

    print("\nCoordenadas:")
    print(
        primera.get(
            "geometry", {}
        ).get("coordinates")
    )


# ============================================================
# 4. CONVERTIR TODAS LAS ESTACIONES A TABLA
# ============================================================

estaciones = []

for feature in features:

    propiedades = feature.get(
        "properties", {}
    )

    coordenadas = feature.get(
        "geometry", {}
    ).get(
        "coordinates", []
    )

    longitud = (
        coordenadas[0]
        if len(coordenadas) >= 2
        else None
    )

    latitud = (
        coordenadas[1]
        if len(coordenadas) >= 2
        else None
    )

    estacion = propiedades.copy()

    estacion["LONGITUD"] = longitud
    estacion["LATITUD"] = latitud

    estaciones.append(estacion)


df = pd.DataFrame(estaciones)


# ============================================================
# 5. MOSTRAR COLUMNAS
# ============================================================

print("\n" + "=" * 70)
print("COLUMNAS DISPONIBLES")
print("=" * 70)

for columna in df.columns:
    print(columna)


# ============================================================
# 6. FILTRAR COLIMA
# ============================================================

df_colima = df[
    df["estado"]
    .astype(str)
    .str.upper()
    .str.strip()
    == "COLIMA"
].copy()


print("\n" + "=" * 70)
print("ESTACIONES DE COLIMA")
print("=" * 70)

print(
    "Número de estaciones encontradas:",
    len(df_colima)
)


# ============================================================
# 7. MOSTRAR INFORMACIÓN PRINCIPAL
# ============================================================

columnas_mostrar = [
    "clave",
    "nombre",
    "municipio",
    "situacion",
    "inicio",
    "mas_reciente",
    "LATITUD",
    "LONGITUD"
]

columnas_existentes = [
    c for c in columnas_mostrar
    if c in df_colima.columns
]

print(
    df_colima[
        columnas_existentes
    ].to_string(index=False)
)


# ============================================================
# 8. COMPROBAR LINKS DE DATOS
# ============================================================

print("\n" + "=" * 70)
print("LINKS DE DESCARGA")
print("=" * 70)

columnas_links = [
    "clave",
    "nombre",
    "diarios",
    "mensuales",
    "extremos"
]

columnas_links = [
    c for c in columnas_links
    if c in df_colima.columns
]

print(
    df_colima[
        columnas_links
    ].head().to_string(index=False)
)


# ============================================================
# 9. GUARDAR RESULTADO
# ============================================================

df_colima.to_csv(
    "estaciones_colima_conagua.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("ARCHIVO CREADO")
print("=" * 70)

print(
    "estaciones_colima_conagua.csv"
)