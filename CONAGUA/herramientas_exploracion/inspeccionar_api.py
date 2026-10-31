import requests
import re
from urllib.parse import urljoin

# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_VISOR = (
    "https://smn.conagua.gob.mx/"
    "tools/GUI/estaciones-climatologicas/"
)

URL_JS = (
    "https://smn.conagua.gob.mx/"
    "tools/GUI/estaciones-climatologicas/"
    "assets/index-B1Trhpqn.js"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131 Safari/537.36"
    )
}


# ============================================================
# 1. DESCARGAR JAVASCRIPT
# ============================================================

print("=" * 70)
print("INSPECCIÓN PROFUNDA DEL VISOR CONAGUA")
print("=" * 70)

response = requests.get(
    URL_JS,
    headers=HEADERS,
    timeout=60
)

response.raise_for_status()

js = response.text

print(f"\nJavaScript descargado correctamente.")
print(f"Tamaño: {len(js):,} caracteres")


# ============================================================
# 2. GUARDAR JS LOCALMENTE
# ============================================================

with open(
    "visor_conagua.js",
    "w",
    encoding="utf-8"
) as f:
    f.write(js)

print("Archivo guardado: visor_conagua.js")


# ============================================================
# 3. BUSCAR URLs ABSOLUTAS
# ============================================================

print("\n" + "=" * 70)
print("URLS ABSOLUTAS ENCONTRADAS")
print("=" * 70)

urls_absolutas = sorted(
    set(
        re.findall(
            r'https?://[^"\'`\s<>]+',
            js
        )
    )
)

for url in urls_absolutas:
    print(url)

print(
    f"\nTotal URLs absolutas: "
    f"{len(urls_absolutas)}"
)


# ============================================================
# 4. BUSCAR POSIBLES ARCHIVOS / ENDPOINTS
# ============================================================

print("\n" + "=" * 70)
print("RECURSOS RELATIVOS POTENCIALES")
print("=" * 70)

patrones = [
    r'["\']([^"\']+\.json[^"\']*)["\']',
    r'["\']([^"\']+\.geojson[^"\']*)["\']',
    r'["\']([^"\']+\.csv[^"\']*)["\']',
    r'["\']([^"\']+\.kml[^"\']*)["\']',
    r'["\']([^"\']+\.kmz[^"\']*)["\']',
    r'["\']([^"\']+\.txt[^"\']*)["\']',
]

recursos = set()

for patron in patrones:

    encontrados = re.findall(
        patron,
        js,
        flags=re.IGNORECASE
    )

    recursos.update(
        encontrados
    )


for recurso in sorted(recursos):

    print(
        urljoin(
            URL_VISOR,
            recurso
        )
    )


# ============================================================
# 5. BUSCAR LLAMADAS FETCH
# ============================================================

print("\n" + "=" * 70)
print("LLAMADAS FETCH")
print("=" * 70)

fetches = re.findall(
    r'fetch\((.{0,300}?)\)',
    js,
    flags=re.IGNORECASE
)

if fetches:

    for i, elemento in enumerate(
        fetches,
        start=1
    ):

        print(
            f"\nFETCH {i}:"
        )

        print(
            elemento[:500]
        )

else:

    print(
        "No se encontraron llamadas fetch "
        "de forma directa."
    )


# ============================================================
# 6. BUSCAR PALABRAS IMPORTANTES CON CONTEXTO
# ============================================================

palabras = [
    "Información estadística",
    "Informacion estadistica",
    "Normales climatológicas",
    "FeatureCollection",
    "geoJSON",
    "clave",
    "precip",
    "diario",
    "estadistica",
    "estaciones",
    ".txt",
    ".csv",
    ".json",
    "/api/",
    "axios",
    "fetch("
]


def mostrar_contexto(
    texto,
    palabra,
    antes=500,
    despues=1000
):

    texto_lower = texto.lower()
    palabra_lower = palabra.lower()

    posiciones = []

    inicio = 0

    while True:

        pos = texto_lower.find(
            palabra_lower,
            inicio
        )

        if pos == -1:
            break

        posiciones.append(pos)

        inicio = pos + len(
            palabra_lower
        )

    return posiciones


print("\n" + "=" * 70)
print("CONTEXTO DE PALABRAS CLAVE")
print("=" * 70)


for palabra in palabras:

    posiciones = mostrar_contexto(
        js,
        palabra
    )

    print(
        f"\n\n{'#' * 70}"
    )

    print(
        f"PALABRA: {palabra}"
    )

    print(
        f"OCURRENCIAS: {len(posiciones)}"
    )

    print(
        "#" * 70
    )


    # Solo mostrar las primeras 5
    for numero, pos in enumerate(
        posiciones[:5],
        start=1
    ):

        inicio = max(
            0,
            pos - 500
        )

        final = min(
            len(js),
            pos + 1500
        )

        fragmento = js[
            inicio:final
        ]

        print(
            f"\n--- Coincidencia {numero} ---"
        )

        print(fragmento)

        print(
            "\n" + "-" * 70
        )


# ============================================================
# 7. GENERAR ARCHIVO DE RESULTADOS
# ============================================================

with open(
    "resultado_inspeccion_api.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "URLS ABSOLUTAS\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    for url in urls_absolutas:
        f.write(url + "\n")


    f.write(
        "\n\nRECURSOS\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    for recurso in sorted(
        recursos
    ):

        f.write(
            urljoin(
                URL_VISOR,
                recurso
            )
            + "\n"
        )


    f.write(
        "\n\nFETCH\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    for elemento in fetches:

        f.write(
            elemento
            + "\n\n"
        )


print("\n" + "=" * 70)

print(
    "INSPECCIÓN TERMINADA"
)

print(
    "Archivo creado:"
)

print(
    "resultado_inspeccion_api.txt"
)

print("=" * 70)