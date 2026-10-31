import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_VISOR = (
    "https://smn.conagua.gob.mx/"
    "tools/GUI/estaciones-climatologicas/"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131 Safari/537.36"
    )
}


# ============================================================
# 1. DESCARGAR HTML DEL VISOR
# ============================================================

def obtener_html():

    print("=" * 70)
    print("INSPECCIÓN DEL VISOR SMN-CONAGUA")
    print("=" * 70)

    response = requests.get(
        URL_VISOR,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    print(
        f"\nVisor cargado correctamente."
        f"\nHTTP: {response.status_code}"
        f"\nTamaño HTML: {len(response.text):,} caracteres"
    )

    return response.text


# ============================================================
# 2. BUSCAR IFRAMES
# ============================================================

def buscar_iframes(soup):

    print("\n" + "=" * 70)
    print("IFRAMES")
    print("=" * 70)

    iframes = soup.find_all(
        "iframe"
    )

    if not iframes:

        print("No se encontraron iframes.")

        return []

    resultados = []

    for iframe in iframes:

        src = iframe.get(
            "src"
        )

        if src:

            url = urljoin(
                URL_VISOR,
                src
            )

            resultados.append(url)

            print(url)

    return resultados


# ============================================================
# 3. BUSCAR ARCHIVOS JAVASCRIPT
# ============================================================

def buscar_scripts(soup):

    print("\n" + "=" * 70)
    print("ARCHIVOS JAVASCRIPT")
    print("=" * 70)

    scripts = []

    for script in soup.find_all(
        "script",
        src=True
    ):

        src = script.get(
            "src"
        )

        url = urljoin(
            URL_VISOR,
            src
        )

        scripts.append(url)

        print(url)

    return scripts


# ============================================================
# 4. BUSCAR RECURSOS DIRECTAMENTE EN HTML
# ============================================================

def buscar_recursos_html(html):

    print("\n" + "=" * 70)
    print("RECURSOS POTENCIALMENTE ÚTILES EN HTML")
    print("=" * 70)

    extensiones = [
        "kmz",
        "kml",
        "json",
        "geojson",
        "csv",
        "txt"
    ]

    encontrados = set()

    for extension in extensiones:

        patron = rf"""[^"'()<>\s]+\.{extension}[^"'()<>\s]*"""

        coincidencias = re.findall(
            patron,
            html,
            flags=re.IGNORECASE
        )

        for resultado in coincidencias:

            url = urljoin(
                URL_VISOR,
                resultado
            )

            encontrados.add(url)

    if encontrados:

        for recurso in sorted(
            encontrados
        ):

            print(recurso)

    else:

        print(
            "No se encontraron recursos "
            "KML/KMZ/JSON/CSV/TXT directamente."
        )

    return encontrados


# ============================================================
# 5. INSPECCIONAR JAVASCRIPT
# ============================================================

def inspeccionar_javascript(
    scripts
):

    print("\n" + "=" * 70)
    print("INSPECCIÓN DE JAVASCRIPT")
    print("=" * 70)

    palabras_clave = [
        ".kmz",
        ".kml",
        ".json",
        ".geojson",
        ".csv",
        ".txt",
        "estacion",
        "station",
        "climat",
        "diario",
        "precip",
        "fetch(",
        "$.ajax",
        "axios",
        "url:"
    ]

    resultados = []

    for numero, url_script in enumerate(
        scripts,
        start=1
    ):

        print(
            f"\n[{numero}/{len(scripts)}] "
            f"Revisando:"
        )

        print(
            url_script
        )

        try:

            response = requests.get(
                url_script,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code != 200:

                print(
                    f"HTTP {response.status_code}"
                )

                continue

            codigo = response.text

            coincidencias_script = []

            for linea in codigo.splitlines():

                linea_lower = (
                    linea.lower()
                )

                if any(
                    palabra.lower()
                    in linea_lower
                    for palabra
                    in palabras_clave
                ):

                    coincidencias_script.append(
                        linea.strip()
                    )

            if coincidencias_script:

                print(
                    ">>> Coincidencias:"
                )

                for linea in coincidencias_script[
                    :30
                ]:

                    print(
                        linea[:500]
                    )

                resultados.append({
                    "script":
                        url_script,

                    "coincidencias":
                        coincidencias_script
                })

            else:

                print(
                    "Sin coincidencias relevantes."
                )

        except Exception as e:

            print(
                f"Error: {e}"
            )

    return resultados


# ============================================================
# 6. GUARDAR HTML PARA INSPECCIÓN
# ============================================================

def guardar_html(html):

    archivo = (
        "visor_conagua.html"
    )

    with open(
        archivo,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print(
        f"\nHTML guardado en: {archivo}"
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    html = obtener_html()

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    buscar_iframes(
        soup
    )

    scripts = buscar_scripts(
        soup
    )

    buscar_recursos_html(
        html
    )

    inspeccionar_javascript(
        scripts
    )

    guardar_html(
        html
    )

    print("\n" + "=" * 70)
    print("INSPECCIÓN TERMINADA")
    print("=" * 70)


if __name__ == "__main__":
    main()