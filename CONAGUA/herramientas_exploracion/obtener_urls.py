"""
Módulo: obtener_urls.py

Objetivo:
Localizar automáticamente los recursos utilizados por el portal
de estaciones climatológicas del SMN-CONAGUA.

Posteriormente este módulo permitirá obtener las URLs de descarga
de los datos históricos diarios de las estaciones.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_PORTAL = (
    "https://smn.conagua.gob.mx/"
    "Es/Climatologia/Informacion-Climatologica/"
    "Informacion-Estadistica-Climatologica"
)

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
# FUNCIÓN 1
# Comprobar conexión con el portal
# ============================================================

def comprobar_portal():

    print("Conectando con SMN-CONAGUA...")

    response = requests.get(
        URL_PORTAL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    print(
        f"Portal accesible. HTTP {response.status_code}"
    )

    return response


# ============================================================
# FUNCIÓN 2
# Comprobar el visor de estaciones
# ============================================================

def comprobar_visor():

    print("\nConectando con visor de estaciones...")

    response = requests.get(
        URL_VISOR,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    print(
        f"Visor accesible. HTTP {response.status_code}"
    )

    return response


# ============================================================
# FUNCIÓN 3
# Buscar archivos KMZ
# ============================================================

def buscar_kmz(html, url_base):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    resultados = []

    # Buscar enlaces normales
    for enlace in soup.find_all("a", href=True):

        href = enlace["href"]

        if ".kmz" in href.lower():

            url = urljoin(
                url_base,
                href
            )

            resultados.append(url)

    # Eliminar duplicados
    resultados = list(
        dict.fromkeys(resultados)
    )

    return resultados


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def obtener_recursos_conagua():

    print("=" * 60)
    print("SMN-CONAGUA")
    print("Búsqueda automática de recursos")
    print("=" * 60)

    # --------------------------------------------------------
    # Portal principal
    # --------------------------------------------------------

    portal = comprobar_portal()

    kmz_portal = buscar_kmz(
        portal.text,
        URL_PORTAL
    )

    # --------------------------------------------------------
    # Visor
    # --------------------------------------------------------

    visor = comprobar_visor()

    kmz_visor = buscar_kmz(
        visor.text,
        URL_VISOR
    )

    # --------------------------------------------------------
    # Combinar resultados
    # --------------------------------------------------------

    urls_kmz = list(
        dict.fromkeys(
            kmz_portal + kmz_visor
        )
    )

    print("\n" + "=" * 60)

    if urls_kmz:

        print(
            f"KMZ encontrados: {len(urls_kmz)}"
        )

        for numero, url in enumerate(
            urls_kmz,
            start=1
        ):

            print(
                f"{numero}. {url}"
            )

    else:

        print(
            "No se encontró un KMZ directamente "
            "en los enlaces HTML."
        )

        print(
            "Esto no significa que no exista. "
            "Puede estar cargado mediante JavaScript."
        )

    print("=" * 60)

    return urls_kmz


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    obtener_recursos_conagua()