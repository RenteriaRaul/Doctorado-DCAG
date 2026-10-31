from pathlib import Path
import sys

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# LECTOR CIENTÍFICO CONAGUA
# ============================================================

from scripts.conagua_reader import leer_estacion_conagua


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_BASE = (
    ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
)

ESTADO_PRUEBA = "COLIMA"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_metadatos(estacion):
    """
    Recupera los metadatos devueltos por conagua_reader.py.
    """

    if not isinstance(estacion, dict):
        return {}

    if "metadata" in estacion:
        metadata = estacion["metadata"]

    elif "metadatos" in estacion:
        metadata = estacion["metadatos"]

    else:
        metadata = {}

    if metadata is None:
        return {}

    return metadata


def obtener_datos(estacion):
    """
    Recupera el DataFrame climático evitando evaluar
    un DataFrame como booleano.
    """

    if not isinstance(estacion, dict):
        return None

    if "data" in estacion:
        return estacion["data"]

    if "datos" in estacion:
        return estacion["datos"]

    if "clima" in estacion:
        return estacion["clima"]

    return None


def obtener_metadata(metadata, claves):
    """
    Busca un valor entre diferentes nombres posibles.
    """

    if not isinstance(metadata, dict):
        return None

    for clave in claves:

        if clave not in metadata:
            continue

        valor = metadata[clave]

        if pd.notna(valor):
            return valor

    return None


# ============================================================
# DETECCIÓN DE ESTADOS
# ============================================================

def detectar_estados_descargados(carpeta_base):
    """
    Devuelve los estados que contienen al menos un archivo XLSX.
    """

    if not carpeta_base.exists():
        return []

    estados = []

    for carpeta in sorted(carpeta_base.iterdir()):

        if not carpeta.is_dir():
            continue

        archivos = [
            archivo
            for archivo in carpeta.glob("*.xlsx")
            if not archivo.name.startswith("~$")
        ]

        if archivos:
            estados.append(carpeta.name)

    return estados


# ============================================================
# DETECCIÓN DE ARCHIVOS
# ============================================================

def detectar_archivos_estado(
    carpeta_base,
    estado
):
    """
    Localiza los XLSX de estaciones descargadas para un estado.
    """

    carpeta_estado = (
        carpeta_base
        / estado
    )

    if not carpeta_estado.exists():

        raise FileNotFoundError(
            f"No existe la carpeta del estado:\n"
            f"{carpeta_estado}"
        )

    archivos = sorted(
        archivo
        for archivo in carpeta_estado.glob("*.xlsx")
        if not archivo.name.startswith("~$")
    )

    return archivos


# ============================================================
# PREPARAR ESTACIÓN PARA GEV
# ============================================================

def preparar_estacion_gev(archivo):
    """
    Lee una estación CONAGUA y construye una estructura
    normalizada preparada para el análisis GEV.

    IMPORTANTE:
    esta función todavía NO ajusta la distribución GEV.
    Solamente valida y prepara la serie.
    """

    estacion = leer_estacion_conagua(
        archivo
    )

    if estacion is None:

        raise ValueError(
            "conagua_reader devolvió None."
        )

    metadata = obtener_metadatos(
        estacion
    )

    datos = obtener_datos(
        estacion
    )

    if datos is None:

        raise ValueError(
            "No se encontró el DataFrame climático."
        )

    if not isinstance(
        datos,
        pd.DataFrame
    ):

        raise TypeError(
            "Los datos climáticos no son un DataFrame."
        )

    if datos.empty:

        raise ValueError(
            "El DataFrame climático está vacío."
        )

    columnas_requeridas = [
        "date",
        "pp",
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if faltantes:

        raise ValueError(
            f"Faltan columnas necesarias para GEV: "
            f"{faltantes}"
        )

    datos = datos[
        [
            "date",
            "pp",
        ]
    ].copy()

    datos["date"] = pd.to_datetime(
        datos["date"],
        errors="coerce"
    )

    datos["pp"] = pd.to_numeric(
        datos["pp"],
        errors="coerce"
    )

    datos = datos.dropna(
        subset=[
            "date",
            "pp",
        ]
    ).copy()

    datos = datos.sort_values(
        "date"
    )

    if datos.empty:

        raise ValueError(
            "No quedaron observaciones válidas."
        )

    # --------------------------------------------------------
    # Máximos anuales
    # --------------------------------------------------------

    datos["year"] = (
        datos["date"]
        .dt
        .year
    )

    maximos_anuales = (
        datos
        .groupby("year")["pp"]
        .max()
        .dropna()
        .sort_index()
    )

    # --------------------------------------------------------
    # Metadatos
    # --------------------------------------------------------

    clave = obtener_metadata(
        metadata,
        [
            "clave",
            "CLAVE",
            "estacion",
            "ESTACION",
        ]
    )

    if clave is None:

        clave = (
            archivo
            .stem
            .split("_")[0]
        )

    nombre = obtener_metadata(
        metadata,
        [
            "nombre",
            "NOMBRE",
        ]
    )

    if nombre is None:

        nombre = archivo.stem

    estado = obtener_metadata(
        metadata,
        [
            "estado",
            "ESTADO",
        ]
    )

    municipio = obtener_metadata(
        metadata,
        [
            "municipio",
            "MUNICIPIO",
        ]
    )

    latitud = obtener_metadata(
        metadata,
        [
            "latitud",
            "LATITUD",
            "latitude",
        ]
    )

    longitud = obtener_metadata(
        metadata,
        [
            "longitud",
            "LONGITUD",
            "longitude",
        ]
    )

    return {
        "archivo":
            archivo,

        "clave":
            str(clave),

        "nombre":
            nombre,

        "estado":
            estado,

        "municipio":
            municipio,

        "latitud":
            latitud,

        "longitud":
            longitud,

        "datos":
            datos,

        "maximos_anuales":
            maximos_anuales,

        "n_registros":
            len(datos),

        "n_maximos_anuales":
            len(maximos_anuales),

        "fecha_inicio":
            datos["date"].min(),

        "fecha_fin":
            datos["date"].max(),
    }


# ============================================================
# PRUEBA PRINCIPAL
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA: DETECCIÓN AUTOMÁTICA CONAGUA -> GEV")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Detectar estados disponibles
    # --------------------------------------------------------

    estados = detectar_estados_descargados(
        CARPETA_BASE
    )

    print()
    print(
        "Estados con datos descargados:"
    )

    for estado in estados:

        print(
            f"  - {estado}"
        )

    assert estados, (
        "No se detectaron estados con estaciones descargadas."
    )

    assert ESTADO_PRUEBA in estados, (
        f"No se detectó {ESTADO_PRUEBA}."
    )

    # --------------------------------------------------------
    # 2. Detectar archivos
    # --------------------------------------------------------

    archivos = detectar_archivos_estado(
        CARPETA_BASE,
        ESTADO_PRUEBA
    )

    print()
    print(
        f"Estado seleccionado: {ESTADO_PRUEBA}"
    )

    print(
        f"Archivos detectados: {len(archivos)}"
    )

    assert archivos, (
        "No se detectaron archivos XLSX."
    )

    # --------------------------------------------------------
    # 3. Preparar estaciones
    # --------------------------------------------------------

    resultados = []
    errores = []

    for numero, archivo in enumerate(
        archivos,
        start=1
    ):

        print()
        print("-" * 72)

        print(
            f"[{numero}/{len(archivos)}] "
            f"{archivo.name}"
        )

        try:

            estacion = preparar_estacion_gev(
                archivo
            )

            resultados.append(
                {
                    "clave":
                        estacion["clave"],

                    "nombre":
                        estacion["nombre"],

                    "estado":
                        estacion["estado"],

                    "municipio":
                        estacion["municipio"],

                    "n_registros":
                        estacion["n_registros"],

                    "n_maximos_anuales":
                        estacion["n_maximos_anuales"],

                    "fecha_inicio":
                        estacion["fecha_inicio"],

                    "fecha_fin":
                        estacion["fecha_fin"],

                    "latitud":
                        estacion["latitud"],

                    "longitud":
                        estacion["longitud"],
                }
            )

            print("  OK")

            print(
                f"  Clave: "
                f"{estacion['clave']}"
            )

            print(
                f"  Estación: "
                f"{estacion['nombre']}"
            )

            print(
                f"  Registros diarios válidos: "
                f"{estacion['n_registros']:,}"
            )

            print(
                f"  Máximos anuales: "
                f"{estacion['n_maximos_anuales']}"
            )

            print(
                f"  Periodo: "
                f"{estacion['fecha_inicio'].date()} "
                f"→ "
                f"{estacion['fecha_fin'].date()}"
            )

        except Exception as error:

            errores.append(
                {
                    "archivo":
                        archivo.name,

                    "tipo_error":
                        type(error).__name__,

                    "error":
                        str(error),
                }
            )

            print(
                f"  ERROR: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # 4. Resumen
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESUMEN")
    print("=" * 72)

    print(
        f"Archivos detectados: "
        f"{len(archivos)}"
    )

    print(
        f"Estaciones preparadas para GEV: "
        f"{len(resultados)}"
    )

    print(
        f"Errores de preparación: "
        f"{len(errores)}"
    )

    # --------------------------------------------------------
    # 5. Tabla
    # --------------------------------------------------------

    if resultados:

        df_resultados = pd.DataFrame(
            resultados
        )

        print()
        print("=" * 72)
        print("ESTACIONES PREPARADAS")
        print("=" * 72)

        print(
            df_resultados.to_string(
                index=False
            )
        )

        # --------------------------------------------
        # Diagnóstico de suficiencia temporal
        # --------------------------------------------

        print()
        print("=" * 72)
        print("DIAGNÓSTICO DE MÁXIMOS ANUALES")
        print("=" * 72)

        suficientes = (
            df_resultados[
                "n_maximos_anuales"
            ] >= 10
        )

        print(
            "Con al menos 10 máximos anuales: "
            f"{int(suficientes.sum())}"
        )

        print(
            "Con menos de 10 máximos anuales: "
            f"{int((~suficientes).sum())}"
        )

        if (~suficientes).any():

            print()
            print(
                "Estaciones con menos de 10:"
            )

            print(
                df_resultados.loc[
                    ~suficientes,
                    [
                        "clave",
                        "nombre",
                        "n_maximos_anuales",
                    ]
                ].to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # 6. Errores
    # --------------------------------------------------------

    if errores:

        print()
        print("=" * 72)
        print("ERRORES")
        print("=" * 72)

        print(
            pd.DataFrame(
                errores
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 7. Validación final
    # --------------------------------------------------------

    print()
    print("=" * 72)

    assert resultados, (
        "Ninguna estación pudo prepararse para GEV."
    )

    if errores:

        print(
            "RESULTADO: DETECCIÓN PARCIAL"
        )

    else:

        print(
            "RESULTADO: DETECCIÓN COMPLETA"
        )

        print(
            f"Las {len(resultados)} estaciones fueron "
            "detectadas y preparadas correctamente."
        )

    print("=" * 72)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()