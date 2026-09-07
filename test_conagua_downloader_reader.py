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
# IMPORTAR LECTOR CONAGUA
# ============================================================

from scripts.conagua_reader import (
    leer_estacion_conagua,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_ESTADO = (
    ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
    / "COLIMA"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_metadatos(estacion):
    """
    Recupera el diccionario de metadatos devuelto por
    leer_estacion_conagua() sin evaluar DataFrames como booleanos.
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
        metadata = {}

    return metadata


def obtener_datos_climaticos(estacion):
    """
    Recupera el DataFrame climático devuelto por
    leer_estacion_conagua() sin utilizar operadores 'or'
    sobre DataFrames.
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


def obtener_valor_metadata(metadata, posibles_claves):
    """
    Busca un valor de metadatos considerando varias posibles
    formas de nombrar una misma variable.
    """

    if not isinstance(metadata, dict):
        return None

    for clave in posibles_claves:

        if clave in metadata:

            valor = metadata[clave]

            if pd.notna(valor):
                return valor

    return None


# ============================================================
# PRUEBA PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("PRUEBA INTEGRACIÓN: DOWNLOADER -> CONAGUA READER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Verificar carpeta
    # --------------------------------------------------------

    if not CARPETA_ESTADO.exists():

        raise FileNotFoundError(
            f"No existe la carpeta:\n{CARPETA_ESTADO}"
        )

    # --------------------------------------------------------
    # 2. Buscar archivos XLSX
    # --------------------------------------------------------

    archivos = sorted(
        archivo
        for archivo in CARPETA_ESTADO.glob("*.xlsx")
        if not archivo.name.startswith("~$")
    )

    if not archivos:

        raise FileNotFoundError(
            "No se encontraron archivos XLSX descargados."
        )

    print()
    print(
        f"Archivos encontrados: {len(archivos)}"
    )

    # --------------------------------------------------------
    # 3. Contenedores de resultados
    # --------------------------------------------------------

    resultados = []
    errores = []

    # --------------------------------------------------------
    # 4. Procesar archivos
    # --------------------------------------------------------

    for numero, archivo in enumerate(
        archivos,
        start=1
    ):

        print()
        print("-" * 70)

        print(
            f"[{numero}/{len(archivos)}] "
            f"{archivo.name}"
        )

        try:

            # =================================================
            # LEER ESTACIÓN
            # =================================================

            estacion = leer_estacion_conagua(
                archivo
            )

            # =================================================
            # VALIDAR RESPUESTA DEL LECTOR
            # =================================================

            if estacion is None:

                raise ValueError(
                    "leer_estacion_conagua() devolvió None."
                )

            if not isinstance(estacion, dict):

                raise TypeError(
                    "leer_estacion_conagua() no devolvió "
                    "un diccionario."
                )

            # =================================================
            # RECUPERAR METADATOS
            # =================================================

            metadata = obtener_metadatos(
                estacion
            )

            # =================================================
            # RECUPERAR DATOS CLIMÁTICOS
            # =================================================

            datos = obtener_datos_climaticos(
                estacion
            )

            if datos is None:

                raise ValueError(
                    "El lector no devolvió el DataFrame climático."
                )

            if not isinstance(datos, pd.DataFrame):

                raise TypeError(
                    "Los datos climáticos devueltos "
                    "no son un DataFrame."
                )

            if datos.empty:

                raise ValueError(
                    "El DataFrame climático está vacío."
                )

            # =================================================
            # VALIDAR COLUMNAS
            # =================================================

            columnas_necesarias = [
                "date",
                "pp",
            ]

            faltantes = [
                columna
                for columna in columnas_necesarias
                if columna not in datos.columns
            ]

            if faltantes:

                raise ValueError(
                    f"Faltan columnas normalizadas: {faltantes}"
                )

            # =================================================
            # NORMALIZAR FECHA POR SEGURIDAD
            # =================================================

            datos = datos.copy()

            datos["date"] = pd.to_datetime(
                datos["date"],
                errors="coerce"
            )

            # =================================================
            # NORMALIZAR PRECIPITACIÓN
            # =================================================

            datos["pp"] = pd.to_numeric(
                datos["pp"],
                errors="coerce"
            )

            # =================================================
            # VALIDAR FECHAS
            # =================================================

            fechas_validas = int(
                datos["date"]
                .notna()
                .sum()
            )

            fechas_invalidas = int(
                datos["date"]
                .isna()
                .sum()
            )

            if fechas_validas == 0:

                raise ValueError(
                    "La estación no tiene fechas válidas."
                )

            # =================================================
            # VALIDAR PRECIPITACIÓN
            # =================================================

            precip_validas = int(
                datos["pp"]
                .notna()
                .sum()
            )

            precip_faltantes = int(
                datos["pp"]
                .isna()
                .sum()
            )

            if precip_validas == 0:

                raise ValueError(
                    "La estación no tiene precipitación válida."
                )

            # =================================================
            # DUPLICADOS DE FECHA
            # =================================================

            duplicados_fecha = int(
                datos["date"]
                .duplicated()
                .sum()
            )

            # =================================================
            # EXTRAER METADATOS
            # =================================================

            clave = obtener_valor_metadata(
                metadata,
                [
                    "clave",
                    "CLAVE",
                    "estacion",
                    "ESTACION",
                    "id",
                    "ID",
                ]
            )

            if clave is None:

                clave = archivo.stem.split("_")[0]

            nombre = obtener_valor_metadata(
                metadata,
                [
                    "nombre",
                    "NOMBRE",
                    "estacion_nombre",
                    "ESTACION_NOMBRE",
                ]
            )

            if nombre is None:

                nombre = archivo.stem

            estado = obtener_valor_metadata(
                metadata,
                [
                    "estado",
                    "ESTADO",
                ]
            )

            municipio = obtener_valor_metadata(
                metadata,
                [
                    "municipio",
                    "MUNICIPIO",
                ]
            )

            situacion = obtener_valor_metadata(
                metadata,
                [
                    "situacion",
                    "SITUACION",
                ]
            )

            latitud = obtener_valor_metadata(
                metadata,
                [
                    "latitud",
                    "LATITUD",
                    "latitude",
                    "LATITUDE",
                ]
            )

            longitud = obtener_valor_metadata(
                metadata,
                [
                    "longitud",
                    "LONGITUD",
                    "longitude",
                    "LONGITUDE",
                ]
            )

            # =================================================
            # PERIODO
            # =================================================

            fecha_inicio = datos["date"].min()
            fecha_fin = datos["date"].max()

            # =================================================
            # PRECIPITACIÓN
            # =================================================

            precip_max = datos["pp"].max()

            eventos_50mm = int(
                (datos["pp"] >= 50).sum()
            )

            # =================================================
            # RESULTADO
            # =================================================

            resultado = {

                "archivo":
                    archivo.name,

                "clave":
                    clave,

                "nombre":
                    nombre,

                "estado":
                    estado,

                "municipio":
                    municipio,

                "situacion":
                    situacion,

                "registros":
                    len(datos),

                "fechas_validas":
                    fechas_validas,

                "fechas_invalidas":
                    fechas_invalidas,

                "precip_validas":
                    precip_validas,

                "precip_faltantes":
                    precip_faltantes,

                "duplicados_fecha":
                    duplicados_fecha,

                "fecha_inicio":
                    fecha_inicio,

                "fecha_fin":
                    fecha_fin,

                "precip_max":
                    precip_max,

                "eventos_50mm":
                    eventos_50mm,

                "latitud":
                    latitud,

                "longitud":
                    longitud,
            }

            resultados.append(
                resultado
            )

            # =================================================
            # SALIDA INDIVIDUAL
            # =================================================

            print("  OK")

            print(
                f"  Clave: {clave}"
            )

            print(
                f"  Estación: {nombre}"
            )

            print(
                f"  Registros: {len(datos):,}"
            )

            print(
                f"  Fechas válidas: "
                f"{fechas_validas:,}"
            )

            print(
                f"  Precipitación válida: "
                f"{precip_validas:,}"
            )

            print(
                f"  Periodo: "
                f"{fecha_inicio} a {fecha_fin}"
            )

            print(
                f"  Precipitación máxima: "
                f"{precip_max} mm"
            )

            print(
                f"  Eventos >= 50 mm: "
                f"{eventos_50mm}"
            )

            print(
                f"  Fechas duplicadas: "
                f"{duplicados_fecha}"
            )

            print(
                f"  Coordenadas: "
                f"{latitud}, {longitud}"
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
                f"{type(error).__name__}: {error}"
            )

    # ========================================================
    # 5. RESUMEN
    # ========================================================

    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)

    print(
        f"Archivos encontrados: "
        f"{len(archivos)}"
    )

    print(
        f"Estaciones compatibles: "
        f"{len(resultados)}"
    )

    print(
        f"Estaciones con error: "
        f"{len(errores)}"
    )

    # ========================================================
    # 6. TABLA DE RESULTADOS
    # ========================================================

    if resultados:

        df_resultados = pd.DataFrame(
            resultados
        )

        print()
        print("=" * 70)
        print("ESTACIONES COMPATIBLES")
        print("=" * 70)

        columnas_resumen = [
            "clave",
            "nombre",
            "registros",
            "fecha_inicio",
            "fecha_fin",
            "precip_max",
            "eventos_50mm",
            "precip_faltantes",
            "duplicados_fecha",
            "latitud",
            "longitud",
        ]

        columnas_resumen = [
            columna
            for columna in columnas_resumen
            if columna in df_resultados.columns
        ]

        print(
            df_resultados[
                columnas_resumen
            ].to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Guardar diagnóstico
        # ----------------------------------------------------

        ruta_resultados = (
            ROOT
            / "diagnostico_downloader_reader.csv"
        )

        df_resultados.to_csv(
            ruta_resultados,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            f"Diagnóstico guardado en:"
        )

        print(
            ruta_resultados
        )

    # ========================================================
    # 7. ERRORES
    # ========================================================

    if errores:

        df_errores = pd.DataFrame(
            errores
        )

        print()
        print("=" * 70)
        print("ERRORES")
        print("=" * 70)

        print(
            df_errores.to_string(
                index=False
            )
        )

        ruta_errores = (
            ROOT
            / "errores_downloader_reader.csv"
        )

        df_errores.to_csv(
            ruta_errores,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "Archivo de errores guardado en:"
        )

        print(
            ruta_errores
        )

    # ========================================================
    # 8. DIAGNÓSTICO DE COORDENADAS
    # ========================================================

    if resultados:

        df_resultados = pd.DataFrame(
            resultados
        )

        coordenadas_completas = (
            df_resultados[
                ["latitud", "longitud"]
            ]
            .notna()
            .all(axis=1)
            .sum()
        )

        coordenadas_faltantes = (
            len(df_resultados)
            - coordenadas_completas
        )

        print()
        print("=" * 70)
        print("DIAGNÓSTICO DE METADATOS ESPACIALES")
        print("=" * 70)

        print(
            f"Estaciones con coordenadas: "
            f"{coordenadas_completas}"
        )

        print(
            f"Estaciones sin coordenadas: "
            f"{coordenadas_faltantes}"
        )

    # ========================================================
    # 9. VALIDACIÓN FINAL
    # ========================================================

    print()
    print("=" * 70)

    if len(resultados) == 0:

        print(
            "RESULTADO: FALLÓ LA INTEGRACIÓN"
        )

        print("=" * 70)

        raise AssertionError(
            "Ningún archivo descargado pudo ser leído "
            "correctamente por conagua_reader.py."
        )

    elif len(errores) == 0:

        print(
            "RESULTADO: INTEGRACIÓN COMPLETA"
        )

        print(
            f"Los {len(resultados)} archivos fueron "
            f"leídos correctamente."
        )

    else:

        print(
            "RESULTADO: INTEGRACIÓN PARCIAL"
        )

        print(
            f"{len(resultados)} archivos compatibles "
            f"de {len(archivos)}."
        )

        print(
            f"{len(errores)} archivos requieren revisión."
        )

    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()