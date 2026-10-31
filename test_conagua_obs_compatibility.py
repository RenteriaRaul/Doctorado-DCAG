from pathlib import Path

import pandas as pd

from scripts.conagua_reader import leer_lote_conagua


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CARPETA_CONAGUA = (
    PROJECT_ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
    / "COLIMA"
)


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 80)
    print("PRUEBA — COMPATIBILIDAD CONAGUA → OBS SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Verificar carpeta
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("1. FUENTE CONAGUA")
    print("-" * 80)

    print(f"Carpeta:\n{CARPETA_CONAGUA}")

    if not CARPETA_CONAGUA.exists():
        raise FileNotFoundError(
            f"No existe la carpeta:\n{CARPETA_CONAGUA}"
        )

    archivos = sorted(
        fp
        for fp in CARPETA_CONAGUA.glob("*.xlsx")
        if not fp.name.startswith("~$")
    )

    print(f"\nArchivos XLSX encontrados: {len(archivos)}")

    if not archivos:
        raise RuntimeError(
            "No se encontraron archivos CONAGUA."
        )

    # --------------------------------------------------------
    # 2. Leer con el lector oficial del proyecto
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. LECTURA CON conagua_reader.py")
    print("-" * 80)

    resultado = leer_lote_conagua(
        CARPETA_CONAGUA,
        patron="*.xlsx",
    )

    # Compatibilidad con las dos formas que puede devolver
    # nuestro lector: lista directa o tupla con metadata.
    if isinstance(resultado, tuple):
        estaciones = resultado[0]
    else:
        estaciones = resultado

    print(
        f"Estaciones devueltas por el lector: "
        f"{len(estaciones)}"
    )

    # --------------------------------------------------------
    # 3. Construir estructura requerida por obs.py
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("3. ESTRUCTURA ESPACIAL PARA obs.py")
    print("-" * 80)

    rows = []
    errores = []

    for estacion in estaciones:

        try:
            metadata = estacion.get("metadata", {})

            station_id = (
                metadata.get("station")
                or metadata.get("clave")
                or metadata.get("station_id")
            )

            lat = (
                metadata.get("latitude")
                or metadata.get("latitud")
                or metadata.get("lat")
            )

            lon = (
                metadata.get("longitude")
                or metadata.get("longitud")
                or metadata.get("lon")
            )

            nombre = (
                metadata.get("name")
                or metadata.get("nombre")
                or ""
            )

            lat = pd.to_numeric(
                pd.Series([lat]),
                errors="coerce",
            ).iloc[0]

            lon = pd.to_numeric(
                pd.Series([lon]),
                errors="coerce",
            ).iloc[0]

            if station_id is None:
                errores.append(
                    {
                        "archivo": estacion.get("path", ""),
                        "problema": "Sin station_id",
                    }
                )
                continue

            if pd.isna(lat) or pd.isna(lon):
                errores.append(
                    {
                        "archivo": estacion.get("path", ""),
                        "problema": "Sin coordenadas válidas",
                    }
                )
                continue

            rows.append(
                {
                    "station_id": str(station_id).strip(),
                    "lat": float(lat),
                    "lon": float(lon),
                    "nombre": nombre,
                }
            )

        except Exception as exc:

            errores.append(
                {
                    "archivo": estacion.get("path", ""),
                    "problema": str(exc),
                }
            )

    conagua_spatial = pd.DataFrame(rows)

    print(
        f"Estaciones espaciales válidas: "
        f"{len(conagua_spatial)}"
    )

    print(
        f"Registros con problemas: "
        f"{len(errores)}"
    )

    if not conagua_spatial.empty:

        print("\nPrimeras estaciones:")

        print(
            conagua_spatial.head(10).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 4. Control de coordenadas
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. CONTROL DE COORDENADAS")
    print("-" * 80)

    if conagua_spatial.empty:
        raise RuntimeError(
            "No fue posible construir la estructura espacial."
        )

    lat_valid = conagua_spatial["lat"].between(
        -90,
        90,
    )

    lon_valid = conagua_spatial["lon"].between(
        -180,
        180,
    )

    coords_validas = lat_valid & lon_valid

    print(
        f"Coordenadas geográficamente válidas: "
        f"{int(coords_validas.sum())}/"
        f"{len(conagua_spatial)}"
    )

    if (~coords_validas).any():

        print("\nCoordenadas fuera de rango:")

        print(
            conagua_spatial.loc[
                ~coords_validas
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 5. Duplicados
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. CONTROL DE IDENTIFICADORES")
    print("-" * 80)

    duplicados = conagua_spatial[
        conagua_spatial["station_id"].duplicated(
            keep=False
        )
    ]

    print(
        f"Station ID duplicados: "
        f"{duplicados['station_id'].nunique()}"
    )

    if not duplicados.empty:

        print("\nDuplicados encontrados:")

        print(
            duplicados.sort_values(
                "station_id"
            ).to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. Compatibilidad exacta con build_neighbors()
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. COMPATIBILIDAD CON obs.py")
    print("-" * 80)

    columnas_requeridas = {
        "station_id",
        "lat",
        "lon",
    }

    faltantes = (
        columnas_requeridas
        - set(conagua_spatial.columns)
    )

    if faltantes:
        print(
            "ERROR: faltan columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )
    else:
        print(
            "OK: estructura compatible con "
            "build_neighbors()."
        )

    # --------------------------------------------------------
    # 7. Resultado final
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("7. VALIDACIÓN")
    print("-" * 80)

    observaciones = []

    if len(conagua_spatial) != len(estaciones):
        observaciones.append(
            "No todas las estaciones leídas pudieron "
            "convertirse a estructura espacial."
        )

    if not coords_validas.all():
        observaciones.append(
            "Existen coordenadas fuera del rango geográfico."
        )

    if not duplicados.empty:
        observaciones.append(
            "Existen identificadores de estación duplicados."
        )

    if faltantes:
        observaciones.append(
            "La estructura no contiene todas las columnas "
            "requeridas por obs.py."
        )

    if observaciones:

        print("PRUEBA FINALIZADA CON OBSERVACIONES:")

        for obs in observaciones:
            print(f"  - {obs}")

    else:

        print(
            "OK: todas las estaciones CONAGUA "
            "tienen estructura espacial utilizable."
        )

        print(
            "OK: las coordenadas son válidas."
        )

        print(
            "OK: los identificadores son únicos."
        )

        print(
            "OK: conagua_reader.py puede alimentar "
            "la etapa espacial de obs.py."
        )

    # --------------------------------------------------------
    # 8. Errores de extracción, si existen
    # --------------------------------------------------------

    if errores:

        print("\n" + "-" * 80)
        print("8. DETALLE DE PROBLEMAS")
        print("-" * 80)

        print(
            pd.DataFrame(errores).to_string(
                index=False
            )
        )

    print("\n" + "=" * 80)
    print("PRUEBA DE COMPATIBILIDAD FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()