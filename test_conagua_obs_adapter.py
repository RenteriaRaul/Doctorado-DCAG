from pathlib import Path

import pandas as pd

from scripts.conagua_reader import leer_lote_conagua
from scripts.sustax.obs import build_conagua_series_lookup


PROJECT_ROOT = Path(__file__).resolve().parent

CARPETA_CONAGUA = (
    PROJECT_ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
    / "COLIMA"
)


def main():

    print("=" * 80)
    print("PRUEBA — ADAPTADOR CONAGUA_READER → OBS SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Leer CONAGUA
    # --------------------------------------------------------

    resultado = leer_lote_conagua(
        CARPETA_CONAGUA,
        patron="*.xlsx",
    )

    estaciones = (
        resultado[0]
        if isinstance(resultado, tuple)
        else resultado
    )

    print("\n1. LECTURA CONAGUA")
    print("-" * 80)

    print(
        f"Estaciones leídas: {len(estaciones)}"
    )

    # --------------------------------------------------------
    # 2. Construir lookup
    # --------------------------------------------------------

    lookup = build_conagua_series_lookup(
        estaciones
    )

    print("\n2. ADAPTACIÓN")
    print("-" * 80)

    print(
        f"Series construidas: {len(lookup)}"
    )

    faltantes = []

    for estacion in estaciones:

        meta = estacion.get("metadata", {})

        sid = (
            meta.get("station")
            or meta.get("clave")
            or meta.get("station_id")
        )

        if sid is None:
            continue

        sid = str(sid).strip()

        if sid not in lookup:
            faltantes.append(sid)

    print(
        f"Estaciones sin serie adaptada: "
        f"{len(faltantes)}"
    )

    if faltantes:
        print(
            "IDs sin serie:",
            ", ".join(faltantes),
        )

    # --------------------------------------------------------
    # 3. Validar estructura
    # --------------------------------------------------------

    print("\n3. ESTRUCTURA")
    print("-" * 80)

    errores = []

    resumen = []

    for station_id, df in lookup.items():

        columnas_ok = (
            "date" in df.columns
            and "pp_mm" in df.columns
        )

        fechas_validas = (
            pd.api.types.is_datetime64_any_dtype(
                df["date"]
            )
            if "date" in df.columns
            else False
        )

        negativos = (
            int((df["pp_mm"] < 0).sum())
            if "pp_mm" in df.columns
            else -1
        )

        n_validos = (
            int(df["pp_mm"].notna().sum())
            if "pp_mm" in df.columns
            else 0
        )

        fecha_ini = (
            df["date"].min()
            if not df.empty
            else None
        )

        fecha_fin = (
            df["date"].max()
            if not df.empty
            else None
        )

        resumen.append(
            {
                "station_id": station_id,
                "registros": len(df),
                "pp_validos": n_validos,
                "fecha_ini": fecha_ini,
                "fecha_fin": fecha_fin,
                "negativos": negativos,
            }
        )

        if not columnas_ok:
            errores.append(
                f"{station_id}: columnas incorrectas"
            )

        if not fechas_validas:
            errores.append(
                f"{station_id}: fecha no datetime"
            )

        if negativos > 0:
            errores.append(
                f"{station_id}: precipitación negativa"
            )

    resumen_df = pd.DataFrame(resumen)

    if not resumen_df.empty:

        resumen_df = resumen_df.sort_values(
            "station_id"
        )

        print(
            resumen_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 4. Estadísticas generales
    # --------------------------------------------------------

    print("\n4. RESUMEN GENERAL")
    print("-" * 80)

    if not resumen_df.empty:

        print(
            f"Series: {len(resumen_df)}"
        )

        print(
            f"Registros totales: "
            f"{resumen_df['registros'].sum():,}"
        )

        print(
            f"Observaciones PP válidas: "
            f"{resumen_df['pp_validos'].sum():,}"
        )

        print(
            f"Fecha más antigua: "
            f"{resumen_df['fecha_ini'].min()}"
        )

        print(
            f"Fecha más reciente: "
            f"{resumen_df['fecha_fin'].max()}"
        )

        print(
            f"Valores negativos: "
            f"{resumen_df['negativos'].sum()}"
        )

    # --------------------------------------------------------
    # 5. Resultado
    # --------------------------------------------------------

    print("\n5. VALIDACIÓN")
    print("-" * 80)

    if errores:

        print(
            f"PRUEBA CON {len(errores)} "
            "PROBLEMA(S):"
        )

        for error in errores:
            print(f"  - {error}")

    else:

        print(
            "OK: estructura date / pp_mm correcta."
        )

        print(
            "OK: fechas normalizadas correctamente."
        )

        print(
            "OK: no existen precipitaciones negativas."
        )

        print(
            "OK: las series están listas para "
            "ser utilizadas por obs.py."
        )

    print("\n" + "=" * 80)
    print("PRUEBA FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()