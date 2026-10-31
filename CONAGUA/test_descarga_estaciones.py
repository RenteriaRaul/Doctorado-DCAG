from descargar_estaciones import (
    obtener_catalogo_conagua,
    filtrar_estaciones,
    procesar_estacion
)

import pandas as pd


# ============================================================
# CONFIGURACIÓN DE LA PRUEBA
# ============================================================

CLAVES_PRUEBA = [
    "6001",
    "6002",
    "6003"
]

CARPETA_SALIDA = (
    "CONAGUA/datos_prueba"
)


# ============================================================
# 1. OBTENER CATÁLOGO
# ============================================================

print("=" * 70)
print("PRUEBA DE DESCARGA DE ESTACIONES")
print("=" * 70)

catalogo = obtener_catalogo_conagua()


# ============================================================
# 2. FILTRAR COLIMA
# ============================================================

colima = filtrar_estaciones(
    catalogo,
    estado="COLIMA"
)

print(
    "\nEstaciones encontradas en Colima:",
    len(colima)
)


# ============================================================
# 3. ASEGURAR CLAVE COMO TEXTO
# ============================================================

colima["clave"] = (
    colima["clave"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. SELECCIONAR SOLO LAS 3 DE PRUEBA
# ============================================================

prueba = colima[
    colima["clave"].isin(
        CLAVES_PRUEBA
    )
].copy()


print("\nEstaciones seleccionadas:")

print(
    prueba[
        [
            "clave",
            "nombre",
            "situacion"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 5. PROCESAR
# ============================================================

resultados = []

errores = []


for _, estacion in prueba.iterrows():

    try:

        resultado = procesar_estacion(
            estacion,
            CARPETA_SALIDA
        )

        resultados.append(
            resultado
        )

        print(
            "  OK"
        )

        print(
            "  Registros:",
            resultado["filas_validas"]
        )

        print(
            "  Periodo:",
            resultado["fecha_inicio"],
            "a",
            resultado["fecha_fin"]
        )

        print(
            "  Máxima precipitación:",
            resultado["precip_max"],
            "mm"
        )

        print(
            "  Eventos >= 50 mm:",
            resultado["eventos_50mm"]
        )


    except Exception as e:

        print(
            "  ERROR:",
            e
        )

        errores.append({
            "clave":
                estacion["clave"],

            "nombre":
                estacion["nombre"],

            "error":
                str(e)
        })


# ============================================================
# 6. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN DE LA PRUEBA")
print("=" * 70)


if resultados:

    df_resultados = pd.DataFrame(
        resultados
    )

    columnas = [
        "clave",
        "nombre",
        "filas_originales",
        "filas_validas",
        "fechas_invalidas",
        "fechas_duplicadas",
        "fecha_inicio",
        "fecha_fin",
        "precip_max",
        "eventos_50mm",
        "faltantes_precip"
    ]

    print(
        df_resultados[
            columnas
        ].to_string(
            index=False
        )
    )


print(
    "\nEstaciones correctas:",
    len(resultados)
)

print(
    "Estaciones con error:",
    len(errores)
)


# ============================================================
# 7. GUARDAR DIAGNÓSTICO
# ============================================================

if resultados:

    pd.DataFrame(
        resultados
    ).to_csv(
        "CONAGUA/diagnostico_prueba.csv",
        index=False,
        encoding="utf-8-sig"
    )


if errores:

    pd.DataFrame(
        errores
    ).to_csv(
        "CONAGUA/errores_prueba.csv",
        index=False,
        encoding="utf-8-sig"
    )


print("\n" + "=" * 70)
print("PRUEBA TERMINADA")
print("=" * 70)