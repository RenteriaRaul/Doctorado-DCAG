from scripts.exceedance import procesar_excedencia_batch_csv

# ======================================================
# CONFIGURACIÓN
# ======================================================

CARPETA = r"G:\My Drive\Doctorado\Probabilidad\Precipitación\smn_downloads\Estaciones_Colima"

# ======================================================
# EJECUCIÓN
# ======================================================

resultados, log, out_master, out_log = (
    procesar_excedencia_batch_csv(
        carpeta_estaciones=CARPETA,
        patron="dat*.csv",
        col_precip="pp",
        col_fecha="date",
        threshold=50.0,
        consolidar_duplicados=True,
        exportar=True,
    )
)

print("=" * 70)
print("RESULTADOS")
print("=" * 70)

print(resultados.head())

print()

print("=" * 70)
print("LOG")
print("=" * 70)

print(log.head())

print()

print("=" * 70)
print("ARCHIVOS")
print("=" * 70)

print(out_master)
print(out_log)