# Doctorado-DCAG "Herramienta detección de inundaciones IOT, IA e Imágenes Satelitales"

Repositorio de trabajo para el análisis de precipitación extrema, excedencias, periodos de retorno y modelado espacial aplicado al estudio de inundaciones en el estado de Colima.

## Estructura del repositorio

- `notebooks/`: notebooks limpios de ejecución y análisis
- `scripts/`: módulos reutilizables en Python
- `data/`: datos de entrada y archivos auxiliares
- `results/`: salidas del análisis (tablas, figuras, rasters)
- `docs/`: documentación metodológica y notas del proyecto

## Notebooks principales

- `01_gev_return_levels.ipynb`: curvas de retorno por estación y procesamiento batch
- `02_exceedance_pipeline.ipynb`: cálculo de excedencias, interpolación, incertidumbre y exportación a QGIS

## Scripts principales

- `bootstrap_utils.py`
- `station_analysis.py`
- `batch_return_levels.py`
- `exceedance.py`
- `interpolation.py`
- `uncertainty.py`
- `raster_export.py`
- `mapping.py`

## Objetivos del proyecto

- Analizar máximos anuales de precipitación
- Estimar niveles de retorno mediante GEV
- Calcular probabilidad de excedencia de lluvia intensa
- Generar mapas interpolados e incertidumbre espacial
- Exportar productos compatibles con QGIS
- Mantener un flujo reproducible para tesis doctoral

## Herramientas utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- SciPy
- GeoPandas
- Rasterio


## Hito de versión

Primer corte estable con estructura modular para curvas de retorno y excedencias.

## Autor

Raúl Uzias Rentería Flores
