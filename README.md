# Doctorado-DCAG "Herramienta detección de inundaciones IOT, IA e Imágenes Satelitales"

Repositorio de trabajo para el análisis de precipitación extrema, excedencias, periodos de retorno y modelado espacial aplicado al estudio de inundaciones en el estado de Colima.

## Estructura del repositorio

- `notebooks/`: notebooks limpios de ejecución y análisis
- `scripts/`: módulos reutilizables en Python
- `data/`: datos de entrada y archivos auxiliares
- `results/`: salidas del análisis (tablas, figuras, rasters)
- `docs/`: documentación metodológica y notas del proyecto

## Notebooks principales

### Módulo CONAGUA (estaciones meteorológicas)

- `01_gev_return_levels.ipynb`: curvas de retorno por estación y procesamiento batch
- `02_exceedance_pipeline.ipynb`: cálculo de excedencias, interpolación, incertidumbre y exportación a QGIS

### Módulo Sustax (precipitación satelital)

Ubicación: `notebooks/sustax/`

- `01_sustax_data_pipeline.ipynb`: limpieza, integración y estructuración de datos Sustax
- `02_sustax_validation_obs_vs_era5.ipynb`: validación histórica entre observaciones (CONAGUA) y ERA5
- `03_sustax_event_analysis_2015.ipynb`: análisis del evento extremo octubre 2015
- `04_sustax_extreme_analysis_heatmaps.ipynb`: generación de heatmaps de percentiles y extremos
- `05_sustax_future_projection.ipynb`: proyección de eventos extremos bajo escenarios SSP
- `06_sustax_flood_detection_proxy.ipynb`: detección proxy de inundaciones basada en extremos hidrometeorológicos

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
- Integrar datos satelitales de precipitación (Sustax)
- Analizar eventos extremos históricos y futuros
- Evaluar escenarios climáticos SSP
- Identificar condiciones potenciales de inundación
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
- Seaborn

## Hito de versión

- Primer corte estable: módulo CONAGUA (curvas de retorno y excedencias)
- Segundo corte estable: integración del módulo Sustax (validación, análisis de extremos y proyección futura)

## Autor

Raúl Uzias Rentería Flores
