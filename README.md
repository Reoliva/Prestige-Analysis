# Prestige Analysis

Este repositorio contiene el código y los datos base utilizados para analizar la relación entre **prestigio** e **influencia grupal** en equipos de trabajo, a partir de un dataset consolidado multimodal.

El flujo general del proyecto se divide en dos partes:

- `prestige_analysis.py`: procesa el archivo de entrada, limpia los datos, calcula variables derivadas de prestigio e influencia, ejecuta análisis estadísticos y genera un reporte consolidado.
- `prestige_plots.py`: toma el archivo de resultados generado por el análisis y crea gráficos en formato `.png` para interpretar los hallazgos.

## Archivos principales

- `consolidado_ordenado.xlsx`  
  Archivo de entrada con los datos consolidados.

- `prestige_analysis.py`  
  Script principal de análisis.

- `prestige_plots.py`  
  Script para generar visualizaciones a partir del reporte de análisis.

## Requisitos

Se recomienda usar Python 3.10+.

Librerías principales:

```bash
pip install pandas numpy scipy scikit-learn networkx openpyxl matplotlib
````

## Uso

### 1. Ejecutar el análisis

```bash
python prestige_analysis.py
```

Este script toma como input `consolidado_ordenado.xlsx` y genera un archivo de resultados consolidado en Excel.

### 2. Generar los gráficos

```bash
python prestige_plots.py
```

Este script usa el archivo de resultados generado en el paso anterior y exporta gráficos `.png` en una carpeta de salida.

## Salidas esperadas

* un archivo Excel con resultados procesados
* tablas de correlación
* rankings de fórmulas de prestigio
* métricas de validación
* gráficos como:

  * heatmaps
  * scatter plots
  * feature importance
  * resúmenes de validación cruzada

## Propósito

El objetivo de este repositorio es dejar trazable y reproducible el pipeline de análisis usado para estudiar cómo distintas dimensiones del prestigio, especialmente la **experiencia**, se relacionan con métricas de influencia como atención visual, tiempo de habla y centralidad estructural.

## Notas

* El repositorio conserva el archivo de entrada original para facilitar la reproducibilidad.
* Los resultados deben interpretarse como parte de un proceso de análisis académico en desarrollo.

