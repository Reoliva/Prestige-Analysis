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
