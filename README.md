# Imputación de datos faltantes en series de tiempo vía procesos Ornstein-Uhlenbeck

Código desarrollado durante mi práctica en el proyecto **FONDECYT 11230184** *("Atmospheric water vapor and precipitation processes in central and southern Chile")*, bajo la supervisión de Raúl Valenzuela.

## Problema

El GNSS provee datos climatológicos (Zenith Total Delay, ZTD) de estaciones a lo largo de Chile, pero las series suelen tener segmentos de datos faltantes. El objetivo es rellenar esos huecos usando solo la información histórica de cada estación, modelando la serie (tras remover su estacionalidad) como un **proceso de Ornstein-Uhlenbeck**:

```
dX_t = -μ X_t dt + σ dB_t
```

## Enfoque

1. **Descomposición**: se remueve la estacionalidad de la serie original `Z_t` (filtro Savitzky-Golay sobre el promedio por hora-del-día a través de los años), obteniendo una serie centrada `X_t`.
2. **Estimación de parámetros**: se implementaron y compararon dos estimadores de `(μ, σ)` — máxima verosimilitud (MLE) y uno basado en distancia de Wasserstein. El MLE resultó más estable en todos los escenarios probados (ver `ou_estimation.py`).
3. **Imputación vía EM**: los segmentos faltantes de `X_t` se rellenan iterativamente simulando **puentes** del proceso OU (Paso E) y re-estimando `(μ, σ)` con los datos ya completados (Paso M), hasta convergencia.
4. **Validación**: se implementó un test de bondad de ajuste (chi-cuadrado) para verificar qué tan bien cada estación se ajusta a un proceso OU.

## Estructura

| Archivo | Contenido |
|---|---|
| `ou_estimation.py` | Simulación Euler-Maruyama del proceso OU, estimadores MLE y Wasserstein |
| `em_imputation.py` | Puentes OU, algoritmo EM de imputación, test de bondad de ajuste |
| `demo.py` | Demo autocontenido con datos sintéticos (sin datos institucionales) |

## Demo rápido

```bash
python demo.py
```

Genera una serie sintética de un proceso OU con parámetros conocidos, le remueve un segmento de datos, y verifica que el algoritmo de imputación recupera parámetros consistentes con los que generaron la serie completa.

## Nota sobre los datos

Los datos reales de estaciones GNSS usados durante la práctica son propiedad del proyecto FONDECYT 11230184 y no se incluyen en este repositorio. El código se presenta como referencia de la implementación; el `demo.py` permite verificar su funcionamiento de extremo a extremo sin depender de esos datos.

## Stack técnico

`Python` · `numpy` · `pandas` · `scipy` (`optimize`, `stats`, `signal`)

## Contexto adicional

El informe completo de la práctica (con resultados detallados, comparación de estimadores, y aplicación a estaciones reales) está disponible como PDF adjunto por separado.