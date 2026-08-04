# Código del libro

Código por capítulo del libro **Privacidad para la ciencia de datos**.
Licencia [MIT](LICENSE); el texto del libro tiene la suya propia (ver
la raíz del repositorio).

## Entorno

- Python ≥ 3.10 con `numpy`, `pandas` y `pyarrow`.
- Los capítulos de la Parte II añaden dependencias propias
  (`opacus`, `torch`, `flwr`…) que documenta el README de cada
  capítulo.

## Mapa de carpetas

| Carpeta | Qué contiene |
|---|---|
| `comun/` | utilidades transversales: semillas (`determinismo.py`), logging con progreso y verificación de artefactos (`registro.py`), mecanismos y contabilidad de privacidad diferencial (`ruido_dp.py`), métricas de reidentificación (`reident.py`), esquema anotado con validación (`esquema.py`) |
| `cap01/` | dataset sintético de cuasi-identificadores y riesgo de partida |
| `cap02/` | los ataques reproducidos: enlace, inferencia de pertenencia, memorización |
| `cap03/` | métricas de riesgo, unicidad poblacional e informe reproducible |
| `cap04/` | ataque de diccionario sobre hashes y particionado de Mondrian |
| `cap05/` | mecanismos de ruido, sensibilidad, composición y presupuesto |

## Convenciones

- PEP8, líneas de máximo 80 caracteres, anotaciones de tipo nativas.
- Reproducibilidad: toda aleatoriedad pasa por `comun/determinismo.py`.
- Las tareas pesadas registran progreso con `comun/registro.py`.
- Todo artefacto guardado imprime 15 observaciones aleatorias como
  verificación final.
