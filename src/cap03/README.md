# Capítulo 3 — Cuantificar el riesgo

## Dependencias

Las del entorno base: `numpy`, `pandas`, `pyarrow`.

## Comandos

```bash
python3 src/cap03/metricas_medidas.py      # escalera y supresión
python3 src/cap03/unicidad_poblacional.py  # estimadores vs verdad
python3 src/cap03/informe_riesgo.py        # el artefacto
```

Requieren el dataset del capítulo 1. El primero recorre una escalera
de generalización midiendo k, l, t, los riesgos, la utilidad y el
error de una consulta real; el segundo compara estimadores de unicidad
poblacional contra la verdad conocida; el tercero genera el informe de
riesgo reproducible en Markdown y JSON.

Las métricas están implementadas en `src/comun/metricas.py`, para
leerse junto al texto del capítulo.
