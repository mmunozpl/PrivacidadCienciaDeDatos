# Capítulo 2 — Ataques a la privacidad

Los cuatro ataques del capítulo, sobre el dataset sintético del libro
y sobre modelos entrenados aquí mismo. **Uso exclusivamente
defensivo**: ejecutarlos sobre datos o modelos ajenos sin autorización
es un tratamiento ilícito.

## Dependencias

Las del entorno base (`numpy`, `pandas`, `pyarrow`) más
`scikit-learn`.

## Comandos

```bash
python3 src/cap02/linkage.py         # enlace con una fuente auxiliar
python3 src/cap02/membership.py      # pertenencia: AUC y ROC log-log
python3 src/cap02/memorizacion.py    # canarios: exposición en bits
python3 src/cap02/vulnerabilidad.py  # quién es vulnerable + atributo
```

Requieren el dataset del capítulo 1
(`python3 src/cap01/generar_dataset.py`). Cada script fija semillas y
escribe su artefacto en `data/processed/`, de donde salen las cifras
de las figuras del capítulo.
