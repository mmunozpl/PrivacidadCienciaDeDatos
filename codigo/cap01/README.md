# Capítulo 1 — El dato personal como objeto de ingeniería

## Dependencias

Las del entorno base del libro: `numpy`, `pandas`, `pyarrow`.

## Comandos

```bash
# genera data/processed/poblacion_sintetica.parquet y mide su riesgo
python3 codigo/cap01/generar_dataset.py
```

El script fija semillas, guarda el dataset transversal de la Parte I
y cierra con el k-anonimato del conjunto {edad, sexo, código postal,
profesión}, la unicidad muestral y las 15 observaciones aleatorias de
verificación.
