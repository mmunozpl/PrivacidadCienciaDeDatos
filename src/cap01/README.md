# Capítulo 1 — El dato personal como objeto de ingeniería

## Dependencias

Las del entorno base del libro: `numpy`, `pandas`, `pyarrow`.

## Comandos

```bash
# marginales reales del INE (si data/ine/ no existe o hay que refrescar)
python3 herramientas/descargar_ine.py

# genera data/processed/poblacion_sintetica.parquet y mide su riesgo
python3 src/cap01/generar_dataset.py

# entropías, unicidad esperada y gemelos poblacionales (lambda)
python3 src/cap01/entropia_cuasi.py
```

El generador muestrea las marginales reales del INE (`data/ine/`,
cifras a 1-1-2025): pirámide de edad y sexo, y código postal con el
prefijo provincial verdadero; profesión y diagnóstico son
estilizados. Fija semillas, valida el esquema anotado, guarda el
dataset transversal de la Parte I y cierra con el k-anonimato, la
unicidad muestral y las 15 observaciones aleatorias de verificación.
