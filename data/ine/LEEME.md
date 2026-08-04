# data/ine/ — marginales reales que calibran el dataset del libro

Cifras oficiales de población del INE (Estadística Continua de
Población / Cifras de Población, **a 1 de enero de 2025**),
descargadas de la API Tempus3 con `herramientas/descargar_ine.py`
el 2026-08-04. Población total: **49 128 297** (cuadra exacta entre
ambas tablas).

| Fichero | Contenido | Tabla INE |
|---|---|---|
| `edad_sexo.csv` | pirámide nacional: edad simple 0–105+ × sexo | 56934 |
| `provincias.csv` | población por provincia, con el código INE (= dos primeros dígitos del CP) | 56945 (filtrada: todas las edades, ambos sexos) |

`codigo/cap01/generar_dataset.py` muestrea de aquí la edad, el sexo y
el prefijo provincial del código postal; profesión, diagnóstico y el
sufijo del CP son estilizados. Para refrescar las cifras: volver a
ejecutar el descargador y regenerar el dataset (los números citados
en la prosa del cap. 1 deberán actualizarse a la vez).
