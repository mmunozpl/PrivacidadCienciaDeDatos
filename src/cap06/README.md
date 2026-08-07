# Capítulo 6 — DP en el entrenamiento

## Dependencias

Además del entorno base (`numpy`, `pandas`, `pyarrow`, `scikit-learn`):

```bash
pip install torch opacus     # opacus 1.6.0 sobre torch 2.11
```

`auditoria.py` usa además `scipy` para el intervalo exacto de
Clopper-Pearson. Todo funciona en CPU; con GPU el barrido de
hiperparámetros baja de decenas de minutos a unos pocos.

## La tarea

El diagnóstico del conjunto base es estilizado y **no depende** de la
edad ni del sexo, de modo que no hay nada aprendible: un bosque
regularizado no supera la clase mayoritaria. Para medir un compromiso
entre exactitud y privacidad hace falta señal, así que
`derivar_tarea.py` construye una —riesgo cardiovascular en tres
niveles— a partir de factores cuya prevalencia por edad y sexo se toma
de `data/ine/prevalencias.csv`. **No toca el conjunto base**: es el
mismo procedimiento con el que el capítulo 5 derivó el registro de
visitas.

El código postal entra como característica de alta cardinalidad y sin
señal: es el combustible de la memorización, y por tanto lo que el
ataque de pertenencia explota.

## Comandos

```bash
python3 src/cap06/derivar_tarea.py     # PRIMERO: construye la tarea
python3 src/cap06/dp_sgd_manual.py     # el mecanismo, a mano y a la vista
python3 src/cap06/contabilidad.py      # RDP, PRV, GDP y la amplificación
python3 src/cap06/muestreo.py          # Poisson frente a barajar
python3 src/cap06/entrenamiento.py     # exactitud contra ataque, por epsilon
python3 src/cap06/hiperparametros.py   # C, tasa de aprendizaje y épocas
python3 src/cap06/sesgo.py             # quién paga la garantía
python3 src/cap06/memorizacion.py      # el coste sobre un modelo que memoriza
python3 src/cap06/unidad.py            # ejemplos frente a personas
python3 src/cap06/auditoria.py         # cota inferior empírica del epsilon
```

Todos salvo el primero exigen `tarea_riesgo.parquet`.
`contabilidad.py` no necesita datos: solo contabiliza.

`dp_sgd_manual.py` implementa las tres líneas de DP-SGD con
`torch.func` y comprueba que sus gradientes por ejemplo coinciden con
los de Opacus hasta la precisión de coma flotante. Mide además un
fallo silencioso que conviene conocer: declararle a Opacus una
`loss_reduction` distinta de la que usa la pérdida multiplica todos
los gradientes por el tamaño del lote, sin excepción ni aviso.

`muestreo.py` comprueba que el cargador de Opacus muestrea de Poisson
—tamaño de lote aleatorio, ejemplos que no salen ninguna vez— y que un
`DataLoader` corriente no. La amplificación por submuestreo se
demuestra sobre el primero; contabilizar como Poisson lo que se
ejecuta barajando anuncia una garantía que no se ha demostrado.

`unidad.py` construye variantes con una, tres y diez filas por
persona y compara las tres maneras de afrontarlo con un objetivo
común de `eps=8` **por persona**: ignorarlo (que gasta 80 y no cuesta
nada porque no se hace nada), calibrar a k, y recortar a un tope. Es
la demostración con números de lo que McMahan et al. llaman coste
«prohibitivo».

`auditoria.py` monta tres adversarios de potencia creciente sobre el
mismo entrenamiento y traduce lo que consiguen en una cota inferior de
epsilon, con intervalos de Clopper-Pearson para que lo que salga sea
una cota y no una estimación.
