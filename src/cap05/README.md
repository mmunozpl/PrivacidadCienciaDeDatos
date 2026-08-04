# Capítulo 5 — Privacidad diferencial

## Dependencias

Las del entorno base: `numpy`, `pandas`, `pyarrow`. Los mecanismos y
la contabilidad viven en `src/comun/ruido_dp.py` y no dependen de
nada más: la calibración exacta del gaussiano se resuelve con
`math.erfc`, sin SciPy.

## Comandos

```bash
python3 src/cap05/mecanismos.py       # perdida, calibracion, eleccion
python3 src/cap05/sensibilidad.py     # recorte, vecindad, unidad
python3 src/cap05/composicion.py      # basica, avanzada, zCDP, paralela
python3 src/cap05/presupuesto.py      # la consulta del libro, con DP
python3 src/cap05/exponencial.py      # moda y mediana privadas
python3 src/cap05/vector_disperso.py  # SVT y auditoria de la variante rota
```

Se ejecutan en cualquier orden salvo `presupuesto.py`, que compara sus
resultados con los CSV del capítulo 3 (`escalera_generalizacion.csv`)
y del 4 (`mondrian.csv`) si existen.

`mecanismos.py` simula cuatro millones de salidas para medir la
variable de pérdida de privacidad, compara la calibración gaussiana
clásica con la exacta de Balle y Wang, y decide qué mecanismo conviene
según la **forma** de la consulta: sobre casillas disjuntas gana
Laplace siempre; sobre consultas anidadas Gauss se impone a partir de
unas veinte componentes.

`sensibilidad.py` mide las tres decisiones que se toman antes del
mecanismo. Deriva un registro de visitas con la frecuentación de
atención primaria española para mostrar que un `eps=1` por fila es un
`eps=63` por persona, con ventaja 0,92 para el adversario ya en el
paciente mediano.

`composicion.py` compara las tres contabilidades y mide lo que vale
reconocer la composición paralela: un factor 310 sobre el histograma
del libro.

`presupuesto.py` responde por tercera vez la consulta que recorren los
capítulos 3 y 4, ahora con garantía, y mide quién la paga: un factor
88 de error entre Madrid y Melilla con el mismo `eps`.

`exponencial.py` compara el mecanismo exponencial con el *report noisy
max* para una salida categórica, y con Laplace para la mediana, donde
la sensibilidad global del dominio lo hace inservible.

`vector_disperso.py` barre el número de consultas para enseñar que el
coste del SVT lo fija el tope `c` y no `m`, y audita empíricamente una
variante publicada incorrecta: gasta 58,81 anunciando 1.
