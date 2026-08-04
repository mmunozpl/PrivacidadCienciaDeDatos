# Capítulo 4 — Seudonimización y anonimización clásica

## Dependencias

Las del entorno base: `numpy`, `pandas`, `pyarrow`.

## Comandos

```bash
python3 src/cap04/hash_no_anonimiza.py   # el ataque de diccionario
python3 src/cap04/mondrian.py            # Mondrian y sus variantes
```

El primero mide la velocidad de hashing de la máquina y calcula el
tiempo de barrido de los espacios de identificadores españoles (móvil,
DNI, matrícula, número de historia); demuestra que la sal pública no
protege y que el HMAC con clave seudonimiza pero no anonimiza.

El segundo implementa Mondrian, lo compara con la generalización
global del capítulo 3 usando el error de una consulta real, y prueba
dos variantes con jerarquía: la provincia como dimensión (peor) y como
clave de partición (error cero, sin suprimir nada).
