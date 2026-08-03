# Privacidad para la ciencia de datos

<!-- Este es el README que viaja al repositorio PÚBLICO. El README.md
     de la raíz privada es otro documento (con notas de proceso) y NO
     se publica. Secciones y orden según FORMATO_PUBLICO.md §5. -->

## Contenido

Privacidad aplicada a la ciencia de datos, con afán divulgativo y
rigor técnico y matemático: 17 capítulos en cuatro partes.

- **I. El dato y su riesgo** — identificadores y cuasi-identificadores,
  ataques a la privacidad y cuantificación del riesgo.
- **II. PETs con código** — anonimización clásica, privacidad
  diferencial (central y local), datos sintéticos, aprendizaje
  federado y computación cifrada.
- **III. Categorías especiales como hilo** — biometría, dato de salud
  e imagen médica, procedencia y autenticidad.
- **IV. La norma como ingeniería** — RGPD operativo, AI Act, privacy
  by design y un caso integral clínico.

## Leer el libro

La edición web, en abierto:
<https://mmunozpl.github.io/PrivacidadCienciaDeDatos/>

Cada capítulo de la edición web omite la sección «Marcos frente a
frente: ENS/UE vs NIST», disponible en la obra completa.

## Ejecutar el código

Python ≥ 3.10 con `numpy`, `pandas` y `pyarrow`; los capítulos con
dependencias propias las documentan en su `README.md`. Por ejemplo:

```bash
python3 codigo/cap01/generar_dataset.py
```

## Licencia

El **texto, las figuras y la edición web** se publican bajo
[CC BY-NC-ND 4.0](LICENSE).
El **código y los ejemplos** de `codigo/`, bajo
[MIT](codigo/LICENSE).
Los datos de terceros conservan su propia licencia.

## Cómo citar

Ver [`CITATION.cff`](CITATION.cff).
