# Privacidad · Ciencia de datos

🇬🇧 [English](README.md) · 🇪🇸 Español

**Manuel Muñoz Plá**

Privacidad aplicada a la ciencia de datos, escrita con afán
divulgativo y rigor técnico y matemático a la vez: del riesgo de
reidentificación —medido en bits sobre una población sintética
calibrada con las cifras oficiales del INE de 2025— a las tecnologías
de privacidad con código que funciona (privacidad diferencial, datos
sintéticos, aprendizaje federado, computación cifrada) y la norma
europea (RGPD, AI Act) tratada como requisitos de ingeniería.
Diecisiete capítulos en cuatro partes: *El dato y su riesgo*, *PETs
con código*, *Categorías especiales como hilo* y *La norma como
ingeniería*.

Este repositorio reúne una **vista previa web navegable** del libro y
el **código reproducible** que genera cada cifra medida. La sección de
cierre «Marcos frente a frente: ENS/UE vs NIST» de cada capítulo vive
solo en la **obra completa** (papel y PDF), que se distribuye por
separado.

> 📘 **Ficha del libro** y más obras del autor:
> [manpla.net/libros/privacidad-ciencia-datos](https://manpla.net/libros/privacidad-ciencia-datos/)

## Contenido

```
.
├── docs/                # edición web (Quarto): un HTML por capítulo,
│                        #   figuras SVG renderizadas con el LaTeX del libro
├── src/                 # código reproducible por capítulo + utilidades comunes
└── data/                # dataset sintético + marginales del INE que lo
                         #   calibran
```

## Leer el libro

La edición web se publica con GitHub Pages desde `docs/`:

> https://mmunozpl.github.io/PrivacidadCienciaDeDatos/

Cada capítulo de la edición web omite la sección «Marcos frente a
frente: ENS/UE vs NIST», disponible en la obra completa.

## Ejecutar el código

Python ≥ 3.10 con `numpy`, `pandas` y `pyarrow`; los capítulos con
dependencias propias las documentan en su `README.md`. Por ejemplo:

```bash
python3 src/cap01/generar_dataset.py
python3 src/cap01/entropia_cuasi.py
```

## Licencia

El **texto, las figuras y la edición web** se publican bajo
[CC BY-NC-ND 4.0](LICENSE).
El **código y los ejemplos** de `src/`, bajo [MIT](src/LICENSE).
Las marginales de población de `data/ine/` proceden de los datos
abiertos del INE y conservan sus condiciones de atribución.

## Cómo citar

Ver [`CITATION.cff`](CITATION.cff).
