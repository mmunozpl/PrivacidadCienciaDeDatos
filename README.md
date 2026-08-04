# Privacy · Data Science

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá**

Applied privacy for data science, written to be readable and
mathematically rigorous at once: from re-identification risk —measured
in bits over a synthetic population calibrated with Spain's official
2025 census figures— to privacy-enhancing technologies with working
code (differential privacy, synthetic data, federated learning,
encrypted computation) and the European rules (GDPR, AI Act) treated
as engineering requirements. Seventeen chapters in four parts: *Data
and its risk*, *PETs with code*, *Special categories as the thread*,
and *Regulation as engineering*.

This repository holds a **browsable web preview** of the book and the
**reproducible code** behind every measured figure. Each chapter's
closing extras —the measured practice, the exercises with their
solutions appendix, and the «EU vs US» section— live only in
the **complete work** (print, PDF and EPUB), distributed separately.

> 📘 **Book page** and more of the author's work:
> [manpla.net/libros/privacidad-ciencia-datos](https://manpla.net/libros/privacidad-ciencia-datos/)

## Contents

```
.
├── docs/                # web edition (Quarto): one HTML page per chapter,
│                        #   SVG figures rendered with the book's own LaTeX
├── src/                 # reproducible code per chapter + shared utilities
└── data/                # synthetic dataset + INE (Spanish statistics office)
                         #   marginals that calibrate it
```

## Read the book

The web edition is published with GitHub Pages from `docs/`:

> https://mmunozpl.github.io/PrivacidadCienciaDeDatos/

Each chapter of the web edition omits its paid closing sections
(measured practice, exercises and «EU vs US»), available in the
complete work.

## Run the code

Python ≥ 3.10 with `numpy`, `pandas` and `pyarrow`; chapters with
extra dependencies document them in their own `README.md`. For
example:

```bash
python3 src/cap01/generar_dataset.py
python3 src/cap01/entropia_cuasi.py
```

## License

The **text, figures and web edition** are released under
[CC BY-NC-ND 4.0](LICENSE).
The **code and examples** in `src/` are released under
[MIT](src/LICENSE).
Population marginals in `data/ine/` come from INE (Instituto Nacional
de Estadística, Spain) open data and keep their own attribution terms.

## How to cite

See [`CITATION.cff`](CITATION.cff).
