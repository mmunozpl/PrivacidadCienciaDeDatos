"""estilo comun de las figuras de datos del libro, con matplotlib.

Por que matplotlib y no solo TikZ. Las figuras conceptuales —bloques,
flechas, fronteras— se componen mejor a mano, y para eso TikZ es
insuperable. Las figuras de DATOS son otra cosa: ejes, escalas
logaritmicas, leyendas, mapas de calor, distribuciones. Colocarlas a
mano obliga a calcular coordenadas, y calcular coordenadas a mano es
exactamente lo que produjo los solapes que hubo que corregir uno a uno
en los capitulos 1 a 5. Un motor de trazado las coloca solo.

Como se mantiene la coherencia tipografica. Se usa el motor PGF de
matplotlib: en vez de dibujar el texto, matplotlib emite un fragmento
LaTeX que el propio LaTeX compone DESPUES, con las fuentes y la
matematica del libro. El resultado es indistinguible de una figura
TikZ en tipografia, y sigue siendo una sola fuente que da PDF (obra de
pago) y SVG (web) por la misma tuberia del Makefile.

Uso desde una figura:

    import sys; sys.path.insert(0, "herramientas")
    from estilo_figuras import PALETA, figura, guardar

    fig, ax = figura(alto=0.55)
    ax.plot(x, y, color=PALETA["base"], label="medido")
    guardar(fig, __file__, "pie de la figura, en cursiva")

`guardar` escribe el .tex standalone completo junto al .py, que es lo
que el Makefile convierte en PDF y en SVG.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# ancho de caja del libro: \linewidth son unos 355 pt en la caja 18x24
ANCHO_PULGADAS = 355 / 72.27

# paleta Okabe-Ito, la misma de latex/figs/fig_preamble.tex. Al tocar
# una hay que tocar la otra.
PALETA = {
    "base": "#0072B2",       # oazul   / cbase
    "trat": "#D55E00",       # obermellon / ctrat
    "tercero": "#009E73",    # overde  / ctercero
    "cuarto": "#E69F00",     # oamarillo
    "gris": "#6A737D",
}
SERIES = [PALETA["base"], PALETA["trat"], PALETA["tercero"],
          PALETA["cuarto"], PALETA["gris"]]

# el preambulo que LaTeX usara al componer el texto de la figura:
# el mismo que el del libro en lo que afecta a fuentes y matematicas
PREAMBULO = r"""
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{amsmath,amssymb}
\newcommand{\eps}{\varepsilon}
"""

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": PREAMBULO,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "grid.linewidth": 0.4,
    "grid.color": "#CCCCCC",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def coma(x: float, _=None) -> str:
    """Formatea un numero con coma decimal, como manda el libro.

    Args:
        x: valor a formatear.
        _: posicion del tick, que matplotlib pasa y no se usa.

    Returns:
        El numero como cadena, con coma en vez de punto.
    """
    s = f"{x:g}"
    return s.replace(".", "{,}") if "." in s else s


FORMATO_COMA = FuncFormatter(coma)


def figura(alto: float = 0.52, ancho: float = 1.0,
           **kwargs) -> tuple:
    """Crea una figura del ancho de la caja del libro.

    Args:
        alto: alto en fracciones del ancho (razon de aspecto).
        ancho: fraccion del ancho de caja a ocupar.
        **kwargs: se pasan a plt.subplots (p. ej. ncols).

    Returns:
        El par (figura, ejes) de matplotlib.
    """
    w = ANCHO_PULGADAS * ancho
    return plt.subplots(figsize=(w, w * alto), **kwargs)


def comas(*ejes) -> None:
    """Aplica la coma decimal a los ejes que se le pasen.

    Args:
        *ejes: objetos Axis (p. ej. ax.xaxis, ax.yaxis).
    """
    for e in ejes:
        e.set_major_formatter(FORMATO_COMA)


def guardar(fig, fuente: str, pie: str = "",
            ancho_pie: float = 12.4) -> Path:
    """Escribe el .tex standalone que el Makefile convertira.

    El fragmento PGF se incrusta en un documento `standalone` con el
    mismo preambulo de figuras del libro, de modo que las reglas del
    Makefile que ya existen producen el PDF y el SVG sin cambios.

    Args:
        fig: figura de matplotlib ya compuesta.
        fuente: ruta del .py que la genera (pasar __file__).
        pie: nota al pie de la figura, en cursiva. Puede ir vacia.
        ancho_pie: ancho del pie, en centimetros.

    Returns:
        La ruta del .tex escrito.
    """
    py = Path(fuente).resolve()
    pgf = py.with_suffix(".pgf")
    fig.savefig(pgf)
    plt.close(fig)

    cuerpo = pgf.read_text(encoding="utf8")
    pgf.unlink()

    nota = ""
    if pie:
        nota = (f"\n\n  \\vspace{{1mm}}\n"
                f"  \\node[etiqueta, text width={ancho_pie}cm] "
                f"{{{pie}}};\n")

    doc = (
        "% GENERADO por " + py.name + " — no editar a mano.\n"
        "% La fuente es el .py de al lado; el Makefile rehace este\n"
        "% fichero, y de el salen el PDF (obra de pago) y el SVG (web).\n"
        "\\documentclass[tikz]{standalone}\n"
        "\\input{../fig_preamble}\n"
        "\\usepackage{pgf}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}\n"
        "  \\node[anchor=north] (grafico) at (0,0) {%\n"
        + cuerpo +
        "  };\n"
        + (f"  \\node[etiqueta, anchor=north, text width={ancho_pie}cm]"
           f" at (grafico.south) {{{pie}}};\n" if pie else "") +
        "\\end{tikzpicture}\n"
        "\\end{document}\n")

    destino = py.with_suffix(".tex")
    destino.write_text(doc, encoding="utf8")
    return destino
