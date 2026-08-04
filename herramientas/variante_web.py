"""genera la variante web de cada capitulo para su paso por pandoc.

Equivale a compilar con \\ifcompleta en falso: los bloques exclusivos
de la obra de pago (papel, PDF y EPUB) se descartan y queda solo su
nota de edicion. Hoy son dos: «soloimpresa» (practica medida y
ejercicios) y «ensnist» (marcos España/UE vs EE. UU.). Ademas reescribe las
rutas de figuras del PDF al SVG de qmd/figs/. El resultado se escribe
en .work/capXX_web.tex.

Uso: python3 herramientas/variante_web.py latex/cap01.tex [...]
"""

import re
import sys
from pathlib import Path

# bloques solo de la obra de pago, con la nota que los sustituye
BLOQUES = {
    "soloimpresa": (
        "\\begin{quote}\\itshape\n"
        "Práctica medida y ejercicios --- disponibles en la obra\n"
        "completa (papel, PDF y EPUB).\n"
        "\\end{quote}\n"
    ),
    "ensnist": (
        "\\begin{quote}\\itshape\n"
        "Marcos frente a frente: España/UE vs EE. UU. --- sección\n"
        "disponible en la obra completa.\n"
        "\\end{quote}\n"
    ),
}

RE_FIGURA = re.compile(
    r"(\\includegraphics(?:\[[^\]]*\])?\{)(figs/[^}]+?)(\})"
)


def re_bloque(entorno: str) -> re.Pattern:
    """Patron de un entorno de pago completo, a inicio de linea."""
    return re.compile(
        rf"^\\begin\{{{entorno}\}}.*?^\\end\{{{entorno}\}}[ \t]*$",
        re.DOTALL | re.MULTILINE,
    )


def variante_web(fuente: Path, destino: Path) -> None:
    """Escribe la variante web de un capitulo.

    Args:
        fuente: ruta del capXX.tex original.
        destino: ruta del capXX_web.tex resultante.
    """
    texto = fuente.read_text(encoding="utf-8")
    retirados = 0
    for entorno, nota in BLOQUES.items():
        # el reemplazo va como funcion: la nota lleva barras invertidas
        texto, n = re_bloque(entorno).subn(lambda _: nota, texto)
        retirados += n
    # se apuntan las figuras al svg generado bajo qmd/figs/
    texto = RE_FIGURA.sub(r"\1\2.svg\3", texto)
    # guarda: ni una linea de la obra de pago puede llegar a la web
    for entorno in BLOQUES:
        if (f"\\begin{{{entorno}}}" in texto
                or f"\\end{{{entorno}}}" in texto):
            raise SystemExit(
                f"✗ {fuente}: queda un resto de {entorno} tras el filtrado"
            )
    destino.write_text(texto, encoding="utf-8")
    print(f"  · {fuente.name} -> {destino} ({retirados} bloques retirados)")


def main() -> None:
    """Procesa los capitulos recibidos por linea de ordenes."""
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    trabajo = Path(".work")
    trabajo.mkdir(exist_ok=True)
    for arg in sys.argv[1:]:
        fuente = Path(arg)
        destino = trabajo / f"{fuente.stem}_web.tex"
        variante_web(fuente, destino)


if __name__ == "__main__":
    main()
