"""genera la variante web de cada capitulo para su paso por pandoc.

Equivale a compilar con \\ifcompleta en falso: el cuerpo de la seccion
«ENS/UE vs NIST» se descarta y queda solo la nota de edicion. Ademas
reescribe las rutas de figuras del PDF al SVG de qmd/figs/. El
resultado se escribe en .work/capXX_web.tex.

Uso: python3 herramientas/variante_web.py latex/cap01.tex [...]
"""

import re
import sys
from pathlib import Path

NOTA = (
    "\\begin{quote}\\itshape\n"
    "Marcos frente a frente: ENS/UE vs NIST --- sección disponible en\n"
    "la obra completa.\n"
    "\\end{quote}\n"
)

RE_ENSNIST = re.compile(
    r"^\\begin\{ensnist\}.*?^\\end\{ensnist\}[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
RE_FIGURA = re.compile(
    r"(\\includegraphics(?:\[[^\]]*\])?\{)(figs/[^}]+?)(\})"
)


def variante_web(fuente: Path, destino: Path) -> None:
    """Escribe la variante web de un capitulo.

    Args:
        fuente: ruta del capXX.tex original.
        destino: ruta del capXX_web.tex resultante.
    """
    texto = fuente.read_text(encoding="utf-8")
    # el reemplazo va como funcion: la nota contiene barras invertidas
    texto, n = RE_ENSNIST.subn(lambda _: NOTA, texto)
    # se apuntan las figuras al svg generado bajo qmd/figs/
    texto = RE_FIGURA.sub(r"\1\2.svg\3", texto)
    # guarda: ni una linea de la seccion de pago puede llegar a la web
    if "\\begin{ensnist}" in texto or "\\end{ensnist}" in texto:
        raise SystemExit(
            f"✗ {fuente}: queda un resto de ensnist tras el filtrado"
        )
    destino.write_text(texto, encoding="utf-8")
    print(f"  · {fuente.name} -> {destino} ({n} sección/es retiradas)")


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
