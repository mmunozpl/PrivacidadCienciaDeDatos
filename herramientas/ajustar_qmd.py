"""da a cada capitulo web el formato de la coleccion (RCienciaDeDatos).

Convierte el H1 que emite pandoc («# Titulo {#ancla}») en la cabecera
YAML «Capítulo N. Titulo» y conserva el ancla como span vacio para que
las referencias internas sigan funcionando.

Uso: python3 herramientas/ajustar_qmd.py qmd/cap01.qmd [...]
"""

import re
import sys
from pathlib import Path

RE_H1 = re.compile(r"^# (.+?)\s*\{#([^}]+)\}\s*$", re.MULTILINE)


def ajustar(ruta: Path) -> None:
    """Reescribe un capXX.qmd al formato de la coleccion.

    Args:
        ruta: fichero qmd generado por pandoc (capNN.qmd).
    """
    numero = int(ruta.stem[3:])
    texto = ruta.read_text(encoding="utf-8")
    m = RE_H1.search(texto)
    if not m:
        raise SystemExit(f"✗ {ruta}: no se encuentra el H1 del capítulo")
    titulo, ancla = m.group(1), m.group(2)
    cabecera = (
        "---\n"
        f'title: "Capítulo {numero}. {titulo}"\n'
        "---\n\n"
        f"[]{{#{ancla}}}\n"
    )
    texto = texto[:m.start()] + cabecera + texto[m.end():].lstrip("\n")
    ruta.write_text(texto, encoding="utf-8")
    print(f"  · {ruta.name}: Capítulo {numero}. {titulo}")


def main() -> None:
    """Procesa los qmd recibidos por linea de ordenes."""
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        ajustar(Path(arg))


if __name__ == "__main__":
    main()
