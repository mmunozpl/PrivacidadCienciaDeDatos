"""logging con progreso y verificacion final de artefactos."""

import logging
import sys
from collections.abc import Iterable, Iterator

import pandas as pd


def crear_registro(nombre: str) -> logging.Logger:
    """Crea un logger con formato uniforme para todo el libro.

    Args:
        nombre: nombre del script o modulo que registra.

    Returns:
        Logger a nivel INFO escribiendo en stderr.
    """
    log = logging.getLogger(nombre)
    if not log.handlers:
        manejador = logging.StreamHandler(sys.stderr)
        manejador.setFormatter(
            logging.Formatter("%(asctime)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        log.addHandler(manejador)
        log.setLevel(logging.INFO)
        # sin propagar al raiz: algunas dependencias (Opacus) lo
        # configuran al importarse y cada mensaje saldria dos veces
        log.propagate = False
    return log


def progreso(elementos: Iterable, total: int, log: logging.Logger,
             cada: int = 1000, tarea: str = "procesando") -> Iterator:
    """Itera registrando avance intermedio en tareas pesadas.

    Args:
        elementos: iterable a recorrer.
        total: numero total de elementos esperado.
        log: logger donde se anota el avance.
        cada: cadencia de registro, en elementos.
        tarea: etiqueta breve de lo que se procesa.

    Yields:
        Los mismos elementos del iterable de entrada.
    """
    for i, elem in enumerate(elementos, start=1):
        if i % cada == 0 or i == total:
            log.info("%s: %d/%d (%.0f%%)", tarea, i, total,
                     100 * i / total)
        yield elem


def muestra_final(df: pd.DataFrame, log: logging.Logger,
                  n: int = 15, semilla: int = 42) -> None:
    """Imprime la verificacion final: n observaciones aleatorias.

    Todo artefacto guardado por un script del libro termina con esta
    llamada (o su equivalente visual con imagenes).

    Args:
        df: tabla recien guardada que se verifica.
        log: logger donde se anota el contexto.
        n: numero de observaciones a mostrar.
        semilla: semilla del muestreo, fija por reproducibilidad.
    """
    log.info("verificacion final: %d observaciones aleatorias de %d",
             min(n, len(df)), len(df))
    with pd.option_context("display.max_columns", None,
                           "display.width", 100):
        print(df.sample(n=min(n, len(df)), random_state=semilla))
