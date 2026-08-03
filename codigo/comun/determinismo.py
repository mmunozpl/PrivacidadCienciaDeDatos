"""semillas y reproducibilidad para todos los scripts del libro."""

import os
import random

import numpy as np

SEMILLA = 42


def fijar_semillas(semilla: int = SEMILLA) -> np.random.Generator:
    """Fija las semillas de todos los generadores en uso.

    Args:
        semilla: valor que se aplica a random, numpy y, si esta
            instalado, torch (cpu y cuda).

    Returns:
        Un generador numpy sembrado, para uso preferente en el codigo
        nuevo frente al estado global.
    """
    # se fija tambien el hash de python, por los ordenes de dict/set
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch

        torch.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
    except ImportError:
        # sin torch instalado basta con random y numpy
        pass
    return np.random.default_rng(semilla)
