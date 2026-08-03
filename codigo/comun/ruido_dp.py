"""mecanismos de laplace y gauss para privacidad diferencial."""

import math

import numpy as np


def mecanismo_laplace(valor: float | np.ndarray, sensibilidad: float,
                      epsilon: float,
                      rng: np.random.Generator) -> float | np.ndarray:
    """Aplica el mecanismo de Laplace a una consulta numerica.

    Garantiza epsilon-DP para una consulta con la sensibilidad L1
    indicada (Dwork et al., 2006).

    Args:
        valor: resultado exacto de la consulta (escalar o vector).
        sensibilidad: sensibilidad L1 de la consulta.
        epsilon: presupuesto de privacidad, estrictamente positivo.
        rng: generador sembrado (ver comun/determinismo.py).

    Returns:
        El valor con ruido Laplace de escala sensibilidad/epsilon.

    Raises:
        ValueError: si epsilon o la sensibilidad no son positivos.
    """
    if epsilon <= 0 or sensibilidad <= 0:
        raise ValueError("epsilon y sensibilidad deben ser positivos")
    escala = sensibilidad / epsilon
    return valor + rng.laplace(loc=0.0, scale=escala,
                               size=np.shape(valor) or None)


def mecanismo_gauss(valor: float | np.ndarray, sensibilidad: float,
                    epsilon: float, delta: float,
                    rng: np.random.Generator) -> float | np.ndarray:
    """Aplica el mecanismo de Gauss a una consulta numerica.

    Garantiza (epsilon, delta)-DP para una consulta con la
    sensibilidad L2 indicada, con la cota clasica de sigma valida
    para epsilon < 1 (Dwork y Roth, 2014, ap. A).

    Args:
        valor: resultado exacto de la consulta (escalar o vector).
        sensibilidad: sensibilidad L2 de la consulta.
        epsilon: presupuesto de privacidad, en (0, 1) para esta cota.
        delta: probabilidad de fallo, en (0, 1).
        rng: generador sembrado (ver comun/determinismo.py).

    Returns:
        El valor con ruido gaussiano de desviacion sigma.

    Raises:
        ValueError: si los parametros salen del rango de la cota.
    """
    if not 0 < epsilon < 1:
        raise ValueError("esta cota clasica exige 0 < epsilon < 1")
    if not 0 < delta < 1 or sensibilidad <= 0:
        raise ValueError("delta en (0,1) y sensibilidad positiva")
    sigma = (sensibilidad * math.sqrt(2 * math.log(1.25 / delta))
             / epsilon)
    return valor + rng.normal(loc=0.0, scale=sigma,
                              size=np.shape(valor) or None)
