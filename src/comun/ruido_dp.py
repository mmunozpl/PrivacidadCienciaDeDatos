"""mecanismos y contabilidad de privacidad diferencial.

Reune lo que comparten los capitulos 5 a 8: los dos mecanismos de
ruido aditivo (Laplace y Gauss), la calibracion exacta del gaussiano,
el mecanismo exponencial para salidas no numericas y las tres formas
de sumar presupuesto cuando se hacen varias consultas.

Ninguna funcion imprime nada: el registro lo hace quien llama.
"""

import math

import numpy as np


# ── mecanismos de ruido aditivo ──────────────────────────────────────

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


def sigma_gauss_clasica(sensibilidad: float, epsilon: float,
                        delta: float) -> float:
    """Sigma del gaussiano por la cota clasica, valida solo si eps<1.

    Es la de Dwork y Roth (2014, teorema A.1, repetida como 3.22), la
    que aparece en casi todos los tutoriales. Su enunciado EXIGE
    epsilon en el intervalo ABIERTO (0,1) y la condicion sobre la
    constante es estricta (c^2 > 2 ln(1,25/delta)): fuera de ahi el
    teorema no dice nada, y usarlo igualmente es un fallo silencioso.

    Args:
        sensibilidad: sensibilidad L2 de la consulta.
        epsilon: presupuesto, obligatoriamente en (0, 1).
        delta: probabilidad de fallo, en (0, 1).

    Returns:
        Desviacion tipica del ruido gaussiano.

    Raises:
        ValueError: si los parametros salen del rango de la cota.
    """
    if not 0 < epsilon < 1:
        raise ValueError("esta cota clasica exige 0 < epsilon < 1")
    if not 0 < delta < 1 or sensibilidad <= 0:
        raise ValueError("delta en (0,1) y sensibilidad positiva")
    return sensibilidad * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def _phi(x: float) -> float:
    """Funcion de distribucion de la normal tipificada."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _delta_gauss(u: float, epsilon: float) -> float:
    """Delta EXACTO de un gaussiano con u = sigma/sensibilidad.

    Condicion de Balle y Wang (2018, teorema 8): el mecanismo es
    (eps,delta)-DP SI Y SOLO SI Phi(1/(2u) - eps*u) - e^eps *
    Phi(-1/(2u) - eps*u) no supera delta. Al ser decreciente en u se
    invierte por biseccion.
    """
    return (_phi(1 / (2 * u) - epsilon * u)
            - math.exp(epsilon) * _phi(-1 / (2 * u) - epsilon * u))


def sigma_gauss_analitica(sensibilidad: float, epsilon: float,
                          delta: float, tol: float = 1e-12) -> float:
    """Sigma EXACTA del gaussiano, valida para cualquier epsilon.

    Invierte por biseccion la condicion necesaria y suficiente de
    Balle y Wang (2018). Frente a la cota clasica: no impone eps<1 y
    no sobra ruido, porque la condicion es exacta y no una cota.

    Args:
        sensibilidad: sensibilidad L2 de la consulta.
        epsilon: presupuesto, estrictamente positivo (sin tope).
        delta: probabilidad de fallo, en (0, 1).
        tol: anchura relativa a la que se detiene la biseccion.

    Returns:
        La menor desviacion tipica que satisface (eps, delta)-DP.

    Raises:
        ValueError: si los parametros no son admisibles.
    """
    if epsilon <= 0 or sensibilidad <= 0:
        raise ValueError("epsilon y sensibilidad deben ser positivos")
    if not 0 < delta < 1:
        raise ValueError("delta debe estar en (0, 1)")
    # se busca un intervalo [lo, hi] con delta(hi) <= delta < delta(lo)
    lo, hi = 1e-9, 1.0
    while _delta_gauss(hi, epsilon) > delta:
        hi *= 2
        if hi > 1e12:
            raise ValueError("no converge: revisar epsilon y delta")
    while (hi - lo) / hi > tol:
        medio = 0.5 * (lo + hi)
        if _delta_gauss(medio, epsilon) > delta:
            lo = medio
        else:
            hi = medio
    return sensibilidad * hi


def mecanismo_gauss(valor: float | np.ndarray, sensibilidad: float,
                    epsilon: float, delta: float,
                    rng: np.random.Generator,
                    analitico: bool = True) -> float | np.ndarray:
    """Aplica el mecanismo de Gauss a una consulta numerica.

    Args:
        valor: resultado exacto de la consulta (escalar o vector).
        sensibilidad: sensibilidad L2 de la consulta.
        epsilon: presupuesto de privacidad.
        delta: probabilidad de fallo, en (0, 1).
        rng: generador sembrado (ver comun/determinismo.py).
        analitico: si es cierto usa la calibracion exacta; si no, la
            cota clasica (que rechaza epsilon >= 1).

    Returns:
        El valor con ruido gaussiano de desviacion sigma.
    """
    sigma = (sigma_gauss_analitica(sensibilidad, epsilon, delta)
             if analitico
             else sigma_gauss_clasica(sensibilidad, epsilon, delta))
    return valor + rng.normal(loc=0.0, scale=sigma,
                              size=np.shape(valor) or None)


# ── salidas no numericas ─────────────────────────────────────────────

def mecanismo_exponencial(candidatos: list, utilidades: np.ndarray,
                          sensibilidad: float, epsilon: float,
                          rng: np.random.Generator):
    """Elige un candidato con probabilidad proporcional a e^(eps*u/2s).

    Resuelve lo que el ruido aditivo no sabe hacer: devolver una
    CATEGORIA (el diagnostico mas frecuente, la mediana) sin que el
    ruido la convierta en un valor imposible.

    El mecanismo es de McSherry y Talwar (2007), cuya definicion 2 no
    lleva el factor 2 en el exponente y da (2*eps*Du)-DP. Aqui se
    implementa la forma de Dwork y Roth (2014, def. 3.4), con el 2 en
    el denominador del exponente, que es eps-DP por su teorema 3.10.
    Son el mismo mecanismo con el presupuesto reparametrizado.

    Args:
        candidatos: salidas posibles, en el mismo orden que utilidades.
        utilidades: puntuacion de cada candidato, a mayor mejor.
        sensibilidad: cuanto puede variar una utilidad si cambia una
            sola persona.
        epsilon: presupuesto de privacidad.
        rng: generador sembrado.

    Returns:
        El candidato elegido.

    Raises:
        ValueError: si las longitudes no casan o epsilon no es positivo.
    """
    if len(candidatos) != len(utilidades):
        raise ValueError("candidatos y utilidades deben ir a la par")
    if epsilon <= 0 or sensibilidad <= 0:
        raise ValueError("epsilon y sensibilidad deben ser positivos")
    puntuacion = epsilon * np.asarray(utilidades, dtype=float) \
        / (2 * sensibilidad)
    # se resta el maximo antes de exponenciar: mismo resultado, sin
    # desbordar cuando epsilon*u/2s es grande
    peso = np.exp(puntuacion - puntuacion.max())
    return candidatos[rng.choice(len(candidatos), p=peso / peso.sum())]


# ── contabilidad del presupuesto ─────────────────────────────────────

def composicion_basica(epsilon: float, delta: float,
                       k: int) -> tuple[float, float]:
    """Suma lineal: k mecanismos (eps,delta) dan (k*eps, k*delta).

    Args:
        epsilon: presupuesto de cada mecanismo.
        delta: delta de cada mecanismo.
        k: numero de mecanismos compuestos.

    Returns:
        Par (epsilon total, delta total).
    """
    return k * epsilon, k * delta


def composicion_avanzada(epsilon: float, delta: float, k: int,
                         delta_prima: float) -> tuple[float, float]:
    """Composicion avanzada: el total crece con la raiz de k.

    Teorema 3.3 de Dwork, Rothblum y Vadhan (2010): la composicion
    ADAPTATIVA k-veces de mecanismos (eps,delta)-DP es (eps',
    k*delta + delta')-DP con
    eps' = sqrt(2k ln(1/delta'))*eps + k*eps*(e^eps - 1). Se paga un
    delta' extra a cambio de que el termino dominante sea sqrt(k) y no
    k; solo compensa cuando k es grande y eps pequeno.

    Args:
        epsilon: presupuesto de cada mecanismo.
        delta: delta de cada mecanismo.
        k: numero de mecanismos compuestos.
        delta_prima: delta adicional que se acepta pagar, en (0, 1).

    Returns:
        Par (epsilon total, delta total).

    Raises:
        ValueError: si delta_prima no esta en (0, 1).
    """
    if not 0 < delta_prima < 1:
        raise ValueError("delta_prima debe estar en (0, 1)")
    eps = (math.sqrt(2 * k * math.log(1 / delta_prima)) * epsilon
           + k * epsilon * (math.exp(epsilon) - 1))
    return eps, k * delta + delta_prima


def rho_de_gauss(sensibilidad: float, sigma: float) -> float:
    """Rho de un gaussiano en zCDP: rho = sensibilidad^2/(2 sigma^2).

    Args:
        sensibilidad: sensibilidad L2 de la consulta.
        sigma: desviacion tipica del ruido aplicado.

    Returns:
        El parametro rho de zCDP (Bun y Steinke, 2016).
    """
    return sensibilidad ** 2 / (2 * sigma ** 2)


def eps_de_rho(rho: float, delta: float) -> float:
    """Traduce rho de zCDP a un epsilon de (eps,delta)-DP.

    Conversion de Bun y Steinke (2016, prop. 1.3): rho-zCDP implica
    (rho + 2*sqrt(rho*ln(1/delta)), delta)-DP. Es la traduccion que usa
    el censo de EE. UU. para anunciar en epsilon un presupuesto que
    contabiliza en rho.

    Args:
        rho: presupuesto en zCDP.
        delta: delta al que se quiere expresar el resultado.

    Returns:
        El epsilon equivalente a ese rho y ese delta.
    """
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def rho_compuesta(rhos: list[float]) -> float:
    """Composicion en zCDP: los rho se suman, sin termino de correccion.

    Esta es la ventaja practica de contabilizar en zCDP frente a
    (eps,delta): la suma es exacta y no hay que elegir delta_prima en
    cada paso; el delta aparece una sola vez, al traducir al final.

    Args:
        rhos: presupuesto rho de cada mecanismo compuesto.

    Returns:
        El rho total.
    """
    return float(sum(rhos))
