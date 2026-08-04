"""los dos mecanismos de ruido aditivo, medidos uno contra otro.

Tres cosas que la definicion de privacidad diferencial dice y que aqui
se COMPRUEBAN sobre numeros, no se enuncian:

1. La perdida de privacidad es una VARIABLE ALEATORIA. Con Laplace
   esta acotada por epsilon y no puede salirse; con Gauss es normal y
   se sale con probabilidad pequena pero no nula. Esa es exactamente
   la diferencia entre epsilon-DP y (epsilon,delta)-DP, y se ve en su
   histograma.
2. La calibracion clasica del gaussiano (Dwork y Roth, 2014) no vale
   para epsilon >= 1 y, donde vale, sobra ruido. La calibracion exacta
   de Balle y Wang (2018) se mide aqui contra ella.
3. Cual de los dos conviene NO depende de cuantas cifras se publican,
   sino de A CUANTAS AFECTA UNA PERSONA. Se miden las dos formas: un
   histograma de casillas disjuntas (cada persona toca dos) y una
   funcion de distribucion acumulada (cada persona toca casi todas).
   En la primera gana Laplace siempre; en la segunda Gauss acaba
   ganando, y aqui se mide donde.

Escribe cuatro CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/mecanismos.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso
from comun.ruido_dp import (_delta_gauss, sigma_gauss_analitica,
                            sigma_gauss_clasica)

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SAL_PERDIDA = RAIZ / "data" / "processed" / "dp_perdida.csv"
SAL_CALIBRA = RAIZ / "data" / "processed" / "dp_calibracion.csv"
SAL_ELECCION = RAIZ / "data" / "processed" / "dp_eleccion.csv"

MUESTRAS = 400_000        # realizaciones para estimar la perdida
DELTA = 1e-5              # delta de referencia en todo el capitulo
EPS_REJILLA = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]


def perdida_laplace(o: np.ndarray, b: float) -> np.ndarray:
    """Perdida de privacidad de Laplace entre los recuentos 0 y 1.

    Vale (|o-1| - |o|)/b. Se evalua a trozos y no por la resta
    directa: para |o| grande esa resta cancela cifras significativas y
    produce valores espurios por encima de epsilon, que es justo lo
    que se quiere comprobar que NO ocurre.

    Args:
        o: realizaciones de la salida del mecanismo.
        b: escala del ruido, igual a sensibilidad/epsilon.

    Returns:
        La perdida de cada realizacion, acotada en [-1/b, 1/b].
    """
    return np.select([o <= 0.0, o >= 1.0],
                     [1.0 / b, -1.0 / b],
                     default=(1.0 - 2.0 * o) / b)


def perdida_gauss(o: np.ndarray, sigma: float) -> np.ndarray:
    """Perdida de privacidad de Gauss entre los recuentos 0 y 1.

    Vale ((o-1)^2 - o^2)/(2 sigma^2) = (1 - 2o)/(2 sigma^2), forma en
    la que no hay cancelacion. Es una normal de media 1/(2 sigma^2) y
    desviacion 1/sigma: no esta acotada.

    Args:
        o: realizaciones de la salida del mecanismo.
        sigma: desviacion tipica del ruido aplicado.

    Returns:
        La perdida de cada realizacion.
    """
    return (1.0 - 2.0 * o) / (2.0 * sigma ** 2)


def medir_perdida(epsilon: float, delta: float,
                  rng: np.random.Generator) -> tuple:
    """Simula la perdida de privacidad de ambos mecanismos.

    Sobre una consulta de recuento (sensibilidad 1) y dos bases
    vecinas cuyo recuento difiere en 1, se simula la salida bajo la
    primera y se evalua log(P[M(D)=o] / P[M(D')=o]): la perdida que la
    definicion exige acotar.

    Args:
        epsilon: presupuesto con el que se calibra cada mecanismo.
        delta: delta del gaussiano.
        rng: generador sembrado.

    Returns:
        Terna (tabla larga de perdidas, sigma del gaussiano, delta
        exacto que esa sigma consigue).
    """
    b = 1.0 / epsilon
    perd_lap = perdida_laplace(rng.laplace(0.0, b, MUESTRAS), b)

    sigma = sigma_gauss_analitica(1.0, epsilon, delta)
    perd_gau = perdida_gauss(rng.normal(0.0, sigma, MUESTRAS), sigma)

    tabla = pd.DataFrame({
        "mecanismo": ["laplace"] * MUESTRAS + ["gauss"] * MUESTRAS,
        "perdida": np.concatenate([perd_lap, perd_gau])})
    return tabla, sigma, _delta_gauss(sigma, epsilon)


def resumen_perdida(tabla: pd.DataFrame,
                    epsilon: float) -> pd.DataFrame:
    """Resume la perdida en las cifras que interesan al capitulo."""
    filas = []
    for nombre, g in tabla.groupby("mecanismo", observed=True):
        p = g["perdida"].to_numpy()
        filas.append({"mecanismo": nombre,
                      "media": float(p.mean()),
                      "desviacion": float(p.std()),
                      "maxima": float(p.max()),
                      "p_supera_eps": float((p > epsilon).mean())})
    return pd.DataFrame(filas)


def calibracion(delta: float) -> pd.DataFrame:
    """Compara la sigma clasica y la exacta a lo largo de epsilon.

    Args:
        delta: delta comun a las dos calibraciones.

    Returns:
        Tabla con sigma clasica (donde existe), sigma exacta y el
        exceso relativo de la primera.
    """
    filas = []
    for eps in EPS_REJILLA:
        exacta = sigma_gauss_analitica(1.0, eps, delta)
        try:
            clasica = sigma_gauss_clasica(1.0, eps, delta)
            exceso = 100 * (clasica / exacta - 1)
        except ValueError:
            clasica, exceso = float("nan"), float("nan")
        filas.append({"epsilon": eps, "sigma_clasica": clasica,
                      "sigma_exacta": exacta, "exceso_pct": exceso})
    return pd.DataFrame(filas)


def sensibilidades(forma: str, dim: int) -> tuple[float, float]:
    """Sensibilidad L1 y L2 de cada forma de consulta vectorial.

    Args:
        forma: «histograma» (casillas disjuntas: una persona cae en
            una sola, y al sustituirla cambian dos en una unidad) o
            «acumulada» (cada persona cuenta en todos los tramos por
            encima de su valor, luego los toca casi todos).
        dim: numero de componentes del vector publicado.

    Returns:
        Par (sensibilidad L1, sensibilidad L2).

    Raises:
        ValueError: si la forma no es una de las dos previstas.
    """
    if forma == "histograma":
        return 2.0, math.sqrt(2.0)
    if forma == "acumulada":
        # en el peor caso la persona sustituida entra y sale de dim
        # tramos: L1 = dim, L2 = raiz(dim)
        return float(dim), math.sqrt(dim)
    raise ValueError(f"forma desconocida: {forma}")


def eleccion_por_forma(delta: float, rng: np.random.Generator,
                       log) -> pd.DataFrame:
    """Mide cual de los dos mecanismos yerra menos, y cuando cambia.

    Args:
        delta: delta del gaussiano.
        rng: generador sembrado.
        log: registro donde anotar el avance.

    Returns:
        Tabla con el error absoluto medio por componente de cada
        mecanismo, por forma de consulta, dimension y epsilon.
    """
    repeticiones = 400
    dims = [1, 2, 6, 20, 52, 100, 311, 1000]
    casos = [(f, d, e) for f in ("histograma", "acumulada")
             for d in dims for e in EPS_REJILLA]
    filas = []
    for forma, dim, eps in progreso(casos, len(casos), log, cada=40,
                                    tarea="forma x dimension x eps"):
        s1, s2 = sensibilidades(forma, dim)
        b = s1 / eps
        sigma = sigma_gauss_analitica(s2, eps, delta)
        err_l = np.abs(rng.laplace(0.0, b, (repeticiones, dim))).mean()
        err_g = np.abs(rng.normal(0.0, sigma,
                                  (repeticiones, dim))).mean()
        filas.append({"forma": forma, "dimension": dim,
                      "epsilon": eps, "error_laplace": float(err_l),
                      "error_gauss": float(err_g),
                      "gana": "gauss" if err_g < err_l else "laplace"})
    return pd.DataFrame(filas)


def main() -> None:
    """Ejecuta las tres mediciones y guarda sus tablas."""
    log = crear_registro("cap05.mecanismos")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    log.info("poblacion de referencia: %d personas · %d provincias · "
             "%d diagnosticos", len(df),
             df["codigo_postal"].astype(str).str[:2].nunique(),
             df["diagnostico"].nunique())

    # ── 1) la perdida de privacidad como variable aleatoria ─────────
    eps_demo = 1.0
    tabla, sigma, delta_exacto = medir_perdida(eps_demo, DELTA, rng)
    res = resumen_perdida(tabla, eps_demo)
    log.info("perdida de privacidad con epsilon=%.1f y delta=%.0e "
             "(sigma del gaussiano: %.3f):", eps_demo, DELTA, sigma)
    for _, f in res.iterrows():
        log.info("  %-8s media %+6.3f · desviacion %5.3f · maxima "
                 "%6.3f · P(perdida > eps) = %.2e",
                 f["mecanismo"], f["media"], f["desviacion"],
                 f["maxima"], f["p_supera_eps"])
    log.info("  Laplace no se sale nunca de [-eps, eps]: es epsilon-DP "
             "puro, sin delta.")
    p_gauss = float(res.loc[res["mecanismo"] == "gauss",
                            "p_supera_eps"].iloc[0])
    log.info("  Gauss se sale con probabilidad %.2e, que es %.0f veces "
             "el delta=%.0e pedido. NO es un fallo: el delta EXACTO de "
             "esa sigma es %.2e, porque la condicion de Balle y Wang "
             "descuenta el peso que la propia cola de e^eps compensa. "
             "Acotar delta por P(perdida > eps) es correcto pero "
             "conservador, y ese descuento es justo lo que hace que la "
             "calibracion exacta necesite menos ruido.",
             p_gauss, p_gauss / DELTA, DELTA, delta_exacto)
    SAL_PERDIDA.parent.mkdir(parents=True, exist_ok=True)
    # se guarda la distribucion en histograma, no las 800 000
    # realizaciones: la figura solo necesita las frecuencias
    bordes = np.linspace(-3, 3, 121)
    hist = []
    for nombre, g in tabla.groupby("mecanismo", observed=True):
        frec, _ = np.histogram(g["perdida"], bins=bordes, density=True)
        hist.append(pd.DataFrame({
            "mecanismo": nombre,
            "centro": (bordes[:-1] + bordes[1:]) / 2,
            "densidad": frec}))
    pd.concat(hist).to_csv(SAL_PERDIDA, index=False)
    res.assign(sigma=sigma, delta_pedido=DELTA,
               delta_exacto=delta_exacto).to_csv(
        SAL_PERDIDA.with_name("dp_perdida_resumen.csv"), index=False)
    log.info("guardado %s", SAL_PERDIDA.relative_to(RAIZ))

    # ── 2) calibracion clasica frente a exacta ──────────────────────
    cal = calibracion(DELTA)
    log.info("calibracion del gaussiano a delta=%.0e:", DELTA)
    for _, f in cal.iterrows():
        if math.isnan(f["sigma_clasica"]):
            log.info("  eps=%-4.2f  clasica: NO APLICA (exige eps<1) · "
                     "exacta %6.3f", f["epsilon"], f["sigma_exacta"])
        else:
            log.info("  eps=%-4.2f  clasica %7.3f · exacta %6.3f · "
                     "sobra un %.1f%% de ruido", f["epsilon"],
                     f["sigma_clasica"], f["sigma_exacta"],
                     f["exceso_pct"])
    cal.to_csv(SAL_CALIBRA, index=False)
    log.info("guardado %s", SAL_CALIBRA.relative_to(RAIZ))

    # ── 3) que mecanismo conviene, segun la FORMA de la consulta ────
    casillas = (df.assign(prov=df["codigo_postal"].astype(str).str[:2])
                .groupby(["prov", "diagnostico"], observed=True)
                .ngroups)
    log.info("histograma de referencia del libro: %d casillas "
             "(provincia x diagnostico)", casillas)
    ele = eleccion_por_forma(DELTA, rng, log)
    for forma in ("histograma", "acumulada"):
        sub = ele[ele["forma"] == forma]
        ganadas = sub[sub["gana"] == "gauss"]
        if not len(ganadas):
            log.info("  %-11s: gana Laplace en TODA la rejilla "
                     "(dim 1-1000, eps 0,1-5). La sensibilidad no "
                     "crece con la dimension: una persona solo toca "
                     "dos casillas.", forma)
            continue
        log.info("  %-11s: Gauss gana en %d de %d casos; el corte por "
                 "dimension es:", forma, len(ganadas), len(sub))
        for dim in sorted(sub["dimension"].unique()):
            g = ganadas[ganadas["dimension"] == dim]["epsilon"]
            log.info("      dim %4d -> %s", dim,
                     f"desde eps={g.min():g}" if len(g) else "nunca")
    ele.to_csv(SAL_ELECCION, index=False)
    log.info("guardado %s", SAL_ELECCION.relative_to(RAIZ))

    muestra_final(ele, log)


if __name__ == "__main__":
    main()
