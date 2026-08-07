"""cuanto presupuesto gasta de verdad un entrenamiento con DP.

Entrenar es componer: cada paso del descenso es una consulta a los
datos, y un entrenamiento corriente da decenas de miles de pasos. Con
la composicion basica del capitulo 5 eso seria inviable —mil pasos de
epsilon=0,1 dan epsilon=100—, y sin embargo se entrena a epsilon=8.
La diferencia la hacen dos cosas que este guion mide por separado:

1. LA AMPLIFICACION POR SUBMUESTREO. Cada paso mira solo una fraccion
   q de los datos, elegida al azar. Quien no ha entrado en el lote no
   ha arriesgado nada en ese paso, y la garantia mejora con q. Es lo
   que convierte el entrenamiento en algo pagable.
2. LA CONTABILIDAD. RDP, PRV y GDP no cambian el mecanismo: cambian
   la cota que se sabe demostrar sobre el mismo mecanismo. Aqui se
   mide cuanto se aprieta al pasar de una a otra, y cuanto ruido de
   menos permite eso para el mismo epsilon.

Mide tambien el efecto del TAMANO DE LOTE, que es el hiperparametro
menos intuitivo de DP-SGD: lotes mas grandes suben q —lo que empeora
cada paso— pero recortan el numero de pasos, y a igual numero de
epocas el saldo suele ser favorable.

Escribe tres CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/contabilidad.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.registro import crear_registro, muestra_final, progreso
from comun.ruido_dp import composicion_basica, eps_de_rho, rho_de_gauss

SAL_CONTABLES = RAIZ / "data" / "processed" / "dp_sgd_contables.csv"
SAL_AMPLIF = RAIZ / "data" / "processed" / "dp_sgd_amplificacion.csv"
SAL_LOTE = RAIZ / "data" / "processed" / "dp_sgd_lote.csv"
SAL_TRAMPA = RAIZ / "data" / "processed" / "dp_sgd_trampa.csv"

N = 10_000                # ejemplos de entrenamiento
DELTA = 1e-5
CONTABLES = ["rdp", "prv", "gdp"]


def eps_de(sigma: float, q: float, pasos: int, delta: float,
           contable: str) -> float:
    """Epsilon que informa un contable para esa configuracion.

    Args:
        sigma: multiplicador de ruido.
        q: tasa de muestreo (tamano de lote entre ejemplos).
        pasos: numero de pasos de entrenamiento.
        delta: delta objetivo.
        contable: «rdp», «prv» o «gdp».

    Returns:
        El epsilon informado, o NaN si el contable no converge.
    """
    from opacus.accountants import create_accountant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuenta = create_accountant(mechanism=contable)
        cuenta.history = [(sigma, q, pasos)]
        try:
            return float(cuenta.get_epsilon(delta=delta))
        except Exception:
            return float("nan")


def sigma_para(eps: float, q: float, pasos: int, delta: float,
               contable: str) -> float:
    """Multiplicador de ruido que hace falta para un epsilon objetivo.

    Args:
        eps: presupuesto objetivo.
        q: tasa de muestreo.
        pasos: numero de pasos.
        delta: delta objetivo.
        contable: contable a usar.

    Returns:
        El sigma necesario, o NaN si no converge.
    """
    from opacus.accountants.utils import get_noise_multiplier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(get_noise_multiplier(
                target_epsilon=eps, target_delta=delta,
                sample_rate=q, steps=pasos, accountant=contable))
        except Exception:
            return float("nan")


def comparar_contables(sigma: float, lote: int, epocas: int,
                       log) -> pd.DataFrame:
    """Compara lo que informa cada contable para el mismo mecanismo.

    Args:
        sigma: multiplicador de ruido fijo.
        lote: tamano de lote.
        epocas: numero de epocas.
        log: registro donde anotar el avance.

    Returns:
        Tabla con el epsilon de cada contable, epoca a epoca.
    """
    q = lote / N
    pasos_por_epoca = N // lote
    filas = []
    rejilla = list(range(1, epocas + 1))
    for e in progreso(rejilla, len(rejilla), log, cada=10,
                      tarea="epocas"):
        pasos = e * pasos_por_epoca
        fila = {"epoca": e, "pasos": pasos}
        for c in CONTABLES:
            fila[c] = eps_de(sigma, q, pasos, DELTA, c)
        # referencia SIN amplificacion: componer los mismos pasos como
        # gaussianos completos, en zCDP, que es la mejor cota simple
        rho = pasos * rho_de_gauss(1.0, sigma)
        fila["sin_amplificar"] = eps_de_rho(rho, DELTA)
        # y la cuenta que haria alguien sin contabilidad fina: cada
        # paso a su epsilon, sumados linealmente
        eps_paso = eps_de(sigma, q, 1, DELTA, "prv")
        fila["basica"] = composicion_basica(eps_paso, 0.0, pasos)[0]
        filas.append(fila)
    return pd.DataFrame(filas)


def barrer_amplificacion(sigma: float, pasos: int,
                         log) -> pd.DataFrame:
    """Mide cuanto aporta la amplificacion segun la tasa de muestreo.

    Args:
        sigma: multiplicador de ruido fijo.
        pasos: numero de pasos fijo.
        log: registro donde anotar el avance.

    Returns:
        Tabla con el epsilon amplificado y sin amplificar por cada q.
    """
    # sin q=1: ahi no hay submuestreo que amplificar, y el
    # contable numerico ademas no converge
    qs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
    rho = pasos * rho_de_gauss(1.0, sigma)
    sin_amp = eps_de_rho(rho, DELTA)
    filas = []
    for q in progreso(qs, len(qs), log, cada=3, tarea="tasas q"):
        amp = eps_de(sigma, q, pasos, DELTA, "prv")
        filas.append({"q": q, "lote_equivalente": int(round(q * N)),
                      "eps_amplificado": amp,
                      "eps_sin_amplificar": sin_amp,
                      "ganancia": sin_amp / amp if amp else np.nan})
    return pd.DataFrame(filas)


def barrer_lote(eps: float, epocas: int, log) -> pd.DataFrame:
    """Mide el efecto del tamano de lote a igual epsilon y epocas.

    Subir el lote sube q, lo que empeora cada paso; pero recorta los
    pasos, que es lo que se compone. El saldo se mide aqui en dos
    magnitudes: el sigma que hace falta y —lo que de verdad importa
    para aprender— el ruido que acaba recibiendo el gradiente MEDIO,
    que es sigma*C/lote.

    Args:
        eps: presupuesto objetivo.
        epocas: numero de epocas.
        log: registro donde anotar el avance.

    Returns:
        Tabla con sigma y ruido efectivo por tamano de lote.
    """
    lotes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    filas = []
    for lote in progreso(lotes, len(lotes), log, cada=2,
                         tarea="tamanos de lote"):
        q = lote / N
        pasos = epocas * (N // lote)
        s = sigma_para(eps, q, pasos, DELTA, "prv")
        filas.append({"lote": lote, "q": q, "pasos": pasos,
                      "sigma": s, "ruido_efectivo": s / lote})
    return pd.DataFrame(filas)


def trampa_de_calibracion(eps: float, q: float, pasos: int,
                          delta: float) -> pd.DataFrame:
    """Mide el desajuste entre calibrar y reportar con contables distintos.

    Opacus tiene dos puertas de entrada con VALORES POR DEFECTO
    DISTINTOS: el motor `PrivacyEngine` usa el contable PRV desde su
    version 1.3, mientras que la funcion suelta `get_noise_multiplier`
    sigue teniendo `accountant="rdp"` en su firma. Quien calibre con
    la segunda y reporte con el primero mezcla dos contabilidades. No
    es inseguro —RDP es mas conservador, luego sobra ruido— pero se
    paga utilidad por nada, y el numero publicado no es el que se uso
    para calibrar.

    Args:
        eps: presupuesto objetivo.
        q: tasa de muestreo.
        pasos: numero de pasos.
        delta: delta objetivo.

    Returns:
        Tabla con el sigma de cada contable y lo que informa el otro.
    """
    filas = []
    for calibra in ("rdp", "prv"):
        s = sigma_para(eps, q, pasos, delta, calibra)
        filas.append({"calibrado_con": calibra, "sigma": s,
                      "eps_segun_rdp": eps_de(s, q, pasos, delta,
                                              "rdp"),
                      "eps_segun_prv": eps_de(s, q, pasos, delta,
                                              "prv")})
    return pd.DataFrame(filas)


def main() -> None:
    """Mide las cuatro cosas y guarda sus tablas."""
    log = crear_registro("cap06.contabilidad")

    from opacus.accountants import create_accountant
    from opacus import PrivacyEngine
    import opacus
    log.info("Opacus %s · contable por defecto de PrivacyEngine: %s",
             opacus.__version__,
             PrivacyEngine().accountant.__class__.__name__)
    log.info("conjunto de referencia: %d ejemplos · delta=%.0e",
             N, DELTA)

    # ── 1) que informa cada contable sobre el mismo mecanismo ───────
    sigma, lote, epocas = 1.0, 256, 60
    log.info("mismo mecanismo (sigma=%.1f, lote=%d, q=%.4f), cuatro "
             "maneras de contabilizarlo:", sigma, lote, lote / N)
    cont = comparar_contables(sigma, lote, epocas, log)
    for _, f in cont.iloc[[0, 4, 9, 29, 59]].iterrows():
        log.info("  epoca %2d (%5d pasos): RDP %6.2f · PRV %6.2f · "
                 "GDP %6.2f · sin amplificar %8.2f · basica %10.1f",
                 int(f["epoca"]), int(f["pasos"]), f["rdp"], f["prv"],
                 f["gdp"], f["sin_amplificar"], f["basica"])
    fin = cont.iloc[-1]
    log.info("  A %d epocas, RDP informa %.2f y PRV %.2f: un %.0f%% "
             "menos por contabilizar mejor, sin tocar el mecanismo.",
             epocas, fin["rdp"], fin["prv"],
             100 * (1 - fin["prv"] / fin["rdp"]))
    log.info("  GDP informa %.2f, todavia menos, pero su analisis "
             "descansa en una aproximacion asintotica y no es una "
             "cota valida en cualquier regimen: conviene no usarlo "
             "para anunciar una garantia.", fin["gdp"])
    SAL_CONTABLES.parent.mkdir(parents=True, exist_ok=True)
    cont.to_csv(SAL_CONTABLES, index=False)
    log.info("guardado %s", SAL_CONTABLES.relative_to(RAIZ))

    # ── 2) lo que aporta la amplificacion por submuestreo ───────────
    pasos = epocas * (N // lote)
    amp = barrer_amplificacion(sigma, pasos, log)
    log.info("amplificacion por submuestreo (%d pasos, sigma=%.1f):",
             pasos, sigma)
    for _, f in amp.iterrows():
        log.info("  q=%.3f (lote %4d): epsilon %8.2f frente a %8.2f "
                 "sin amplificar · gana un factor %.1f", f["q"],
                 int(f["lote_equivalente"]), f["eps_amplificado"],
                 f["eps_sin_amplificar"], f["ganancia"])
    log.info("  Sin submuestreo, entrenar es sencillamente "
             "impagable. La amplificacion no es una optimizacion: es "
             "la razon de que DP-SGD exista.")
    amp.to_csv(SAL_AMPLIF, index=False)
    log.info("guardado %s", SAL_AMPLIF.relative_to(RAIZ))

    # ── 3) el tamano de lote, a igual epsilon y epocas ──────────────
    eps_obj = 3.0
    lot = barrer_lote(eps_obj, 20, log)
    log.info("a epsilon=%.0f y 20 epocas, segun el tamano de lote:",
             eps_obj)
    for _, f in lot.iterrows():
        log.info("  lote %5d (q=%.4f, %5d pasos): sigma %6.3f · "
                 "ruido por ejemplo del gradiente medio %.5f",
                 int(f["lote"]), f["q"], int(f["pasos"]), f["sigma"],
                 f["ruido_efectivo"])
    mejor = lot.loc[lot["ruido_efectivo"].idxmin()]
    peor = lot.loc[lot["ruido_efectivo"].idxmax()]
    log.info("  El sigma SUBE con el lote —cada paso mira a mas "
             "gente—, pero el ruido que recibe el gradiente medio BAJA "
             "un factor %.0f entre el lote %d y el lote %d, porque se "
             "reparte entre mas ejemplos y se dan menos pasos. Por eso "
             "los mejores resultados publicados con DP-SGD usan lotes "
             "enormes, del orden de miles.",
             peor["ruido_efectivo"] / mejor["ruido_efectivo"],
             int(peor["lote"]), int(mejor["lote"]))
    lot.to_csv(SAL_LOTE, index=False)
    log.info("guardado %s", SAL_LOTE.relative_to(RAIZ))

    # ── 4) la trampa: calibrar con un contable y reportar con otro ──
    tr = trampa_de_calibracion(eps_obj, 512 / N, 20 * (N // 512),
                               DELTA)
    log.info("Opacus tiene dos puertas con contables por defecto "
             "DISTINTOS: PrivacyEngine usa PRV desde la version 1.3, y "
             "get_noise_multiplier sigue con «rdp» en su firma. Para "
             "epsilon=%.0f:", eps_obj)
    for _, f in tr.iterrows():
        log.info("  calibrando con %-4s → sigma %.4f · ese sigma da "
                 "epsilon %.3f segun RDP y %.3f segun PRV",
                 f["calibrado_con"], f["sigma"], f["eps_segun_rdp"],
                 f["eps_segun_prv"])
    a, b = tr.iloc[0], tr.iloc[1]
    log.info("  Calibrar con RDP y reportar con PRV no es inseguro "
             "—sobra ruido, no falta—, pero cuesta un %.1f%% de sigma "
             "de mas y publica un epsilon (%.3f) que no es el que se "
             "pidio. Fijar el contable UNA vez y usarlo en los dos "
             "sitios.", 100 * (a["sigma"] / b["sigma"] - 1),
             a["eps_segun_prv"])
    tr.to_csv(SAL_TRAMPA, index=False)
    log.info("guardado %s", SAL_TRAMPA.relative_to(RAIZ))

    muestra_final(cont, log)


if __name__ == "__main__":
    main()
