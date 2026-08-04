"""el vector disperso: pagar solo por las respuestas que interesan.

Hay una situacion muy comun en vigilancia epidemiologica: se lanzan
cientos de consultas de umbral —«¿supera esta combinacion de
diagnostico, sexo y franja de edad las N personas?»— y solo importan
las POCAS que lo superan. Como las consultas se solapan (una misma
persona entra en muchas), la composicion paralela no aplica, y
responderlas todas con Laplace obliga a repartir el presupuesto entre
todas: cuantas mas se preguntan, peor sale cada una.

La tecnica del vector disperso (Dwork y Roth, 2014, seccion 3.6)
rompe esa cuenta: el coste depende de cuantas respuestas POSITIVAS se
devuelven, no de cuantas consultas se hacen. Se puede preguntar mil
veces y pagar por diez.

El script hace dos cosas:
  1. Barre el numero de consultas y mide como se degrada cada metodo.
     El reparto ingenuo cae con m; el vector disperso no se entera.
  2. Mide la FUGA de una variante rota. Lyu, Su y Li (2017)
     catalogaron seis versiones publicadas de esta tecnica y
     comprobaron que varias no son privadas. Aqui se implementa el
     fallo mas repetido —calibrar el ruido para un tope de respuestas
     y luego no pararse en el— y se AUDITA empiricamente: se estima
     por debajo el epsilon que de verdad consume.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/vector_disperso.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso
from comun.ruido_dp import mecanismo_laplace

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SAL_SVT = RAIZ / "data" / "processed" / "dp_svt.csv"
SAL_AUDIT = RAIZ / "data" / "processed" / "dp_svt_auditoria.csv"

EPS_TOTAL = 1.0
TOPE = 10                 # respuestas positivas admitidas (c)
REPETICIONES = 400
SIMULACIONES = 60_000     # realizaciones de la auditoria empirica
PASOS = [20, 10, 5, 2, 1]         # paso del corte de edad, en anos


def bateria_de_consultas(df: pd.DataFrame,
                         paso: int) -> tuple[np.ndarray, float]:
    """Bateria de consultas ACUMULADAS con el detalle que se pida.

    Para cada diagnostico y cada corte de edad se cuenta cuanta gente
    tiene ese diagnostico y al menos esa edad. Son consultas ANIDADAS:
    quien cuenta en «al menos 40» cuenta tambien en «al menos 39», de
    modo que una persona entra en muchisimas y no hay composicion
    paralela que valga.

    Afinar el paso multiplica el NUMERO de consultas sin encoger los
    recuentos —siguen siendo acumulados—, que es justo lo que hace
    falta para aislar el efecto del numero de consultas del efecto de
    tener menos datos en cada una.

    Args:
        df: poblacion con edad y diagnostico.
        paso: separacion entre cortes de edad, en anos.

    Returns:
        Par (vector de recuentos, umbral fijado al percentil 90).
    """
    conteos = []
    for diag in sorted(df["diagnostico"].unique()):
        edades = df.loc[df["diagnostico"] == diag, "edad"].to_numpy()
        for c in range(0, 100, paso):
            conteos.append(float((edades >= c).sum()))
    v = np.array(conteos)
    return v, float(np.percentile(v, 90))


def svt(conteos: np.ndarray, umbral: float, eps: float, tope: int,
        rng: np.random.Generator,
        con_tope: bool = True) -> tuple[np.ndarray, int, int]:
    """Tecnica del vector disperso, correcta o rota.

    La mitad del presupuesto va al ruido del UMBRAL, que se sortea una
    sola vez; la otra mitad se reparte entre las «tope» respuestas
    positivas admitidas, con ruido nuevo en cada consulta. Al llegar
    al tope se PARA: esa parada es lo que acota la fuga, porque cada
    positiva revela informacion y las negativas casi no.

    Args:
        conteos: valor exacto de cada consulta.
        umbral: umbral que se compara.
        eps: presupuesto total.
        tope: respuestas positivas para las que se calibra el ruido.
        rng: generador sembrado.
        con_tope: si es falso, el ruido se calibra igual pero el
            mecanismo NO se detiene. Es la variante rota.

    Returns:
        Terna (vector con 1/0/-1, respuestas dadas, positivas).
    """
    # sensibilidad 1 en cada consulta de recuento
    umbral_ruidoso = umbral + rng.laplace(0.0, 2.0 / eps)
    escala = 4.0 * tope / eps
    salida = np.full(len(conteos), -1, dtype=int)
    positivas = dadas = 0
    for i, q in enumerate(conteos):
        if con_tope and positivas >= tope:
            break
        salida[i] = int(q + rng.laplace(0.0, escala) >= umbral_ruidoso)
        positivas += salida[i]
        dadas += 1
    return salida, dadas, positivas


def laplace_todas(conteos: np.ndarray, umbral: float, eps: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Alternativa ingenua: repartir el presupuesto entre todas."""
    ruidosos = mecanismo_laplace(conteos, 1.0, eps / len(conteos), rng)
    return (ruidosos >= umbral).astype(int)


def calidad(verdad: np.ndarray, salida: np.ndarray) -> dict:
    """Precision y exhaustividad de las respuestas positivas."""
    dadas = salida == 1
    aciertos = int((dadas & verdad).sum())
    return {"positivas": int(dadas.sum()),
            "precision": aciertos / max(int(dadas.sum()), 1),
            "exhaustividad": aciertos / max(int(verdad.sum()), 1)}


def barrido(df: pd.DataFrame, rng: np.random.Generator,
            log) -> pd.DataFrame:
    """Compara los dos metodos segun cuantas consultas se hagan."""
    filas = []
    for paso in progreso(PASOS, len(PASOS), log, cada=1,
                         tarea="granularidades"):
        conteos, umbral = bateria_de_consultas(df, paso)
        verdad = conteos >= umbral
        for nombre, fn in (
                ("vector disperso",
                 lambda: svt(conteos, umbral, EPS_TOTAL, TOPE, rng)[0]),
                ("Laplace en todas",
                 lambda: laplace_todas(conteos, umbral, EPS_TOTAL,
                                       rng))):
            res = pd.DataFrame([calidad(verdad, fn())
                                for _ in range(REPETICIONES)]).mean()
            filas.append({"metodo": nombre, "paso": paso,
                          "consultas": len(conteos),
                          "superan_de_verdad": int(verdad.sum()),
                          "eps_por_consulta_ingenuo":
                              EPS_TOTAL / len(conteos),
                          **res.to_dict()})
    return pd.DataFrame(filas)


def eps_empirico(muestras_d: np.ndarray,
                 muestras_dp: np.ndarray) -> float:
    """Cota INFERIOR del epsilon real, por el evento mas discriminante.

    Se recorren los eventos «el estadistico alcanza al menos t» y se
    devuelve el mayor logaritmo del cociente de sus probabilidades
    bajo las dos bases vecinas. Es una cota por debajo: el epsilon
    real es al menos ese. Si supera al anunciado, el mecanismo no
    cumple lo que dice.

    Args:
        muestras_d: valores del estadistico bajo la base D.
        muestras_dp: valores del estadistico bajo la base vecina D'.

    Returns:
        La mayor separacion logaritmica hallada.
    """
    mejor = 0.0
    lo = int(min(muestras_d.min(), muestras_dp.min()))
    hi = int(max(muestras_d.max(), muestras_dp.max()))
    for t in range(lo, hi + 2):
        for a, b in ((muestras_dp, muestras_d), (muestras_d,
                                                 muestras_dp)):
            p, q = float((a >= t).mean()), float((b >= t).mean())
            # se exige masa suficiente en el denominador para que el
            # cociente no lo fije el ruido de Monte Carlo
            if p > 0 and q > 20 / len(b):
                mejor = max(mejor, np.log(p / q))
    return float(mejor)


def auditar(m: int, rng: np.random.Generator, log) -> pd.DataFrame:
    """Estima el epsilon que consumen de verdad las dos variantes.

    Se construyen dos bases vecinas cuyos recuentos difieren en 1 en
    TODAS las consultas: es el caso de una persona que entra en todas
    ellas, que es exactamente lo que ocurre con consultas anidadas
    («mayores de 30», «mayores de 31», …). Todas quedan justo en el
    umbral, que es donde el ruido decide.

    Se auditan dos estadisticos, porque las dos variantes tienen
    formas de salida distintas: cuantas respuestas se dieron (la
    correcta se detiene antes o despues) y cuantas fueron positivas.
    El epsilon empirico es el mayor de los dos.

    Args:
        m: numero de consultas anidadas de la bateria.
        rng: generador sembrado.
        log: registro donde anotar el avance.

    Returns:
        Tabla con el epsilon empirico de cada variante.
    """
    umbral = 100.0
    bases = {"D": np.full(m, umbral), "D'": np.full(m, umbral + 1.0)}
    filas = []
    for nombre, con_tope, tope in (
            ("correcta (se detiene en c)", True, TOPE),
            ("rota (calibra a c=1 y no para)", False, 1)):
        est = {}
        for etiqueta, datos in bases.items():
            sim = np.array([
                svt(datos, umbral, EPS_TOTAL, tope, rng,
                    con_tope=con_tope)[1:]
                for _ in progreso(range(SIMULACIONES), SIMULACIONES,
                                  log, cada=20000,
                                  tarea=f"{nombre} · {etiqueta}")])
            est[etiqueta] = sim
        eps = max(eps_empirico(est["D"][:, j], est["D'"][:, j])
                  for j in (0, 1))
        # gasto REAL por la propia contabilidad del mecanismo: la
        # mitad del presupuesto en el umbral, mas una fraccion
        # eps/(2*tope) por cada respuesta POSITIVA devuelta. Si se
        # devuelven mas de «tope», el mecanismo gasta de mas y lo
        # sabe: no hace falta auditar nada para verlo.
        positivas = float(est["D'"][:, 1].mean())
        gasto = EPS_TOTAL / 2 + positivas * EPS_TOTAL / (2 * tope)
        filas.append({"variante": nombre, "consultas": m,
                      "eps_anunciado": EPS_TOTAL, "eps_empirico": eps,
                      "eps_contable": gasto, "tope_calibrado": tope,
                      "positivas_D": float(est["D"][:, 1].mean()),
                      "positivas_Dp": positivas})
    return pd.DataFrame(filas)


def main() -> None:
    """Mide la ganancia del vector disperso y audita la variante rota."""
    log = crear_registro("cap05.vector_disperso")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    log.info("bateria de consultas de umbral sobre %d personas "
             "(diagnostico x sexo x franja de edad), con presupuesto "
             "total epsilon=%.1f y tope c=%d", len(df), EPS_TOTAL,
             TOPE)

    # ── 1) lo que se gana, segun cuantas consultas se hagan ─────────
    bar = barrido(df, rng, log)
    for m in sorted(bar["consultas"].unique()):
        sub = bar[bar["consultas"] == m]
        f0 = sub.iloc[0]
        log.info("  %3d consultas (cortes cada %d anos, %d superan el "
                 "umbral; el reparto ingenuo da %.5f a cada una):",
                 int(m), int(f0["paso"]),
                 int(f0["superan_de_verdad"]),
                 f0["eps_por_consulta_ingenuo"])
        for _, f in sub.iterrows():
            log.info("      %-18s precision %.3f · exhaustividad %.3f",
                     f["metodo"], f["precision"], f["exhaustividad"])
    svt_b = bar[bar["metodo"] == "vector disperso"]
    lap_b = bar[bar["metodo"] == "Laplace en todas"]
    log.info("  Al pasar de %d a %d consultas, la precision del "
             "reparto ingenuo cae de %.3f a %.3f mientras la del "
             "vector disperso va de %.3f a %.3f. Es la propiedad que "
             "lo hace util: su coste lo fija c, no m.",
             int(bar["consultas"].min()), int(bar["consultas"].max()),
             lap_b.iloc[0]["precision"], lap_b.iloc[-1]["precision"],
             svt_b.iloc[0]["precision"], svt_b.iloc[-1]["precision"])
    SAL_SVT.parent.mkdir(parents=True, exist_ok=True)
    bar.to_csv(SAL_SVT, index=False)
    log.info("guardado %s", SAL_SVT.relative_to(RAIZ))

    # ── 2) la variante rota, auditada ───────────────────────────────
    log.info("auditoria empirica: %d consultas ANIDADAS (una persona "
             "entra en todas), todas en el umbral, %d simulaciones por "
             "base", 200, SIMULACIONES)
    aud = auditar(200, rng, log)
    for _, f in aud.iterrows():
        log.info("  %-30s devuelve %6.1f positivas (calibrado para "
                 "%d) · gasto CONTABLE %6.2f · cota empirica >= %.2f "
                 "· anunciado %.1f", f["variante"], f["positivas_Dp"],
                 int(f["tope_calibrado"]), f["eps_contable"],
                 f["eps_empirico"], f["eps_anunciado"])
    rota = aud.iloc[1]
    buena = aud.iloc[0]
    log.info("  La correcta cumple: devuelve las %d positivas para "
             "las que se calibro y su gasto contable es %.2f, el "
             "anunciado. La rota se calibro para UNA respuesta y "
             "devuelve %.0f: por el teorema de composicion ha gastado "
             "%.1f, o sea %.0f veces lo que dice. Eso no hay que "
             "auditarlo, se lee en su propia contabilidad.",
             TOPE, buena["eps_contable"], rota["positivas_Dp"],
             rota["eps_contable"], rota["eps_contable"] / EPS_TOTAL)
    log.info("  La cota EMPIRICA, en cambio, solo llega a %.2f para la "
             "rota frente a %.2f para la correcta. Es una leccion en "
             "si misma: auditar da cotas POR DEBAJO, y son flojas. Que "
             "una auditoria salga limpia no demuestra que el mecanismo "
             "cumpla; lo que demuestra la garantia es la DEMOSTRACION, "
             "y por eso los mecanismos no se escriben a mano.",
             rota["eps_empirico"], buena["eps_empirico"])
    aud.to_csv(SAL_AUDIT, index=False)
    log.info("guardado %s", SAL_AUDIT.relative_to(RAIZ))

    muestra_final(bar, log)


if __name__ == "__main__":
    main()
