"""cuando el modelo memoriza, la privacidad sale gratis.

El guion `entrenamiento.py` mide el coste de DP-SGD sobre un modelo
BIEN AJUSTADO, y sale que es de decimas de punto. Ese resultado tiene
una letra pequena: sobre un modelo que generaliza, el ataque de
pertenencia tampoco funcionaba de entrada, asi que la garantia no se
compra contra ninguna amenaza medible.

Aqui se mide el otro caso, que es el que motiva el capitulo entero.
Se entrena la misma tarea con la configuracion que el capitulo 2
llamaba MEMORIZADOR —red ancha, sin normalizar, muchas epocas, Adam—
hasta que el ataque funciona de verdad. Y entonces se aplica DP-SGD.

El resultado es el que hace falta entender para decidir si merece la
pena: sobre un modelo que memoriza, la privacidad diferencial no
cuesta exactitud. La MEJORA, porque lo que destruye es memorizacion,
y la memorizacion no era conocimiento util sino sobreajuste. El
precio que se paga esta en otro sitio —tiempo de computo, ajuste mas
delicado— y no en la calidad del modelo.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/memorizacion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import (auc_pertenencia, dispositivo,
                                  entrenar_red, exactitud, tpr_a_fpr)
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_CURVA = RAIZ / "data" / "processed" / "dp_sgd_memorizacion.csv"
SAL_ROC = RAIZ / "data" / "processed" / "dp_sgd_memorizacion_roc.csv"

# la configuracion que memoriza: ancha, sin normalizar, Adam y muchas
# epocas. Es exactamente lo que se obtiene por descuido, no una
# construccion artificiosa
OCULTA = 512
EPOCAS = 300
LOTE = 256
APRENDIZAJE = 1e-3
C = 1.0
DELTA = 1e-5
PRESUPUESTOS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEMILLAS = [42, 43, 44]


def cargar_tarea() -> tuple:
    """Carga la tarea derivada y su particion."""
    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    df = pd.read_parquet(TAREA)
    cols = [c for c in df.columns if c.startswith("x_")]
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int64)
    dentro = df["particion"].to_numpy() == "entrenamiento"
    return x[dentro], y[dentro], x[~dentro], y[~dentro]


def main() -> None:
    """Mide el ataque y su desaparicion sobre el modelo memorizador."""
    log = crear_registro("cap06.memorizacion")
    fijar_semillas()
    disp = dispositivo()

    x_in, y_in, x_out, y_out = cargar_tarea()
    clases = int(max(y_in.max(), y_out.max()) + 1)
    log.info("configuracion memorizadora: %d neuronas por capa, SIN "
             "normalizar, Adam a %.0e, %d epocas, lote %d · %d "
             "ejemplos dentro y %d fuera", OCULTA, APRENDIZAJE,
             EPOCAS, LOTE, len(y_in), len(y_out))

    filas, curvas = [], []
    escenarios = [None] + PRESUPUESTOS
    for eps in progreso(escenarios, len(escenarios), log, cada=1,
                        tarea="presupuestos"):
        medidas, fpr, tpr = [], None, None
        for s in SEMILLAS:
            modelo, gastado, segundos = entrenar_red(
                x_in, y_in, clases, eps, disp, lote=LOTE,
                epocas=EPOCAS, aprendizaje=APRENDIZAJE, c=C,
                delta=DELTA, oculta=OCULTA, normalizada=False,
                adam=True, semilla=s)
            a, f, tp = auc_pertenencia(modelo, x_in, y_in, x_out,
                                       y_out, disp)
            medidas.append({
                "exactitud_dentro": exactitud(modelo, x_in, y_in,
                                              disp),
                "exactitud_fuera": exactitud(modelo, x_out, y_out,
                                             disp),
                "mia_auc": a,
                "mia_tpr_1pct": tpr_a_fpr(f, tp, 0.01),
                "mia_tpr_01pct": tpr_a_fpr(f, tp, 0.001),
                "segundos": segundos})
            if fpr is None:
                fpr, tpr = f, tp
        m = pd.DataFrame(medidas).mean()
        nombre = "sin DP" if eps is None else f"eps={eps:g}"
        filas.append({"escenario": nombre,
                      "epsilon": np.inf if eps is None else eps,
                      "epsilon_gastado": gastado,
                      "brecha": (m["exactitud_dentro"]
                                 - m["exactitud_fuera"]),
                      **m.to_dict()})
        log.info("  %-8s dentro %.3f · fuera %.3f · brecha %+.3f · "
                 "MIA AUC %.3f · TPR@0,1%% %.4f", nombre,
                 m["exactitud_dentro"], m["exactitud_fuera"],
                 m["exactitud_dentro"] - m["exactitud_fuera"],
                 m["mia_auc"], m["mia_tpr_01pct"])
        paso = max(1, len(fpr) // 200)
        curvas.append(pd.DataFrame({"escenario": nombre,
                                    "fpr": fpr[::paso],
                                    "tpr": tpr[::paso]}))

    tabla = pd.DataFrame(filas)
    SAL_CURVA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SAL_CURVA, index=False)
    pd.concat(curvas).to_csv(SAL_ROC, index=False)

    base = tabla.iloc[0]
    con_dp = tabla.iloc[1:]
    mejor = con_dp.loc[con_dp["exactitud_fuera"].idxmax()]
    log.info("lecturas:")
    log.info("  Sin proteccion, el modelo acierta %.3f sobre lo que ya "
             "vio y %.3f sobre lo que no: una brecha de %+.3f. Eso no "
             "es aprender, es recordar, y el ataque lo detecta con AUC "
             "%.3f —frente al 0,5 que daba el modelo bien ajustado—.",
             base["exactitud_dentro"], base["exactitud_fuera"],
             base["brecha"], base["mia_auc"])
    log.info("  Con DP-SGD el ataque desaparece: AUC %.3f a "
             "epsilon=%.0f y %.3f a epsilon=%.1f. Y la exactitud "
             "sobre datos nuevos SUBE de %.3f a %.3f, %+.1f puntos.",
             con_dp.iloc[-1]["mia_auc"], con_dp.iloc[-1]["epsilon"],
             con_dp.iloc[0]["mia_auc"], con_dp.iloc[0]["epsilon"],
             base["exactitud_fuera"], mejor["exactitud_fuera"],
             100 * (mejor["exactitud_fuera"]
                    - base["exactitud_fuera"]))
    log.info("  Conviene entender por que, porque no es magia y no "
             "siempre pasa: lo que DP-SGD destruye aqui es "
             "memorizacion, y la memorizacion no era conocimiento "
             "util. El recorte y el ruido actuan como una "
             "regularizacion muy fuerte, y este modelo estaba "
             "sobreajustado. Donde el modelo YA generaliza —el caso de "
             "entrenamiento.py— no hay nada de eso que quitar y la "
             "privacidad si cuesta, aunque poco.")
    log.info("  La lectura conjunta de los dos guiones es la tesis del "
             "capitulo: el coste de la garantia depende de si el "
             "modelo estaba memorizando. Si lo estaba, es gratis o "
             "mejor. Si no lo estaba, cuesta decimas. En ningun caso "
             "cuesta lo que la fama de DP-SGD sugiere, al menos en "
             "este tamano de problema.")
    log.info("guardado %s y %s", SAL_CURVA.relative_to(RAIZ),
             SAL_ROC.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
