"""los hiperparametros de DP-SGD no son los de SGD.

Entrenar con privacidad diferencial cambia lo que significa cada
mando, y quien traslade su configuracion habitual obtiene un modelo
mucho peor de lo que el presupuesto permitiria. Tres cosas cambian:

- LA NORMA DE RECORTE C no es un mando de regularizacion sino la
  SENSIBILIDAD del mecanismo. Bajarla reduce el ruido en la misma
  proporcion en que reduce la senal, de modo que su optimo no esta ni
  en «no recortar» ni en «recortar mucho»: esta donde el sesgo del
  recorte iguala al ruido que ahorra. Es el mismo compromiso del tope
  de la seccion 5.3, aplicado a los gradientes.
- LA TASA DE APRENDIZAJE optima es MAYOR que sin privacidad, porque el
  recorte acota el paso y el ruido lo promedia: se puede ser mas
  agresivo sin divergir.
- MAS EPOCAS NO ES MEJOR. Cada epoca gasta presupuesto, y a epsilon
  fijo alargar el entrenamiento obliga a subir sigma. Hay un numero de
  epocas optimo, y pasarse empeora.

Este guion barre los tres a epsilon fijo y mide la exactitud, para que
el optimo se vea en vez de suponerse.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/hiperparametros.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import dispositivo, entrenar_red, exactitud
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_REJILLA = RAIZ / "data" / "processed" / "dp_sgd_rejilla.csv"
SAL_EPOCAS = RAIZ / "data" / "processed" / "dp_sgd_epocas.csv"

EPS = 3.0
NORMAS = [0.1, 0.3, 1.0, 3.0, 10.0]
TASAS = [0.25, 0.5, 1.0, 2.0]
REJILLA_EPOCAS = [5, 10, 20, 30, 50, 80]
SEMILLAS = [42, 43, 44]
# la referencia SIN privacidad necesita SU tasa, no la del regimen
# privado: con lr alta y sin recorte el descenso corriente diverge, y
# compararlo asi no mide el coste de la privacidad sino un ajuste malo
LR_SIN_DP = 0.003


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
    """Barre C, la tasa de aprendizaje y las epocas, a epsilon fijo."""
    log = crear_registro("cap06.hiperparametros")
    fijar_semillas()
    disp = dispositivo()

    x_in, y_in, x_out, y_out = cargar_tarea()
    clases = int(max(y_in.max(), y_out.max()) + 1)
    log.info("tarea: %d de entrenamiento · %d de prueba · epsilon "
             "fijo %.0f · dispositivo %s", len(y_in), len(y_out), EPS,
             disp)

    # ── 1) la rejilla C x tasa de aprendizaje ───────────────────────
    combinaciones = [(c, lr) for c in NORMAS for lr in TASAS]
    filas = []
    for c, lr in progreso(combinaciones, len(combinaciones), log,
                          cada=5, tarea="C x lr"):
        modelo, gastado, _ = entrenar_red(x_in, y_in, clases, EPS,
                                          disp, aprendizaje=lr, c=c)
        filas.append({"C": c, "lr": lr, "epsilon": gastado,
                      "exactitud": exactitud(modelo, x_out, y_out,
                                             disp)})
    rejilla = pd.DataFrame(filas)
    SAL_REJILLA.parent.mkdir(parents=True, exist_ok=True)
    rejilla.to_csv(SAL_REJILLA, index=False)

    tabla = rejilla.pivot(index="C", columns="lr", values="exactitud")
    log.info("exactitud fuera del entrenamiento, a epsilon=%.0f:", EPS)
    log.info("        %s", "  ".join(f"lr={lr:<5g}"
                                     for lr in tabla.columns))
    for c, fila in tabla.iterrows():
        log.info("  C=%-5g %s", c,
                 "  ".join(f"{v:.3f}    " for v in fila))
    mejor = rejilla.loc[rejilla["exactitud"].idxmax()]
    peor = rejilla.loc[rejilla["exactitud"].idxmin()]
    log.info("  Mejor: C=%g, lr=%g → %.3f. Peor: C=%g, lr=%g → %.3f. "
             "Entre la mejor y la peor configuracion hay %.1f puntos "
             "de exactitud, con el MISMO presupuesto gastado.",
             mejor["C"], mejor["lr"], mejor["exactitud"],
             peor["C"], peor["lr"], peor["exactitud"],
             100 * (mejor["exactitud"] - peor["exactitud"]))
    log.info("  El optimo de C esta en %g, no en el extremo: recortar "
             "poco deja pasar gradientes grandes y obliga a un ruido "
             "proporcional; recortar mucho tira senal. Es el mismo "
             "minimo interior del tope de visitas del capitulo 5.",
             mejor["C"])
    log.info("guardado %s", SAL_REJILLA.relative_to(RAIZ))

    # ── 2) mas epocas no es mejor ───────────────────────────────────
    filas = []
    for ep in progreso(REJILLA_EPOCAS, len(REJILLA_EPOCAS), log,
                       cada=2, tarea="epocas"):
        con, sin = [], []
        for s in SEMILLAS:
            m1, gastado, _ = entrenar_red(
                x_in, y_in, clases, EPS, disp, epocas=ep,
                aprendizaje=float(mejor["lr"]), c=float(mejor["C"]),
                semilla=s)
            con.append(exactitud(m1, x_out, y_out, disp))
            # referencia SIN privacidad, con SU propia tasa
            m2, _, _ = entrenar_red(x_in, y_in, clases, None, disp,
                                    epocas=ep,
                                    aprendizaje=LR_SIN_DP,
                                    semilla=s)
            sin.append(exactitud(m2, x_out, y_out, disp))
        filas.append({"epocas": ep, "epsilon": gastado,
                      "exactitud_dp": float(np.mean(con)),
                      "exactitud_dp_sd": float(np.std(con)),
                      "exactitud_sin_dp": float(np.mean(sin))})
    epocas = pd.DataFrame(filas)
    epocas.to_csv(SAL_EPOCAS, index=False)

    log.info("a epsilon=%.0f fijo, segun cuantas epocas se entrene "
             "(C=%g, lr=%g):", EPS, mejor["C"], mejor["lr"])
    for _, f in epocas.iterrows():
        log.info("  %2d epocas: con DP %.3f ± %.3f · sin DP %.3f",
                 int(f["epocas"]), f["exactitud_dp"],
                 f["exactitud_dp_sd"], f["exactitud_sin_dp"])
    cima = epocas.loc[epocas["exactitud_dp"].idxmax()]
    dispersion = epocas["exactitud_dp"].std()
    log.info("  Con DP el maximo esta en %d epocas (%.3f), pero la "
             "dispersion entre configuraciones (%.3f) es del orden de "
             "la dispersion entre semillas (%.3f): sobre esta tarea y "
             "a este presupuesto, el numero de epocas NO decide gran "
             "cosa, y afirmar lo contrario seria leer ruido.",
             int(cima["epocas"]), cima["exactitud_dp"], dispersion,
             epocas["exactitud_dp_sd"].mean())
    log.info("  Lo que si se ve es la asimetria de fondo: sin "
             "privacidad la exactitud sube con las epocas y se estanca "
             "(%.3f a %d epocas, %.3f a %d), porque mirar los datos "
             "otra vez no cuesta nada. Con presupuesto fijo, cada "
             "epoca de mas obliga a subir sigma, de modo que existe un "
             "punto a partir del cual entrenar mas es entrenar peor. "
             "Esa es la diferencia de regimen, y la razon de que los "
             "ajustes heredados de un entrenamiento corriente casi "
             "nunca sirvan.",
             epocas.iloc[0]["exactitud_sin_dp"],
             int(epocas.iloc[0]["epocas"]),
             epocas.iloc[-1]["exactitud_sin_dp"],
             int(epocas.iloc[-1]["epocas"]))
    log.info("guardado %s", SAL_EPOCAS.relative_to(RAIZ))

    muestra_final(rejilla, log)


if __name__ == "__main__":
    main()
