"""proteger ejemplos no es proteger personas, medido.

DP-SGD, tal como lo implementan todas las bibliotecas, protege
EJEMPLOS. Si cada persona aporta uno, ambas cosas coinciden. Si aporta
k —varias visitas, varias imagenes, varios episodios—, la garantia por
persona se degrada a k*epsilon por privacidad de grupo, linealmente y
sin que nada avise.

McMahan et al. (2018) califican de «prohibitivo» el coste de traducir
de una a otra, y la afirmacion se cita mucho sin numeros. Este guion
los pone, sobre los datos del libro y en las dos direcciones:

1. Lo que se PIERDE si no se corrige: con epsilon=8 por ejemplo y k
   filas por persona, la garantia real por persona es 8k, y la ventaja
   del mejor test posible para decidir si esa persona esta en el
   conjunto es 1 - e^{-8k/2}, que satura casi de inmediato.
2. Lo que CUESTA corregirlo: para tener epsilon=8 POR PERSONA hay que
   calibrar a 8/k por ejemplo. Aqui se entrena con ese presupuesto y
   se mide la exactitud que queda.

La tercera via —recortar a un tope de filas por persona y calibrar a
ese tope— se mide tambien, porque es la que se usa en la practica y
la que decide si el proyecto es viable.

Escribe un CSV en data/processed/ para la figura del libro.

Uso: python3 src/cap06/unidad.py
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
SALIDA = RAIZ / "data" / "processed" / "dp_sgd_unidad.csv"

EPS_PERSONA = 8.0         # la garantia que se quiere POR PERSONA
KS = [1, 3, 10]           # filas por persona
TOPE = 3                  # tope de filas al que recortar en la tercera via
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


def replicar(x: np.ndarray, y: np.ndarray, k: int,
             rng: np.random.Generator) -> tuple:
    """Convierte una fila por persona en k filas por persona.

    Es lo que ocurre al pasar de un padron a un registro de contactos:
    la misma persona aparece k veces, con variaciones pequenas que no
    cambian quien es. Se anade un ruido leve a las variables continuas
    para que las filas no sean identicas, como no lo son dos visitas.

    Args:
        x: caracteristicas, una fila por persona.
        y: etiquetas.
        k: filas por persona.
        rng: generador sembrado.

    Returns:
        Par (X replicado, y replicado).
    """
    if k == 1:
        return x, y
    xr = np.repeat(x, k, axis=0).astype(np.float32)
    # solo las dos primeras columnas son continuas (edad y sexo
    # codificado); el resto son indicadores que no deben moverse
    xr[:, 0] += rng.normal(0.0, 0.02, len(xr)).astype(np.float32)
    return xr, np.repeat(y, k)


def ventaja(k: int, eps_ejemplo: float) -> float:
    """Ventaja del mejor test para detectar a una persona de k filas.

    Por privacidad de grupo la garantia por persona es k*eps, y la
    distancia de variacion total entre las dos distribuciones vale
    1 - e^{-k*eps/2}. Es la misma cuenta de la seccion 5.3, aplicada
    al entrenamiento.

    Args:
        k: filas que aporta la persona.
        eps_ejemplo: presupuesto por ejemplo.

    Returns:
        Ventaja en [0, 1].
    """
    return float(1.0 - np.exp(-k * eps_ejemplo / 2.0))


def main() -> None:
    """Mide las tres vias sobre la misma tarea."""
    log = crear_registro("cap06.unidad")
    rng = fijar_semillas()
    disp = dispositivo()

    x_in, y_in, x_out, y_out = cargar_tarea()
    clases = int(max(y_in.max(), y_out.max()) + 1)
    log.info("tarea: %d personas dentro del entrenamiento · %d fuera "
             "· objetivo: epsilon=%.0f POR PERSONA", len(y_in),
             len(y_out), EPS_PERSONA)

    filas = []
    casos = [(k, via) for k in KS
             for via in ("ignorar", "calibrar a k", "recortar a tope")]
    for k, via in progreso(casos, len(casos), log, cada=3,
                           tarea="k x via"):
        if via == "ignorar":
            # se entrena a eps=8 POR EJEMPLO y se anuncia 8, que es lo
            # que hace casi todo el mundo. La garantia real es 8k.
            eps_ejemplo, k_efectiva = EPS_PERSONA, k
        elif via == "calibrar a k":
            # correcto y caro: 8/k por ejemplo da 8 por persona
            eps_ejemplo, k_efectiva = EPS_PERSONA / k, k
        else:
            # la via practica: recortar a TOPE filas por persona y
            # calibrar a ese tope
            eps_ejemplo = EPS_PERSONA / min(k, TOPE)
            k_efectiva = min(k, TOPE)

        xk, yk = replicar(x_in, y_in, k, rng)
        if via == "recortar a tope" and k > TOPE:
            # se conservan TOPE filas de cada persona, tomadas al azar
            idx = np.concatenate([
                rng.choice(np.arange(i * k, (i + 1) * k), TOPE,
                           replace=False)
                for i in range(len(y_in))])
            xk, yk = xk[idx], yk[idx]

        exac = np.mean([
            exactitud(entrenar_red(xk, yk, clases, eps_ejemplo, disp,
                                   semilla=s)[0], x_out, y_out, disp)
            for s in SEMILLAS])
        filas.append({
            "k": k, "via": via, "filas": len(yk),
            "eps_por_ejemplo": eps_ejemplo,
            "eps_por_persona": eps_ejemplo * k_efectiva,
            "ventaja_adversario": ventaja(k_efectiva, eps_ejemplo),
            "exactitud": float(exac)})

    tabla = pd.DataFrame(filas)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)

    for k in KS:
        sub = tabla[tabla["k"] == k]
        log.info("con %d filas por persona:", k)
        for _, f in sub.iterrows():
            log.info("    %-16s eps por ejemplo %5.2f → POR PERSONA "
                     "%5.1f · ventaja %.3f · exactitud %.3f",
                     f["via"], f["eps_por_ejemplo"],
                     f["eps_por_persona"], f["ventaja_adversario"],
                     f["exactitud"])

    ign = tabla[tabla["via"] == "ignorar"]
    cal = tabla[tabla["via"] == "calibrar a k"]
    rec = tabla[tabla["via"] == "recortar a tope"]
    log.info("lecturas:")
    log.info("  Ignorarlo no cuesta exactitud —de %.3f a %.3f— porque "
             "no se hace nada: solo se anuncia un epsilon que no es. "
             "Con %d filas por persona el presupuesto REAL es %.0f, y "
             "la ventaja del adversario para detectar a esa persona "
             "vale %.3f, o sea certeza.",
             ign.iloc[0]["exactitud"], ign.iloc[-1]["exactitud"],
             KS[-1], ign.iloc[-1]["eps_por_persona"],
             ign.iloc[-1]["ventaja_adversario"])
    log.info("  Calibrar a k SI cuesta: la exactitud baja de %.3f con "
             "una fila por persona a %.3f con %d, porque el "
             "presupuesto por ejemplo se divide por %d. Eso es lo que "
             "McMahan et al. llaman coste «prohibitivo», con numeros.",
             cal.iloc[0]["exactitud"], cal.iloc[-1]["exactitud"],
             KS[-1], KS[-1])
    log.info("  Recortar a un tope de %d filas es el termino medio que "
             "se usa en la practica: da la garantia buena por persona "
             "(%.0f) con exactitud %.3f, a cambio de tirar el %.0f%% "
             "de las filas de quien mas aporta.", TOPE,
             rec.iloc[-1]["eps_por_persona"], rec.iloc[-1]["exactitud"],
             100 * (1 - TOPE / KS[-1]))
    log.info("guardado %s", SALIDA.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
