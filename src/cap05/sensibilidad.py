"""de que depende el ruido: sensibilidad, vecindad y unidad.

El epsilon no fija el ruido por si solo. Lo fija junto con tres
decisiones que se toman ANTES y que casi nunca se documentan:

1. La SENSIBILIDAD de la consulta. Un recuento vale 1; una suma vale
   lo que valga el mayor sumando posible. Cuando esa magnitud no tiene
   tope publico —el numero de visitas de un paciente no lo tiene— no
   existe NINGUN mecanismo que la responda: hay que imponer un tope, y
   el tope introduce sesgo. Aqui se mide ese compromiso donde de
   verdad aprieta, que es en las provincias pequenas.
2. La VECINDAD. Sustituir a una persona (vecindad acotada, la del
   censo) no es lo mismo que anadirla o quitarla (vecindad no
   acotada): sobre un histograma la primera da sensibilidad 2 y la
   segunda 1, o sea el DOBLE de ruido por la misma garantia nominal.
3. La UNIDAD de privacidad. Si una persona aporta varias filas y el
   mecanismo protege filas, la garantia por persona no es epsilon:
   es k*epsilon, con k el maximo de filas por persona. Es el primero
   de los «riesgos de privacidad» que enumera la guia NIST SP 800-226,
   y aqui se mide la ventaja que le deja al adversario.

Escribe tres CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/sensibilidad.py
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
SAL_RECORTE = RAIZ / "data" / "processed" / "dp_recorte.csv"
SAL_VECINDAD = RAIZ / "data" / "processed" / "dp_vecindad.csv"
SAL_UNIDAD = RAIZ / "data" / "processed" / "dp_unidad.csv"

EPS = 1.0
REPETICIONES = 4000
FRECUENTACION = 7.0       # contactos medios al ano en primaria
TOPES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 70, 100]
# tres provincias por tamano: la mayor, una mediana y la menor
MUESTRA_PROV = {"28": "la mayor", "26": "una mediana",
                "52": "la menor"}


def registro_de_visitas(df: pd.DataFrame,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Deriva un registro de visitas con frecuentacion realista.

    La tabla de poblacion tiene una fila por PERSONA. Un registro
    clinico real tiene una fila por CONTACTO, y el numero de contactos
    por persona es muy asimetrico: la mayoria acude poco y una minoria
    acumula muchas visitas. Se simula con una geometrica de media
    igual a la frecuentacion de atencion primaria en Espana.

    Args:
        df: tabla de poblacion, una fila por persona.
        rng: generador sembrado.

    Returns:
        La misma tabla con una columna visitas por persona.
    """
    return df.assign(
        provincia=df["codigo_postal"].astype(str).str[:2],
        visitas=rng.geometric(p=1 / FRECUENTACION, size=len(df)))


def coste_del_recorte(visitas: np.ndarray, epsilon: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Mide sesgo y ruido de la media de visitas segun donde se recorte.

    La suma de visitas tiene sensibilidad igual al tope que se imponga.
    Recortar en C la baja a C y con ella el ruido, pero introduce un
    sesgo determinista: todo el que pase de C cuenta como C. El error
    total es la suma de los dos, y tiene un minimo interior.

    Args:
        visitas: visitas de cada persona del grupo consultado.
        epsilon: presupuesto de la consulta.
        rng: generador sembrado.

    Returns:
        Tabla con sesgo, ruido y error total por cada tope.
    """
    n = len(visitas)
    verdadera = float(visitas.mean())
    filas = []
    for c in TOPES:
        recortadas = np.minimum(visitas, c)
        sesgo = float(recortadas.mean()) - verdadera
        # el tamano del grupo es publico (el padron lo es), asi que
        # todo el presupuesto va a la suma y se divide por n exacto
        est = np.array([float(mecanismo_laplace(
            float(recortadas.sum()), float(c), epsilon, rng)) / n
            for _ in range(REPETICIONES)])
        filas.append({"tope": c, "n": n, "sesgo": sesgo,
                      "ruido": float(est.std()),
                      "error_total": float(np.abs(est
                                                  - verdadera).mean()),
                      "pct_recortados": float(100
                                              * (visitas > c).mean())})
    return pd.DataFrame(filas)


def coste_de_la_vecindad(conteos: np.ndarray, epsilon: float,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Compara el error del histograma bajo las dos vecindades.

    Args:
        conteos: recuentos exactos del histograma.
        epsilon: presupuesto de la consulta.
        rng: generador sembrado.

    Returns:
        Tabla con el error medio por casilla bajo cada vecindad.
    """
    filas = []
    for nombre, sens in (("no acotada (anadir o quitar)", 1.0),
                         ("acotada (sustituir)", 2.0)):
        errores = [float(np.abs(mecanismo_laplace(
            conteos.astype(float), sens, epsilon, rng)
            - conteos).mean()) for _ in range(200)]
        filas.append({"vecindad": nombre, "sensibilidad": sens,
                      "error_medio": float(np.mean(errores))})
    return pd.DataFrame(filas)


def ventaja_del_adversario(k: int, epsilon: float) -> float:
    """Ventaja optima al distinguir si una persona de k filas esta.

    Con un mecanismo de Laplace calibrado a epsilon POR FILA, quitar a
    una persona de k filas desplaza la respuesta k unidades. La mejor
    prueba posible acierta con la distancia de variacion total entre
    las dos distribuciones, que para dos Laplace de escala 1/epsilon
    desplazadas k vale 1 - e^(-epsilon*k/2).

    Args:
        k: filas que aporta la persona.
        epsilon: presupuesto por fila.

    Returns:
        Ventaja en [0, 1]; 0 es indistinguible y 1 es certeza.
    """
    return 1.0 - np.exp(-epsilon * k / 2.0)


def coste_de_la_unidad(vis: pd.DataFrame, epsilon: float,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Mide que garantia por PERSONA da un mecanismo por FILA.

    Args:
        vis: tabla de poblacion con la columna visitas.
        epsilon: presupuesto nominal, por fila.
        rng: generador sembrado.

    Returns:
        Tabla con la garantia efectiva y la ventaja empirica del
        adversario en varios cuantiles de frecuentacion.
    """
    k = vis["visitas"]
    total = float(k.sum())
    filas = []
    for etiqueta, valor in (("mediana", int(k.median())),
                            ("p90", int(k.quantile(0.90))),
                            ("p99", int(k.quantile(0.99))),
                            ("maximo", int(k.max()))):
        # comprobacion empirica de la ventaja: se simulan las dos
        # respuestas y se separa por el umbral optimo (el punto medio)
        con = mecanismo_laplace(np.full(40000, total), 1.0, epsilon,
                                rng)
        sin_ = mecanismo_laplace(np.full(40000, total - valor), 1.0,
                                 epsilon, rng)
        umbral = total - valor / 2.0
        filas.append({
            "cuantil": etiqueta, "visitas": valor,
            "eps_por_fila": epsilon,
            "eps_por_persona": epsilon * valor,
            "ventaja_teorica": ventaja_del_adversario(valor, epsilon),
            "ventaja_medida": float((con > umbral).mean()
                                    - (sin_ > umbral).mean())})
    return pd.DataFrame(filas)


def main() -> None:
    """Ejecuta las tres mediciones y guarda sus tablas."""
    log = crear_registro("cap05.sensibilidad")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    vis = registro_de_visitas(df, rng)
    k = vis["visitas"]
    log.info("registro de visitas derivado de la poblacion: %d "
             "contactos de %d personas · media %.1f · mediana %d · "
             "p99 %d · maximo %d — sin tope publico posible",
             int(k.sum()), len(vis), k.mean(), int(k.median()),
             int(k.quantile(0.99)), int(k.max()))

    # ── 1) donde recortar, y por que depende del tamano del grupo ───
    log.info("media de visitas por provincia con epsilon=%.1f: sesgo "
             "del recorte contra ruido del mecanismo", EPS)
    trozos = []
    for cod, papel in progreso(list(MUESTRA_PROV.items()),
                               len(MUESTRA_PROV), log, cada=1,
                               tarea="provincias"):
        v = vis.loc[vis["provincia"] == cod, "visitas"].to_numpy()
        rec = coste_del_recorte(v, EPS, rng).assign(provincia=cod,
                                                    papel=papel)
        mejor = rec.loc[rec["error_total"].idxmin()]
        peor = rec.loc[rec["tope"] == max(TOPES)].iloc[0]
        log.info("  provincia %s (%s, %d personas, media real %.2f): "
                 "optimo en el tope %d con error %.3f visitas, que es "
                 "un %.2f%% de la media; sin recortar (tope %d) el "
                 "error es %.3f, o sea %.1f veces peor. El sesgo del "
                 "optimo es %+.3f.",
                 cod, papel, len(v), v.mean(), int(mejor["tope"]),
                 mejor["error_total"],
                 100 * mejor["error_total"] / v.mean(), max(TOPES),
                 peor["error_total"],
                 peor["error_total"] / mejor["error_total"],
                 mejor["sesgo"])
        trozos.append(rec)
    rec_todas = pd.concat(trozos, ignore_index=True)
    SAL_RECORTE.parent.mkdir(parents=True, exist_ok=True)
    rec_todas.to_csv(SAL_RECORTE, index=False)
    log.info("  Dos lecturas. El tope optimo no es una propiedad de la "
             "magnitud sino del grupo: baja de 50 a 30 al pasar de "
             "2903 personas a 138. Y sobre todo, el coste de la "
             "garantia NO se reparte por igual: con el mismo epsilon, "
             "el error relativo va del 0,25%% en la provincia mayor al "
             "15%% en la menor. La privacidad diferencial protege a "
             "todos igual y cobra a los grupos pequenos mucho mas.")
    log.info("guardado %s", SAL_RECORTE.relative_to(RAIZ))

    # ── 2) que vecindad se ha elegido ───────────────────────────────
    conteos = (vis.groupby(["provincia", "diagnostico"],
                           observed=True).size().to_numpy())
    vec = coste_de_la_vecindad(conteos, EPS, rng)
    log.info("histograma de %d casillas, epsilon=%.1f:", len(conteos),
             EPS)
    for _, f in vec.iterrows():
        log.info("  vecindad %-28s sensibilidad %.0f · error medio "
                 "%.3f personas por casilla", f["vecindad"],
                 f["sensibilidad"], f["error_medio"])
    log.info("  El MISMO epsilon nominal cuesta el doble de ruido bajo "
             "la vecindad acotada. Anunciar epsilon sin decir la "
             "vecindad deja la garantia indeterminada.")
    vec.to_csv(SAL_VECINDAD, index=False)
    log.info("guardado %s", SAL_VECINDAD.relative_to(RAIZ))

    # ── 3) que unidad se protege ────────────────────────────────────
    uni = coste_de_la_unidad(vis, EPS, rng)
    log.info("un mecanismo calibrado a epsilon=%.1f POR FILA sobre el "
             "registro de contactos:", EPS)
    for _, f in uni.iterrows():
        log.info("  %-8s %3d visitas -> garantia real por persona "
                 "epsilon=%2.0f · ventaja del adversario %.3f "
                 "(teorica %.3f)", f["cuantil"], f["visitas"],
                 f["eps_por_persona"], f["ventaja_medida"],
                 f["ventaja_teorica"])
    peor = uni.iloc[-1]
    log.info("  Con epsilon=1 por fila, al paciente MEDIANO ya se le "
             "detecta con ventaja %.2f, y al de la cola con %.2f: la "
             "garantia por persona es epsilon=%.0f, no 1. Para "
             "proteger PERSONAS hay que recortar las filas por persona "
             "a un tope y calibrar a ese tope.",
             uni.iloc[0]["ventaja_medida"], peor["ventaja_medida"],
             peor["eps_por_persona"])
    uni.to_csv(SAL_UNIDAD, index=False)
    log.info("guardado %s", SAL_UNIDAD.relative_to(RAIZ))

    muestra_final(rec_todas, log)


if __name__ == "__main__":
    main()
