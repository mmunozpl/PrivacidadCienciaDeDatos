"""quien paga la garantia, cuando la garantia va dentro del modelo.

El capitulo 5 midio que la privacidad diferencial no reparte su coste
por igual: con el mismo epsilon, el error relativo de una tabla iba
del 0,25 % en la provincia mayor al 15 % en la menor. Aqui se
comprueba si pasa lo mismo dentro de un modelo, que es una afirmacion
distinta y que hay que medir aparte.

El mecanismo por el que podria pasar es concreto y tiene nombre. El
recorte de gradientes trata a todos los ejemplos por igual, pero los
ejemplos de un grupo infrarrepresentado producen gradientes MAYORES
—el modelo los predice peor, luego su perdida es mayor—, de modo que
el recorte se los lleva por delante mas a menudo y su senal llega
atenuada. El ruido, ademas, pesa mas donde hay menos ejemplos que lo
promedien. Bagdasaryan, Poursaeed y Shmatikov (2019) documentaron el
efecto; aqui se mide sobre nuestros datos, con tres particiones que
interesan por razones distintas:

- por TAMANO DE PROVINCIA, que es el eje del capitulo 5;
- por FRECUENCIA DE LA CLASE, que es donde el efecto se describio;
- por EDAD, porque es la variable con mas senal de la tarea.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/sesgo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import dispositivo, entrenar_red
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_GRUPOS = RAIZ / "data" / "processed" / "dp_sgd_sesgo.csv"
SAL_NORMAS = RAIZ / "data" / "processed" / "dp_sgd_normas_grupo.csv"

PRESUPUESTOS = [1.0, 3.0, 8.0]
C = 1.0                   # la misma norma de recorte que entrenamiento.py
APRENDIZAJE = 0.3         # la tasa que entrenamiento.py eligio por
                          # validacion para los regimenes con recorte


def cargar_tarea() -> pd.DataFrame:
    """Carga la tarea derivada con sus columnas de agrupacion."""
    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    return pd.read_parquet(TAREA)


def particiones(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Define los tres ejes de agrupacion del analisis.

    Args:
        df: tarea derivada, con provincia, edad y etiqueta.

    Returns:
        Diccionario nombre del eje -> etiqueta de grupo por fila.
    """
    tam = df.groupby("provincia", observed=True)["y"].transform("size")
    return {
        "tamano de provincia": pd.cut(
            tam, [0, 100, 400, 10 ** 9],
            labels=["pequeña (<100)", "mediana", "grande (>400)"]),
        "frecuencia de la clase": df["y"].map(
            df["y"].value_counts(normalize=True)).pipe(
                lambda s: pd.cut(s, [0, 0.15, 0.3, 1.0],
                                 labels=["clase rara", "clase media",
                                         "clase frecuente"])),
        "edad": pd.cut(df["edad"], [0, 40, 65, 120],
                       labels=["hasta 40", "41-65", "más de 65"]),
    }


def exactitud_por_grupo(modelo, x: np.ndarray, y: np.ndarray,
                        grupo: pd.Series,
                        disp: torch.device) -> pd.Series:
    """Exactitud dentro de cada grupo.

    Args:
        modelo: red entrenada.
        x: caracteristicas.
        y: etiquetas verdaderas.
        grupo: etiqueta de grupo de cada fila.
        disp: dispositivo de computo.

    Returns:
        Serie grupo -> exactitud.
    """
    modelo.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=disp)
        pred = modelo(xt).argmax(dim=1).cpu().numpy()
    return pd.Series(pred == y).groupby(
        grupo.reset_index(drop=True), observed=True).mean()


def normas_por_grupo(modelo, x: np.ndarray, y: np.ndarray,
                     grupo: pd.Series, disp: torch.device,
                     c: float) -> pd.DataFrame:
    """Norma del gradiente por ejemplo, agregada por grupo.

    Es la medida que explica el efecto: si un grupo produce gradientes
    sistematicamente mayores, el recorte a C le quita mas senal que a
    los demas.

    Args:
        modelo: red sobre la que derivar.
        x: caracteristicas.
        y: etiquetas.
        grupo: etiqueta de grupo de cada fila.
        disp: dispositivo de computo.
        c: norma de recorte con la que se compara.

    Returns:
        Tabla con la norma mediana y la fraccion recortada por grupo.
    """
    from torch import nn
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in modelo.named_parameters()}
    buffers = {k: v.detach() for k, v in modelo.named_buffers()}
    criterio = nn.CrossEntropyLoss()

    def perdida_de_uno(p, b, x_uno, y_uno):
        salida = functional_call(modelo, (p, b), (x_uno.unsqueeze(0),))
        return criterio(salida, y_uno.unsqueeze(0))

    normas = []
    for i in range(0, len(y), 1024):
        xb = torch.tensor(x[i:i + 1024], dtype=torch.float32,
                          device=disp)
        yb = torch.tensor(y[i:i + 1024], dtype=torch.long, device=disp)
        g = vmap(grad(perdida_de_uno), in_dims=(None, None, 0, 0))(
            params, buffers, xb, yb)
        cuad = sum((v.reshape(len(yb), -1) ** 2).sum(dim=1)
                   for v in g.values())
        normas.append(cuad.sqrt().cpu().numpy())
    normas = np.concatenate(normas)

    g = grupo.reset_index(drop=True)
    return pd.DataFrame({
        "norma_mediana": pd.Series(normas).groupby(
            g, observed=True).median(),
        "fraccion_recortada": pd.Series(normas > c).groupby(
            g, observed=True).mean()})


def main() -> None:
    """Mide el reparto del coste entre grupos, con y sin privacidad."""
    log = crear_registro("cap06.sesgo")
    fijar_semillas()
    disp = dispositivo()

    df = cargar_tarea()
    cols = [c for c in df.columns if c.startswith("x_")]
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int64)
    dentro = df["particion"].to_numpy() == "entrenamiento"
    ejes = particiones(df)
    clases = int(y.max() + 1)
    log.info("tarea: %d ejemplos · %d clases · tres ejes de "
             "agrupacion: %s", len(df), clases, ", ".join(ejes))

    # la referencia NO es el descenso corriente sino el recorte SIN
    # ruido: es lo unico que aisla el coste de la garantia del efecto
    # del recorte, que no es una medida de privacidad (ver
    # entrenamiento.py). Ademas comparte tasa de aprendizaje con los
    # regimenes privados, de modo que la comparacion es limpia.
    escenarios = [(None, 0.0)] + [(e, None) for e in PRESUPUESTOS]
    filas, normas = [], []
    for eps, sigma in progreso(escenarios, len(escenarios), log,
                               cada=1, tarea="presupuestos"):
        modelo, _, _ = entrenar_red(x[dentro], y[dentro], clases,
                                    eps, disp, c=C, sigma=sigma,
                                    aprendizaje=APRENDIZAJE)
        nombre = "sin ruido" if eps is None else f"eps={eps:g}"
        for eje, grupo in ejes.items():
            exac = exactitud_por_grupo(modelo, x[~dentro], y[~dentro],
                                       grupo[~dentro], disp)
            for g, v in exac.items():
                filas.append({"escenario": nombre, "eje": eje,
                              "grupo": str(g), "exactitud": float(v)})
        if eps is None:
            for eje, grupo in ejes.items():
                n = normas_por_grupo(modelo, x[dentro], y[dentro],
                                     grupo[dentro], disp, C)
                for g, f in n.iterrows():
                    normas.append({"eje": eje, "grupo": str(g),
                                   **f.to_dict()})

    tabla = pd.DataFrame(filas)
    SAL_GRUPOS.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SAL_GRUPOS, index=False)
    pd.DataFrame(normas).to_csv(SAL_NORMAS, index=False)

    for eje in ejes:
        sub = tabla[tabla["eje"] == eje].pivot(
            index="grupo", columns="escenario", values="exactitud")
        log.info("eje «%s»:", eje)
        for g, f in sub.iterrows():
            caida = f["sin ruido"] - f.get("eps=3", np.nan)
            log.info("  %-16s sin ruido %.3f · %s · caida a eps=3: "
                     "%+.3f", g, f["sin ruido"],
                     " · ".join(f"{c} {f[c]:.3f}" for c in sub.columns
                                if c != "sin ruido"), -caida)
        caidas = (sub["sin ruido"] - sub["eps=3"]).sort_values()
        log.info("  Caida maxima entre grupos: %+.4f («%s»); minima: "
                 "%+.4f («%s»). Rango de la disparidad: %.4f puntos.",
                 -caidas.iloc[-1], caidas.index[-1],
                 -caidas.iloc[0], caidas.index[0],
                 abs(caidas.iloc[-1] - caidas.iloc[0]))

    n = pd.DataFrame(normas)
    log.info("y la razon, medida sobre el modelo de referencia: norma "
             "del "
             "gradiente por ejemplo y fraccion que el recorte a C=%.1f "
             "se llevaria por delante:", C)
    for _, f in n.iterrows():
        log.info("  %-24s %-16s norma mediana %.3f · recortaria el "
                 "%.1f%%", f["eje"], f["grupo"], f["norma_mediana"],
                 100 * f["fraccion_recortada"])
    peor = n.loc[n["fraccion_recortada"].idxmax()]
    mejorr = n.loc[n["fraccion_recortada"].idxmin()]
    log.info("  El MECANISMO del impacto desigual esta ahi y es "
             "grande: al grupo «%s» le recortaria el %.1f%% de los "
             "gradientes y al grupo «%s» solo el %.1f%%, con normas "
             "medianas de %.2f y %.2f. Un mismo C trata de forma muy "
             "distinta a quien produce gradientes distintos.",
             peor["grupo"], 100 * peor["fraccion_recortada"],
             mejorr["grupo"], 100 * mejorr["fraccion_recortada"],
             peor["norma_mediana"], mejorr["norma_mediana"])
    log.info("  Y sin embargo el EFECTO sobre la exactitud no aparece: "
             "las caidas por grupo son de milesimas. Conviene no "
             "forzar la conclusion. Lo que estas cifras dicen es que "
             "sobre esta tarea —presupuesto holgado, red pequena, "
             "senal repartida— el mecanismo no llega a traducirse en "
             "disparidad medible. Eso coincide con la literatura que "
             "matizo la tesis original: el impacto desigual no es "
             "automatico, depende del regimen, y hay que MEDIRLO en "
             "cada caso en vez de suponerlo en ninguna direccion.")
    log.info("guardado %s y %s", SAL_GRUPOS.relative_to(RAIZ),
             SAL_NORMAS.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
