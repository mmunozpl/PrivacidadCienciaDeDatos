"""deriva la tarea supervisada con la que se mide el capitulo 6.

El diagnostico del conjunto base es ESTILIZADO: sus categorias y sus
pesos son plausibles, pero no dependen de la edad ni del sexo (asi lo
declara src/cap01/generar_dataset.py). Eso basta para los capitulos 1
a 5, que miden riesgo de reidentificacion y no capacidad predictiva,
y no basta para este: sin senal aprendible no hay compromiso entre
exactitud y privacidad que medir, porque no habria nada que perder.

Se deriva por tanto una tarea con dependencia REAL, sin tocar el
conjunto base —el mismo procedimiento que uso el capitulo 5 para el
registro de visitas—. La variable objetivo es el RIESGO
CARDIOVASCULAR en tres niveles, construido a partir de factores cuya
prevalencia por edad y sexo se toma de las cifras publicadas para
Espana (data/ine/prevalencias.csv).

Tres decisiones que conviene tener a la vista, porque condicionan
todo lo que el capitulo mide despues:

1. La senal esta en EDAD y SEXO, que es donde esta en la realidad.
2. El CODIGO POSTAL entra como caracteristica de alta cardinalidad y
   SIN senal: es el combustible de la memorizacion, igual que en el
   capitulo 2, y por tanto lo que el ataque de pertenencia explota.
3. La particion es mitad y mitad, como la del capitulo 2, para que el
   ataque de pertenencia se evalue igual. El tamano no coincide —aqui
   son 15 y mas anos, que es el universo de la encuesta—, asi que lo
   comparable entre los dos capitulos es el METODO y la metrica, no
   la cifra absoluta.

Escribe data/processed/tarea_riesgo.parquet.

Uso: python3 src/cap06/derivar_tarea.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
PREVALENCIAS = RAIZ / "data" / "ine" / "prevalencias.csv"
SALIDA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"

NIVELES = ["bajo", "moderado", "alto"]


def prevalencia_de(tabla: pd.DataFrame, factor: str, edad: np.ndarray,
                   sexo: np.ndarray) -> np.ndarray:
    """Interpola la prevalencia de un factor para cada persona.

    La tabla viene por tramos de edad y sexo; se asigna a cada persona
    la del tramo que le corresponde. No se interpola dentro del tramo:
    las cifras publicadas son medias de tramo y fingir continuidad
    seria inventarse precision que la fuente no da.

    Args:
        tabla: prevalencias por factor, tramo y sexo, en tanto por uno.
        factor: nombre del factor de riesgo.
        edad: edades de la poblacion.
        sexo: sexo de cada persona, «M» o «F».

    Returns:
        Vector con la prevalencia que le toca a cada persona.
    """
    sub = tabla[tabla["factor"] == factor]
    salida = np.zeros(len(edad))
    for _, f in sub.iterrows():
        toca = ((edad >= f["edad_min"]) & (edad <= f["edad_max"])
                & (sexo == f["sexo"]))
        salida[toca] = f["prevalencia"]
    return salida


def construir_riesgo(df: pd.DataFrame, tabla: pd.DataFrame,
                     rng: np.random.Generator) -> tuple:
    """Sortea los factores de riesgo y compone el nivel resultante.

    Cada factor se sortea de forma independiente con la prevalencia
    real de su tramo de edad y sexo. El riesgo se compone sumando los
    factores presentes con los pesos habituales de las escalas
    clinicas —la edad pesa mas que ninguna otra cosa— y se discretiza
    en tres niveles por terciles de la puntuacion.

    Args:
        df: poblacion base con edad, sexo y diagnostico.
        tabla: prevalencias reales por factor, tramo y sexo.
        rng: generador sembrado.

    Returns:
        Terna (nivel de riesgo por persona, matriz de factores,
        nombres de los factores).
    """
    edad = df["edad"].to_numpy()
    sexo = df["sexo"].to_numpy()
    factores = sorted(tabla["factor"].unique())

    presentes = np.column_stack([
        rng.random(len(df)) < prevalencia_de(tabla, f, edad, sexo)
        for f in factores]).astype(np.int8)

    # pesos: la edad domina, como en cualquier escala de riesgo
    # cardiovascular; los factores metabolicos suman entre si
    pesos = {"hipertension": 1.6, "diabetes": 1.8,
             "colesterol_elevado": 1.0, "obesidad": 0.9,
             "tabaquismo": 1.2}
    puntos = sum(pesos.get(f, 1.0) * presentes[:, i]
                 for i, f in enumerate(factores))
    puntos = puntos + 0.055 * (edad - edad.mean()) / edad.std() * 10
    puntos = puntos + 0.35 * (sexo == "M")
    # ruido irreducible: ni la mejor escala clinica acierta siempre
    puntos = puntos + rng.normal(0.0, 0.9, len(df))

    cortes = np.quantile(puntos, [1 / 3, 2 / 3])
    return np.digitize(puntos, cortes), presentes, factores


def caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica las caracteristicas que ve el modelo.

    Args:
        df: poblacion base.

    Returns:
        Tabla de columnas x_* listas para la red.
    """
    x = pd.DataFrame(index=df.index)
    edad = df["edad"].to_numpy(dtype=np.float32)
    x["x_edad"] = (edad - edad.mean()) / edad.std()
    x["x_sexo"] = (df["sexo"] == "M").astype(np.float32)
    # la provincia, una columna por cada una: tiene sentido geografico
    prov = df["codigo_postal"].astype(str).str[:2]
    for p in sorted(prov.unique()):
        x[f"x_prov_{p}"] = (prov == p).astype(np.float32)
    # el sufijo del CP, normalizado: alta cardinalidad y CERO senal,
    # que es justo lo que un modelo con capacidad acaba memorizando
    sufijo = df["codigo_postal"].astype(str).str[2:].astype(np.float32)
    x["x_cp_sufijo"] = (sufijo - sufijo.mean()) / sufijo.std()
    for p in sorted(df["profesion"].unique()):
        x[f"x_prof_{p}"] = (df["profesion"] == p).astype(np.float32)
    return x


def main() -> None:
    """Deriva la tarea y la guarda."""
    log = crear_registro("cap06.derivar_tarea")
    rng = fijar_semillas()

    if not PREVALENCIAS.exists():
        raise SystemExit(
            f"falta {PREVALENCIAS.relative_to(RAIZ)}: la tabla de "
            "prevalencias reales por edad y sexo")
    df = pd.read_parquet(DATOS)
    # la encuesta cubre poblacion de 15 y mas anos, y un modelo de
    # riesgo cardiovascular tampoco se aplicaria a menores: la tarea
    # se restringe a ese universo en vez de extrapolar
    df = df[df["edad"] >= 15].reset_index(drop=True)
    tabla = pd.read_csv(PREVALENCIAS, dtype={"sexo": str})
    log.info("poblacion base (15 y mas anos): %d personas · "
             "prevalencias de %d "
             "factores en %d tramos (%s)", len(df),
             tabla["factor"].nunique(),
             tabla[["edad_min", "edad_max"]].drop_duplicates().shape[0],
             tabla["fuente"].iloc[0])

    y, presentes, factores = construir_riesgo(df, tabla, rng)
    log.info("factores sorteados con la prevalencia real de su tramo:")
    for i, f in enumerate(factores):
        log.info("  %-20s presente en el %5.2f%% de la poblacion "
                 "sintetica", f, 100 * presentes[:, i].mean())

    x = caracteristicas(df)
    orden = rng.permutation(len(df))
    particion = np.array(["prueba"] * len(df), dtype=object)
    particion[orden[: len(df) // 2]] = "entrenamiento"

    salida = pd.concat([
        x,
        pd.DataFrame({"y": y, "particion": particion,
                      "edad": df["edad"].to_numpy(),
                      "sexo": df["sexo"].to_numpy(),
                      "provincia": df["codigo_postal"].astype(str)
                      .str[:2].to_numpy()}, index=df.index)], axis=1)

    log.info("tarea derivada: %d caracteristicas · %d clases · "
             "reparto %s", x.shape[1], len(NIVELES),
             " / ".join(f"{NIVELES[i]} {100 * (y == i).mean():.1f}%"
                        for i in range(len(NIVELES))))
    log.info("particion: %d en entrenamiento, %d en prueba",
             int((particion == "entrenamiento").sum()),
             int((particion == "prueba").sum()))

    # cuanta senal hay de verdad, para saber contra que se compara
    from sklearn.ensemble import RandomForestClassifier
    ent = particion == "entrenamiento"
    cols = [c for c in salida.columns if c.startswith("x_")]
    bosque = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=20,
                                    random_state=42, n_jobs=-1)
    bosque.fit(salida.loc[ent, cols], y[ent])
    techo = bosque.score(salida.loc[~ent, cols], y[~ent])
    mayoritaria = float(np.bincount(y[~ent]).max() / (~ent).sum())
    log.info("techo de referencia: un bosque regularizado acierta "
             "%.3f fuera del entrenamiento, frente al %.3f de la clase "
             "mayoritaria. Hay %.1f puntos de senal aprendible, que es "
             "lo que el capitulo 6 va a poner en la balanza contra la "
             "privacidad.", techo, mayoritaria,
             100 * (techo - mayoritaria))

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))

    muestra_final(salida[["edad", "sexo", "provincia", "y",
                          "particion"]], log)


if __name__ == "__main__":
    main()
