"""mide la memorizacion con canarios, al modo de «The Secret Sharer».

Inserta en el entrenamiento registros artificiales («canarios») con un
secreto que no aparece en ningun otro sitio, repetidos un numero
distinto de veces, y mide despues cuanto delata el modelo ese secreto:

    exposicion = log2(|candidatos|) - log2(rango del secreto)

Cero exposicion = el secreto es indistinguible de los |candidatos|
posibles; exposicion maxima = el modelo lo pone el primero.

Escribe data/processed/memorizacion.csv para la figura del libro.

Uso: python3 src/cap02/memorizacion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, progreso

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "memorizacion.csv"

# espacio de secretos: 1000 codigos postales REALES de la tabla. Tienen
# que estar dentro de la distribucion: si el secreto cayera fuera del
# rango, todos los candidatos irian a parar a la misma hoja del arbol y
# la medida no distinguiria nada (empatarian todos).
N_CANDIDATOS = 1000
REPETICIONES = [1, 2, 4, 8, 16, 32]
DX_CANARIO = "dermatitis"

# mismo experimento con dos capacidades: el modelo que memoriza y el
# regularizado (hojas grandes = ningun registro vive solo en su hoja)
MODELOS = {
    "memorizador": dict(min_samples_leaf=1),
    "regularizado": dict(min_samples_leaf=50),
}


def preparar(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Codifica la tabla base para el modelo.

    Args:
        df: tabla del libro.

    Returns:
        Caracteristicas y etiquetas.
    """
    x = np.column_stack([
        df["edad"].to_numpy(),
        (df["sexo"] == "M").astype(int).to_numpy(),
        df["codigo_postal"].astype(int).to_numpy(),
        pd.factorize(df["profesion"])[0],
    ])
    return x, df["diagnostico"].to_numpy()


def exposicion(modelo, candidatos: np.ndarray, secreto: int,
               clase: str, edad: int, sexo: int,
               prof: int) -> tuple[float, int]:
    """Calcula la exposicion del secreto frente a sus candidatos.

    Args:
        modelo: clasificador entrenado con los canarios dentro.
        candidatos: espacio de secretos posibles.
        secreto: el codigo postal del canario.
        clase: diagnostico del canario.
        edad: edad del canario.
        sexo: sexo codificado del canario.
        prof: profesion codificada del canario.

    Returns:
        Exposicion en bits y rango del secreto (1 = el primero).
    """
    rejilla = np.column_stack([
        np.full(len(candidatos), edad),
        np.full(len(candidatos), sexo),
        candidatos,
        np.full(len(candidatos), prof),
    ])
    p = modelo.predict_proba(rejilla)
    col = list(modelo.classes_).index(clase)
    puntuacion = p[:, col]
    suyo = puntuacion[candidatos == secreto][0]
    # rango MEDIO: los empates no delatan nada, asi que cuentan a
    # medias (si todos empatan, el rango es ~n/2 y la exposicion ~0)
    mayores = int((puntuacion > suyo).sum())
    empates = int((puntuacion == suyo).sum())
    rango = mayores + (empates + 1) / 2
    return float(np.log2(len(candidatos)) - np.log2(rango)), int(rango)


def main() -> None:
    """Entrena con canarios repetidos y mide su exposicion."""
    log = crear_registro("cap02.memorizacion")
    fijar_semillas()

    base = pd.read_parquet(DATOS)
    x0, y0 = preparar(base)
    rng = np.random.default_rng(42)
    cps = np.unique(base["codigo_postal"].astype(int).to_numpy())
    candidatos = np.sort(rng.choice(cps, size=N_CANDIDATOS,
                                    replace=False))
    secreto = int(candidatos[len(candidatos) // 2])
    edad, sexo, prof = 42, 1, 0
    tope = np.log2(len(candidatos))
    log.info("espacio de secretos: %d codigos postales reales; el "
             "canario usa el %d (exposicion maxima %.1f bits)",
             len(candidatos), secreto, tope)

    filas = []
    total = len(MODELOS) * len(REPETICIONES)
    casos = [(n, p, k) for n, p in MODELOS.items() for k in REPETICIONES]
    for nombre, params, k in progreso(casos, total, log, cada=3,
                                      tarea="canarios"):
        canario_x = np.tile([edad, sexo, secreto, prof], (k, 1))
        canario_y = np.full(k, DX_CANARIO)
        modelo = RandomForestClassifier(
            n_estimators=60, random_state=42, n_jobs=-1, **params,
        )
        modelo.fit(np.vstack([x0, canario_x]),
                   np.concatenate([y0, canario_y]))
        exp, rango = exposicion(modelo, candidatos, secreto,
                                DX_CANARIO, edad, sexo, prof)
        log.info("%-12s repetido %2d -> rango %4d de %d, "
                 "exposicion %.1f bits (de %.1f)",
                 nombre, k, rango, len(candidatos), exp, tope)
        filas.append({"modelo": nombre, "repeticiones": k,
                      "rango": rango, "exposicion": exp})

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))


if __name__ == "__main__":
    main()
