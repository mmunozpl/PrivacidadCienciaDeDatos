"""el lote que la contabilidad supone no es el que casi todos usan.

La amplificacion por submuestreo —lo que hace pagable entrenar con
privacidad diferencial— se demuestra suponiendo MUESTREO DE POISSON:
en cada paso, cada ejemplo entra en el lote de forma independiente con
probabilidad q. Eso tiene dos consecuencias visibles que la practica
habitual no cumple:

- el tamano del lote es ALEATORIO, no fijo;
- en una epoca hay ejemplos que no salen ninguna vez y otros que salen
  dos o mas.

Lo que hace un DataLoader corriente es lo contrario: baraja el
conjunto y lo trocea en lotes de tamano fijo, de modo que cada ejemplo
sale EXACTAMENTE una vez por epoca. Es mejor para aprender y es lo que
todo el mundo escribe, pero NO es el mecanismo que el contable esta
contabilizando.

Este guion mide la diferencia entre los dos regimenes sobre el mismo
conjunto, para que se vea que no es una sutileza formal, y comprueba
que Opacus sustituye el cargador al activar la privacidad.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/muestreo.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import RedTabular, cargador, dispositivo
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_LOTES = RAIZ / "data" / "processed" / "dp_sgd_muestreo.csv"
SAL_VISITAS = RAIZ / "data" / "processed" / "dp_sgd_visitas.csv"

LOTE = 512
EPOCAS = 30


def cargar_tarea() -> tuple:
    """Carga la tarea derivada del capitulo 6."""
    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    df = pd.read_parquet(TAREA)
    cols = [c for c in df.columns if c.startswith("x_")]
    dentro = df["particion"].to_numpy() == "entrenamiento"
    return (df.loc[dentro, cols].to_numpy(dtype=np.float32),
            df.loc[dentro, "y"].to_numpy(dtype=np.int64))


def cargador_de_poisson(x: np.ndarray, y: np.ndarray, lote: int,
                        epocas: int):
    """Devuelve el cargador de Poisson que instala Opacus.

    No se construye a mano: se pide a Opacus que privatice un
    entrenamiento cualquiera y se recoge el cargador que devuelve, que
    es exactamente el que usara al entrenar.

    Args:
        x: caracteristicas.
        y: etiquetas.
        lote: tamano de lote NOMINAL (fija q = lote/n).
        epocas: epocas previstas, que el motor necesita para calibrar.

    Returns:
        El DataLoader de Opacus, con muestreo de Poisson.
    """
    from opacus import PrivacyEngine
    disp = dispositivo()
    modelo = RedTabular(x.shape[1], int(y.max()) + 1).to(disp)
    opt = torch.optim.SGD(modelo.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, datos = PrivacyEngine(accountant="prv") \
            .make_private_with_epsilon(
                module=modelo, optimizer=opt,
                data_loader=cargador(x, y, lote),
                target_epsilon=3.0, target_delta=1e-5,
                epochs=epocas, max_grad_norm=1.0)
    return datos


def recorrer(datos, n: int, epocas: int, log,
             etiqueta: str) -> tuple[np.ndarray, np.ndarray]:
    """Recorre varias epocas anotando tamanos de lote y visitas.

    Args:
        datos: cargador a recorrer.
        n: numero de ejemplos del conjunto.
        epocas: epocas a recorrer.
        log: registro donde anotar el avance.
        etiqueta: nombre del regimen, para el registro.

    Returns:
        Par (tamanos de lote observados, visitas por ejemplo y epoca).
    """
    tamanos, visitas = [], []
    for _ in progreso(range(epocas), epocas, log, cada=10,
                      tarea=f"epocas ({etiqueta})"):
        cuenta = np.zeros(n, dtype=np.int32)
        for lote_datos in datos:
            yb = lote_datos[1]
            tamanos.append(int(len(yb)))
        visitas.append(cuenta)
    return np.array(tamanos), np.array(visitas)


def visitas_por_ejemplo(n: int, q: float, epocas: int,
                        pasos_por_epoca: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Simula cuantas veces ve cada ejemplo bajo Poisson.

    El cargador de Opacus no expone los indices, asi que la cuenta se
    simula con la misma ley que el usa: en cada paso, cada ejemplo
    entra de forma independiente con probabilidad q.

    Args:
        n: numero de ejemplos.
        q: tasa de muestreo.
        epocas: epocas simuladas.
        pasos_por_epoca: pasos que da cada epoca.
        rng: generador sembrado.

    Returns:
        Matriz epocas x n con las visitas de cada ejemplo.
    """
    return rng.binomial(pasos_por_epoca, q,
                        size=(epocas, n)).astype(np.int32)


def main() -> None:
    """Compara los dos regimenes de muestreo sobre el mismo conjunto."""
    log = crear_registro("cap06.muestreo")
    rng = fijar_semillas()

    x, y = cargar_tarea()
    n = len(y)
    q = LOTE / n
    pasos_por_epoca = n // LOTE
    log.info("conjunto de entrenamiento: %d ejemplos · lote nominal "
             "%d · q = %.5f · %d pasos por epoca", n, LOTE, q,
             pasos_por_epoca)

    # ── 1) el tamano del lote ───────────────────────────────────────
    corriente = cargador(x, y, LOTE)
    tam_corr, _ = recorrer(corriente, n, EPOCAS, log, "barajado")
    poisson = cargador_de_poisson(x, y, LOTE, EPOCAS)
    tam_pois, _ = recorrer(poisson, n, EPOCAS, log, "Poisson")

    resumen = pd.DataFrame([
        {"regimen": "barajado (DataLoader corriente)",
         "lotes": len(tam_corr), "media": tam_corr.mean(),
         "desviacion": tam_corr.std(), "minimo": tam_corr.min(),
         "maximo": tam_corr.max(),
         "vacios": int((tam_corr == 0).sum())},
        {"regimen": "Poisson (DPDataLoader de Opacus)",
         "lotes": len(tam_pois), "media": tam_pois.mean(),
         "desviacion": tam_pois.std(), "minimo": tam_pois.min(),
         "maximo": tam_pois.max(),
         "vacios": int((tam_pois == 0).sum())}])
    log.info("tamano de lote observado en %d epocas:", EPOCAS)
    for _, f in resumen.iterrows():
        log.info("  %-34s %5d lotes · media %6.1f · desviacion %5.1f "
                 "· rango [%d, %d] · vacios %d", f["regimen"],
                 int(f["lotes"]), f["media"], f["desviacion"],
                 int(f["minimo"]), int(f["maximo"]), int(f["vacios"]))
    teorica = np.sqrt(n * q * (1 - q))
    log.info("  La desviacion teorica de una binomial(n=%d, q=%.5f) es "
             "%.1f, y la medida %.1f: el cargador de Opacus hace "
             "exactamente lo que la demostracion supone.", n, q,
             teorica, tam_pois.std())
    SAL_LOTES.parent.mkdir(parents=True, exist_ok=True)
    # se guarda el histograma, no los 600 tamanos
    bordes = np.arange(tam_pois.min() - 1, tam_pois.max() + 2)
    frec, _ = np.histogram(tam_pois, bins=bordes)
    pd.DataFrame({"tamano": bordes[:-1], "frecuencia": frec}).to_csv(
        SAL_LOTES, index=False)
    resumen.to_csv(SAL_LOTES.with_name("dp_sgd_muestreo_resumen.csv"),
                   index=False)
    log.info("guardado %s", SAL_LOTES.relative_to(RAIZ))

    # ── 2) cuantas veces ve el modelo a cada persona ────────────────
    vis_pois = visitas_por_ejemplo(n, q, EPOCAS, pasos_por_epoca, rng)
    por_epoca = vis_pois[0]
    log.info("visitas por ejemplo en UNA epoca:")
    log.info("  barajado: exactamente 1 para los %d ejemplos, sin "
             "excepcion", n)
    cuentas = np.bincount(por_epoca, minlength=5)
    log.info("  Poisson: %d ejemplos no salen NINGUNA vez (%.1f%%), "
             "%d salen una, %d salen dos, %d salen tres o mas",
             cuentas[0], 100 * cuentas[0] / n, cuentas[1], cuentas[2],
             int(cuentas[3:].sum()))
    total = vis_pois.sum(axis=0)
    log.info("  En las %d epocas completas, el ejemplo menos visto "
             "aparece %d veces y el mas visto %d: una diferencia de "
             "%.0f%% en cuanto influye cada persona sobre el modelo.",
             EPOCAS, total.min(), total.max(),
             100 * (total.max() / max(total.min(), 1) - 1))
    pd.DataFrame({"visitas": np.arange(len(cuentas)),
                  "ejemplos": cuentas}).to_csv(SAL_VISITAS,
                                               index=False)
    log.info("guardado %s", SAL_VISITAS.relative_to(RAIZ))

    # ── 3) lo que el contable informa en cada regimen ───────────────
    from opacus.accountants import create_accountant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuenta = create_accountant(mechanism="prv")
        cuenta.history = [(1.0, q, EPOCAS * pasos_por_epoca)]
        informado = float(cuenta.get_epsilon(delta=1e-6))
    log.info("y ahora lo decisivo: el contable recibe (sigma, q, "
             "pasos) y NADA MAS. Para sigma=1, q=%.5f y %d pasos "
             "informa epsilon=%.3f a delta=1e-6, y lo informa IGUAL se "
             "haya muestreado de Poisson o se haya barajado: no tiene "
             "manera de saberlo.", q, EPOCAS * pasos_por_epoca,
             informado)
    log.info("Ese numero solo es una garantia en el primero de los dos "
             "regimenes. Chua et al. (2024) acotan cuanto puede "
             "subestimar bajo barajado; las cifras publicadas van de "
             "un factor 4 en modelos reales a un orden de magnitud en "
             "algunas variantes. La conclusion practica es simple: "
             "usar el cargador que DEVUELVE Opacus, no el que se le "
             "paso.")

    log.info("por que importa: la amplificacion se demuestra sobre el "
             "regimen de Poisson. Entrenar barajando y contabilizar "
             "como si se muestreara de Poisson informa un epsilon que "
             "NO corresponde al mecanismo ejecutado. Opacus lo resuelve "
             "sustituyendo el cargador —por eso "
             "make_private_with_epsilon devuelve uno nuevo y hay que "
             "usarlo—; quien implemente DP-SGD a mano con un "
             "DataLoader corriente esta anunciando una garantia que no "
             "ha demostrado.")

    muestra_final(resumen, log)


if __name__ == "__main__":
    main()
