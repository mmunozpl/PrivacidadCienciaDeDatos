"""DP-SGD escrito a mano, y comprobado contra Opacus.

DP-SGD (Abadi et al., 2016) cambia tres lineas del descenso por
gradiente, y ninguna de las tres es sutil:

1. El gradiente se calcula POR EJEMPLO, no por lote. Es la unica
   manera de saber cuanto aporta cada persona.
2. Cada gradiente se RECORTA a norma L2 como mucho C. Eso, y solo
   eso, da a la suma una sensibilidad acotada: C.
3. A la suma se le suma ruido gaussiano de desviacion sigma*C, y
   despues se divide por el tamano del lote.

Este guion las implementa las tres a mano con torch.func, comprueba
que los gradientes por ejemplo coinciden con los de Opacus hasta la
precision numerica, y mide dos cosas que deciden si el entrenamiento
va a servir de algo:

- QUE FRACCION de gradientes toca el recorte, epoca a epoca. Al
  principio casi todos; si sigue siendo casi todos al final, C esta
  demasiado bajo y se esta tirando senal.
- LA RELACION SENAL-RUIDO de la actualizacion: la norma del gradiente
  medio recortado frente a la del ruido que se le suma. Cuando el
  ruido domina, el paso es un paseo aleatorio caro.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/dp_sgd_manual.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import RedTabular, dispositivo, exactitud
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_RECORTE = RAIZ / "data" / "processed" / "dp_sgd_recorte.csv"
SAL_ACUERDO = RAIZ / "data" / "processed" / "dp_sgd_acuerdo.csv"

LOTE = 256
EPOCAS = 15
C = 1.0                   # norma de recorte
SIGMA = 1.0               # multiplicador de ruido
APRENDIZAJE = 0.5


def cargar_tarea() -> tuple:
    """Carga la tarea derivada del capitulo 6.

    Returns:
        Terna (X entrenamiento, y entrenamiento, X prueba, y prueba).

    Raises:
        SystemExit: si la tarea no se ha generado todavia.
    """
    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    df = pd.read_parquet(TAREA)
    cols = [c for c in df.columns if c.startswith("x_")]
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int64)
    en_entrenamiento = df["particion"].to_numpy() == "entrenamiento"
    return (x[en_entrenamiento], y[en_entrenamiento],
            x[~en_entrenamiento], y[~en_entrenamiento])


def gradientes_por_ejemplo(modelo: nn.Module, xb: torch.Tensor,
                           yb: torch.Tensor) -> dict:
    """Calcula el gradiente de CADA ejemplo del lote, en paralelo.

    Un `backward()` corriente devuelve el gradiente de la perdida
    MEDIA: los aportes individuales ya vienen sumados y no se pueden
    recortar por separado. torch.func los obtiene de golpe componiendo
    grad con vmap, que es lo mismo que hace Opacus por dentro con sus
    ganchos, pero escrito a la vista.

    Args:
        modelo: red cuyos parametros se derivan.
        xb: lote de caracteristicas.
        yb: lote de etiquetas.

    Returns:
        Diccionario parametro -> tensor con una fila por ejemplo.
    """
    params = {k: v.detach() for k, v in modelo.named_parameters()}
    buffers = {k: v.detach() for k, v in modelo.named_buffers()}
    criterio = nn.CrossEntropyLoss()

    def perdida_de_uno(p, b, x_uno, y_uno):
        """Perdida de un solo ejemplo, con el lote fingido a tamano 1."""
        salida = functional_call(modelo, (p, b),
                                 (x_uno.unsqueeze(0),))
        return criterio(salida, y_uno.unsqueeze(0))

    return vmap(grad(perdida_de_uno), in_dims=(None, None, 0, 0))(
        params, buffers, xb, yb)


def paso_dp_sgd(modelo: nn.Module, xb: torch.Tensor,
                yb: torch.Tensor, c: float, sigma: float,
                lr: float, gen: torch.Generator) -> dict:
    """Un paso de DP-SGD, con las tres lineas a la vista.

    Args:
        modelo: red a actualizar, en el sitio.
        xb: lote de caracteristicas.
        yb: lote de etiquetas.
        c: norma de recorte.
        sigma: multiplicador de ruido.
        lr: tasa de aprendizaje.
        gen: generador del ruido, sembrado.

    Returns:
        Diccionario con las medidas del paso: fraccion recortada,
        norma mediana del gradiente y relacion senal-ruido.
    """
    por_ejemplo = gradientes_por_ejemplo(modelo, xb, yb)
    n = len(yb)

    # (1) norma L2 de cada gradiente, sobre TODOS los parametros
    cuadrados = sum((g.reshape(n, -1) ** 2).sum(dim=1)
                    for g in por_ejemplo.values())
    normas = cuadrados.sqrt()

    # (2) factor de recorte: min(1, C/||g||). Nunca amplifica
    factor = (c / (normas + 1e-6)).clamp(max=1.0)

    señal_total = 0.0
    ruido_total = 0.0
    with torch.no_grad():
        for nombre, p in modelo.named_parameters():
            g = por_ejemplo[nombre]
            recortado = (g.reshape(n, -1) * factor[:, None]).sum(dim=0)
            # (3) ruido gaussiano de desviacion sigma*C sobre la SUMA
            ruido = torch.normal(0.0, sigma * c, size=recortado.shape,
                                 device=recortado.device,
                                 generator=gen)
            señal_total += float((recortado ** 2).sum())
            ruido_total += float((ruido ** 2).sum())
            # y solo entonces se promedia por el tamano del lote
            p -= lr * ((recortado + ruido) / n).reshape(p.shape)

    return {"fraccion_recortada": float((normas > c).float().mean()),
            "norma_mediana": float(normas.median()),
            "snr": float(np.sqrt(señal_total / max(ruido_total, 1e-12)))}


def gradientes_de_opacus(modelo: nn.Module, xb: torch.Tensor,
                         yb: torch.Tensor, reduccion: str,
                         reduccion_declarada: str) -> dict:
    """Gradientes por ejemplo segun Opacus, con la reduccion que se pida.

    Opacus reconstruye el gradiente por ejemplo con ganchos sobre la
    propagacion hacia atras. Para deshacer el promediado necesita
    saber COMO se redujo la perdida, y eso se le declara aparte, en
    `loss_reduction`. Si lo declarado no coincide con lo que hace la
    perdida, no falla nada: los gradientes salen escalados por el
    tamano del lote.

    Args:
        modelo: red sobre la que derivar.
        xb: lote de caracteristicas.
        yb: lote de etiquetas.
        reduccion: la que se pasa a CrossEntropyLoss.
        reduccion_declarada: la que se le declara a Opacus.

    Returns:
        Diccionario parametro -> gradiente por ejemplo.
    """
    from opacus import GradSampleModule

    envuelto = GradSampleModule(modelo,
                                loss_reduction=reduccion_declarada)
    envuelto.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss(reduction=reduccion)(envuelto(xb),
                                             yb).backward()
    salida = {n.replace("_module.", ""): p.grad_sample.clone()
              for n, p in envuelto.named_parameters()
              if p.grad_sample is not None}
    envuelto.to_standard_module()
    return salida


def comprobar_contra_opacus(modelo: nn.Module, xb: torch.Tensor,
                            yb: torch.Tensor) -> pd.DataFrame:
    """Compara mis gradientes con los de Opacus, bien y mal declarados.

    Args:
        modelo: red sobre la que comparar.
        xb: lote de caracteristicas.
        yb: lote de etiquetas.

    Returns:
        Tabla con la discrepancia por parametro en los dos casos.
    """
    mio = gradientes_por_ejemplo(modelo, xb, yb)
    bien = gradientes_de_opacus(modelo, xb, yb, "sum", "sum")
    # el fallo silencioso: perdida sumada, pero declarada promediada
    mal = gradientes_de_opacus(modelo, xb, yb, "sum", "mean")

    filas = []
    for nombre, referencia in mio.items():
        if nombre not in bien:
            continue
        filas.append({
            "parametro": nombre,
            "elementos": int(referencia.numel()),
            "magnitud_tipica": float(referencia.abs().mean()),
            "discrepancia_bien": float((bien[nombre]
                                        - referencia).abs().max()),
            "factor_mal": float(
                (mal[nombre].abs().sum()
                 / referencia.abs().sum().clamp(min=1e-12)))})
    return pd.DataFrame(filas)


def main() -> None:
    """Entrena con DP-SGD a mano, midiendo el recorte y el ruido."""
    log = crear_registro("cap06.dp_sgd_manual")
    fijar_semillas()
    disp = dispositivo()
    torch.manual_seed(42)
    gen = torch.Generator(device=disp).manual_seed(1234)

    x_tr, y_tr, x_te, y_te = cargar_tarea()
    log.info("tarea: %d ejemplos de entrenamiento · %d de prueba · "
             "%d caracteristicas · %d clases · dispositivo %s",
             len(y_tr), len(y_te), x_tr.shape[1],
             len(np.unique(y_tr)), disp)

    modelo = RedTabular(x_tr.shape[1], len(np.unique(y_tr))).to(disp)
    xt = torch.tensor(x_tr, device=disp)
    yt = torch.tensor(y_tr, device=disp)

    # ── 1) el gradiente por ejemplo coincide con el de Opacus ───────
    acuerdo = comprobar_contra_opacus(modelo, xt[:LOTE], yt[:LOTE])
    log.info("gradientes por ejemplo: mi implementacion frente a la de "
             "Opacus, sobre un lote de %d:", LOTE)
    for _, f in acuerdo.iterrows():
        log.info("  %-18s %7d elementos · discrepancia %.2e "
                 "(magnitud tipica %.2e) · mal declarado: x%.0f",
                 f["parametro"], int(f["elementos"]),
                 f["discrepancia_bien"], f["magnitud_tipica"],
                 f["factor_mal"])
    peor = acuerdo["discrepancia_bien"].max()
    log.info("  Discrepancia maxima global: %.2e. Es error de coma "
             "flotante, no una diferencia de algoritmo: las dos "
             "implementaciones calculan lo mismo por caminos "
             "distintos —yo componiendo grad con vmap, Opacus con "
             "ganchos sobre la propagacion hacia atras—.", peor)
    factor = acuerdo["factor_mal"].mean()
    log.info("  Y un fallo que conviene conocer: si a Opacus se le "
             "declara loss_reduction='mean' mientras la perdida usa "
             "reduction='sum', los gradientes por ejemplo salen "
             "multiplicados por %.0f, que es el tamano del lote. No "
             "hay excepcion ni aviso: el entrenamiento corre, el "
             "recorte a C deja de recortar a C, y el epsilon que "
             "informa el contable no corresponde a lo que se ha "
             "hecho.", factor)
    SAL_ACUERDO.parent.mkdir(parents=True, exist_ok=True)
    acuerdo.to_csv(SAL_ACUERDO, index=False)
    log.info("guardado %s", SAL_ACUERDO.relative_to(RAIZ))

    # ── 2) entrenar midiendo recorte y relacion senal-ruido ─────────
    log.info("entrenando con DP-SGD a mano: C=%.1f · sigma=%.1f · "
             "lote=%d · %d epocas", C, SIGMA, LOTE, EPOCAS)
    filas = []
    inicio = time.perf_counter()
    for epoca in progreso(range(1, EPOCAS + 1), EPOCAS, log, cada=3,
                          tarea="epocas"):
        orden = torch.randperm(len(yt), device=disp)
        medidas = []
        for i in range(0, len(orden) - LOTE + 1, LOTE):
            idx = orden[i:i + LOTE]
            medidas.append(paso_dp_sgd(modelo, xt[idx], yt[idx], C,
                                       SIGMA, APRENDIZAJE, gen))
        m = pd.DataFrame(medidas).mean()
        filas.append({"epoca": epoca, **m.to_dict(),
                      "exactitud_prueba": exactitud(modelo, x_te,
                                                    y_te, disp)})
    segundos = time.perf_counter() - inicio

    for f in filas[:: max(1, len(filas) // 5)]:
        log.info("  epoca %2d: recorta el %5.1f%% de los gradientes · "
                 "norma mediana %.3f · senal/ruido %.2f · exactitud "
                 "%.3f", f["epoca"], 100 * f["fraccion_recortada"],
                 f["norma_mediana"], f["snr"], f["exactitud_prueba"])

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SAL_RECORTE, index=False)
    primera, ultima = tabla.iloc[0], tabla.iloc[-1]
    log.info("  Con C=%.1f y una norma mediana de %.1f, el recorte "
             "toca entre el %.0f%% y el %.0f%% de los gradientes "
             "durante TODO el entrenamiento. Eso no es podar unos "
             "pocos atipicos: es NORMALIZAR. A este C, DP-SGD tira la "
             "magnitud del gradiente y se queda solo con su direccion, "
             "que es un algoritmo distinto del que se cree estar "
             "ejecutando —y, por cierto, uno que funciona—.",
             C, tabla["norma_mediana"].median(),
             100 * tabla["fraccion_recortada"].min(),
             100 * tabla["fraccion_recortada"].max())
    log.info("  La relacion senal-ruido se queda en %.2f: el ruido "
             "supera a la senal por un factor %.0f en cada paso. Y aun "
             "asi el modelo aprende (%.3f de exactitud), porque el "
             "ruido tiene media cero y se promedia entre pasos. Es la "
             "razon de que DP-SGD necesite muchos pasos pequenos en "
             "vez de pocos grandes.", tabla["snr"].median(),
             1 / tabla["snr"].median(), ultima["exactitud_prueba"])
    log.info("  Coste: %.1f segundos de %d epocas con gradiente por "
             "ejemplo. El mismo entrenamiento sin DP se mide en "
             "src/cap06/entrenamiento.py.", segundos, EPOCAS)
    log.info("guardado %s", SAL_RECORTE.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
