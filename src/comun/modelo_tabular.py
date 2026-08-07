"""el modelo tabular que comparten los capitulos de aprendizaje.

Una sola definicion de red, de bucle de entrenamiento y de evaluacion,
para que todo lo que mida el libro sobre modelos sea comparable entre
capitulos: el coste en exactitud del capitulo 6, el reparto federado
del 9 y lo que se compare con sinteticos en el 8.

La red es deliberadamente pequena y sin normalizacion por lotes. Lo
primero, porque el problema es tabular y una red grande solo anadiria
varianza; lo segundo, porque BatchNorm mezcla ejemplos dentro del lote
y eso rompe el gradiente POR EJEMPLO del que depende DP-SGD (Opacus lo
rechaza por ese motivo, y hace bien). Donde hace falta normalizar se
usa GroupNorm o LayerNorm, que operan ejemplo a ejemplo.

Ninguna funcion imprime nada: el registro lo hace quien llama.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RedTabular(nn.Module):
    """Perceptron multicapa para datos tabulares.

    Args:
        entradas: numero de caracteristicas.
        clases: numero de clases de salida.
        oculta: anchura de las dos capas ocultas.
        capas_normalizadas: si es cierto, intercala LayerNorm, que es
            compatible con el gradiente por ejemplo (BatchNorm no).
    """

    def __init__(self, entradas: int, clases: int, oculta: int = 128,
                 capas_normalizadas: bool = True):
        super().__init__()
        def bloque(dentro: int, fuera: int) -> list[nn.Module]:
            capas: list[nn.Module] = [nn.Linear(dentro, fuera)]
            if capas_normalizadas:
                capas.append(nn.LayerNorm(fuera))
            capas.append(nn.ReLU())
            return capas
        self.red = nn.Sequential(*bloque(entradas, oculta),
                                 *bloque(oculta, oculta),
                                 nn.Linear(oculta, clases))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga la entrada y devuelve los logits."""
        return self.red(x)


def dispositivo() -> torch.device:
    """Devuelve la GPU si la hay, y si no la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cargador(x: np.ndarray, y: np.ndarray, lote: int,
             barajar: bool = True) -> DataLoader:
    """Envuelve dos arrays en un DataLoader corriente.

    OJO: este cargador BARAJA y trocea en lotes de tamano fijo, que no
    es lo que supone la contabilidad de DP-SGD. Para entrenar con
    privacidad hay que sustituirlo por el DPDataLoader de Opacus, que
    muestrea de Poisson; la diferencia se mide en
    src/cap06/muestreo.py y no es cosmetica.

    Args:
        x: matriz de caracteristicas.
        y: vector de etiquetas enteras.
        lote: tamano del lote.
        barajar: si se baraja en cada epoca.

    Returns:
        Un DataLoader sobre tensores en CPU.
    """
    conjunto = TensorDataset(torch.tensor(x, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.long))
    return DataLoader(conjunto, batch_size=lote, shuffle=barajar,
                      num_workers=0)


def entrenar_epoca(modelo: nn.Module, cargador_datos: DataLoader,
                   optimizador: torch.optim.Optimizer,
                   disp: torch.device) -> float:
    """Recorre una epoca y devuelve la perdida media.

    Args:
        modelo: red a entrenar, ya en el dispositivo.
        cargador_datos: cargador de entrenamiento.
        optimizador: optimizador (envuelto por Opacus si hay DP).
        disp: dispositivo de computo.

    Returns:
        Perdida media de la epoca.
    """
    modelo.train()
    criterio = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for xb, yb in cargador_datos:
        xb, yb = xb.to(disp), yb.to(disp)
        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(xb), yb)
        perdida.backward()
        optimizador.step()
        # con muestreo de Poisson el lote puede venir vacio
        if len(yb):
            total += float(perdida) * len(yb)
            n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def exactitud(modelo: nn.Module, x: np.ndarray, y: np.ndarray,
              disp: torch.device) -> float:
    """Fraccion de aciertos sobre un conjunto.

    Args:
        modelo: red entrenada.
        x: caracteristicas.
        y: etiquetas verdaderas.
        disp: dispositivo de computo.

    Returns:
        Exactitud en [0, 1].
    """
    modelo.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=disp)
    pred = modelo(xt).argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


@torch.no_grad()
def perdidas_por_ejemplo(modelo: nn.Module, x: np.ndarray,
                         y: np.ndarray,
                         disp: torch.device) -> np.ndarray:
    """Perdida logaritmica de cada ejemplo: el estadistico del ataque.

    Es el mismo criterio que usa el ataque del capitulo 2, para que
    las cifras de los dos capitulos sean comparables: si el modelo
    esta mas seguro de un registro, probablemente lo vio al entrenar.

    Args:
        modelo: red entrenada.
        x: caracteristicas.
        y: etiquetas verdaderas.
        disp: dispositivo de computo.

    Returns:
        Vector de perdidas, una por ejemplo.
    """
    modelo.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=disp)
    yt = torch.tensor(y, dtype=torch.long, device=disp)
    criterio = nn.CrossEntropyLoss(reduction="none")
    return criterio(modelo(xt), yt).cpu().numpy()


def auc_pertenencia(modelo: nn.Module, x_dentro: np.ndarray,
                    y_dentro: np.ndarray, x_fuera: np.ndarray,
                    y_fuera: np.ndarray,
                    disp: torch.device) -> tuple[float, np.ndarray,
                                                 np.ndarray]:
    """Ataque de pertenencia por perdida, como en el capitulo 2.

    Args:
        modelo: red entrenada.
        x_dentro, y_dentro: ejemplos que SI se usaron al entrenar.
        x_fuera, y_fuera: ejemplos que no.
        disp: dispositivo de computo.

    Returns:
        Terna (AUC, tasas de falsa alarma, tasas de acierto).
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    puntuacion = np.concatenate([
        -perdidas_por_ejemplo(modelo, x_dentro, y_dentro, disp),
        -perdidas_por_ejemplo(modelo, x_fuera, y_fuera, disp)])
    etiqueta = np.concatenate([np.ones(len(y_dentro)),
                               np.zeros(len(y_fuera))])
    fpr, tpr, _ = roc_curve(etiqueta, puntuacion)
    return float(roc_auc_score(etiqueta, puntuacion)), fpr, tpr


def entrenar_red(x: np.ndarray, y: np.ndarray, clases: int,
                 eps: float | None, disp: torch.device, *,
                 lote: int = 512, epocas: int = 30,
                 aprendizaje: float = 1.0, c: float = 1.0,
                 delta: float = 1e-5, sigma: float | None = None,
                 oculta: int = 128, normalizada: bool = True,
                 adam: bool = False,
                 semilla: int = 42) -> tuple[nn.Module, float, float]:
    """Entrena la red tabular, en uno de los tres regimenes del cap. 6.

    Los tres hacen falta para separar dos efectos que se confunden. El
    recorte de gradientes NO es una medida de privacidad: es una
    decision de optimizacion que, por si sola, cambia como aprende la
    red (a C pequeno equivale a normalizar el gradiente). Lo que aporta
    la privacidad es el RUIDO. Comparar «sin DP» con «DP» sin separar
    las dos cosas atribuye al presupuesto lo que hizo el recorte.

    - eps dado: DP-SGD completo. Opacus calibra sigma, envuelve el
      optimizador y SUSTITUYE el cargador por uno de Poisson —lo
      tercero es lo que se olvida, y sin ello la contabilidad no
      corresponde a lo ejecutado (ver src/cap06/muestreo.py)—.
    - sigma dado (con eps a None): recorte SIN ruido si sigma=0. No da
      ninguna garantia; sirve de referencia intermedia.
    - los dos a None: descenso por gradiente corriente.

    Args:
        x: caracteristicas de entrenamiento.
        y: etiquetas de entrenamiento.
        clases: numero de clases.
        eps: presupuesto objetivo, o None.
        disp: dispositivo de computo.
        lote: tamano de lote nominal.
        epocas: numero de epocas.
        aprendizaje: tasa de aprendizaje.
        c: norma de recorte de gradientes.
        delta: delta objetivo.
        sigma: multiplicador de ruido explicito, si no se da eps.
        oculta: anchura de las capas ocultas.
        normalizada: si la red lleva LayerNorm.
        adam: si se usa Adam en vez de SGD.
        semilla: semilla de la inicializacion, fija por comparabilidad.

    Returns:
        Terna (modelo desnudo entrenado, epsilon gastado, segundos).
    """
    import time
    import warnings

    torch.manual_seed(semilla)
    modelo = RedTabular(x.shape[1], clases, oculta=oculta,
                        capas_normalizadas=normalizada).to(disp)
    datos = cargador(x, y, lote)
    optimizador = (torch.optim.Adam(modelo.parameters(),
                                    lr=aprendizaje) if adam
                   else torch.optim.SGD(modelo.parameters(),
                                        lr=aprendizaje, momentum=0.0))
    motor = None

    if eps is not None or sigma is not None:
        from opacus import PrivacyEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            motor = PrivacyEngine(accountant="prv")
            if eps is not None:
                modelo, optimizador, datos = \
                    motor.make_private_with_epsilon(
                        module=modelo, optimizer=optimizador,
                        data_loader=datos, target_epsilon=eps,
                        target_delta=delta, epochs=epocas,
                        max_grad_norm=c)
            else:
                modelo, optimizador, datos = motor.make_private(
                    module=modelo, optimizer=optimizador,
                    data_loader=datos, noise_multiplier=sigma,
                    max_grad_norm=c)

    criterio = nn.CrossEntropyLoss()
    inicio = time.perf_counter()
    for _ in range(epocas):
        modelo.train()
        for xb, yb in datos:
            if not len(yb):        # Poisson puede dar lotes vacios
                continue
            xb, yb = xb.to(disp), yb.to(disp)
            optimizador.zero_grad(set_to_none=True)
            criterio(modelo(xb), yb).backward()
            optimizador.step()
    segundos = time.perf_counter() - inicio

    gastado = float("inf")
    if motor is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gastado = float(motor.get_epsilon(delta))
            except Exception:
                gastado = float("inf")   # sigma=0 no da garantia
        # se devuelve la red DESNUDA: to_standard_module() no solo
        # desenvuelve, tambien retira los ganchos que Opacus instalo
        # sobre la marcha atras. Sin eso torch.func no puede derivar
        # el modelo despues —los ganchos son autograd.Function sin
        # setup_context— y cualquier analisis posterior falla
        if hasattr(modelo, "to_standard_module"):
            modelo = modelo.to_standard_module()
        else:
            modelo = getattr(modelo, "_module", modelo)
    return modelo, gastado, segundos


def tpr_a_fpr(fpr: np.ndarray, tpr: np.ndarray,
              objetivo: float) -> float:
    """Interpola la TPR a una FPR objetivo.

    Carlini et al. (2022) insisten en que el AUC promedia demasiado:
    lo que hace peligroso a un ataque es acertar mucho cuando acusa
    poco, o sea la TPR a tasas de falsa alarma muy bajas.

    Args:
        fpr: tasas de falsa alarma de la curva ROC.
        tpr: tasas de acierto correspondientes.
        objetivo: FPR de interes (p. ej. 0,001).

    Returns:
        TPR alcanzada a esa FPR.
    """
    return float(np.interp(objetivo, fpr, tpr))
