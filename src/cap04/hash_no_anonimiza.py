"""demuestra que hashear un identificador NO lo anonimiza.

El fallo no esta en la funcion hash sino en el ESPACIO DE ENTRADA: si
el identificador procede de un conjunto pequeno y enumerable —un DNI,
un movil espanol, una matricula—, el adversario recorre el espacio,
hashea cada candidato y construye la tabla inversa. Es el ataque de
diccionario que documentaron la AEPD y el EDPS con moviles espanoles y
el que reventó los registros de taxis de Nueva York.

Uso: python3 src/cap04/hash_no_anonimiza.py
"""

import hashlib
import hmac
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro

# espacios de identificadores reales y su tamano
ESPACIOS = {
    "móvil español (6xx/7xx)": 2 * 10 ** 8,
    "DNI (8 dígitos + letra)": 10 ** 8,
    "matrícula española": 8000 * 20 ** 3,
    "número de historia (6 dígitos)": 10 ** 6,
}
MUESTRA = 200_000          # candidatos que se hashean para cronometrar


def sha256(dato: str) -> str:
    """Hash SHA-256 en hexadecimal."""
    return hashlib.sha256(dato.encode()).hexdigest()


def hash_con_clave(dato: str, clave: bytes) -> str:
    """HMAC-SHA256: hash con clave secreta (seudonimizacion seria)."""
    return hmac.new(clave, dato.encode(), hashlib.sha256).hexdigest()


def main() -> None:
    """Mide el coste real de invertir un hash sin clave."""
    log = crear_registro("cap04.hash")
    rng = fijar_semillas()

    # ── 1) velocidad real de esta maquina ───────────────────────────
    candidatos = [f"6{n:08d}" for n in range(MUESTRA)]
    inicio = time.perf_counter()
    tabla = {sha256(c): c for c in candidatos}
    segundos = time.perf_counter() - inicio
    por_segundo = MUESTRA / segundos
    log.info("velocidad medida en esta máquina: %.0f hashes/segundo "
             "(Python puro, un solo núcleo)", por_segundo)

    # ── 2) el ataque completo, sobre un movil ───────────────────────
    # la victima esta dentro del tramo que el adversario ha barrido:
    # con el espacio entero (122 s, mas abajo) estaria siempre
    victima = "600042042"
    huella = sha256(victima)
    log.info("huella publicada del móvil de la víctima: %s...",
             huella[:24])
    recuperado = tabla.get(huella)
    log.info("¿está en la tabla inversa de %s candidatos? %s",
             f"{MUESTRA:,}".replace(",", "."),
             "SÍ, el móvil era " + recuperado if recuperado
             else "no (fuera del tramo barrido)")

    # ── 3) cuanto costaria barrer cada espacio ──────────────────────
    log.info("tiempo para recorrer el espacio ENTERO, a esta "
             "velocidad y con hardware dedicado:")
    for nombre, tam in ESPACIOS.items():
        seg_python = tam / por_segundo
        # una GPU de gama media hace del orden de 10^10 SHA-256/s
        seg_gpu = tam / 1e10
        log.info("  %-32s %12s candidatos · %8.1f s en Python · "
                 "%.6f s en GPU", nombre,
                 f"{tam:,}".replace(",", "."), seg_python, seg_gpu)

    # ── 4) la sal no basta si la sal se publica ─────────────────────
    sal = "sal-publica-2026"
    tabla_salada = {sha256(sal + c): c for c in candidatos}
    log.info("con sal CONOCIDA, el mismo ataque recupera %s: %s",
             victima,
             tabla_salada.get(sha256(sal + victima), "no (fuera de "
                              "la muestra, pero el espacio sigue "
                              "siendo enumerable)"))

    # ── 5) lo que sí funciona: clave secreta y separada ─────────────
    clave = rng.bytes(32)
    seudonimo = hash_con_clave(victima, clave)
    log.info("HMAC con clave secreta: %s... — el adversario no puede "
             "construir la tabla porque le falta la clave; el dato "
             "sigue siendo PERSONAL (art. 4.5), no anónimo",
             seudonimo[:24])


if __name__ == "__main__":
    main()
