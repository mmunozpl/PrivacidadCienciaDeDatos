"""genera el informe de riesgo de reidentificacion, reproducible.

Es el artefacto del capitulo 3: un documento que se genera con un
comando, dice de que dataset habla, con que hipotesis, que mide y con
que resultado, y caduca. No es un PDF que alguien escribe a mano: es
la salida de un script versionado.

Uso: python3 src/cap03/informe_riesgo.py
Escribe data/processed/informe_riesgo.md y .json
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import SEMILLA, fijar_semillas
from comun.metricas import k_anonimato, l_diversidad, riesgos, t_cercania
from comun.registro import crear_registro

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA_MD = RAIZ / "data" / "processed" / "informe_riesgo.md"
SALIDA_JSON = RAIZ / "data" / "processed" / "informe_riesgo.json"

# hipotesis declaradas del informe: sin esto, ningun numero significa
# nada (cap. 3, seccion del informe)
HIPOTESIS = {
    "publicacion": "tabla completa, sin identificadores directos",
    "cuasi_identificador": ["edad5", "sexo", "provincia"],
    "sensible": "diagnostico",
    "poblacion_referencia": "residentes en España (INE, 1-1-2025)",
    "n_poblacion": 49_128_297,
    "adversario": ("dispone de padrón o fuente equivalente y sabe que "
                   "su objetivo está en la tabla (modelo del fiscal)"),
    "umbral_aceptado": {"riesgo_medio": 0.05, "k_minimo": 5},
}


def version_codigo() -> str:
    """Devuelve el commit del manuscrito, si lo hay, para trazabilidad."""
    try:
        return subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "sin control de versiones"
    except Exception:
        return "sin control de versiones"


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la generalizacion declarada en las hipotesis."""
    cp = df["codigo_postal"].astype(str)
    return df.assign(provincia=cp.str[:2],
                     edad5=(df["edad"] // 5) * 5)


def evaluar(df: pd.DataFrame) -> dict:
    """Calcula todas las metricas del informe.

    Args:
        df: tabla ya generalizada.

    Returns:
        Diccionario con las metricas y el veredicto.
    """
    cuasi = HIPOTESIS["cuasi_identificador"]
    sens = HIPOTESIS["sensible"]
    r = riesgos(df, cuasi, n_poblacion=HIPOTESIS["n_poblacion"])
    m = {
        "filas": len(df),
        "clases_equivalencia": int(df.groupby(cuasi,
                                              observed=True).ngroups),
        "k": k_anonimato(df, cuasi),
        "l_distinct": l_diversidad(df, cuasi, sens),
        "l_entropica": round(l_diversidad(df, cuasi, sens,
                                          entropica=True), 2),
        "t": round(t_cercania(df, cuasi, sens), 3),
        "riesgo_fiscal_max": round(r["fiscal_max"], 4),
        "riesgo_fiscal_medio": round(r["fiscal_medio"], 4),
        "riesgo_periodista_max": round(r["periodista_max"], 6),
        "filas_unicas_pct": round(100 * r["unicas"], 2),
    }
    umbral = HIPOTESIS["umbral_aceptado"]
    m["cumple_riesgo_medio"] = m["riesgo_fiscal_medio"] <= umbral["riesgo_medio"]
    m["cumple_k"] = m["k"] >= umbral["k_minimo"]
    m["veredicto"] = ("publicable con las hipótesis declaradas"
                      if m["cumple_riesgo_medio"] and m["cumple_k"]
                      else "NO publicable tal cual")
    return m


def redactar(m: dict, commit: str) -> str:
    """Escribe el informe en markdown."""
    h = HIPOTESIS
    poblacion_es = f"{h['n_poblacion']:,}".replace(',', '.')
    filas = "\n".join(
        f"| {k} | {v} |" for k, v in [
            ("filas", m["filas"]),
            ("clases de equivalencia", m["clases_equivalencia"]),
            ("k-anonimato", m["k"]),
            ("l-diversidad (distinct)", int(m["l_distinct"])),
            ("l-diversidad (entrópica)", m["l_entropica"]),
            ("t-cercanía", m["t"]),
            ("riesgo del fiscal (máx.)", m["riesgo_fiscal_max"]),
            ("riesgo del fiscal (medio)", m["riesgo_fiscal_medio"]),
            ("riesgo del periodista (máx.)", m["riesgo_periodista_max"]),
            ("filas únicas (%)", m["filas_unicas_pct"]),
        ])
    return f"""# Informe de riesgo de reidentificación

**Conjunto**: `{DATOS.name}` · **semilla**: {SEMILLA} ·
**versión del código**: {commit}

## Hipótesis declaradas

- Publicación evaluada: {h['publicacion']}.
- Cuasi-identificador: {', '.join(h['cuasi_identificador'])}.
- Atributo sensible: {h['sensible']}.
- Población de referencia: {h['poblacion_referencia']}
  ({poblacion_es} personas).
- Adversario: {h['adversario']}.
- Umbrales aceptados: riesgo medio <= {h['umbral_aceptado']['riesgo_medio']},
  k >= {h['umbral_aceptado']['k_minimo']}.

## Medidas

| medida | valor |
|---|---|
{filas}

## Veredicto

**{m['veredicto']}**
(cumple el umbral de riesgo medio: {'sí' if m['cumple_riesgo_medio'] else 'no'};
cumple el umbral de k: {'sí' if m['cumple_k'] else 'no'})

## Caducidad

Este informe vale para el conjunto, el cuasi-identificador y el modelo
de adversario declarados arriba. Se repite si cambia cualquiera de los
tres, y en todo caso al cambiar el conjunto de datos o al aparecer
fuentes auxiliares nuevas.
"""


def main() -> None:
    """Genera el informe y lo deja en data/processed/."""
    log = crear_registro("cap03.informe")
    fijar_semillas()
    df = preparar(pd.read_parquet(DATOS))
    m = evaluar(df)
    commit = version_codigo()

    SALIDA_MD.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_MD.write_text(redactar(m, commit), encoding="utf-8")
    SALIDA_JSON.write_text(
        json.dumps({"hipotesis": HIPOTESIS, "medidas": m,
                    "version_codigo": commit},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    for clave in ("k", "riesgo_fiscal_medio", "riesgo_periodista_max",
                  "filas_unicas_pct", "veredicto"):
        log.info("%s = %s", clave, m[clave])
    log.info("guardados %s y %s", SALIDA_MD.relative_to(RAIZ),
             SALIDA_JSON.relative_to(RAIZ))


if __name__ == "__main__":
    main()
