# Informe de riesgo de reidentificación

**Conjunto**: `poblacion_sintetica.parquet` · **semilla**: 42 ·
**versión del código**: dd09346

## Hipótesis declaradas

- Publicación evaluada: tabla completa, sin identificadores directos.
- Cuasi-identificador: edad5, sexo, provincia.
- Atributo sensible: diagnostico.
- Población de referencia: residentes en España (INE, 1-1-2025)
  (49.128.297 personas).
- Adversario: dispone de padrón o fuente equivalente y sabe que su objetivo está en la tabla (modelo del fiscal).
- Umbrales aceptados: riesgo medio <= 0.05,
  k >= 5.

## Medidas

| medida | valor |
|---|---|
| filas | 20000 |
| clases de equivalencia | 1874 |
| k-anonimato | 1 |
| l-diversidad (distinct) | 1 |
| l-diversidad (entrópica) | 1.0 |
| t-cercanía | 0.949 |
| riesgo del fiscal (máx.) | 1.0 |
| riesgo del fiscal (medio) | 0.0937 |
| riesgo del periodista (máx.) | 0.000407 |
| filas únicas (%) | 1.12 |

## Veredicto

**NO publicable tal cual**
(cumple el umbral de riesgo medio: no;
cumple el umbral de k: no)

## Caducidad

Este informe vale para el conjunto, el cuasi-identificador y el modelo
de adversario declarados arriba. Se repite si cambia cualquiera de los
tres, y en todo caso al cambiar el conjunto de datos o al aparecer
fuentes auxiliares nuevas.
