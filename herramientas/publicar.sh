#!/usr/bin/env bash
# publicar.sh — puebla Publico/ por LISTA BLANCA, según la norma de la
# editorial (metodo/FORMATO_PUBLICO.md) y el formato de los repos
# publicados (ejemplo: mmunozpl/PythonCienciaDeDatos).
#
#   herramientas/publicar.sh          # construye y verifica Publico/
#
# Al público van SOLO: README.md (desde README_publico.md), LEEME.md
# (desde LEEME_publico.md), LICENSE, CITATION.cff, docs/, src/ y los
# datos redistribuibles. JAMÁS: latex/, qmd/, el PDF, Bibliografia.md
# ni ningún .md de implementación.
#
# Este script NO crea el repositorio público ni hace commit ni push:
# eso lo decide el autor.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d docs ] || { echo "✗ no hay docs/ — ejecuta antes: make web" >&2; exit 1; }
[ -f latex/main.pdf ] || echo "aviso: latex/main.pdf no existe (make completa)"

echo "▶ construyendo Publico/ por lista blanca"
rm -rf Publico
mkdir -p Publico

# ── lista blanca ─────────────────────────────────────────────────────
cp README_publico.md Publico/README.md
cp LEEME_publico.md  Publico/LEEME.md
cp LICENSE CITATION.cff Publico/

cp -r docs Publico/docs
touch Publico/docs/.nojekyll

mkdir -p Publico/src
cp src/LICENSE src/README.md Publico/src/
for d in src/comun src/cap*; do
  [ -d "$d" ] || continue
  mkdir -p "Publico/$d"
  find "$d" -maxdepth 1 -type f \( -name '*.py' -o -name 'README.md' \) \
    -exec cp {} "Publico/$d/" \;
done

mkdir -p Publico/data/processed Publico/data/ine
cp data/processed/poblacion_sintetica.parquet Publico/data/processed/
cp data/ine/edad_sexo.csv data/ine/provincias.csv Publico/data/ine/

# ── guardas (si abortan, se corrige el ORIGEN, nunca la guarda) ──────
echo "▶ guardas"

# 1) cero rastro del asistente (site_libs trae .bi-claude* de serie)
if grep -ril 'claude\|anthropic' Publico --exclude-dir=site_libs | grep -q .; then
  echo "✗ rastro del asistente en Publico/:" >&2
  grep -ril 'claude\|anthropic' Publico --exclude-dir=site_libs >&2
  exit 1
fi

# 2) nada oculto salvo .nojekyll
if find Publico -name '.*' ! -name '.nojekyll' ! -path '*/\.git*' | grep -q .; then
  echo "✗ ficheros ocultos inesperados:" >&2
  find Publico -name '.*' ! -name '.nojekyll' >&2
  exit 1
fi

# 3) ni fuentes LaTeX, ni qmd, ni PDF (el PDF es el producto de pago)
if find Publico \( -name '*.tex' -o -name '*.qmd' -o -name '*.pdf' \) | grep -q .; then
  echo "✗ fuentes o PDF en Publico/:" >&2
  find Publico \( -name '*.tex' -o -name '*.qmd' -o -name '*.pdf' \) >&2
  exit 1
fi

# 4) ningún .md fuera de los permitidos (los de implementación no viajan)
if find Publico -name '*.md' \
     ! -path 'Publico/README.md' ! -path 'Publico/LEEME.md' \
     ! -path 'Publico/src/README.md' ! -path 'Publico/src/cap*/README.md' \
     | grep -q .; then
  echo "✗ .md no permitido en Publico/:" >&2
  find Publico -name '*.md' ! -path 'Publico/README.md' \
    ! -path 'Publico/LEEME.md' ! -path 'Publico/src/README.md' \
    ! -path 'Publico/src/cap*/README.md' >&2
  exit 1
fi

echo "✓ Publico/ listo y verificado ($(find Publico -type f | wc -l) ficheros)"
cat <<'EOF'

Siguientes pasos (los decide el AUTOR, nunca por iniciativa propia):
  1. cd Publico && git init -b main && instalar el hook commit-msg
  2. git add -A && git add -f docs/.nojekyll && commit con identidad mmunozpl
  3. gh repo create (público) + push
  4. Pages: main /docs  ·  gh repo edit --homepage "https://mmunozpl.github.io/<REPO>/"
  5. bin/verificar-publico.sh privacidad-ciencia-datos
EOF
