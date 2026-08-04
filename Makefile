# Makefile — orquesta figuras, conversión LaTeX -> Quarto y obra completa.
#
#   make figuras    figuras TikZ: PDF (obra completa) + SVG (web)
#   make qmd        variante web de los capítulos PUBLICADOS
#   make web        qmd + quarto render -> docs/
#   make borrador   como qmd, pero incluyendo los aún no publicados
#   make completa   PDF de la obra completa (latex/main.pdf)
#   make limpiar    borra artefactos y generados (qmd/, .work/)
#
# qmd/ y docs/ son GENERADOS: nunca se editan a mano.

# Capítulos PUBLICADOS en la web. Publicar uno es moverlo aquí desde
# CAPS_BORRADOR y darle su entrada en herramientas/quarto_armazon.yml
# y herramientas/index_base.md.
CAPS := cap01 cap02 cap03 cap04

# Terminados y compilados en el PDF de pago, pero NO publicados. Se
# previsualizan con `make borrador`; `make web` no los toca y además
# borra su .qmd si quedó de una previsualización anterior, para que no
# puedan llegar a docs/ por descuido.
CAPS_BORRADOR := cap05

# lo que se convierte a .qmd; `make borrador` lo amplía
CAPS_QMD ?= $(CAPS)

LATEX_DIR := latex
QMD_DIR   := qmd
WORK      := .work

FIG_SRC := $(wildcard $(LATEX_DIR)/figs/cap*/fig_*.tex)
FIG_PDF := $(FIG_SRC:.tex=.pdf)
FIG_SVG := $(patsubst $(LATEX_DIR)/figs/%.tex,$(QMD_DIR)/figs/%.svg,$(FIG_SRC))

# figuras de las SOLUCIONES: contenido de pago — solo PDF, jamás
# SVG ni web (por eso quedan fuera de FIG_SRC/FIG_SVG)
SOL_SRC := $(wildcard $(LATEX_DIR)/figs/soluciones/fig_*.tex)
SOL_PDF := $(SOL_SRC:.tex=.pdf)

.PHONY: figuras qmd borrador web completa limpiar

# ── figuras ──────────────────────────────────────────────────────────
figuras: $(FIG_PDF) $(FIG_SVG) $(SOL_PDF)

# pdf junto al fuente: main.tex lo resuelve con \includegraphics
$(LATEX_DIR)/figs/%.pdf: $(LATEX_DIR)/figs/%.tex
	cd $(dir $<) && pdflatex -interaction=nonstopmode $(notdir $<) > /dev/null
	cd $(dir $<) && rm -f $(notdir $(basename $<)).aux $(notdir $(basename $<)).log

# svg bajo qmd/figs/: la web lo referencia en ruta relativa al .qmd
$(QMD_DIR)/figs/%.svg: $(LATEX_DIR)/figs/%.tex
	mkdir -p $(dir $@)
	cd $(dir $<) && latex -interaction=nonstopmode -output-format=dvi \
	  $(notdir $<) > /dev/null
	dvisvgm --font-format=woff $(basename $<).dvi -o $@ 2> /dev/null
	rm -f $(basename $<).dvi $(basename $<).aux $(basename $<).log

# ── conversión LaTeX -> Quarto ───────────────────────────────────────
# variante_web.py retira la sección ensnist (solo obra de pago), deja
# la nota de edición y reescribe las figuras PDF -> SVG; aborta si el
# resultado conserva cualquier resto de ensnist.
qmd: figuras
	mkdir -p $(WORK) $(QMD_DIR)
	# se retira el .qmd de todo capítulo que no toque convertir: si
	# quedó de un `make borrador`, quarto lo renderizaría igual
	for c in $(CAPS) $(CAPS_BORRADOR); do \
	  echo " $(CAPS_QMD) " | grep -q " $$c " || \
	    rm -f $(QMD_DIR)/$$c.qmd; \
	done
	python3 herramientas/variante_web.py \
	  $(patsubst %,$(LATEX_DIR)/%.tex,$(CAPS_QMD))
	for c in $(CAPS_QMD); do \
	  quarto pandoc $(WORK)/$${c}_web.tex -f latex -t markdown \
	    -o $(QMD_DIR)/$$c.qmd || exit 1; \
	done
	python3 herramientas/ajustar_qmd.py $(patsubst %,$(QMD_DIR)/%.qmd,$(CAPS_QMD))
	cp herramientas/index_base.md $(QMD_DIR)/index.qmd
	cp herramientas/quarto_armazon.yml $(QMD_DIR)/_quarto.yml
	cp herramientas/estilo.scss $(QMD_DIR)/estilo.scss
	$(MAKE) $(QMD_DIR)/figs/portada_web.jpg

# portada para la web: pagina 1 del PDF compuesto (titulo + logo +
# autor), exportada a JPEG. Requiere la obra completa compilada.
$(QMD_DIR)/figs/portada_web.jpg: $(LATEX_DIR)/main.pdf
	mkdir -p $(QMD_DIR)/figs
	pdftoppm -f 1 -l 1 -r 150 -jpeg -jpegopt quality=92 \
	  $(LATEX_DIR)/main.pdf $(QMD_DIR)/figs/portada_web
	# pdftoppm rellena el numero de pagina segun el total (-1, -01,
	# -001...): se recoge con comodin para que no dependa del tamano
	mv $(QMD_DIR)/figs/portada_web-*.jpg $@

$(LATEX_DIR)/main.pdf:
	$(MAKE) completa

# docs/ se regenera DE CERO: es salida pura de quarto. El .nojekyll lo
# repone el propio objetivo: sin él Pages ignora todo lo que empieza
# por guion bajo, y como el gitignore global tapa los ocultos hay que
# forzarlo con -f al versionarlo. Se perdió dos veces por el rm -rf.
web: qmd
	rm -rf docs
	quarto render $(QMD_DIR)
	touch docs/.nojekyll
	git add -f docs/.nojekyll 2> /dev/null || true

# previsualización local de lo aún no publicado: llega hasta el .qmd y
# NO renderiza docs/, que es lo que se sube
borrador:
	$(MAKE) qmd CAPS_QMD="$(CAPS) $(CAPS_BORRADOR)"

# ── obra completa ────────────────────────────────────────────────────
completa: $(FIG_PDF) $(SOL_PDF)
	cd $(LATEX_DIR) && latexmk -pdf -interaction=nonstopmode main.tex

limpiar:
	cd $(LATEX_DIR) && latexmk -C main.tex 2> /dev/null || true
	rm -f $(LATEX_DIR)/main.bbl $(LATEX_DIR)/main.run.xml
	rm -rf $(WORK) $(QMD_DIR) .quarto
	find $(LATEX_DIR)/figs \( -name '*.aux' -o -name '*.log' \
	  -o -name '*.dvi' -o -name '*.fls' -o -name '*.fdb_latexmk' \) \
	  -delete
