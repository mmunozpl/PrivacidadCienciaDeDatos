# Makefile — orquesta figuras, conversión LaTeX -> Quarto y obra completa.
#
#   make figuras    figuras TikZ: PDF (obra completa) + SVG (web)
#   make qmd        variante web de cada capítulo -> qmd/capXX.qmd
#   make web        qmd + quarto render -> docs/
#   make completa   PDF de la obra completa (latex/main.pdf)
#   make limpiar    borra artefactos y generados (qmd/, .work/)
#
# qmd/ y docs/ son GENERADOS: nunca se editan a mano.

CAPS := cap01

LATEX_DIR := latex
QMD_DIR   := qmd
WORK      := .work

FIG_SRC := $(wildcard $(LATEX_DIR)/figs/cap*/fig_*.tex)
FIG_PDF := $(FIG_SRC:.tex=.pdf)
FIG_SVG := $(patsubst $(LATEX_DIR)/figs/%.tex,$(QMD_DIR)/figs/%.svg,$(FIG_SRC))

.PHONY: figuras qmd web completa limpiar

# ── figuras ──────────────────────────────────────────────────────────
figuras: $(FIG_PDF) $(FIG_SVG)

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
	python3 herramientas/variante_web.py \
	  $(patsubst %,$(LATEX_DIR)/%.tex,$(CAPS))
	for c in $(CAPS); do \
	  quarto pandoc $(WORK)/$${c}_web.tex -f latex -t markdown \
	    -o $(QMD_DIR)/$$c.qmd || exit 1; \
	done
	cp herramientas/index_base.qmd $(QMD_DIR)/index.qmd
	cp herramientas/quarto_armazon.yml $(QMD_DIR)/_quarto.yml
	$(MAKE) $(QMD_DIR)/figs/portada_web.jpg

# portada para la web: pagina 1 del PDF compuesto (titulo + logo +
# autor), exportada a JPEG. Requiere la obra completa compilada.
$(QMD_DIR)/figs/portada_web.jpg: $(LATEX_DIR)/main.pdf
	mkdir -p $(QMD_DIR)/figs
	pdftoppm -f 1 -l 1 -r 150 -jpeg -jpegopt quality=92 \
	  $(LATEX_DIR)/main.pdf $(QMD_DIR)/figs/portada_web
	mv $(QMD_DIR)/figs/portada_web-01.jpg $@

$(LATEX_DIR)/main.pdf:
	$(MAKE) completa

# docs/ se regenera DE CERO: es salida pura de quarto. Si algún día
# lleva CNAME o .nojekyll, van como resources del armazón, no a mano.
web: qmd
	rm -rf docs
	quarto render $(QMD_DIR)

# ── obra completa ────────────────────────────────────────────────────
completa: $(FIG_PDF)
	cd $(LATEX_DIR) && latexmk -pdf -interaction=nonstopmode main.tex

limpiar:
	cd $(LATEX_DIR) && latexmk -C main.tex 2> /dev/null || true
	rm -f $(LATEX_DIR)/main.bbl $(LATEX_DIR)/main.run.xml
	rm -rf $(WORK) $(QMD_DIR) .quarto
	find $(LATEX_DIR)/figs \( -name '*.aux' -o -name '*.log' \
	  -o -name '*.dvi' -o -name '*.fls' -o -name '*.fdb_latexmk' \) \
	  -delete
