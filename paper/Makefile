# Makefile for LaTeX paper compilation

PAPER = main
SECTIONS = sections/01_introduction.tex sections/02_related_work.tex sections/03_methods.tex sections/04_experiments.tex sections/05_results.tex sections/06_conclusion.tex

.PHONY: all clean pdf view

all: pdf

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(SECTIONS) references.bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

view: $(PAPER).pdf
	open $(PAPER).pdf  # macOS
	# xdg-open $(PAPER).pdf  # Linux
	# start $(PAPER).pdf  # Windows

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

# Quick compile (single pass, for drafts)
draft:
	pdflatex $(PAPER)

# Watch for changes and auto-compile (requires entr)
watch:
	ls $(PAPER).tex $(SECTIONS) | entr make draft
