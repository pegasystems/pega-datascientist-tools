# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

# Move the articles in the examples folder to source/articles
# Run docs
# Remove articles from source/articles

%: Makefile
	mkdir -p source/articles
	cp ../../examples/datamart/Example_ADM_Analysis.ipynb ../../examples/valuefinder/* ../../examples/adm/AGBModelVisualisation.ipynb ../../examples/adm/ADMBinningInsights.ipynb ../../examples/hds/Example_Data_Anonymization.ipynb ../../examples/articles/*.ipynb source/articles
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf source/articles/output
	rm source/articles/*
