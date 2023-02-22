# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

# First we move the example files into the docs folder
# Then we clear the top-level docs Python folder of any pre-existing docs
# Next we run the sphinx-build command
# Then we clean up the articles folder,
# Move the docs into the top-level docs folder
# And finally remove the buildinfo so we do a clean build every time

%: Makefile
	mkdir -p source/articles
	cp ../../examples/datamart/Example_ADM_Analysis.ipynb ../../examples/graph_gallery/graph_gallery.ipynb ../../examples/helloworld/hello_cdhtools.ipynb ../../examples/valuefinder/* ../../examples/adm/AGBModelVisualisation.ipynb ../../examples/hds/Example_Data_Anonymization.ipynb source/articles
	mkdir ../../docs/Python/tmp
	rm -r ../../docs/Python/*
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm source/articles/*
	mv build/html/* ../../docs/Python
	rm build/html/.buildinfo