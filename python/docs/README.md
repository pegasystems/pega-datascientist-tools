# Python documentation

The python documentation uses *Sphinx* to generate the docs, *nbsphinx* to convert the jupyter notebooks to markdown, and *Furo* as the Sphinx template.

These requirements are included in the docs-requirements.txt file. By installing it (`uv pip install -r docs-requirements.txt`) you should be able to generate the docs. This is provided you have *Pandoc* installed, which is not a pip package so can't be included in there, you need to install that seperately.

Building the docs is done through the Makefile: it
- First moves the example files into the docs folder,
- Then clears the top-level docs Python folder of any pre-existing docs
- Then runs the sphinx-build command
- Then cleans up the articles folder,
- Moves the docs into the top-level docs folder
- And finally removes the buildinfo so we do a clean build every time

## Creating locally

To generate a new version of the docs, simply navigate to the top-level folder and run the following command:

`(cd python/docs && make html)`

## To add new content:

1. Copy the notebooks/articles by editing the Makefile
2. Include the new article via the index.rst file in the docs/source folder




