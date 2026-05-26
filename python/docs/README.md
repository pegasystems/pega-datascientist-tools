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

## Versioned publishing

The docs are published to GitHub Pages under per-version subdirectories:

```
https://<owner>.github.io/<repo>/latest/   ← built from master
https://<owner>.github.io/<repo>/5.1.0/    ← built from tag v5.1.0
https://<owner>.github.io/<repo>/5.0.0/    ← built from tag v5.0.0
```

The root URL redirects to `latest/`. A `versions.json` manifest at the
gh-pages root drives the version-switcher dropdown that the Furo sidebar
template renders (`source/_templates/sidebar/version-switcher.html` plus
the JS in `source/_static/version-switcher.js`).

The deploy workflow (`.github/workflows/Docs deploy.yml`) publishes to a
`gh-pages` branch using `peaceiris/actions-gh-pages`. It runs on:

- pushes to `master` → builds into `latest/`
- pushes of tags matching `v*` → builds into `<semver>/` (the leading `v`
  is stripped)

A second job rewrites `versions.json` after each deploy so the dropdown
sees newly-published versions.

### Local testing of the switcher

`make html` produces a self-contained build under `build/html/`. The
switcher gracefully no-ops when there is no sibling `versions.json`. To
preview the populated dropdown locally, drop a fake manifest one
directory above the build and serve from there:

```bash
(cd python/docs && make html)
mkdir -p /tmp-or-anywhere/site/latest
cp -r python/docs/build/html/. /tmp-or-anywhere/site/latest/
cat > /tmp-or-anywhere/site/versions.json <<'JSON'
[
  {"version": "latest", "url": "/latest/", "preferred": false},
  {"version": "5.1.0",  "url": "/5.1.0/", "preferred": true}
]
JSON
python -m http.server -d /tmp-or-anywhere/site
```

(Substitute any scratch directory for `/tmp-or-anywhere`.)
