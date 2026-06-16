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

```text
https://<owner>.github.io/<repo>/dev/      ← built from master
https://<owner>.github.io/<repo>/5.1.0/    ← built from tag V5.1.0
https://<owner>.github.io/<repo>/5.0.0/    ← built from tag V5.0.0
```

The root URL redirects to the latest version that has made it to PyPI, not to
`dev/`. That keeps the default landing page on the most recent released docs
while still publishing unreleased `master` docs under `dev/`.

The deploy workflow (`.github/workflows/Docs deploy.yml`) publishes to a
`gh-pages` branch using `peaceiris/actions-gh-pages`. It runs on:

- pushes to `master` → builds into `dev/`
- pushes of tags matching `V*` or `v*` → builds into `<semver>/` (the leading
  `V`/`v` is stripped)
- manual dispatches → rebuild the selected branch or tag, or backfill an older
  tag by supplying `source_ref`

After each deploy, `python/docs/versioned_docs.py` regenerates:

- `versions.json`, consumed by the sidebar switcher
- the root `index.html` redirect, pointed at the latest PyPI-published release

`Python release.yml` triggers a second docs deploy after PyPI publish completes,
so the redirect flips to the newly-released version only once PyPI has the
package.

### Backfilling older tags

Older tags that already existed before this workflow lands will not publish
automatically. To backfill them:

1. Open **Actions → Build & Deploy docs → Run workflow**
2. Run it from the branch that contains this workflow
3. Set `source_ref` to the historical tag, for example `V4.7.1`
4. Leave `publish_slug` empty unless you need a non-standard output directory

The workflow checks out that tag for the build, publishes into the normalized
slug (`4.7.1/` for `V4.7.1`), and then refreshes `versions.json` plus the root
redirect.

### Local testing of the switcher

`make html` produces a self-contained build under `build/html/`. The switcher
gracefully no-ops when there is no sibling `versions.json`. To preview the
populated dropdown locally, drop a fake manifest one directory above the build
and serve from there:

```bash
(cd python/docs && make html)
mkdir -p /tmp-or-anywhere/site/dev
cp -r python/docs/build/html/. /tmp-or-anywhere/site/dev/
cat > /tmp-or-anywhere/site/versions.json <<'JSON'
[
  {"version": "dev", "url": "/dev/", "preferred": false},
  {"version": "5.1.0", "url": "/5.1.0/", "preferred": true}
]
JSON
python -m http.server -d /tmp-or-anywhere/site
```

Substitute any scratch directory for `/tmp-or-anywhere`.
