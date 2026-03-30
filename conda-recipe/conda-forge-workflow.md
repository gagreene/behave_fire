# conda-forge Publishing Workflow

## What was created

`meta.yaml` — a standard conda-forge recipe. It lives in your repo for
reference, but the copy that actually gets published lives in a fork of
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes).

---

## Step-by-step process

### 1 — Publish to PyPI first (conda-forge pulls from there)

```bash
python -m build
twine upload dist/*
```

### 2 — Get the sdist SHA-256 hash

After uploading, grab the hash from PyPI and paste it into `meta.yaml`:

```bash
# Option A — from the file you just built
python -c "import hashlib; print(hashlib.sha256(open('dist/behave_fire-1.0.0.tar.gz','rb').read()).hexdigest())"

# Option B — download and hash directly from PyPI
pip download --no-deps --no-binary :all: behave-fire==1.0.0 -d /tmp/cf
python -c "import hashlib; print(hashlib.sha256(open('/tmp/cf/behave_fire-1.0.0.tar.gz','rb').read()).hexdigest())"
```

Replace `SET_THIS_AFTER_UPLOADING_TO_PYPI` in `meta.yaml` with the result.

### 3 — Fork staged-recipes and open a PR

```bash
# Fork https://github.com/conda-forge/staged-recipes on GitHub, then:
git clone https://github.com/YOUR-USERNAME/staged-recipes
cd staged-recipes
mkdir -p recipes/behave-fire
cp /path/to/behave_py/conda-recipe/meta.yaml recipes/behave-fire/
git checkout -b add-behave-fire
git add recipes/behave-fire/meta.yaml
git commit -m "Add behave-fire recipe"
git push origin add-behave-fire
# Open a PR against conda-forge/staged-recipes
```

The conda-forge bots will build and test the recipe automatically.
A human maintainer reviews and merges it (~1–7 days).
After merge, a `behave-fire-feedstock` repo is created automatically and
`conda install -c conda-forge behave-fire` becomes available.

---

## Things to fill in before submitting

| Placeholder | Where | What to put |
|---|---|---|
| `SET_THIS_AFTER_UPLOADING_TO_PYPI` | `meta.yaml` `sha256:` | SHA-256 of the sdist `.tar.gz` from PyPI |
| `your-org` | `meta.yaml` URLs | Your GitHub username / org |
| `your-github-username` | `meta.yaml` `recipe-maintainers:` | Your GitHub username |

---

## Why `noarch: python`?

The package is pure Python (no C extensions), so the same wheel runs on
Windows, macOS, and Linux across all architectures. `noarch: python` tells
conda-forge to build it only once instead of once per platform — faster CI
and smaller download for users.

