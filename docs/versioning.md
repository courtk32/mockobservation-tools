# Versioning and releases

`mockobservation_tools` uses a declarative `pyproject.toml` and
[setuptools_scm](https://setuptools-scm.readthedocs.io/) for versioning.
You never edit a version string by hand anywhere in the source tree.

## Where the version comes from

setuptools_scm derives the version from the repository's **git tags** at
build time:

- On a commit that is exactly a tag `vX.Y.Z`, the version is `X.Y.Z`.
- On a commit after the latest tag, it is a development version such as
  `X.Y.(Z+1).devN+g<hash>`, where `N` is the number of commits since the
  tag.

setuptools records that version in the installed package's metadata, and
`mockobservation_tools.__version__` reads it back at runtime through
`importlib.metadata`, so the attribute always matches the version you
installed.

## How to cut a release

1. Commit the changes you want to release.
2. Tag that commit with a `v`-prefixed PEP 440 version, then push the
   tag:

   ```
   git tag v1.2.3
   git push origin v1.2.3
   ```

   Pre-releases use an `a`/`b`/`rc` suffix, for example `v1.2.3b1` (beta
   1) or `v1.2.3rc1` (release candidate 1).
3. That is the whole release. Anyone installing from that tag gets the
   clean version; anyone on a later commit gets a development version
   that sorts above it, so they can tell they are ahead of the release.

Do not move a tag once you have pushed it and something may have
installed from it. To correct a release, cut a new tag.

## Why there is no setup.py

The package used to carry a `setup.py` whose first lines imported the
package to read its version. That import ran at build time, before pip
installed the runtime dependencies (numpy and friends), so building the
package in an isolated environment failed with `ModuleNotFoundError`.
Declarative metadata in `pyproject.toml` removes that import entirely,
so the package builds cleanly under normal build isolation.
