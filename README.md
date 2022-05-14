# saliancy-metrics
A Unified Framework for Benchmarking Explanation Methods in Computer Vision.

---
[![Run unit tests](https://github.com/sandylaker/saliancy-metrics/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/sandylaker/saliancy-metrics/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/sandylaker/saliancy-metrics/branch/main/graph/badge.svg?token=ipPQ5VZivM)](https://codecov.io/gh/sandylaker/saliancy-metrics)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
---

:warning: **Note: The first version _(0.1.0)_ of this package is still a work in progress.**



## Supported Benchmarks

* [ ] [Sanity Check](https://arxiv.org/abs/1810.03292)
* [ ] [Sensitivity-N](https://arxiv.org/abs/1711.06104)
* [ ] [Insertion/Deletion](https://arxiv.org/abs/1509.06321)
* [ ] [ROAR](https://arxiv.org/abs/1806.10758)

## Contributing Guideline

### Set up environment
1. Fork the repo and then clone it.
2. Install pre-commit hooks: `pip install pre-commit` and then `pre-commit install`. The hooks will be
automatically triggered when running `git commit`. One can also manually trigger the hooks with
`pre-commit run --all-files`.
3. (Optional) Install `torch`.
4. Install the requirements for development: `pip install -r requirements_dev.txt`.
5. Install the package in editable mode `pip install -e .` .
6. (Optional) Add custom directories in `.git/info/exclude` to make them ignored by git. Note that
some common directories like `.idea`, `.vscode` are already in the `.gitignore` file.

### Code linter, formatter, and unit tests
* [flake8](https://flake8.pycqa.org/en/latest/)
* [isort](https://pycqa.github.io/isort/)
* [black](https://black.readthedocs.io/en/stable/)
* [mypy](https://mypy.readthedocs.io/en/stable/)
* [pytest](https://docs.pytest.org/en/7.1.x/). Run unit tests with `sh tests/run_tests.sh`

### Build docs locally

1. Go to `docs/` and install the requirements: `cd docs/ && pip install -r requirements_doc.txt`.
2. Now the current directory should be under `docs/`. Build the html webpage: `make html`.
3. Go to `docs/build/` and then host the webpage locally: `cd build/ && python -m http.server <port> `, where `port` is a number (e.g., 1234).
4. Open the webpage `localhost:<port>` in a browser.
