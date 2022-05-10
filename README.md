# saliancy-metrics
A Unified Framework for Benchmarking Explanation Methods in Computer Vision.

---
[![Run unit tests](https://github.com/sandylaker/saliancy-metrics/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/sandylaker/saliancy-metrics/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/sandylaker/saliancy-metrics/branch/main/graph/badge.svg?token=ipPQ5VZivM)](https://codecov.io/gh/sandylaker/saliancy-metrics)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
---

## Contributing Guideline

### Setups
1. Fork the repo and then clone it.
2. Install pre-commit hooks: `pip install pre-commit` and then `pre-commit install`. The hooks will be
automatically triggered when running `git commit`. One can also manually trigger the hooks with
`pre-commit run --all-files`.
3. Install `torch`.
4. Install the requirements for development: `pip install -r requirements_dev.txt`.
5. Install the package in editable mode `pip install -e .` .
6. (Optional) Add custom directories in `.git/info/exclude` to make them ignored by git. Note that
some common directories like `.idea`, `.vscode` are already in the `.gitignore` file.

### Code linter, formatter, and unit tests
* [flake8](https://flake8.pycqa.org/en/latest/)
* [isort](https://pycqa.github.io/isort/)
* [black](https://black.readthedocs.io/en/stable/)
* [mypy](https://mypy.readthedocs.io/en/stable/)
* [pytest](https://docs.pytest.org/en/7.1.x/). Run unit tests with
`pytest --tx 4*popen//python=python --color yes --cov kgt --cov-report term-missing --cov-report xml -vvv tests`
