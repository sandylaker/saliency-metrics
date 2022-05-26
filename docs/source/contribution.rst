:github_url: https://github.com/sandylaker/saliency-metrics

Contribution Guideline
======================


Set up the environment
----------------------

#. Fork the repo and then clone it.
#. ⚠️ Install python >= 3.8.
#. Install pre-commit hooks: ``pip install pre-commit`` and then ``pre-commit install``. The hooks will be automatically triggered when running `git commit`. One can also manually trigger the hooks with ``pre-commit run --all-files``.
#. (Optional) Install ``torch``.
#. Install the requirements for development: ``pip install -r requirements_dev.txt``.
#. Install the package in editable mode ``pip install -e .`` .
#. (Optional) Add custom directories in ``.git/info/exclude`` to make them ignored by git. Note that some common directories like ``.idea``, ``.vscode`` are already in the ``.gitignore`` file.


Code linting, type hints, and unit tests
----------------------------------------
We use `flake8`_, `isort`_, and `black`_ for the code linting, and `mypy`_ for checking type hints. We write the unit tests with `pytest`_.

.. _flake8: https://flake8.pycqa.org/en/latest/
.. _isort: https://pycqa.github.io/isort/
.. _black: https://black.readthedocs.io/en/stable/
.. _mypy: https://mypy.readthedocs.io/en/stable/
.. _pytest: https://docs.pytest.org/en/7.1.x/


Build docs locally
------------------
#. Go to ``docs/`` and install the requirements: ``cd docs/ && pip install -r requirements_doc.txt``.
#. Now the current directory should be under ``docs/``. Build the html webpage: ``make html``.
#. Go to ``docs/build/`` and then host the webpage locally: ``cd build/ && python -m http.server <port>`` , where ``port`` is a number (e.g., 1234).
#. Open the webpage ``localhost:<port>`` in a browser.
