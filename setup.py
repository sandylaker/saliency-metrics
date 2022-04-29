import os.path as osp
from setuptools import find_packages, setup


def get_version() -> str:
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), "saliency_metrics", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]  # noqa: E741
    # get the version number without the double quote symbols: e.g. 0.1.0 rather than "0.1.0"
    version = version_line.split("=")[-1].strip().strip('"')
    return version


setup(
    version=get_version(),
    packages=find_packages(exclude=("tests",)),
)
