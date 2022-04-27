import os.path as osp
import pkg_resources
from setuptools import find_packages, setup
from typing import List


def get_requirements(mode: str) -> List[str]:
    if mode not in ("install", "dev"):
        raise ValueError(f"name should be one of ('install', 'dev'), but got {mode}")

    with open(osp.join("requirements", f"{mode}.txt"), "r") as f:
        requirements: List[str] = [str(r) for r in pkg_resources.parse_requirements(f)]

    return requirements


def get_version() -> str:
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), "saliency_metrics", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]  # noqa: E741
    # get the version number without the double quote symbols: e.g. 0.1.0 rather than "0.1.0"
    version = version_line.split("=")[-1].strip().strip('"')
    return version


setup(
    name="saliency-metrics",
    version=get_version(),
    description="A Unified Framework for Benchmarking Explanation methods in Computer Vision",
    keywords=["Deep Learning", "Computer Vision", "Explainable AI"],
    packages=find_packages(exclude=("tests",)),
    url="https://github.com/slds-lmu/saliancy-metrics",
    author="TODO",
    author_email="TODO",
    install_requires=get_requirements("install"),
    zip_safe=False,
)
