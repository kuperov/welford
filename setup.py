from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as buff:
    long_description = buff.read()


def _parse_requirements(req_path):
    with open(path.join(here, req_path)) as req_file:
        return [
            line.rstrip()
            for line in req_file
            if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="welford",
    version="0.0.2",
    description="Welford algorithms",
    long_description=long_description,
    author="Alex Cooper",
    author_email="alex@acooper.org",
    url="https://github.com/kuperov/welford",
    license="LICENSE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["welford"],
    install_requires=_parse_requirements("requirements.txt"),
    include_package_data=True,
)
