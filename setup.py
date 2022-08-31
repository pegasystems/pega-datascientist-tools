import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pdstools",
    version="2.2.0",
    author="Stijn Kas",
    author_email="stijn.kas@pega.com",
    description="Open source tooling that helps Data Scientists to analyze Pega models and conduct impactful analyses.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pegasystems/pega-datascientist-tools",
    project_urls={
        "Bug Tracker": "https://github.com/pegasystems/pega-datascientist-tools/issues",
        "Wiki": "https://github.com/pegasystems/pega-datascientist-tools/wiki",
    },
    license="Apache-2.0",
    packages=["pdstools"],
    package_dir={"": "python"},
    install_requires=[
        "pandas",
        "plotly>=5.5.0",
        "seaborn",
        "sklearn",
        "requests",
        "nbformat",
        "jupyterlab",
        "ipywidgets",
        "pydot",
        "polars",
        "pyarrow",
        "tqdm"
    ],
    keywords=[
        "pega",
        "pegasystems",
        "cdh",
        "pdstools",
        "customer decision hub",
        "datascientist",
        "tools",
    ],
    python_required=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
)
