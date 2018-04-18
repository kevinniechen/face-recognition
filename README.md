# Face Recognition

### Dependencies
- Python3
- Run `pip install -r requirements` to install required Python libaries

### Quick Start
Run `jupyter notebook` or `jupyter lab` (requires Python module installation of `jupyter` and `jupyterlab`, respectively) and traverse `notebooks/` to run the notebooks.

Run `make run` to run classifier models. (might be buggy- run jupyter notebook if unavailable)

### Directory Structure
`notebooks/` contains all exploratory code in Jupyter Notebooks
`src/` contains refined code i.e classifiers, modules, etc
`data/` contains raw data in Matlab files, and is populated with images after preprocessing
`Makefile` contains commands to clean data, run models, and perform other helpful tasks
`report.pdf` contains the finalized report
`reports` contains plots used in the report

All code in this repository is written in Python 3.6.
