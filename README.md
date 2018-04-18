# Face Recognition

### Dependencies
- Python3
- Run `pip install -r requirements` to install required Python libaries

### Quick Start
Run `make run` or `python3 -W ignore test.py` to run classifier models from `src/`. NOTE: You must be in 'src/' before running `test.py` or else relative paths will fail. These models contain a fraction of the code in the notebooks.

Run `jupyter notebook` or `jupyter lab` (requires Python module installation of `jupyter` and `jupyterlab`, respectively) and traverse `notebooks/` to run the notebooks. `analysis.ipynb` contains all exploratory analysis (plots, etc) performed in the report.

Run `python3 process_data.py` to manually generate the data from the raw inputs from `src/`. This shouldn't be necessary since the data is already in the .zip file. This code is from the `process_data.ipynb` notebook.

### Directory Structure
`notebooks/` contains all exploratory code in Jupyter Notebooks
`src/` contains refined code i.e classifiers, modules, etc
`data/` contains raw data in Matlab files, and is populated with images after preprocessing
`Makefile` contains commands to clean data, run models, and perform other helpful tasks
`report.pdf` contains the finalized report
`reports` contains plots used in the report

All code in this repository is written in Python 3.6.
