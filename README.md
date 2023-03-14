# Detecting Extreme Events using Gaussian Mixture Models

This repository provides a Python implementation of the Gaussian Mixture Model (GMM) algorithm for detecting extreme events in time series data. 

> Paçal, A., Hassler, B., Weigel, K., Wehner, M., Kurnaz, M. L., & Eyring, V. (2023). Detecting Extreme Events using Gaussian Mixture Models.
------------------------------------------------------------------------

## Usage

CMIP6 datasets are available at [DKRZ](https://esgf-data.dkrz.de/search/cmip6-dkrz/). Regional grid files are generated using the [recipe](esmvaltool/recipe_gmm_ssp.yml) with [ESMValTool](https://github.com/ESMValGroup/ESMValTool). Shapefiles for IPCC regions are freely available at the [ATLAS GitHub](https://github.com/SantanderMetGroup/ATLAS) repository. Separate shapefiles for each region should be placed into the `auxiliary_data` directory of the ESMValTool. Shapefiles for the regions are available in the [esmvaltool/auxiliary_data](esmvaltool/auxiliary_data/) directory of this repository.

The GMM algorithm is implemented in the [diagnostic script](esmvaltool/diag_scripts/gmm/gmm_analysis.py) for [ESMValTool v2.5.0](https://github.com/ESMValGroup/ESMValTool/releases/tag/v2.5.0). This script takes the 'ESMValTool' output as input and produces the GMM results for each grid cell and each model.

1. This repository uses Python 3.8. You can use this repository with ESMValTool conda environment. Follow the installation steps in [ESMValTool Mamba/Conda Installation](https://docs.esmvaltool.org/en/latest/quickstart/installation.html#mamba-conda-installation) page. To install the required packages, run the following command:

        mamba env create --name esmvaltool --file environment.yml

2. Return periods for regions under different GWL scenarios are calculated using the [return_period.py](return_period.py) script. This script takes the GMM results from ESMValTool recipe and produces the return periods for each region and each model under GWL scenarios. `INPUT_PATH` is the path to the ESMValTool output directory, and `OUTPUT_PATH` is the path to save the results.

        return_period.py [-h] [INPUT_PATH] [OUTPUT_PATH]

3. Figures in the paper can be plotted using the [result_plotter.py](result_plotter.py) script. This script takes the return periods as input and produces the figures in the paper. [INPUT_PATH] is the path to the directory containing the regional return period files. Files should be named as `{region}.csv`. The script will produce the figures and save them in a new `plot` directory.

        result_plotter.py [-h] [INPUT_PATH]
