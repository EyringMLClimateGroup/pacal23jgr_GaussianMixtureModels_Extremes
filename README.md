# Detecting Extreme Events using Gaussian Mixture Models

This repository provides a Python implementation of the Gaussian Mixture Model (GMM) algorithm for detecting extreme events in time series data. The algorithm is described in the following paper:

> Paçal, A., Hassler, B., Weigel, K., Wehner, M., Kurnaz, M. L., & Eyring, V. (2023). Detecting Extreme Events using Gaussian Mixture Models.
------------------------------------------------------------------------
## Usage

CMIP6 datasets are available at [https://esgf-data.dkrz.de/search/cmip6-dkrz/](https://esgf-data.dkrz.de/search/cmip6-dkrz/). Regional grid files are generated using the [recipe](esmvaltool/recipe_gmm_ssp.yml) with [ESMValTool](https://github.com/ESMValGroup/ESMValTool). Shapefiles for IPCC land regions are freely available at the ATLAS GitHub repository: [https://github.com/SantanderMetGroup/ATLAS](https://github.com/SantanderMetGroup/ATLAS) and provided in the `esmvaltool/auxiliary_data/sep/` directory with the recipe.

The GMM algorithm is implemented in the [diagnostic script](esmvaltool/diag_scripts/gmm/gmm_analysis.py) for ESMValTool. This script takes the 'ESMValTool' output as input and produces the GMM results for each grid cells and each model. 

This repository uses Python 3.8. The required packages are listed in `environment.yaml`. To install the required packages, run the following command:

    mamba env create --file environment.yaml

Return periods for regions under different GWL scenarios are calculated using the [return_period.py](return_period.py) script. This script takes the GMM results as input and produces the return periods for each region and each model under GWL scenarios.

    return_period.py [-h] [INPUT_PATH] [OUTPUT_PATH]

Figures in the paper can be plotted using the [result_plotter.py](result_plotter.py) script. This script takes the return periods as input and produces the figures in the paper.

    results_plotter.py [-h] [INPUT_PATH]