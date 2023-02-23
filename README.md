# Detecting Extreme Events using Gaussian Mixture Models

This repository provides a Python implementation of the Gaussian Mixture Model (GMM) algorithm for detecting extreme events in time series data. The algorithm is described in the following paper:

> **Detecting Extreme Events in Time Series Data using Gaussian Mixture Models**

> *A. Paçal, B. Hassler, K. Weigel, M. Wehner, M. L. Kurnaz, V. Eyring *

------------------------------------------------------------------------
## Usage

This repository uses Python 3.6. The required packages are listed in `requirements.txt`. To install the required packages, run the following command:

    pip install -r requirements.txt

CMIP6 datasets are available at [https://esgf-data.dkrz.de/search/cmip6-dkrz/](https://esgf-data.dkrz.de/search/cmip6-dkrz/). Regional grid files are generated using the [recipe](esmvaltool/recipe_gmm_ssp.yml) with [ESMValTool](https://github.com/ESMValGroup/ESMValTool). Shapefiles for IPCC land regions are freely available at the ATLAS GitHub repository: [https://github.com/SantanderMetGroup/ATLAS](https://github.com/SantanderMetGroup/ATLAS) and provided in the `esmvaltool/auxiliary_data/sep/` directory with the recipe.

The GMM algorithm is implemented in the [diagnostic script](esmvaltool/diag_scripts/gmm/gmm_analysis.py) for ESMValTool. This script takes the 'ESMValTool' output as input and produces the GMM results for each region and each model under GWL scenarios. 



