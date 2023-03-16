# Detecting Extreme Events using Gaussian Mixture Models

This repository provides a Python implementation of the Gaussian Mixture Model (GMM) algorithm for detecting extreme events in time series data. 

> Paçal, A., Hassler, B., Weigel, K., Wehner, M., Kurnaz, M. L., & Eyring, V. (2023). Detecting Extreme Events using Gaussian Mixture Models.

Author: Aytaç Paçal, [aytac.pacal@dlr.de](mailto:aytac.pacal@dlr.de)

------------------------------------------------------------------------

## Installation

To use this implementation of the GMM algorithm, you will need to have Python 3 installed on your system. You can use this repository with [ESMValTool environment](https://github.com/ESMValGroup/ESMValTool/blob/main/environment.yml). Follow the installation steps on the [ESMValTool installation](https://docs.esmvaltool.org/en/latest/quickstart/installation.html#mamba-conda-installation) page. Alternatively, the [environment_v2.5.0.yml](environment_v2.5.0.yml) file for the ESMValTool version(v2.5.0) used in this study is provided in this repository. You can create a new environment using this file with the following command:

    mamba env create -f environment_v2.5.0.yml

## Data

The CMIP6 datasets, which are a collection of climate model simulations produced by research institutions worldwide, are available on the [ESGF](https://esgf-data.dkrz.de/search/cmip6-dkrz/) servers.

Regional grid files are generated using [the recipe](esmvaltool/recipe_gmm_ssp.yml) with [ESMValTool](https://github.com/ESMValGroup/ESMValTool), which is a community-driven Python-based evaluation tool for Earth system models. ESMValTool provides a set of predefined recipes for the evaluation of various aspects of Earth system models. Shapefiles for IPCC (Intergovernmental Panel on Climate Change) regions are freely available at the [ATLAS GitHub](https://github.com/SantanderMetGroup/ATLAS) repository. Separate shapefiles for each region should be placed into the [auxiliary_data](esmvaltool/auxiliary_data/) directory of the ESMValTool.

## Usage

The GMM algorithm is implemented in [the diagnostic script](esmvaltool/diag_scripts/gmm/gmm_analysis.py) for [ESMValTool v2.5.0](https://github.com/ESMValGroup/ESMValTool/releases/tag/v2.5.0). You can use [the ESMValTool tutorial](https://esmvalgroup.github.io/ESMValTool_Tutorial/) to learn how to run a recipe. [The recipe](esmvaltool/recipe_gmm_ssp.yml) can be run using the following command:

    esmvaltool run recipe_gmm_ssp.yml

This script takes the CMIP6 data, applies preprocessing functions, and creates regional output files. Then, the diagnostic script takes the ESMValTool output as input and produces the GMM results for each grid cell and each model.

Return periods for regions under different Global Warming Levels (GWL) are calculated using the [return_period.py](return_period.py) script. This script takes the GMM results from the ESMValTool recipe and produces the return periods for each region and each model under GWL scenarios. `INPUT_PATH` is the path to the ESMValTool output directory, and `OUTPUT_PATH` is the path to save the results.

    return_period.py [-h] [INPUT_PATH] [OUTPUT_PATH]

Figures in the paper can be plotted using the [result_plotter.py](result_plotter.py) script. This script takes the return periods as input and produces the figures in the paper. [INPUT_PATH] is the path to the directory containing the regional return period files. Files should be named as `{region_abbreviation}.csv`. The script will produce the figures and save them in a new `plot` directory.

    result_plotter.py [-h] [INPUT_PATH]

## License
This code is released under Apache 2.0. See [LICENSE](LICENSE) for more information.
