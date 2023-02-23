import math
import os.path
import random

import cftime
import iris
import numpy
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.mixture import GaussianMixture


def unimodality_check(dataset, best_gmm, best_n):
    clf = best_gmm

    mu1_p = min(clf.means_)[0]
    mu2_p = max(clf.means_)[0]
    lhs = abs(mu1_p - mu2_p)
    rhs = 2 * min([math.sqrt(item) for item in clf.covariances_])

    if lhs <= rhs:
        return True
    else:
        return False


def grid_gmm_analysis(grid_dictionary):
    """

    Parameters
    ----------
    data_holder
    grid_dictionary

    Returns
    -------
    GMM fit for grid cell

    """

    limit_range = False
    if grid_dictionary['experiment'] != 'historical' or not limit_range:
        if grid_dictionary['experiment'] == 'era5':
            grid_dictionary['data'].convert_units('celsius')
        data_holder = grid_dictionary['data'].data.compressed()

    else:
        d1 = cftime.DatetimeNoLeap(1980, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        d2 = cftime.DatetimeNoLeap(2000, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        historical_timerange = iris.Constraint(
            time=lambda cell: d1 <= cell.point < d2)
        data_holder = grid_dictionary['data'].extract(historical_timerange).data.compressed()

    # Check if grid is empty
    if data_holder.size == 0:
        return

    if grid_dictionary['experiment'] != 'reanalysis':
        data_holder = data_holder.reshape(-1, 1)
    else:
        data_holder = data_holder.reshape(-1, 1)

    n_components_range = range(1, 4)
    bic_score_component = {}
    best_n = 1
    lowest_bic = np.infty
    bic = []
    best_gmm = 0
    # gmm_result_dictionary = {
    #     'grid_number': grid_dictionary['grid_number'],
    #     'dataset': grid_dictionary['dataset'],
    #     'region': grid_dictionary['region'],
    #     'lon': grid_dictionary['grid_lon'],
    #     'lat': grid_dictionary['grid_lat'],
    #     'experiment': grid_dictionary['experiment']
    # }

    gmm_result_dictionary = {
        grid_dictionary['dataset']: {
            grid_dictionary['region']: {
                grid_dictionary['experiment']: {
                    'grid_number': grid_dictionary['grid_number'],
                    'lon': grid_dictionary['grid_lon'],
                    'lat': grid_dictionary['grid_lat'],
                    'raw_data': grid_dictionary['cube_file'],
                    'gwl': grid_dictionary['gwl']
                }
            }
        }
    }
    cdfs = {}
    gmm_dict = {}
    bin_number = math.ceil((max(data_holder) - min(data_holder)) / 1)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full',
                              n_init=3,
                              )
        fit_holder = gmm.fit(X=data_holder)
        gmm_dict[n_components] = {'gmm': gmm,
                                  'gmm_fit': fit_holder,
                                  'n_comp': n_components,
                                  }

        para_dict = {}
        x_list = []
        for i, (mean, cov, weight) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
            temp = {'mean': mean[0],
                    'stdev': math.sqrt(cov),
                    'weight': weight}
            # gauss_key = 'Gauss_{number}'.format(number=i + 1)
            para_dict[i + 1] = temp
            x_list.append(np.random.normal(mean[0], math.sqrt(cov), size=int(len(data_holder) * weight)))
        gmm_dict[n_components]['parameters'] = para_dict
        cdf_data = np.concatenate(x_list)
        while len(cdf_data) < len(data_holder):
            cdf_data = np.append(cdf_data, random.choice(cdf_data))
        count, bins_count = np.histogram(cdf_data, bins=bin_number)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        cdfs[n_components] = dict(pdf=pdf, cdf=cdf, X=cdf_data)

        temp_bic = gmm.bic(X=data_holder)
        bic.append(temp_bic)
        bic_score_component[n_components] = temp_bic
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            best_n = n_components


    ks_dict = {}
    for k, v in cdfs.items():
        ks_dict[k] = stats.kstest(data_holder.flatten(), v['X'].flatten())

    gmm_result_dictionary[
        grid_dictionary['dataset']][
        grid_dictionary['region']][
        grid_dictionary['experiment']]['n_comp_initial'] = best_n

    gmm_result_dictionary[
        grid_dictionary['dataset']][
        grid_dictionary['region']][
        grid_dictionary['experiment']]['ks_results'] = ks_dict


    gmm_result_dictionary[
        grid_dictionary['dataset']][
        grid_dictionary['region']][
        grid_dictionary['experiment']][
        'n_comp_initial'] = best_n

    if best_n == 1:
        gmm_result_dictionary[
            grid_dictionary['dataset']][
            grid_dictionary['region']][
            grid_dictionary['experiment']][
            'best_fit_gmm_dict'] = gmm_dict[best_n]
    else:
        # if GMM fits 2 Gaussian and not uni-modal, return the best fit, i.e. multi-modal fit
        if best_n == 2 and not unimodality_check(data_holder, best_gmm, best_n):
            # print("GMM fits 2 Gaussian and uni-modality check is False => returned multimodal Gaussian")
            gmm_result_dictionary[
                grid_dictionary['dataset']][
                grid_dictionary['region']][
                grid_dictionary['experiment']][
                'best_fit_gmm_dict'] = gmm_dict[best_n]

        # if GMM fits 2 Gaussian but uni-modality check is True, return single Gaussian
        elif best_n == 2 and unimodality_check(data_holder, best_gmm, best_n):
            # print("GMM fits 2 Gaussian but uni-modality check is True => returned single Gaussian")
            gmm_result_dictionary[
                grid_dictionary['dataset']][
                grid_dictionary['region']][
                grid_dictionary['experiment']][
                'best_fit_gmm_dict'] = gmm_dict[1]

        # if GMM fits more than 2 Gaussian's, check for largest gradient change between components
        else:
            # print("GMM fits more than 2 Gaussian's check for largest gradient")
            largest_change = 0
            for i in range(1, 3):
                diff = bic_score_component[i] - bic_score_component[i + 1]
                if diff < 0 and i == 1:
                    best_n = i
                    # print("Break at {best_n}".format(best_n=best_n))
                    best_gmm = gmm_dict[1]['gmm']
                    gmm_result_dictionary[
                        grid_dictionary['dataset']][
                        grid_dictionary['region']][
                        grid_dictionary['experiment']][
                        'best_fit_gmm_dict'] = gmm_dict[1]
                elif diff > largest_change:
                    largest_change = diff
                    best_n = i + 1
                else:
                    continue

            if best_n == 2 and unimodality_check(data_holder, gmm_dict[best_n]['gmm'], best_n):
                gmm_result_dictionary[
                    grid_dictionary['dataset']][
                    grid_dictionary['region']][
                    grid_dictionary['experiment']][
                    'best_fit_gmm_dict'] = gmm_dict[1]
            elif best_n == 3:
                # print("OVERWRITE SINGLE GAUSSIAN")
                gmm_result_dictionary[
                    grid_dictionary['dataset']][
                    grid_dictionary['region']][
                    grid_dictionary['experiment']][
                    'best_fit_gmm_dict'] = gmm_dict[1]
            else:
                # print("Largest gradient change is between {comp1} and {comp2}, and not a single Gaussian "
                #       "choose {comp2}".format(comp1=best_n - 1, comp2=best_n))
                gmm_result_dictionary[
                    grid_dictionary['dataset']][
                    grid_dictionary['region']][
                    grid_dictionary['experiment']][
                    'best_fit_gmm_dict'] = gmm_dict[best_n]

    return gmm_result_dictionary


def gmm_select_plot(dataset, best_gmm, bic, region_name, model_name, period, output_folder_path):
    import matplotlib.pyplot as plt
    n_components_range = range(1, len(bic) + 1)
    clf = best_gmm
    bin_number = math.ceil((max(dataset) - min(dataset)) / 1)

    # Plot the original data
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.hist(dataset, bins=bin_number, density=True, label="Original data")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    # Plot predicted histograms
    x_list = []
    for mean, cov, weight in zip(clf.means_, clf.covariances_, clf.weights_):
        x_list.append(np.random.normal(mean[0], math.sqrt(cov), size=int(len(dataset) * weight)))
    X = np.concatenate(x_list)
    plt.hist(X, bins=bin_number, density=True, histtype=u'step', label="Predicted histogram")

    plt.title("Histograms for the original data and the predicted parameters at "
              "\n{region} region for {model} model in {period} ".format(region=region_name, model=model_name,
                                                                        period=period))
    plt.xlabel("T")
    plt.legend()

    # Plot the BIC scores
    bic = np.array(bic)
    spl = plt.subplot(2, 1, 2)
    xpos = np.array(n_components_range)
    plt.bar(xpos, bic, width=.2, color='green')
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per component')
    # xpos = np.mod(best_gmm.n_components,
    #               len(n_components_range)) + .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(best_gmm.n_components - .03, bic.min() * 0.97 + .03 * bic.max(), '**', color='r', fontsize=14)
    spl.set_xlabel('Number of components')
    plt.savefig(output_folder_path)
    plt.close()
    return "GMM selection plot end"


def region_mean_save(cube, input_dictionary, exp, output_dir_path):
    grid_areas = iris.analysis.cartography.area_weights(cube)
    if cube.coord('latitude').bounds is None:
        print("Guess bounds")
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    region_mean_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
    region_mean_array = region_mean_cube.data.compressed()
    region_csv_filename = '{region}_{dataset}_{exp}_{gwl}_{year}_region_mean.csv'.format(
        dataset=input_dictionary['dataset'],
        region=input_dictionary['region'],
        exp=exp,
        gwl=input_dictionary['gwl'],
        year=str(input_dictionary['start_year']) + '-' + str(input_dictionary['end_year']),
    )
    region_csv_path = os.path.join(output_dir_path, "means", region_csv_filename)
    numpy.savetxt(region_csv_path, region_mean_array, delimiter=',', fmt='%f')
    return