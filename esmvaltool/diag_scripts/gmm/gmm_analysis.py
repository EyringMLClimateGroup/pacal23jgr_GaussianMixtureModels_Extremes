import csv
import math
import os
import sys
from os import chdir, getcwd, getpid
from os.path import join

import cftime
import dask
import pandas as pd
import yaml
from joblib import Parallel, delayed, parallel_backend
from utils._gmm_utils import grid_gmm_analysis, region_mean_save
from utils._my_functions import (
    _mp_collector_distributor,
    _mp_collector_flatten,
    files_to_dict,
)

if hasattr(dask, 'config'):
    # Works at Dask 0.19.4
    dask.config.set(scheduler='single-threaded')
else:
    # Works at Dask 0.17.1
    dask.set_options(get=dask.get)
import datetime as dt
import time

import iris
import numpy as np
from esmvalcore.preprocessor import area_statistics
from sklearn.mixture import GaussianMixture

from esmvaltool.diag_scripts.shared import (
    group_metadata,
    run_diagnostic,
    select_metadata,
)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def region_experiments_to_grid_slicer(input_dictionary, output_dir_path, cfg):
    """
    Iterate over all grid cells and map to multiprocess function
    """

    start_distributor = time.time()
    exp_collector = []

    for exp_gwl, exp_dict in input_dictionary.items():
        print('/n')
        print(exp_dict['region'])
        print(exp_dict['dataset'])
        print(exp_gwl)
        print(exp_dict['gwl'])
        print(exp_dict['timerange'])

        cube = iris.load_cube(exp_dict["filename"])

        cube_size = cube.shape[1] * cube.shape[2]
        print(
            '{region} region {exp} GWL{gwl} for {dataset} has {numberofgrid}(Shape={shape}) for MP run'
            .format(region=exp_dict['region'],
                    dataset=exp_dict['dataset'],
                    numberofgrid=cube_size,
                    shape=cube.shape,
                    exp=exp_dict["exp"],
                    gwl=exp_dict['gwl']))

        mp_data = []
        for grid_no, grid_slice in enumerate(cube.slices(['time'])):
            lon = grid_slice.coord('longitude').points
            lat = grid_slice.coord('latitude').points

            grid_dictionary = {
                'data': grid_slice,
                'cube_file': exp_dict["filename"],
                'grid_number': grid_no,
                'grid_lon': round(lon[0], 2),
                'grid_lat': round(lat[0], 2),
                'output_directory': output_dir_path,
                'auxiliary_data_dir': cfg['auxiliary_data_dir'],
                'gwl': exp_dict['gwl'],
                'region': exp_dict['region'],
                'dataset': exp_dict['dataset'],
                'experiment': exp_dict["exp"],
            }

            if grid_no % (cube_size / 10) == 0:
                print('Queued {grid_no} grids for {region}'.format(
                    grid_no=grid_no, region=exp_dict['region']))
            mp_data.append(grid_dictionary)

        print("********  Starting MP ************")
        start_mp = time.time()

        if cube_size > 5000:
            number_jobs = 256
        else:
            number_jobs = 256

        with parallel_backend('loky', n_jobs=number_jobs):
            mp_val = Parallel(verbose=10,
                              batch_size="auto")(delayed(grid_gmm_analysis)(i)
                                                 for i in mp_data)
        mp_val = list([x for x in mp_val if x is not None])

        end_mp = time.time()
        print('Multi process ended in {time}'.format(
            time=str(dt.timedelta(seconds=end_mp - start_mp))))
        exp_collector.extend(mp_val)

    # return list of dictionaries for each grid cell and GMM parameters
    final_dataframe = _mp_collector_distributor(exp_collector, output_dir_path)

    return final_dataframe


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def extreme_events_with_GMM(input_dictionary, cfg):
    """

    Parameters
    ----------
    input_dictionary
    cfg

    Returns
    -------
    This function unpack input dictionary and send regions dictionary to
    run with MP

    """

    # Get time for output folder
    # filter_dataset=[int(x) for x in sys.argv[2].split('_')]

    dt_string = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print('Create folders and distribute dict.\n')
    '''
    Input dictionary structure:
    Region (i.e. ARP)
        Dataset (i.e. BCC-CSM2-MR)
            historical
                'activity', 'alias', 'dataset', 'end_year', 'ensemble', 'exp', 'filename', 
                'frequency', 'long_name', 'short_name', 'start_year'
            ssp-gwl (i.e. SSP126-15)
                ...
            ssp-gwl (i.e. SSP126-20)
                ...
    '''

    skipped_data = []
    save_input_dict = True

    completed_regions_fp = join(cfg['plot_dir'], 'completed_regions.csv')
    completed_regions = []
    with open(completed_regions_fp, newline='') as inputfile:
        for row in csv.reader(inputfile):
            completed_regions.append(row[0])

    print(completed_regions)

    for region, dataset_dict in input_dictionary.items():
        region_name = region.split('_')[1]

        if region_name not in ['RAR']:
            print(
                'Region {region_} already exists.'.format(region_=region_name))
            continue

        output_directory_path = cfg[
            'plot_dir'] + '_' + dt_string + '/' + region_name + '/'
        # CREATE DIRECTORIES IF NOT EXIST
        if not os.path.exists(output_directory_path):
            os.makedirs(join(output_directory_path, "means"))
        if save_input_dict:
            dict_to_json(
                input_dictionary,
                join(cfg['plot_dir'] + '_' + dt_string + '/',
                     'files_dictionary'))
            save_input_dict = False

        print("Start analyses for {region} -> {output_path}".format(
            region=region_name, output_path=output_directory_path))

        for dataset, exp_dict in dataset_dict.items():

            if dataset in [
                    'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR',
                    'BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1-HR', 'CNRM-CM6-1',
                    'CNRM-ESM2-1', 'EC-Earth3-AerChem', 'EC-Earth3-Veg-LR',
                    'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-g3',
                    'GFDL-EMS4', 'HadGEM3-GC31-LL'
            ]:
                print('Dataset {dataset_name} already exists.'.format(
                    dataset_name=dataset))
                continue
            print("Start experiments analyses for {dataset} dataset".format(
                dataset=dataset))
            region_grids_gmm_results_dataframe = region_experiments_to_grid_slicer(
                exp_dict, output_directory_path, cfg)
            if isinstance(region_grids_gmm_results_dataframe, dict):
                print("Skip {dataset} {region} too big".format(
                    dataset=dataset, region=region_name))
                print(region_grids_gmm_results_dataframe)
                skipped_data.extend([region_grids_gmm_results_dataframe])
                continue

            # Remove raw_data column and save region results
            dataframe_to_save = region_grids_gmm_results_dataframe.drop(
                ['best_gmm', 'best_gmm_fit'], axis=1)

            filename = "{region}_{dataset}.csv".format(
                dataset=dataset,
                region=region_name,
            )
            save_filename = join(output_directory_path, filename)
            dataframe_to_save.to_csv(save_filename, index_label='row_number')

            print(
                '********{dataset} ENDED********\n\n'.format(dataset=dataset))

        append_new_line(completed_regions_fp, region_name)
        print('**{region} ENDED**\n\n\n\n'.format(region=region_name))

    print(skipped_data)
    skipped_filename = join(cfg['plot_dir'] + '_' + dt_string,
                            'skipped_dataset')
    skipped_df = pd.DataFrame(skipped_data)
    skipped_df.to_csv(skipped_filename)

    return "GMM FUNCTION COMPLETED"


def files_to_csv(input_files):
    holder = []
    print("start CSV")
    for item in input_files:
        print(item)
        with open(item, 'r') as file:
            metadata_yml = yaml.safe_load(file)
        temp = pd.DataFrame.from_dict(metadata_yml, orient='index')
        temp = temp.reset_index(level=0)

        temp = temp.join(temp['variable_group'].str.split(
            '_', expand=True).reindex(
                range(3), axis=1).iloc[:, 1:].add_prefix('string').fillna(0))
        temp.index
        temp = temp.join(temp['diagnostic'].str.split(
            '_', expand=True).iloc[:, 1:2].add_prefix('region'))
        temp = temp.drop(columns=[
            'diagnostic', 'exp', 'recipe_dataset_index', 'variable_group',
            'index'
        ])
        temp = temp.rename(columns={
            'region1': 'region',
            'string1': 'exp',
            'string2': 'gwl'
        })

        temp.set_index(['exp', 'dataset', 'region', 'gwl'], inplace=True)
        holder.append(temp)
    print("loop end")
    df = pd.concat(holder)
    df = df.reset_index()
    df["gwl"] = df["gwl"].astype(float) / 10
    df = df.set_index(['region', 'dataset', 'exp', 'gwl'])
    df.sort_index(axis=0,
                  level=['region', 'dataset', 'exp', 'gwl'],
                  inplace=True)

    print("saving...")
    df.reset_index().to_csv("regional_ssp_gwl_datasets.csv")
    print('done')
    return df


def dict_to_json(d, filename):
    import json

    import yaml
    print('save json')
    with open(filename + '.json', 'w') as fp:
        json.dump(d, fp, sort_keys=True, indent=4)
    print('save yml')
    with open(filename + '.yml', 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)
    print('done')


def main(cfg):
    """
    Organized metadata dictionary will have the following keys:
    Region -> Model -> Experiment-GWL
    i.e. MED -> HadGEM2-ES  -> historical
    i.e. MED -> HadGEM2-ES  -> ssp126-15
    """
    era5_path = '/work/bd1083/b309178/era5cli/out/era5_regions_1x1/'
    era5_dictionary = files_to_dict(era5_path)

    my_files_dict = {}
    for k1, v1 in group_metadata(cfg['input_data'].values(),
                                 'preprocessor').items():
        my_files_dict[k1] = {}
        for k2, v2 in group_metadata(v1, 'dataset').items():
            my_files_dict[k1][k2] = group_metadata(v2, 'variable_group')

    for k1, v1 in my_files_dict.items():
        print(k1)
        region_name = k1.split('_')[1]
        v1['ECMWF-ERA5'] = {
            'reanalysis': [{
                'filename': era5_dictionary[region_name],
                'exp': 'reanalysis',
                'start_year': 1980,
                'end_year': 2010,
                'dataset': 'ECMWF-ERA5',
                'short_name': 'tasmax',
                'long_name': 'Daily Maximum Near-Surface Air Temperature',
                'project': 'OBS',
                'timerange': '1980/2010',
                "frequency": "day",
                "standard_name": "air_temperature",
            }]
        }

        for k2, v2 in v1.items():
            print(k2)
            for k3, v3 in v2.items():
                print(k3)
                if "historical" not in k3:
                    if "cmip6" in k3:
                        gwl_extract = int(k3.split('_')[-1]) / 10
                        v3[0]['exp'] = k3.split('_')[1]
                    else:
                        gwl_extract = 0

                else:
                    gwl_extract = 0
                v3[0]['gwl'] = gwl_extract
                v3[0]['region'] = region_name

                my_files_dict[k1][k2][k3] = v3[0]

    print('*******START\n\n\n\n')

    extreme_events_with_GMM(my_files_dict, cfg)

    return "MAIN DONE"


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)

    with run_diagnostic() as config:
        # list here the functions that need to run
        main_start_time = time.time()
        main(config)
        print('\n\n\n')
        print("PROGRAM END. Elapsed time : {eta_program}******\n\n\n\n".format(
            eta_program=str(
                dt.timedelta(seconds=time.time() - main_start_time))))
        print("*******DONE**********")
        print('\n\n\n')
