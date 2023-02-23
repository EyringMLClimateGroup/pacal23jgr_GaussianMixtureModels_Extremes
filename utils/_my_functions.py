import iris
import os
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_backend
import dask
import cftime
import sys
if hasattr(dask, 'config'):
    # Works at Dask 0.19.4
    dask.config.set(scheduler='single-threaded')
else:
    # Works at Dask 0.17.1
    dask.set_options(get=dask.get)


def dict_path_replacer(dict_to_change):
    for key, value in dict_to_change.items():
        old_mistral_path = "/mnt/lustre02/work/bd1083/b309178/gmmDiag_esmvaltool/recipe_gmm_multiple_shape_gwl15_20211001_231754"
        old_shp_path = "/pf/b/b309178/shapefiles"
        new_path = "/mnt/d/Documents/DATA/gwl15"
        new_shp_path = "/mnt/d/Documents/DATA"
        if isinstance(value, str) and old_mistral_path in value:
            dict_to_change[key] = path_converter(value.replace(old_mistral_path, new_path))
        if isinstance(value, str) and old_shp_path in value:
            dict_to_change[key] = path_converter(value.replace(old_shp_path, new_shp_path))
        elif isinstance(value, list):
            dict_to_change[key] = [path_converter(sub.replace(old_mistral_path, new_path)) for sub in value]
        elif isinstance(value, dict):
            dict_path_replacer(value)

    final_dict = {}
    for old_key, value in dict_to_change.items():
        old_mistral_path = path_converter("/mnt/lustre02/work/bd1083/b309178/gmmDiag_esmvaltool"
                                          "/recipe_gmm_multiple_shape_gwl15_20211001_231754")
        new_path = path_converter("/mnt/d/Documents/DATA/gwl15")

        if old_mistral_path in old_key:
            new_key = old_key.replace(old_mistral_path, new_path)
            final_dict[new_key] = value
        else:
            final_dict[old_key] = value

    return final_dict


def print_dict_tree(dict_to_print, indent=0):
    if not isinstance(dict_to_print, dict) and not isinstance(dict_to_print, list):
        print("    " * indent + str(dict_to_print))
    elif isinstance(dict_to_print, list):
        for i in dict_to_print:
            if not isinstance(i, list):
                print_dict_tree(i, indent + 1)
            else:
                print("    " * indent + str(i))
    else:
        for key, value in dict_to_print.items():
            print("    " * indent + str(key) + ':')
            if isinstance(dict_to_print, dict):
                print_dict_tree(value, indent + 1)
        print("\n")


def cube_to_array(nc_fp):
    cube = iris.load_cube(nc_fp)

    if cube.units == 'degC':
        return cube.data.compressed()
    else:
        return cube.convert_units('celsius').data.compressed()


def path_converter(path):
    _dict = {"/mnt/d/": "D:\\",
             "/": "\\"}
    if os.name == 'nt':
        for i, j in _dict.items():
            path = path.replace(i, j)
        return path
    else:
        return path


def files_to_dict(files_path):
    # Create list from file names
    nc_files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
    file_dict = {}
    for i, file in enumerate(nc_files):
        param = file.split('_')
        region = param[0]
        # Assign file paths for regions
        file_dict[region] = os.path.join(files_path, file)
    return file_dict


def _mp_collector_flatten(exp_collector_list, *output_directory_path):
    main_list = []
    columns = [
        'dataset',
        'region',
        'exp',
        'grid_number',
        'lon',
        'lat',
        'raw_data',
        'raw_data_length',
        'n_comp_initial',
        'best_gmm',
        'best_gmm_fit',
        'n_comp',
        'ks_test1',
        'ks_test2',
        'mean_1',
        'stdev_1',
        'weight_1',
        'mean_2',
        'stdev_2',
        'weight_2',
        'mean_3',
        'stdev_3',
        'weight_3',
        'days>mean-stdev',
        'txx',
    ]

    for item in exp_collector_list:
        if not isinstance(item, dict):
            continue
        for dataset, dataset_dict in item.items():
            for region, region_dict in dataset_dict.items():
                for exp, exp_dict in region_dict.items():
                    cube = iris.load_cube(exp_dict['raw_data'])
                    raw_data = cube.extract(iris.Constraint(latitude=exp_dict['lat'], longitude=exp_dict['lon'])).data.compressed()
                    raw_length = raw_data.size
                    temp = [
                        dataset,
                        region,
                        exp,
                        exp_dict['grid_number'],
                        exp_dict['lon'],
                        exp_dict['lat'],
                        exp_dict['raw_data'],
                        raw_length,
                        exp_dict['n_comp_initial'],
                        exp_dict['best_fit_gmm_dict']['gmm'],
                        exp_dict['best_fit_gmm_dict']['gmm_fit'],
                        exp_dict['best_fit_gmm_dict']['n_comp'],
                    ]

                    ks_list = []
                    for k, v in exp_dict['ks_results'].items():
                        ks_list.append(
                            [k,
                             round(v.pvalue,7)]
                        )
                    ks_list = sorted(ks_list, key=lambda x: x[0])
                    while len(ks_list) < 2:
                        ks_list.append([None, None])
                    temp.extend([ks_list[0][1], ks_list[1][1]])


                    # Export Gauss parameters as list of list
                    para_list = []
                    for Gauss_comp, parameters in exp_dict['best_fit_gmm_dict']['parameters'].items():
                        para_list.append(
                            [parameters['mean'],
                             parameters['stdev'],
                             parameters['weight']]
                        )
                    # Add None value to have same number of columns
                    para_list = sorted(para_list, key=lambda x: x[0])
                    while len(para_list) < 3:
                        para_list.append([None, None, None])
                    # Flatten the list of list
                    para_list = [item for sublist in para_list for item in sublist]
                    temp.extend(para_list)

                    mean_back_sigma = para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 3] - para_list[
                        3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 2]
                    # Calculate number of days hotter than mean-sigma
                    hotter_days = sum(
                        x >=
                        para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 3]
                        - para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 2]
                        for x in raw_data.flatten()
                    )
                    txx = raw_data.max()

                    temp.extend([hotter_days, txx])

                    main_list.append(temp)

    df = pd.DataFrame(main_list, columns=columns)
    grids = list(dict.fromkeys(df['grid_number'].tolist()))

    if exp != 'reanalysis':
        for grid in grids:
            cube = iris.load_cube(df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'raw_data'].values[0])
            ssp585_raw_data = cube.extract(iris.Constraint(
                latitude=df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'lat'].values[0],
                longitude=df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'lon'].values[0]
            )).data.compressed()
            historical_txx = df.loc[((df.grid_number == grid) & (df.exp == 'historical')), 'txx'].values[0]
            df.loc[(df.grid_number == grid) & (df.exp == 'ssp585'), "days>txx"] = sum(x >= historical_txx for x in ssp585_raw_data)

    return df



def _mp_collector_distributor(exp_collector_list, *output_directory_path):
    main_list = []
    columns = [
        'dataset',
        'region',
        'exp',
        'gwl',
        'grid_number',
        'lon',
        'lat',
        'raw_data',
        'raw_data_length',
        'n_comp_initial',
        'best_gmm',
        'best_gmm_fit',
        'n_comp',
        'ks_test1',
        'ks_test2',
        'mean_1',
        'stdev_1',
        'weight_1',
        'mean_2',
        'stdev_2',
        'weight_2',
        'mean_3',
        'stdev_3',
        'weight_3',
        'days>mean-stdev',
        'txx',
    ]
    
    exp_collector = [item for item in exp_collector_list if isinstance(item, dict)]
    

    with parallel_backend('loky', n_jobs=256):
        exp_collector_df  = Parallel(verbose=10, batch_size='auto')(delayed(mpcollector_mproc)(i) for i in exp_collector)
    
    main_list.extend(exp_collector_df)
    main_list = [item for item in main_list if item != None]
    df = pd.DataFrame(main_list, columns=columns)
    
    return df
    # grids = list(dict.fromkeys(df['grid_number'].tolist()))
    # grids = [[df, i] for i in grids]
    
    # with parallel_backend('loky', n_jobs=120):
        # grid_collector  = Parallel(verbose=10, batch_size=8)(delayed(grid_mproc)(i) for i in grids)
    # grid_collector = [item for sublist in grid_collector for item in sublist]

    # final_dataframe =pd.DataFrame(grid_collector, columns=columns)
    # final_dataframe.sort_values(by=['dataset',  'region', 'exp', 'grid_number'], inplace=True)
    # print(final_dataframe)
    # # for grid in grids:
        # # if df.loc[(df.grid_number == grid), 'exp' ].values[0] != 'reanalysis':
            # # cube = iris.load_cube(df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'raw_data'].values[0])
            # # ssp585_raw_data = cube.extract(iris.Constraint(
                # # latitude=df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'lat'].values[0],
                # # longitude=df.loc[((df.grid_number == grid) & (df.exp == 'ssp585')), 'lon'].values[0]
            # # )).data.compressed()
            # # historical_txx = df.loc[((df.grid_number == grid) & (df.exp == 'historical')), 'txx'].values[0]
            # # df.loc[(df.grid_number == grid) & (df.exp == 'ssp585'), "days>txx"] = sum(x >= historical_txx for x in ssp585_raw_data)

 


# def grid_mproc(item):
    
#     df = item[0]
#     grid = item[1]
#     exp_holder = []

#     for exp  in df.loc[(df.grid_number == grid), 'exp' ].to_numpy():
#         if exp == 'ssp585':
#             cube = iris.load_cube(df.loc[((df.grid_number == grid) & (df.exp == exp)), 'raw_data'].values[0])
#             ssp585_raw_data = cube.extract(iris.Constraint(
#                 latitude=df.loc[((df.grid_number == grid) & (df.exp == exp)), 'lat'].values[0],
#                 longitude=df.loc[((df.grid_number == grid) & (df.exp ==exp)), 'lon'].values[0]
#             )).data.compressed()
#             historical_txx = df.loc[((df.grid_number == grid) & (df.exp == 'historical')), 'txx'].values[0]
#             print(historical_txx)
#             df.loc[(df.grid_number == grid) & (df.exp ==exp), "days>txx"] = sum(x >= historical_txx for x in ssp585_raw_data)
#             print(df)
#             exp_holder.append[df.loc[(df.grid_number == grid) & (df.exp == exp)].values.flatten().tolist()]
#         elif exp == 'historical':
#             print(df.loc[(df.grid_number == grid) & (df.exp == exp)].values.flatten().tolist())
#             exp_holder.append[df.loc[(df.grid_number == grid) & (df.exp == exp)].values.flatten().tolist()]
#         else:
#             exp_holder.append[df.loc[(df.grid_number == grid) & (df.exp == exp)].values.flatten().tolist()]
#     print(exp_holder)
#     print(df.columns)
#     return exp_holder


def mpcollector_mproc(item):
    import json
    print("\n\n")
    for dataset, dataset_dict in item.items():
        for region, region_dict in dataset_dict.items():
            for exp, exp_dict in region_dict.items():
                cube = iris.load_cube(exp_dict['raw_data'])
                raw_data = cube.extract(iris.Constraint(latitude=exp_dict['lat'], longitude=exp_dict['lon'])).data.compressed()
                raw_length = raw_data.size
                if raw_length == 0:
                    continue
                temp = [
                    dataset,
                    region,
                    exp,
                    exp_dict['gwl'],
                    exp_dict['grid_number'],
                    exp_dict['lon'],
                    exp_dict['lat'],
                    exp_dict['raw_data'],
                    raw_length,
                    exp_dict['n_comp_initial'],
                    exp_dict['best_fit_gmm_dict']['gmm'],
                    exp_dict['best_fit_gmm_dict']['gmm_fit'],
                    exp_dict['best_fit_gmm_dict']['n_comp'],
                ]
             
                ks_list = []
                for k, v in exp_dict['ks_results'].items():
                    ks_list.append(
                        [k,
                         round(v.pvalue,7)]
                    )
                ks_list = sorted(ks_list, key=lambda x: x[0])
                while len(ks_list) < 2:
                    ks_list.append([None, None])
                temp.extend([ks_list[0][1], ks_list[1][1]])
      

                # Export Gauss parameters as list of list
                para_list = []
                
                for Gauss_comp, parameters in exp_dict['best_fit_gmm_dict']['parameters'].items():
                    para_list.append(
                        [parameters['mean'],
                         parameters['stdev'],
                         parameters['weight']]
                    )
                
                # Add None value to have same number of columns
                
                para_list = sorted(para_list, key=lambda x: x[0])
                while len(para_list) < 3:
                    para_list.append([None, None, None])
                # Flatten the list of list
                para_list = [item for sublist in para_list for item in sublist]
                temp.extend(para_list)
              
                mean_back_sigma = para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 2] - para_list[
                    3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 1]
                # Calculate number of days hotter than mean-sigma
                hotter_days = sum(
                    x >=
                    para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 3]
                    - para_list[3 * exp_dict['best_fit_gmm_dict']['n_comp'] - 2]
                    for x in raw_data.flatten()
                )
                txx = raw_data.max()
                
                temp.extend([hotter_days, txx])

                # print(temp)
                return temp






# def test_hist_plot(grid_dictionary):
#     holder = grid_dictionary[0]['data'].data.compressed()
#     if holder.size == 0:
#         return
#
#     holder = holder.reshape(-1, 1)
#
#     output_path = "/mnt/d/Google Drive/PHD/1st paper/era5_grid_test/output"
#     filename = "{dataset}_{region}_{grid_no}_lon-{lon:.2f}_lat-{lat:.2f}_{PID}.png".format(
#         dataset=grid_dictionary['dataset'],
#         region=grid_dictionary['region'],
#         grid_no=grid_dictionary['grid_number'],
#         lon=grid_dictionary['grid_lon'],
#         lat=grid_dictionary['grid_lat'],
#         PID=getpid(),
#     )
#     save_filename = join(output_path, filename)
#     print(save_filename)
#     return grid_gmm_analysis(holder, grid_dictionary, save_filename)
#
#
# def cube_slicer_test(cube):
#     mp_data = []
#     ic()
#     for i, t_slice in enumerate(cube.slices(['time'])):
#         # if i < 100 or i > 200:
#         #     continue
#         # holder = t_slice.data.compressed()
#         # if holder.size == 0:
#         #      print('pass %s' % i)
#         #      continue
#
#         lon = t_slice.coord('longitude').points
#         lat = t_slice.coord('latitude').points
#         grid_dictionary = {'data': t_slice,
#                            'grid_number': i,
#                            'grid_lon': lon[0],
#                            'grid_lat': lat[0],
#                            'dataset': 'BCC-CSM2-MR',
#                            'region': 'ARP',
#                            }
#
#         mp_data.append(grid_dictionary)
#
#     # ic(mp_data)
#     # Assign jobs to multiprocess and keep time
#     start_mp = time.time()
#     ic(type(mp_data[0]))
#     with parallel_backend('multiprocessing', n_jobs=4):
#         mp_val = Parallel()(delayed(test_hist_plot)(i) for i in mp_data)
#     end_mp = time.time()
#
#     print(mp_val)
#     eta_multiprocessing = str(datetime.timedelta(seconds=end_mp - start_mp))
#     ic(eta_multiprocessing)
#
#
# def test_joblib():
#     # nc_file = '/mnt/d/Google Drive/PHD/1st paper/era5_grid_test/
#     # arp_OBS_ERA5-org_day_reanalysis_v1_tasmax_1980-2010.nc'
#     # nc_file = '/mnt/d/Documents/DATA/era5_regions_1x1/ARP_OBS_ERA5_day_reanalysis_v1_tasmax_1980-2010.nc'
#     nc_file = '/mnt/d/Documents/DATA/gwl15/preproc/extract_ARP_region/maximum_temperature' \
#               '/CMIP6_BCC-CSM2-MR_day_historical_r1i1p1f1_tasmax_1980-2010.nc'
#
#     cube = iris.load_cube(nc_file)
#     mp_data = []
#     for i, t_slice in enumerate(cube.slices(['time'])):
#         lon = t_slice.coord('longitude').points
#         lat = t_slice.coord('latitude').points
#         grid_dictionary = {'data': t_slice,
#                            'grid_number': i,
#                            'grid_lon': lon[0],
#                            'grid_lat': lat[0],
#                            'dataset': 'BCC-CSM2-MR',
#                            'region': 'ARP',
#                            }
#         mp_data.append(([grid_dictionary]))
#
#     ic('start mp')
#     with parallel_backend('multiprocessing', n_jobs=4):
#         mp_val = Parallel(verbose=10)(delayed(test_hist_plot)(i) for i in mp_data)
#     return mp_val
