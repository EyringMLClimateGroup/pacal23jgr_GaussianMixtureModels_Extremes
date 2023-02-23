import math
import os
import time
import datetime
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from joblib import parallel_backend, Parallel, delayed
from scipy import special as sps
import datetime as dt
from collections import defaultdict


def list_files_in_directory(parent_directory):
    list_of_files = list()
    infos = parent_directory.split('/')
    gwl = infos[5]
    region_dictionary = {}
    for dirname in os.listdir(parent_directory):
        if dirname == 'return_analysis':
            continue
        if isfile(join(parent_directory, dirname)):
            continue

        #dataset = dirname.split('_')[1]
        # file_dictionary[dataset] = {}
        region_dictionary[dirname] = {}
        for file in os.listdir(join(parent_directory, dirname)):
            if 'means' in file:
                continue
            params = os.path.splitext(file)
            dataset = params[0].split('_')[1]

            region_dictionary[dirname][dataset] = join(parent_directory, dirname, file)

    return region_dictionary



def grids_to_region_average(dataframe):
    dataframe.columns = dataframe.columns.astype(str)

    model = dataframe['dataset'].values[0]
    region = dataframe['region'].values[0]
    ssp = dataframe['exp'].values[0]
    gwl = dataframe['gwl'].values[0]
    
    historical_average_cold_mean = dataframe['historical_mean_cold'].mean()
    historical_std_cold_mean = dataframe['historical_mean_cold'].std()
    historical_average_hot_mean = dataframe['historical_mean_hot'].mean()
    historical_std_hot_mean = dataframe['historical_mean_hot'].std()
    historical_peak_diff = historical_average_hot_mean - historical_average_cold_mean
    historical_hot_period_length = dataframe['historical_hot_period_length'].mean()
    historical_one_component = len(dataframe.loc[dataframe['historical_n_comp'] == 1]['historical_n_comp']) / len(dataframe['historical_n_comp'])
    historical_two_component = len(dataframe.loc[dataframe['historical_n_comp'] == 2]['historical_n_comp']) / len(dataframe['historical_n_comp'])

    future_average_cold_mean = dataframe['future_mean_cold'].mean()
    future_std_cold_mean = dataframe['future_mean_cold'].std()
    future_average_hot_mean = dataframe['future_mean_hot'].mean()
    future_std_hot_mean = dataframe['future_mean_hot'].std()
    future_peak_diff = future_average_hot_mean - future_average_cold_mean
    future_hot_period_length = dataframe['future_hot_period_length'].mean()
    future_one_component = len(dataframe.loc[dataframe['future_n_comp'] == 1]['future_n_comp']) / len(dataframe['future_n_comp'])
    future_two_component = len(dataframe.loc[dataframe['future_n_comp'] == 2]['future_n_comp']) / len(dataframe['future_n_comp'])

    average_hot_peak_diff = future_average_hot_mean - historical_average_hot_mean
    average_cold_peak_diff = future_average_cold_mean - historical_average_cold_mean

    peak_moving_direction = average_cold_peak_diff - average_hot_peak_diff
    # cold peak converging to hot if positive
    # hot peak diverging from cold if negative

    mean_std_dict = dict(
        region=region,
        dataset=model,
        exp = ssp,
        gwl=gwl,

        historical_average_cold_mean=historical_average_cold_mean,
        historical_std_cold_mean=historical_std_cold_mean,
        historical_average_hot_mean=historical_average_hot_mean,
        historical_std_hot_mean=historical_std_hot_mean,
        historical_peak_diff=historical_peak_diff,
        historical_one_component=historical_one_component,
        historical_two_component=historical_two_component,

        future_average_cold_mean=future_average_cold_mean,
        future_std_cold_mean=future_std_cold_mean,
        future_average_hot_mean=future_average_hot_mean,
        future_std_hot_mean=future_std_hot_mean,
        future_peak_diff=future_peak_diff,
        future_one_component=future_one_component,
        future_two_component=future_two_component,

        historical_hot_period_length=historical_hot_period_length,
        future_hot_period_length=future_hot_period_length,
        average_hot_peak_diff=average_hot_peak_diff,
        average_cold_peak_diff=average_cold_peak_diff,
        peak_moving_direction=peak_moving_direction
    )

    for n in [1, 5, 10, 20, 30]:
        return_temp = {
            str(n) + '-year_mean': dataframe[str(n)].mean(),
            str(n) + '-year_stdev': dataframe[str(n)].std(),
            str(n) + '-year_freq_day_mean': dataframe[str(n) + '-year_future_freq_day'].mean(),
            str(n) + '-year_freq_day_std': dataframe[str(n) + '-year_future_freq_day'].std(),
        }
        mean_std_dict.update(return_temp)

    return mean_std_dict


def _calculate_return_distributor(data_list):

    mp_distributor = []
    for item in data_list:
        output_path = data_list[1]
        for region, dataset_dict in item[0].items():
            for model, filepath in dataset_dict.items():
                df = pd.read_csv(filepath, index_col='row_number')
                grids = list(dict.fromkeys(df['grid_number'].tolist()))
                gwl_list = list(dict.fromkeys(df['gwl'].tolist()))
                exp_list = list(dict.fromkeys(df['exp'].tolist()))
                for exp in exp_list:
                    for gwl in gwl_list:
                        mp_distributor.append(dict(region=region, model=model, exp=exp, gwl=gwl, output_path=output_path, filepath=filepath))
                    

def _calculate_return_for_grid_cell(data_list):
    
    output_path = data_list[1]
    region_result_collector = []
    for region, dataset_dict in data_list[0].items():
        if region not in ['RAR', 'EAN']:
            continue
        for model, filepath in dataset_dict.items():
            df = pd.read_csv(filepath, index_col='row_number')
            grids = list(dict.fromkeys(df['grid_number'].tolist()))
            gwl_list = list(dict.fromkeys(df['gwl'].tolist()))
            ssp_list = list(dict.fromkeys(df['exp'].tolist()))
            
            for ssp in ssp_list:
                if ssp == 'historical':
                    continue
                
                for gwl in gwl_list:
                    if gwl == 0:
                        continue   
                    print("Starting {region} region - {model} model - {ssp} scenario - GWL{gwl}C".format(
                        model=model,
                        region=region,
                        ssp=ssp,
                        gwl=gwl))
                                            
                    filename = "{region}_{model}_{ssp}_{gwl}_grids.csv".format(
                        model=model,
                        region=region,
                        ssp=ssp,
                        gwl=gwl
                    )
                    
                    grid_return_results_path = join(output_path, 'grid_returns', filename)

                    if not os.path.exists(join(output_path, 'grid_returns')):
                        os.makedirs(join(output_path, 'grid_returns'), exist_ok=True)

                    # Check if the files exist from previous runs
                    if os.path.isfile(grid_return_results_path):
                        print("Reading grid return file for {region} region - {model} model - {ssp} scenario - GWL{gwl}C".format(
                        model=model,
                        region=region,
                        ssp=ssp,
                        gwl=gwl))
                        
                        grids_dataframe = pd.read_csv(grid_return_results_path)
                        region_average = grids_to_region_average(grids_dataframe)

                        # Append model average of region for GWLs
                        region_result_collector.append(region_average)
                    
                    else:
                        # Skip ssp and gwl if not exist
                        mid_df = df.loc[df.exp.isin([ssp]) & df.gwl.isin([gwl])]
                        if mid_df.empty:
                            print("{SSP} does not exceed GWL{GWL}".format(SSP=ssp, GWL=gwl))
                            continue
                                                
                        grid_results_list = []
                        for grid in grids:
                            grid_historical_df = df.loc[
                                df.grid_number.isin([grid]) 
                                & df.exp.isin(['historical'])].reset_index()
                            
                            grid_future_df = df.loc[
                                df.grid_number.isin([grid]) & 
                                (df.exp.isin([ssp]) & df.gwl.isin([gwl]))].reset_index()
                            
                            grid_df = pd.concat([grid_historical_df, grid_future_df])
                            
                            historical_n_comp = grid_df.loc[grid_df['exp'] == 'historical']['n_comp'].values[0]
                            future_n_comp = grid_df.loc[grid_df['exp'] == ssp]['n_comp'].values[0]
                            
                            historical_hot_mean_string = 'mean_' + str(historical_n_comp)
                            historical_hot_stdev_string = 'stdev_' + str(historical_n_comp)
                            historical_hot_weight_string = 'weight_' + str(historical_n_comp)
                            
                            historical_hot_mean = grid_df.loc[grid_df['exp'] == 'historical'][historical_hot_mean_string].values[0] 
                            historical_hot_stdev = grid_df.loc[grid_df['exp'] == 'historical'][historical_hot_stdev_string].values[0] 
                            historical_hot_weight = grid_df.loc[grid_df['exp'] == 'historical'][historical_hot_weight_string].values[0] 
                            historical_cold_mean = grid_df.loc[grid_df['exp'] == 'historical']['mean_1'].values[0] 
                            historical_cold_stdev = grid_df.loc[grid_df['exp'] == 'historical']['stdev_1'].values[0] 
                            historical_cold_weight = grid_df.loc[grid_df['exp'] == 'historical']['weight_1'].values[0] 
                            
                            future_hot_mean_string = 'mean_' + str(future_n_comp)
                            future_hot_stdev_string = 'stdev_' + str(future_n_comp)
                            future_hot_weight_string = 'weight_' + str(future_n_comp)
                            future_hot_mean = grid_df.loc[grid_df['exp'] == ssp][future_hot_mean_string].values[0] 
                            future_hot_stdev = grid_df.loc[grid_df['exp'] == ssp][future_hot_stdev_string].values[0] 
                            future_hot_weight = grid_df.loc[grid_df['exp'] == ssp][future_hot_weight_string].values[0] 
                            future_cold_mean = grid_df.loc[grid_df['exp'] == ssp]['mean_1'].values[0] 
                            future_cold_stdev = grid_df.loc[grid_df['exp'] == ssp]['stdev_1'].values[0] 
                            future_cold_weight = grid_df.loc[grid_df['exp'] == ssp]['weight_1'].values[0] 
                            
                            if historical_n_comp == future_n_comp:
                                historical_hot_period_length = (
                                    grid_df.loc[grid_df['exp'] == 'historical']['raw_data_length'].values[0] * grid_df.loc[grid_df['exp'] == 'historical'][historical_hot_weight_string].values[0])\
                                / round(((grid_df.loc[grid_df['exp'] == 'historical']['raw_data_length'].values[0]) / 365))

                                future_hot_period_length = (grid_df.loc[grid_df['exp'] == ssp]['raw_data_length'].values[0] 
                                                        * grid_df.loc[grid_df['exp'] == ssp][historical_hot_weight_string].values[0])  / 20

                                historical_mean_difference = historical_hot_mean  - historical_cold_mean 
                                future_mean_difference = future_hot_mean - future_cold_mean
                                change_in_mean_diff = future_mean_difference - historical_mean_difference
                                temp = {
                                    'dataset': model,
                                    'region': region,
                                    'exp': ssp,
                                    'gwl': gwl,
                                    'grid_number': grid,
                                    'lon': grid_df.loc[grid_df['exp'] == 'historical']['lon'].values[0] ,
                                    'lat': grid_df.loc[grid_df['exp'] == 'historical']['lat'].values[0] ,

                                    'historical_time_range': grid_df.loc[grid_df['exp'] == 'historical']['raw_data_length'].values[0] ,
                                    'historical_n_comp': historical_n_comp,
                                    'historical_mean_cold': historical_cold_mean,
                                    'historical_stdev_cold': historical_cold_stdev ,
                                    'historical_weight_cold': historical_cold_weight,
                                    'historical_mean_hot': historical_hot_mean,
                                    'historical_stdev_hot': historical_hot_stdev ,
                                    'historical_weight_hot': historical_hot_weight ,

                                    'future_time_range': grid_df.loc[grid_df['exp'] == ssp]['raw_data_length'].values[0] ,
                                    'future_n_comp': future_n_comp,
                                    'future_mean_cold': future_cold_mean ,
                                    'future_stdev_cold': future_cold_stdev ,
                                    'future_weight_cold': future_cold_weight,
                                    'future_mean_hot': future_hot_mean,
                                    'future_stdev_hot': future_hot_stdev,
                                    'future_weight_hot': future_hot_weight,

                                    'historical_hot_period_length': historical_hot_period_length,
                                    'future_hot_period_length': future_hot_period_length,
                                    'historical_mean_difference': historical_mean_difference,
                                    'future_mean_difference': future_mean_difference,
                                    'change_in_mean_diff': change_in_mean_diff,
                                    }
                                
                                for n in [1, 5, 10, 20, 30]:
                                    event_name = str(n) + '-year'
                                    # paste functions to https://latex.codecogs.com/eqneditor/editor.php

                                    # expected frequency of n-year events in the past
                                    # f_{n}^{historical} =
                                    # n * |\mathcal{N}(\mu_{hot}^{historical}, \sigma_{hot}^{historical})|
                                    historical_return_period_day = n * historical_hot_period_length

                                    # sigma range of n-year event
                                    # x^{historical} = \textup{erf}^{-1} \left( 1 - \frac{1}{f_n} \right) \ sqrt2
                                    historical_sigma_range = math.sqrt(2) * sps.erfinv(1 - (1 / historical_return_period_day))
                                
                                    # temperature threshold of range
                                    # \tau = \mu_{hot}^{historical} + x^{historical} * \sigma_{hot}^{historical}
                                    tau = historical_hot_mean + (historical_sigma_range * historical_hot_stdev)
                                    
                                    # future sigma range for n-year event
                                    # x^{future} = \frac{\tau - \mu_{hot}^{future}}{\sigma_{hot}^{future}}
                                    future_sigma_range = (tau - future_hot_mean) / future_hot_stdev
                                    
                                    # future frequency of historical n-year event
                                    # f_{\dot{n}}^{future} =
                                    # \frac{1}{1 - \textup{erf}\left(\frac{x^{future}}{\sqrt2} \right )}
                                    future_return_period_day = 1 / (1 - sps.erf(future_sigma_range / math.sqrt(2)))

                                    # future n-year event value
                                    # \dot{n} =
                                    # \frac{f_{\dot{n}}^{future}}{|\mathcal{N}(\mu_{hot}^{future}, \sigma_{hot}^{future})|}
                                    future_return_period_n = future_return_period_day / future_hot_period_length
                                    if future_return_period_day > n * future_hot_period_length:
                                        return_dict = {event_name + "_historical_freq": None,
                                                    event_name + "_event_threshold_temp": None,
                                                    event_name + "_future_freq_day": None,
                                                    n: None,
                                                    }
                                    else:
                                        return_dict = {event_name + "_historical_freq": historical_return_period_day,
                                                    event_name + "_event_threshold_temp": tau,
                                                    event_name + "_future_freq_day": future_return_period_day,
                                                    n: future_return_period_n,
                                                    }
                                    temp.update(return_dict)
                                    
                            else:
                                historical_mean_difference = historical_hot_mean - historical_cold_mean
                                future_mean_difference = future_hot_mean - future_cold_mean     
                                temp = {
                                    'dataset': model,
                                    'region': region,
                                    'exp': ssp,
                                    'gwl': gwl,
                                    'grid_number': grid,
                                    'lon': grid_df.loc[grid_df['exp'] == 'historical']['lon'].values[0] ,
                                    'lat': grid_df.loc[grid_df['exp'] == 'historical']['lat'].values[0] ,

                                    'historical_time_range': grid_df.loc[grid_df['exp'] == 'historical']['raw_data_length'].values[0] ,
                                    'historical_n_comp': historical_n_comp,
                                    'historical_mean_cold': historical_cold_mean,
                                    'historical_stdev_cold': historical_cold_stdev ,
                                    'historical_weight_cold': historical_cold_weight,
                                    'historical_mean_hot': historical_hot_mean,
                                    'historical_stdev_hot': historical_hot_stdev ,
                                    'historical_weight_hot': historical_hot_weight ,

                                    'future_time_range': grid_df.loc[grid_df['exp'] == ssp]['raw_data_length'].values[0] ,
                                    'future_n_comp': future_n_comp,
                                    'future_mean_cold': future_cold_mean ,
                                    'future_stdev_cold': future_cold_stdev ,
                                    'future_weight_cold': future_cold_weight,
                                    'future_mean_hot': future_hot_mean,
                                    'future_stdev_hot': future_hot_stdev,
                                    'future_weight_hot': future_hot_weight,

                                    'historical_mean_difference': historical_mean_difference,
                                    'future_mean_difference': future_mean_difference,
                                    }
                            
                            # Add grid results to main list 
                            grid_results_list.append(temp)
                        
                        # Convert all grid cell results for a model from list to Dataframe and save
                        grids_dataframe = pd.DataFrame(grid_results_list)
                        grids_dataframe.to_csv(grid_return_results_path, index_label='row_number')
                        
                        # Calculate model average from  grid cells
                        region_average = grids_to_region_average(grids_dataframe)    
                        region_result_collector.append(region_average)
        
    region_results = pd.DataFrame(region_result_collector)
    filename = "{region}.csv".format(region=region)
    region_result_filename = join(output_path, filename)
    region_results.to_csv(region_result_filename)                

    return


def main(): 
    # Choose the path for gmm_analysis results
    
    start_time = time.time()
    folder_path = '/work/bd1083/b309178/gmmDiag_esmvaltool/recipe_gmm_ssp_20220903_113433/plots/extract_WSB_region/gmm_analysis_20220906_191030/'
    
    files = list_files_in_directory(folder_path)
    
    # Path for saving output
    # parent_directory_path = '/mnt/d/Documents/DATA/'
    parent_directory_path = '/work/bd1083/b309178/gmmDiag_esmvaltool/return_analysis/'
    dt_string = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = join(parent_directory_path, 'return_analysis', dt_string)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mp_list = []
    for k, v in files.items():
        mp_list.append([{k: v}, output_path])
    
    n_jobs = 256
    with parallel_backend('loky', n_jobs=n_jobs):
        mp_val = Parallel(verbose=10)(delayed(_calculate_return_for_grid_cell)(i) for i in mp_list)
    mp_val = list([x for x in mp_val if x is not None])
    
    print('END')
    print(str(datetime.timedelta(seconds=time.time() - start_time)))


if __name__ == '__main__':
    main()
