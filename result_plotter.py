import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import parallel_backend, Parallel, delayed
from scipy import special as sps
from utils._plot_functions import list_item_to_df, single_global_nyear_plot_exp, grid_modality_calculator, grid_item_to_df, region_nyear_on_map, global_nyear_plot_exp, region_n_components_percentage_from_grids, multi_model_peak_change_plot_cold_ordered_from_grids, multi_model_peak_change_plot_from_grids
import datetime as dt
from collections import defaultdict


# collect all files in a folder
def _file_collector(filepath):
    print('file collector')
    onlyfiles = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    return onlyfiles


def main():

    # define paths for input and output
    # region_files = "/mnt/d/Google Drive/PHD/1st paper/DATA/return_analysis/2022-09-13_13-28-26/"
    # grid_files = "/mnt/d/Google Drive/PHD/1st paper/DATA/return_analysis/2022-09-13_13-28-26/grid_returns/"
    
    region_files = "/mnt/c/Users/paca_ay/Documents/DATA/return_analysis/2022-09-13_13-28-26/"
    grid_files = "/mnt/c/Users/paca_ay/Documents/DATA/return_analysis/2022-09-13_13-28-26/grid_returns/"

    # create output path
    output_path = join(region_files, 'plots/')
    # create a folder with the current date and time
    dt_string = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    plot_path = join(output_path, dt_string)
    
    # check if paths exist, if not create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # define output paths for dataframes
    grid_final_df_output_path = join(output_path , 'all_grid_cells.csv')
    region_final_df_output_path = join(output_path , 'all_regions.csv')

    # check if dataframes exist, if exist read them, if not create them
    if os.path.isfile(region_final_df_output_path):
        print("Read data for all regions")
        region_final_df = pd.read_csv(region_final_df_output_path)
        
    else:
        region_files = _file_collector(region_files)
        region_file_list = [[file, output_path] for file in region_files]
    
        with parallel_backend('loky', n_jobs=6):
            mp_val = Parallel(verbose=10)(delayed(list_item_to_df)(i) for i in region_file_list)
        mp_val = list([x for x in mp_val if x is not None])
        region_final_df = pd.concat(mp_val, ignore_index=True)
        region_final_df.to_csv(region_final_df_output_path)
        
    # Plot regional n-year return period subplots on a world map for GWL 1.5, 2, 3 and 4
    region_nyear_on_map(region_final_df, output_path=plot_path)
    
    # Plot global n-year return period boxplots for GWL 1.5, 2, 3 and 4
    global_nyear_plot_exp(region_final_df, output_path=plot_path)
    single_global_nyear_plot_exp(region_final_df, output_path=output_path)
    

    # check if dataframes exist, if exist read them, if not create them
    if os.path.isfile(grid_final_df_output_path):
        print("Reading data for all grid cells")
        grid_final_df = pd.read_csv(grid_final_df_output_path)
    else:
        grid_files = _file_collector(grid_files)
        grid_file_list = []
        for file in grid_files:
            region =  Path(file).stem.split('_')[0]
            model = Path(file).stem.split('_')[1]
            exp =  Path(file).stem.split('_')[2]
            gwl =  Path(file).stem.split('_')[3]
            grid_file_list.append([file, output_path, region, model, exp, gwl])

        with parallel_backend('loky', n_jobs=6):
            mp_val = Parallel(verbose=10)(delayed(grid_item_to_df)(i) for i in grid_file_list)
        mp_val = list([x for x in mp_val if x is not None])
        grid_final_df = pd.concat(mp_val, ignore_index=True)
        grid_final_df.to_csv(grid_final_df_output_path)

    
    # dark blue-light blue bar plot, percentage of region grid modalities
    region_n_components_percentage_from_grids(grid_final_df, plot_path)
    grid_modality_calculator(grid_final_df, plot_path)

    # Bar plot, showing peak diff and meand for regions grouped by continent
    multi_model_peak_change_plot_from_grids(grid_final_df, plot_path)
    multi_model_peak_change_plot_cold_ordered_from_grids(grid_final_df, plot_path)


if __name__ == '__main__':
    main()
