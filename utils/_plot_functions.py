from cgitb import text
import json
import math

import numpy as np
import pandas as pd
from matplotlib import cm, colors
import matplotlib as mpl
import datetime as dt


def ensemble_mean(data_list):
    df = pd.read_csv(data_list[0], index_col=0)
    output_path = data_list[1]
    temp = []
    for gwl in ['gwl15', 'gwl20', 'gwl30']:
        df_gwl = df.loc[(df.gwl == gwl)]
        region = list(set(df_gwl['region'].to_list()))[0]
        mean_df = df_gwl.mean().to_frame().T
        mean_df.insert(0, 'region', '')
        mean_df.insert(1, 'gwl', '')
        mean_df['region'] = region
        mean_df['gwl'] = gwl
        temp.append(mean_df)

    final_df = pd.concat(temp, ignore_index=True)
    return final_df


def figure_saving(plt, plot_path, savefilename):
    plt.tight_layout()
    savefilename = plot_path + savefilename  
    plt.savefig(savefilename + '.jpg', facecolro='white', transparent=False)
    plt.savefig(savefilename + '.pdf', facecolor='white')
    plt.close()
    return


def redefine_centeroid(region_shape):
    if region_shape.attributes['Acronym'] == 'ARP':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 12
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 8
    if region_shape.attributes['Acronym'] == 'CAU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 20
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 5
    if region_shape.attributes['Acronym'] == 'CAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 5
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 8
    if region_shape.attributes['Acronym'] == 'CAR':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 2
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 4
    if region_shape.attributes['Acronym'] == 'CNA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 2
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'ECA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 18
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 12
    if region_shape.attributes['Acronym'] == 'EAS':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 15
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 4
    if region_shape.attributes['Acronym'] == 'EAU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 8
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 3
    if region_shape.attributes['Acronym'] == 'EEU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 5
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 5
    if region_shape.attributes['Acronym'] == 'ENA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 6
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 6
    if region_shape.attributes['Acronym'] == 'ESB':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 7
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 3
    if region_shape.attributes['Acronym'] == 'ESAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 4
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 20
    if region_shape.attributes['Acronym'] == 'GIC':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord']
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 5
    if region_shape.attributes['Acronym'] == 'MED':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 5
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 7
    if region_shape.attributes['Acronym'] == 'MDG':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 18
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 6
    if region_shape.attributes['Acronym'] == 'NAU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord']
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 3
    if region_shape.attributes['Acronym'] == 'NCA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 13
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 11
    if region_shape.attributes['Acronym'] == 'NEAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 4
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'NEN':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 8
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 2
    if region_shape.attributes['Acronym'] == 'NES':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 8
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'NEU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord']
    if region_shape.attributes['Acronym'] == 'NSA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 2
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'NWS':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 11
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 7
    if region_shape.attributes['Acronym'] == 'NZ':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 2
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 8
    if region_shape.attributes['Acronym'] == 'RAR':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 9
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] + 1
    if region_shape.attributes['Acronym'] == 'RFE':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 5
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 5
    if region_shape.attributes['Acronym'] == 'SAH':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 25
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'SAU':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord']
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 9
    if region_shape.attributes['Acronym'] == 'SAM':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 3
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 10
    if region_shape.attributes['Acronym'] == 'SAS':
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 25
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 5
    if region_shape.attributes['Acronym'] == 'SEAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 2
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 13
    if region_shape.attributes['Acronym'] == 'SCA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 4
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord']
    if region_shape.attributes['Acronym'] == 'SES':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 17
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 9
    if region_shape.attributes['Acronym'] == 'SSA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 8
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 10
    if region_shape.attributes['Acronym'] == 'SWS':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 15
    if region_shape.attributes['Acronym'] == 'TIB':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord']
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 8
    if region_shape.attributes['Acronym'] == 'WAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 8
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 12
    if region_shape.attributes['Acronym'] == 'WAN':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 5
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 8
    if region_shape.attributes['Acronym'] == 'WNA':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 2
    if region_shape.attributes['Acronym'] == 'WSAF':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] - 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 10
    if region_shape.attributes['Acronym'] == 'WSB':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord'] - 5
    if region_shape.attributes['Acronym'] == 'WCE':
        region_shape.attributes['xcoord'] = region_shape.attributes['xcoord'] + 10
        region_shape.attributes['ycoord'] = region_shape.attributes['ycoord']

    return region_shape


def list_item_to_df(data_list):
    df = pd.read_csv(data_list[0], index_col=0)
    return df


def grid_item_to_df(data_list):
    print(data_list[2], data_list[3],data_list[4], data_list[5])
    try:
        df = pd.read_csv(data_list[0], index_col=0)
    except pd.errors.EmptyDataError:
      print(str(data_list), " is empty")
    # df['gwl'] = data_list[2]
    return df


def grid_modality_calculator(df, output_path):
    modality_list = []

    exp_list = list(sorted(df.exp.unique()))
    gwl_list = list(sorted(df.gwl.unique()))
    dataset_list = list(sorted(df.dataset.unique()))
    region_list = list(sorted(df.region.unique()))
    
    for region in region_list:
        for exp in exp_list:
            if 'ERA5' in exp:
                continue
            region_exp_file = df.loc[
                        ((df.region == region) & (df.exp == exp))][['dataset', 'region', 'exp', 'gwl', 'grid_number','lon','lat','historical_n_comp', 'future_n_comp' ]]
            for gwl in gwl_list:
                for dataset in dataset_list:
                    print(region, exp, gwl, dataset)
                    region_model_file = region_exp_file.loc[((region_exp_file.gwl == gwl) & (region_exp_file.dataset == dataset))]
                    if region_model_file.empty:
                        print("empty")
                        continue
                    grid_list = list(set(region_model_file['grid_number'].values))
                    grid_count = len(grid_list)
                    uu, ub, bu, bb = 0, 0, 0, 0
                    for grid in grid_list:
                        grid_df = region_model_file.loc[region_model_file['grid_number'] == grid]
                        grid_historical_comp =  grid_df['historical_n_comp'].values[0]
                        grid_future_comp =  grid_df['future_n_comp'].values[0]
                        if grid_historical_comp == grid_future_comp:
                            if grid_historical_comp == 1:
                                uu += 1
                            else:
                                bb += 1
                        else:
                            if grid_historical_comp == 1:
                                ub += 1
                            else:
                                bu += 1

                    temp = [region, dataset, exp, gwl, uu, ub, bu, bb, grid_count, uu/grid_count, ub/grid_count, bu/grid_count, bb/grid_count]
                    modality_list.append(temp)
                    print(temp)
    print(modality_list)
    main_df = pd.DataFrame(modality_list, columns=[
    'region', 'dataset', 'exp', 'gwl', 'uu', 'ub', 'bu', 'bb', 'count', 'uu/gc','ub/gc','bu/gc','bb/gc'])

    main_df.to_csv(output_path + 'grid_count.csv')  
    return modality_list


def single_global_nyear_plot_exp(df, output_path):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import cartopy.io.shapereader as shpreader
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from cartopy.feature import ShapelyFeature
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    
    exp_list = list(sorted(df.exp.unique()))
    gwl_list = list(sorted(df.gwl.unique()))
    # Loop over n-year events
    box_plot_data = []
    
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols,
                       sharex='col', 
                       sharey='row',
                       figsize=(10,10))
    
    for row, n_year in enumerate(['1-year', '5-year', '10-year', '20-year']):
        for col, exp in enumerate(exp_list):
            gwl_holder = []
            
            tick_labels = []
            number_of_models_dict = {}
            for gwl in gwl_list:
                gwl_holder.append(df.loc[(df.gwl == gwl) & (df.exp == exp)][n_year + '_freq_day_mean'])
                number_of_models = len(list(sorted(df.loc[(df.gwl == gwl) & (df.exp == exp)]["dataset"].unique())))
                number_of_models_dict[gwl] = number_of_models
                tick_label = '{gwl}$^\circ$C\n({number_of_ds})'.format(gwl=gwl, number_of_ds=number_of_models)
                tick_labels.append(tick_label)
                print(tick_label)
            gwl_glob_plot = axes[row, col].boxplot(
                gwl_holder,
                labels=['1.5$^\circ$C', '2.0$^\circ$C', '3.0$^\circ$C', '4.0$^\circ$C'],
                vert=True,
                patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'pink', 'sienna']
            
            for patch, color in zip(gwl_glob_plot['boxes'], colors):
                patch.set_facecolor(color)

    for ax, col in zip(axes[0], exp_list):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], ['1-year', '5-year', '10-year', '20-year']):
        ax.set_ylabel(row, rotation=90, size='large', labelpad=10) 
        
        # historical_length = df.loc[df.exp == exp]['historical_hot_period_length'].median()    
    # plt.suptitle('Global {nyear} return periods (Base {day} days)'.format(day=int(historical_length) * int(n_year.split('-')[0]), nyear=n_year))
    fig.supylabel('Occurrence (days)')
    plt.xticks(rotation=90, fontsize='large')
    savefilename = output_path + 'global_allssp.jpg'
    figure_saving(plt, output_path, savefilename)
        

def global_nyear_plot_exp(df, output_path):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import cartopy.io.shapereader as shpreader
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from cartopy.feature import ShapelyFeature
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    
    exp_list = list(sorted(df.exp.unique()))
    gwl_list = list(sorted(df.gwl.unique()))
    colormaps_list = ['Blues', 'Greens', 'Wistia', 'Reds']
    
    def ytick_arange(max, step):
        step = step - 1
        if int(max / step) <= 100:
            value = int(max / step) - int(max / step) % 10
        elif 100 < int(max / step) <= 1000:
            value = int(max / step) - int(max / step) % 100
        elif int(max / step) > 1000:
            value = int(max / step) - int(max / step) % 1000
        else:
            value = int(max / step)
        return value
    
    # Loop over n-year events
    box_plot_data = []
    for n_year in ['1-year', '5-year', '10-year', '20-year']:
        
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        labels = ['a)', 'b)', 'c)', 'd)']
        for exp, ax, colormap_name, label in zip(exp_list, axs.ravel(), colormaps_list, labels):
            gwl_holder = []
            
            tick_labels = []
            number_of_models_dict = {}
            ax_max = df[n_year + '_freq_day_mean'].max()
            for gwl in gwl_list:
                gwl_holder.append(df.loc[(df.gwl == gwl) & (df.exp == exp)][n_year + '_freq_day_mean'])
                number_of_models = len(list(sorted(df.loc[(df.gwl == gwl) & (df.exp == exp)]["dataset"].unique())))
                number_of_models_dict[gwl] = number_of_models
                tick_label = '{gwl}\n({number_of_ds})'.format(gwl=gwl, number_of_ds=number_of_models)
                tick_labels.append(tick_label)
                print(tick_label)
            gwl_glob_plot = ax.boxplot(
                gwl_holder,
                labels=['1.5', '2.0', '3.0', '4.0'],
                vert=True,
                patch_artist=True)
            
            ax.set_ylim(0, ax_max)
            
            ax.set_xticklabels(tick_labels, rotation='vertical', fontsize=20)
            ax.set_yticks(np.arange(0, ax_max, ytick_arange(ax_max, 6)))
            ax.tick_params(axis='y', labelsize=20)
            # ax.set_title('{exp}'.format(exp=exp[:3].upper()+exp[3]+'-'+exp[4]+'.'+exp[5]), fontsize=20)
            ax.set_title(label, fontfamily='serif', loc='left', fontsize='xx-large')
            cmap = cm.get_cmap(colormap_name)
            ssp_colors = iter(cmap(np.linspace(0.4, 1.2, len(gwl_list))))
            for patch, color in zip(gwl_glob_plot['boxes'], ssp_colors):
                patch.set_facecolor(color)
                
        historical_length = df.loc[df.exp == exp]['historical_hot_period_length'].median()    
        # plt.suptitle('Global {nyear} return periods (Base {day} days)'.format(day=int(historical_length) * int(n_year.split('-')[0]), nyear=n_year), fontsize=20)
        fig.supylabel('Occurrence (days)', fontsize=20)
        fig.supxlabel('GWL ($^\circ$C)', fontsize=20)
        savefilename = 'global_{n_year}'.format(n_year=n_year)
        figure_saving(plt, output_path, savefilename)    


def region_nyear_on_map(df, output_path):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import cartopy.io.shapereader as shpreader
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from cartopy.feature import ShapelyFeature
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt

    small_figure_width = 2
    colormaps_list = ['Blues', 'Greens', 'Wistia', 'Reds']

    def get_box_plot_data(region, n_year, labels, bp, exp):
        rows_list = []
        for i in range(len(labels)):
            dict1 = {}
            dict1['label'] = labels[i]
            dict1['region'] = region
            dict1['exp'] = exp
            dict1['n_year'] = n_year
            dict1['lower_whisker'] = bp['whiskers'][i * 2].get_ydata()[1]
            dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
            dict1['median'] = bp['medians'][i].get_ydata()[1]
            dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
            dict1['upper_whisker'] = bp['whiskers'][(i * 2) + 1].get_ydata()[1]
            rows_list.append(dict1)
        return pd.DataFrame(rows_list)


    def ytick_arange(max, step):
        step = step - 1
        if int(max / step) <= 100:
            value = int(max / step) - int(max / step) % 10
        elif 100 < int(max / step) <= 1000:
            value = int(max / step) - int(max / step) % 100
        elif int(max / step) > 1000:
            value = int(max / step) - int(max / step) % 1000
        else:
            value = int(max / step)
        return value


    exp_list = list(sorted(df.exp.unique()))
    gwl_list = list(sorted(df.gwl.unique()))
    # Loop over n-year events
    box_plot_data = []
    for n_year in ['1-year', '5-year', '10-year', '20-year']:
        for exp, colormap_name in zip(exp_list, colormaps_list):
            proj = ccrs.PlateCarree()
            ipcc_ax = plt.axes(projection=proj)
            # Add land and ocean
            land_reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/ne_110m_land.shp')
            ocean_reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/ne_110m_ocean.shp')
            land = list(land_reader.geometries())
            ocean = list(ocean_reader.geometries())
            LAND = cf.ShapelyFeature(land, proj)
            OCEAN = cf.ShapelyFeature(ocean, proj)

            # Get region names
            regions = list(set(df['region'].to_list() + ['RAR', 'EAN']))

            # Set color of land and ocean
            facecolor_land = (252 / 255, 255 / 255, 232 / 255, 1)
            facecolor_ocean = (201 / 255, 250 / 255, 255 / 255, 1)

            # Add land and ocean to figure
            ipcc_ax.add_feature(LAND, facecolor=facecolor_land, edgecolor=None)
            ipcc_ax.add_feature(OCEAN, facecolor=facecolor_ocean, edgecolor=None)

            # Make figure larger
            plt.gcf().set_size_inches(36, 18)

            # Read IPCC regions shape file
            reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/only_lands_id_order_centeroids.shp')

            small_hist_width = 0.2
            xpos = [0, 0.05]
            # Loop over region keys and data values
            ax_max = df[n_year + '_freq_day_mean'].max()
         
            for region in sorted(regions):
                # Read region shape from shpfile
                region_shape = [rgnshp for rgnshp in reader.records() if rgnshp.attributes["Acronym"] == region][0]
                # Draw shape
                shape_feature = ShapelyFeature([region_shape.geometry],
                                            proj,
                                            facecolor=(0, 0, 0, 0),
                                            edgecolor=(150 / 255, 150 / 255, 150 / 255, 0.4),
                                            lw=1)
                # Add shape to figure
                ipcc_ax.add_feature(shape_feature, zorder=2)
                redefine_centeroid(region_shape)
                # Add overlay axis for small plots
                ax_h = inset_axes(ipcc_ax, width=small_figure_width,
                                height=small_figure_width,
                                loc=3,
                                bbox_to_anchor=(
                                    region_shape.attributes['xcoord'] - 10, region_shape.attributes['ycoord'] - 6),
                                bbox_transform=ipcc_ax.transData,
                                borderpad=0,
                                axes_kwargs={'alpha': 0.35, 'visible': True})
                ax_title = region_shape.attributes['Acronym']
                # ax_h.set_title(ax_title, fontsize='xx-large')
                ax_h.set_ylim(0, ax_max)
                ax_h.text(0.7, 0.6, ax_title, 
                          horizontalalignment='center', 
                          verticalalignment='center', 
                          fontsize=22, 
                          transform=ax_h.transAxes, 
                          rotation=315, 
                          color='k', 
                          alpha=0.5)
                ax_h.axes.xaxis.set_visible(False)
                ax_h.tick_params(axis='y', labelsize='x-large')
                ax_h.patch.set_alpha(0.6)

                ax_h.set_yticks(np.arange(0, ax_max, ytick_arange(ax_max, 6)))

                # if region in ['RAR', 'EAN']:
                #      continue
                # Get all GWL data for region
                gwl_holder = []
                historical_length_mean = df.loc[(df.region == region) & (df.exp == exp)]['historical_hot_period_length'].mean()
                historical_length = df.loc[(df.region == region)  & (df.exp == exp)]['historical_hot_period_length'].median()
                
                for gwl in gwl_list:
                    gwl_holder.append(df.loc[((df.region == region) & (df.gwl == gwl)) & (df.exp == exp)][n_year + '_freq_day_mean'])
                    
                gwl_plot = ax_h.boxplot(
                    gwl_holder,
                    vert=True,
                    labels=['1.5$^\circ$C', '2.0$^\circ$C', '3.0$^\circ$C', '4.0$^\circ$C'],
                    patch_artist=True) # Make this True and uncomment below for loop to extract data for box plot

                # box_plot_data.append(get_box_plot_data(region,
                #                                        n_year,
                #                                        ['GWL 1.5$^\circ$C', 'GWL 2.0$^\circ$C', 'GWL 3.0$^\circ$C', 'GWL 4.0$^\circ$C'],
                #                                        gwl_plot,
                #                                        exp
                #                                        )
                #                      )

                # colors = ['lightblue', 'lightgreen', 'pink']
                cmap = cm.get_cmap(colormap_name)
                ssp_colors = iter(cmap(np.linspace(0.4, 1.2, len(gwl_list))))
                for patch, color in zip(gwl_plot['boxes'], ssp_colors):
                    patch.set_facecolor(color)
                
                gwl_holder =  [item for item in gwl_holder if not item.empty]
                ax_h.text(0.17,
                        0.9,
                        '{day} days'.format(day=int(historical_length) * int(n_year.split('-')[0])),
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3},
                        fontsize='xx-large',
                        transform=ax_h.transAxes)

            # ipcc_ax.set_title('Future return periods of {n_year} events compared to the 1980-2010 period under \n'
            #                   '1.5, 2, 3 and 4°C global warming levels relative to 1850-1900 baseline for {exp} scenario'.format(
            #     n_year=n_year, exp=exp), fontsize=32)
            # ipcc_ax.set_title('{n_year} events for {exp} scenario'.format(n_year=n_year, exp=exp[:3].upper()+exp[3]+'-'+exp[4]+'.'+exp[5]), fontsize=36)
            
            # Add global plot
            ax_glob = inset_axes(ipcc_ax, width=4,
                                height=5.5,
                                loc=3,
                                bbox_to_anchor=(-165, -55),
                                bbox_transform=ipcc_ax.transData,
                                borderpad=0,
                                axes_kwargs={'alpha': 0.35, 'visible': True})

            ax_glob.patch.set_alpha(0.6)
            gwl_holder = []
            number_of_models_dict = {}
            for gwl in gwl_list:
                gwl_holder.append(df.loc[(df.gwl == gwl) & (df.exp == exp)][n_year + '_freq_day_mean'])
                number_of_models = len(list(sorted(df.loc[(df.gwl == gwl) & (df.exp == exp)]["dataset"].unique())))
                number_of_models_dict[gwl] = number_of_models
            
            string_datasets = 'Number of datasets: '+ ' '.join(f'GWL{key}$^\circ$C = {value}, ' for key, value in number_of_models_dict.items())
                
            # ipcc_ax.annotate(text=string_datasets,
            #              xy=(20,-50), 
            #              xycoords='axes pixels', 
            #              fontsize=24, 
            #              ) 
            
            gwl_labels = [f'{key}$^\circ$C({value})' for key, value in number_of_models_dict.items()]
            gwl_glob_plot = ax_glob.boxplot(
                gwl_holder,
                labels=gwl_labels,
                vert=True,
                patch_artist=True)
            # box_plot_data.append(get_box_plot_data('global',
            #                                            n_year,
            #                                            ['GWL 1.5$^\circ$C', 'GWL 2.0$^\circ$C', 'GWL 3.0$^\circ$C', 'GWL 4.0$^\circ$C'],
            #                                            gwl_glob_plot,
            #                                            exp
            #                                            )
            #                          )
            
            colors = ['lightblue', 'lightgreen', 'pink', 'sienna']
            ax_glob.set_xticklabels(gwl_labels,
                                    rotation='vertical',
                                    fontsize=24)
            ax_glob.tick_params(axis='y', labelsize=24)
            
            cmap = cm.get_cmap(colormap_name)
            ssp_colors = iter(cmap(np.linspace(0.4, 1.2, len(gwl_list))))
            for patch, color in zip(gwl_glob_plot['boxes'], ssp_colors):
                patch.set_facecolor(color)
            ax_glob.set_ylim(0, ax_max)

            ax_glob.set_yticks(np.arange(0, ax_max, ytick_arange(ax_max, 6)))
            ax_glob.set_ylabel('Occurrence (days)', fontsize='22')
            ax_glob.set_title('Global', fontsize=22)
            historical_length = df.loc[df.exp == exp]['historical_hot_period_length'].median()
            ax_glob.text(0.31,
                        0.94,
                        'Base {day} days'.format(day=int(historical_length) * int(n_year.split('-')[0])),
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3},
                        fontsize=24,
                        transform=ax_glob.transAxes)

            savefilename = 'All_IPCC_return_periods_freq_{n_year}_{exp}'.format(n_year=n_year, exp=exp)
            figure_saving(plt, output_path, savefilename)
    # box_plot_data = pd.concat(box_plot_data, axis=0, ignore_index=True)
    # box_plot_data.to_csv(output_path + 'box_plot_date.csv')


def region_n_components_percentage_from_grids(df, output_path):
    import matplotlib.pyplot as plt
    import cartopy.io.shapereader as shpreader
    from matplotlib import ticker

    title_fontsize = 18
    legend_fontsize = 14
    box_fontsize = 13
    region_fontsize = 12
    continent_fontsize = 12
    ytick_fontsize = 12
    ylabel_fontsize = 16
    
    reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/only_lands_id_order_centeroids.shp')
    historical_flag = False
    modality_list = []
    exp_list = ['historical']
    exp_list.extend(list(sorted(df.exp.unique())))
    gwl_list = list(sorted(df.gwl.unique()))
    for exp in exp_list:
        for gwl in gwl_list:
            if exp == 'historical':
                df_gwl = df.loc[(df.gwl == gwl) & (df.exp == 'ssp585')]
            else:
                df_gwl = df.loc[(df.gwl == gwl) & (df.exp == exp)]
            if df_gwl.empty:
                continue
            region_shape = [rgnshp for rgnshp in reader.records()]
            continent_regions = {}
            for item in region_shape:
                if item.attributes['Continent'] in continent_regions:
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])
                else:
                    continent_regions[item.attributes['Continent']] = []
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])

            #print(continent_regions)

            add_legend = True
            if historical_flag and exp == 'historical':
                continue
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            x_end = 0
            ax_x_pos = []
            continent_x_pos = []
            region_labels = []
            continent_labels = []
            x_ticker = [-1]

            for continent, regions in continent_regions.items():
                gwl_df = []

                continent_labels.append(continent.replace('-','\n'))

                for region in regions:
                    print(region)
                    try:
                        historical_components = df_gwl.loc[(df_gwl.region == region), ['historical_n_comp']].value_counts(normalize=True)*100
                        future_components = df_gwl.loc[(df_gwl.region == region), ['future_n_comp']].value_counts(normalize=True)*100

                        data = {'historical_one_component': historical_components.loc[(1,)], 
                                'historical_two_component': historical_components.loc[(2,)], 
                                'future_one_component': future_components.loc[(1,)], 
                                'future_two_component': future_components.loc[(2,)]}
                        temp = pd.DataFrame([data])
                    except:
                        data = {'historical_one_component': 0, 
                                'historical_two_component': 0, 
                                'future_one_component':0, 
                                'future_two_component':0   }
                        temp = pd.DataFrame([data])
                    temp['region'] = region
                    temp['Continent'] = continent.replace('-','\n')
                    temp['gwl'] = gwl
                    temp['exp'] = exp

                    gwl_df.append(temp)
                    
                
                main_df = pd.concat(gwl_df)
                modality_list.append([gwl, exp, main_df])
                width = 0.5
                x = np.arange(x_end, x_end + len(regions), 1)
               
                ax_x_pos.extend(x)
                continent_x_pos.append((x_end-1 + x_end + len(regions)) / 2 )
                x_end = max(x) + 2
                x_ticker.extend([x_end-1])
                main_df = main_df.sort_values(by=['historical_one_component',])
                region_labels.extend(main_df['region'])
                
                if exp != 'historical':
                    label_exp = 'Future'
                else:
                    label_exp = 'Historical'
                print(len(x))
                print(x)
                print(len(main_df['future_one_component']))
                print(main_df['future_one_component'])
                # if 47 in x:
                #     x = x[:-1]
                ax.bar(x,
                    main_df['future_one_component'],
                    width,
                    label=label_exp + ' unimodal'if add_legend else "",
                    align='center',
                    color='royalblue',
                    zorder=8)
                ax.bar(x,
                    main_df['future_two_component'],
                    width,
                    align='center',
                    bottom=main_df['future_one_component'],
                    label=label_exp + " bimodal" if add_legend else "",
                    color='lightblue',
                    zorder=8)
                add_legend = False
            ax.tick_params(axis='y', which='major', labelsize=ytick_fontsize)
            ax.yaxis.set_ticks_position('both')
            ax.set_xticks(ax_x_pos, region_labels, rotation='vertical', fontsize=region_fontsize)
            ax.set_ylabel("% of grids", fontsize=ylabel_fontsize)
            ax.grid(visible=True, which='major', axis='y', linestyle=':', zorder=0)
            ax.legend(fontsize=legend_fontsize).set_zorder(102)
            if exp == 'historical':
                title= 'Percentages of grid cell modalities for historical period 1980-2010'
            else:
                title= '{exp} scenario under GWL{gwl} $^\circ$C'.format(gwl=gwl, exp=exp[:3].upper()+exp[3]+'-'+exp[4]+'.'+exp[5])
            # plt.title(title, fontsize=title_fontsize)

            global_historical_components = df_gwl['historical_n_comp'].value_counts(normalize=True)*100
            global_future_components = df_gwl['future_n_comp'].value_counts(normalize=True)*100
            
            data = {'historical_one_component': global_historical_components.loc[(1,)], 
                    'historical_two_component': global_historical_components.loc[(2,)], 
                    'future_one_component': global_future_components.loc[(1,)], 
                    'future_two_component': global_future_components.loc[(2,)]}
            global_df = pd.DataFrame([data])
            
            number_of_models = len(list(sorted(df_gwl.dataset.unique())))
            print(number_of_models)
            if exp == 'historical':
                ax.text(0.4,
                    0.8,
                    'Historical unimodal: {histone}%\n'
                    'Historical bimodal: {histtwo}%\n'
                    'Datasets: {number_of_models}'.format(
                        histone=round(global_df['historical_one_component'][0],2),
                        histtwo=round(global_df['historical_two_component'][0],2),
                        number_of_models=number_of_models),
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3},
                    transform=ax.transAxes,
                    zorder=10,
                    fontsize=box_fontsize)
            else:
                ax.text(0.4,
                        0.7,
                        'Historical unimodal: {histone}%\n'
                        'Historical bimodal: {histtwo}%\n'
                        'Future unimodal: {sspone}%\n'
                        'Future bimodal: {ssptwo}%\n'
                        'Datasets: {number_of_models}'.format(
                            histone=round(global_df['historical_one_component'][0], 2),
                            histtwo=round(global_df['historical_two_component'][0], 2),
                            sspone=round(global_df['future_one_component'][0], 2),
                            ssptwo=round(global_df['future_two_component'][0], 2),
                            number_of_models=number_of_models,),
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3},
                        transform=ax.transAxes,
                        zorder=10,
                        fontsize=box_fontsize)

            # Second X-axis
            ax2 = ax.twiny()
            ax2.spines["bottom"].set_position(("axes", -0.20))
            ax2.tick_params('both', length=0, width=0, which='minor')
            ax2.tick_params('both', direction='in', which='major')
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")

            ax2.set_xticks(x_ticker)
            ax2.xaxis.set_major_formatter(ticker.NullFormatter())
            ax2.xaxis.set_minor_locator(ticker.FixedLocator(continent_x_pos))
            ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(continent_labels))
            ax2.set_xbound(ax.get_xbound())
            ax2.tick_params(axis="x", which="both", rotation=90, labelsize=continent_fontsize)
            
            savefilename = 'Modalities_of_regions_{exp}_{gwl}'.format(gwl=gwl, exp=exp) + '_fromgrids'
            figure_saving(plt, output_path, savefilename)
            
            # if exp == 'historical':
            #     historical_flag = True
            #     return
    print(modality_list)
    final_df = []
    for item in modality_list:
        temp = item[2]
        temp['gwl'] = item[0]
        temp['exp'] = item[1]
        final_df.append(temp)
    final_df = pd.concat(final_df)
    final_df.to_csv(output_path + 'modalities_fromgrids.csv')


def multi_model_peak_change_plot_from_grids(df, output_path):
    import matplotlib.pyplot as plt
    import cartopy.io.shapereader as shpreader
    from matplotlib import ticker
    
    cm = 1/2.54
    mm = 1/25.4
    
    results = []

    hot_markersize = 4.0
    cold_markersize = 4.0
    legend_fontsize = 'medium'
    bar_data_fontsize = 'small'
    y_axes_label_fontsize = 'large'
    continent_fontsize = 'small'
    region_fontsize = 'small'
    y_tick_fontsize = 'large'
    title_fontsize = 'x-large'
    
    gwl_list = list(sorted(df.gwl.unique()))
    exp_list = list(sorted(df.exp.unique()))
    for exp in exp_list:
        for gwl in gwl_list:
            df_gwl = df.loc[((df.gwl == gwl) & (df.exp == exp)) & ((df.historical_n_comp == 2) & (df.future_n_comp == 2))]
            reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/only_lands_id_order_centeroids.shp')
            region_shape = [rgnshp for rgnshp in reader.records()]
            continent_regions = {}
            for item in region_shape:
                if item.attributes['Continent'] in continent_regions:
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])
                else:
                    continent_regions[item.attributes['Continent']] = []
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])

            fig, ax1 = plt.subplots(figsize=(230*mm, 130*mm), dpi=300)
            x_end = 0
            region_label_x_pos = []
            continent_label_x_pos = []
            region_labels = []
            continent_labels = []
            continent_x_ticker = []
            add_legend = True

            ax2 = ax1.twinx()
            ax2_color = 'green'
            ax2_color = (0/255,112/255,0/255,0.5)
            ax2.spines['right'].set_color('green')
            xpos = [-0.2, 0.9]
            
            for continent, regions in continent_regions.items():
                continent_labels.append(continent.replace('-','\n'))
                gwl_df = []
                continent_x_ticker.append(xpos[0]-1.95)
                for region in regions:
                    temp = df_gwl.loc[(df_gwl.region == region), ['historical_mean_cold',
                                                                'historical_mean_hot',
                                                                'future_mean_cold',
                                                                'future_mean_hot']].mean().to_frame().T
                    temp['region'] = region
                    temp['Continent'] = continent
                    temp['peak_moving_direction'] = (temp['future_mean_cold'] - temp['historical_mean_cold']) \
                                                    - (temp['future_mean_hot'] - temp['historical_mean_hot'])
                    temp['cold_difference'] = temp['future_mean_cold'] - temp['historical_mean_cold']
                    temp['hot_difference'] = temp['future_mean_hot'] - temp['historical_mean_hot']
                    temp['gwl'] = gwl
                    temp['exp'] = exp
                    gwl_df.append(temp)
                main_df = pd.concat(gwl_df).sort_values(by=['historical_mean_cold'])
                results.append(main_df)
                continent_label_pos_holder = []
                for region in main_df['region']:
                    cold_means = [
                        main_df.loc[(main_df.region == region), 'historical_mean_cold'].values[0],
                        main_df.loc[(main_df.region == region), 'future_mean_cold'].values[0]
                    ]

                    hot_means = [
                        main_df.loc[(main_df.region == region), 'historical_mean_hot'].values[0],
                        main_df.loc[(main_df.region == region), 'future_mean_hot'].values[0]
                    ]

                    historical = [
                        main_df.loc[(main_df.region == region), 'historical_mean_cold'].values[0],
                        main_df.loc[(main_df.region == region), 'historical_mean_hot'].values[0]
                    ]

                    future = [
                        main_df.loc[(main_df.region == region), 'future_mean_cold'].values[0],
                        main_df.loc[(main_df.region == region), 'future_mean_hot'].values[0]
                    ]


                    region_labels.append(region)

                    barpos = sum(xpos) / len(xpos)

                    change_label = "$\Delta T=\Delta \mu_{cold}^{future-historical}$-$\Delta \mu_{hot}^{future-historical}$"
                    cold_diff_label = "$\Delta \mu_{cold}^{future-historical}$"
                    hot_diff_label = "$\Delta \mu_{hot}^{future-historical}$"
                    historical_cold_label = "$\mu_{cold}^{historical}$"
                    historical_hot_label = "$\mu_{hot}^{historical}$"
                    future_cold_label = "$\mu_{cold}^{future}$"
                    future_hot_label = "$\mu_{hot}^{future}$"

                    ax1.plot(
                        (xpos, xpos),
                        ([i for i in hot_means], [j for j in cold_means]),
                        c='grey',
                        lw=0.5,
                        zorder=1)

                    chng = ax2.bar(
                        barpos,
                        main_df.loc[(main_df.region == region), 'peak_moving_direction'].values[
                            0],
                        color=ax2_color,
                        label=change_label if add_legend else "",
                        width=0.8,
                        data=main_df.loc[(main_df.region == region), 'peak_moving_direction'].values[0],
                        zorder=2)

                    # cold_diff = ax2.bar(
                    #     barpos-0.6,
                    #     main_df.loc[(main_df.region == region), 'cold_difference'].values[
                    #         0],
                    #     color='lightblue',
                    #     alpha=0.5,
                    #     label=cold_diff_label if add_legend else "",
                    #     width=0.6,
                    #     data=main_df.loc[(main_df.region == region), 'cold_difference'].values[0],
                    #     zorder=2)
                    #
                    # hot_diff = ax2.bar(
                    #     barpos+0.6,
                    #     main_df.loc[(main_df.region == region), 'hot_difference'].values[0],
                    #     color='lightcoral',
                    #     alpha=0.5,
                    #     label=hot_diff_label if add_legend else "",
                    #     width=0.6,
                    #     data=main_df.loc[(main_df.region == region), 'hot_difference'].values[0],
                    #     zorder=2)

                    ax2.bar_label(chng, padding=3.0, fmt='%.2f', rotation='vertical', fontsize=bar_data_fontsize, zorder=3)

                    hist_cold_plt, = ax1.plot(
                        xpos[0],
                        historical[0],
                        'bo',
                        markersize=cold_markersize,
                        label=historical_cold_label if add_legend else "",
                        zorder=20)

                    hist_hot_plt, = ax1.plot(
                        xpos[0],
                        historical[1],
                        'bs',
                        markersize=hot_markersize,
                        label=historical_hot_label if add_legend else "",
                        zorder=20)

                    future_cold_plt, = ax1.plot(
                        xpos[1],
                        future[0],
                        'ro',
                        markersize=cold_markersize,
                        label=future_cold_label if add_legend else "",
                        zorder=20)

                    future_hot_plt, = ax1.plot(
                        xpos[1],
                        future[1],
                        'rs',
                        markersize=hot_markersize,
                        label=future_hot_label if add_legend else "",
                        zorder=20)

                    region_label_x_pos.append(sum(xpos)/len(xpos))
                    continent_label_pos_holder.extend(xpos)
                    last_xpos = xpos[1]
                    xpos = [x + 3 for x in xpos]
                    add_legend = False
                continent_label_x_pos.append((sum(continent_label_pos_holder)/len(continent_label_pos_holder)))
                xpos = [x + 2 for x in xpos]
            continent_x_ticker.append(last_xpos+1.95)
            
            
            number_of_models = len(list(sorted(df_gwl.dataset.unique())))
            print(number_of_models)
            string_datasets = '{number_of_models}'.format(number_of_models=number_of_models)
            title = '{exp}({number_of_models}) GWL{gwl}$^\circ$C'.format(exp=exp[:3].upper()+exp[3]+'-'+exp[4]+'.'+exp[5], number_of_models=number_of_models, gwl=gwl)
    
            # ax1.annotate(text=string_datasets,
            #              xy=(80, 45), 
            #              xycoords='data', 
            #              fontsize=15, 
            #              ) 
            # pk_min = df['peak_moving_direction'].min()
            # pk_max = df['peak_moving_direction'].max()

            ax1.set_xticks(region_label_x_pos)
            print(region_label_x_pos)
            ax1.set_xticklabels(labels=region_labels, rotation='vertical', fontsize=region_fontsize)
            ax1.set_ylabel('T ($^\circ$C)', fontsize=y_axes_label_fontsize)
            # ax1.set_title(title, fontsize=title_fontsize)
            ax1.tick_params(axis='y', labelsize=y_tick_fontsize)
            ax1.grid(visible=True, which='major', axis='y', linestyle='dotted')
            ax1.set_ylim(-42.0, 52.0)
            ax1.set_xlim(ax1.get_xbound()[0]+5,
                        ax1.get_xbound()[1]-5)

            ax2.axhline(0, color='grey', linewidth=0.8)
            ax2.set_ylabel('$\Delta T$ ($^{\circ}$C)', color=ax2_color, fontsize=y_axes_label_fontsize)
            ax2.xaxis.label.set_color('green')
            ax2.tick_params(axis='y', colors=ax2_color, labelsize=y_tick_fontsize)
            ax2.tick_params(axis='y', which='minor')
            ax2.set_ylim(-4.2, 5.2)

            # ax2.set_ylim(pk_min,pk_max)

            # Second X-axis
            ax3 = ax1.twiny()
            ax3.spines["bottom"].set_position(("axes", -0.15))
            ax3.tick_params('both', length=0, width=0, which='minor')
            ax3.tick_params('both', direction='in', which='major')
            ax3.xaxis.set_ticks_position("bottom")
            ax3.xaxis.set_label_position("bottom")

            # Continent labels
            print(continent_x_ticker)
            ax3.set_xticks(continent_x_ticker)
            ax3.xaxis.set_major_formatter(ticker.NullFormatter())
            ax3.xaxis.set_minor_locator(ticker.FixedLocator([i+1 for i in continent_label_x_pos]))
            ax3.xaxis.set_minor_formatter(ticker.FixedFormatter(continent_labels))
            for tick in ax3.xaxis.get_minor_ticks():
                tick.label1.set_horizontalalignment('center')
                
            ax3.set_xbound(ax1.get_xbound())
            ax3.tick_params(axis="x", which="both", rotation=90.0, length=5.0, labelsize=continent_fontsize)
            ax3.grid(visible=True, which='major', axis='x', linestyle='dashdot')
            
            peak_plts = [hist_cold_plt, hist_hot_plt, future_cold_plt,  future_hot_plt, chng]
            ax1.legend(
                peak_plts,
                [historical_cold_label, historical_hot_label,
                future_cold_label, future_hot_label,
                change_label],
                loc=4,
                ncol=3,
                fontsize=legend_fontsize
            )
            
            savefilename = '{exp}_{gwl}_multi_model_peak_change_from_grids_with_DIFF'.format(gwl=gwl, exp=exp,)
            figure_saving(plt, output_path, savefilename)

    results = pd.concat(results)
    results.to_csv(output_path + 'mean_peak_movements_.csv')



def multi_model_peak_change_plot_cold_ordered_from_grids(df, output_path):
    import matplotlib.pyplot as plt
    import cartopy.io.shapereader as shpreader
    from matplotlib import ticker


    hot_markersize = 4
    cold_markersize = 4
    legend_fontsize = 'medium'
    bar_data_fontsize = 'medium'
    y_axes_label_fontsize = 'x-large'
    continent_fontsize = 'medium'
    region_fontsize = 'large'
    y_tick_fontsize = 'x-large'
    title_fontsize = 'xx-large'

    
    gwl_list = list(sorted(df.gwl.unique()))
    exp_list = list(sorted(df.exp.unique()))
    
    for exp in exp_list:
        for gwl in gwl_list:
            df_gwl = df.loc[((df.gwl == gwl) & (df.exp == exp)) & ((df.historical_n_comp == 2) & (df.future_n_comp == 2))]
            if df_gwl.empty:
                continue
            reader = shpreader.Reader('/mnt/c/Users/paca_ay/Documents/DATA/NaturalEarth/only_lands_id_order_centeroids.shp')
            region_shape = [rgnshp for rgnshp in reader.records()]
            continent_regions = {}

            regions = list(set(df_gwl['region'].tolist()))
            for item in region_shape:
                if item.attributes['Continent'] in continent_regions:
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])
                else:
                    continent_regions[item.attributes['Continent']] = []
                    continent_regions[item.attributes['Continent']].append(item.attributes['Acronym'])

            fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
            x_end = 0
            ax_x_pos = []
            continent_x_pos = []
            region_labels = []
            continent_labels = []
            x_ticker = [-1]
            add_legend = True

            ax2 = ax1.twinx()
            ax2_color = 'green'
            ax2_color = (0 / 255, 112 / 255, 0 / 255, 0.5)
            ax2.spines['right'].set_color('green')
            xpos = [-0.2, 0.9]
            gwl_df = []
            x_ticker.append(xpos[0] - 1)
            for region in regions:
                print(region)
                temp = df_gwl.loc[(df_gwl.region == region), ['historical_mean_cold',
                                                            'historical_mean_hot',
                                                            'future_mean_cold',
                                                            'future_mean_hot']].mean().to_frame().T
                temp['region'] = region
                temp['exp'] = exp
                temp['gwl'] = gwl
                temp['peak_moving_direction'] = (temp['future_mean_cold'] - temp['historical_mean_cold']) \
                                                - (temp['future_mean_hot'] - temp['historical_mean_hot'])

                gwl_df.append(temp)
            
            main_df = pd.concat(gwl_df).sort_values(by=['historical_mean_cold'])

            c_label_hold = []
            for region in main_df['region']:
                cold_means = [
                    main_df.loc[(main_df.region == region), 'historical_mean_cold'].values[0],
                    main_df.loc[(main_df.region == region), 'future_mean_cold'].values[0]
                ]

                hot_means = [
                    main_df.loc[(main_df.region == region), 'historical_mean_hot'].values[0],
                    main_df.loc[(main_df.region == region), 'future_mean_hot'].values[0]
                ]

                historical = [
                    main_df.loc[(main_df.region == region), 'historical_mean_cold'].values[0],
                    main_df.loc[(main_df.region == region), 'historical_mean_hot'].values[0]
                ]

                future = [
                    main_df.loc[(main_df.region == region), 'future_mean_cold'].values[0],
                    main_df.loc[(main_df.region == region), 'future_mean_hot'].values[0]
                ]
                if main_df.loc[(main_df.region == region)].empty:
                    continue

                region_labels.append(region)

                barpos = sum(xpos) / len(xpos)
                
                
                chng = ax2.bar(
                    barpos,
                    main_df.loc[(main_df.region == region), 'peak_moving_direction'].values[
                        0],
                    color=ax2_color,
                    label="$\Delta T_{cold}^{future-historical}$-$\Delta T_{hot}^{future-historical}$" if add_legend else "",
                    width=0.6,
                    data=main_df.loc[(main_df.region == region), 'peak_moving_direction'].values[0],
                    zorder=2)
                ax2.bar_label(chng, padding=3, fmt='%.2f', rotation='vertical', fontsize=bar_data_fontsize, zorder=2)
                hist_plt, = ax1.plot(
                    (xpos[0], xpos[0]),
                    historical,
                    'bo',
                    markersize=hot_markersize,
                    label="Historical" if add_legend else "",
                    zorder=10)

                future_plt, = ax1.plot(
                    (xpos[1], xpos[1]),
                    future,
                    'ro',
                    markersize=hot_markersize,
                    label="Future" if add_legend else "",
                    zorder=10)

                ax1.plot(
                    (xpos, xpos),
                    ([i for i in hot_means], [j for j in cold_means]),
                    c='gray',
                    lw=0.5,
                    zorder=10)
                ax_x_pos.append(sum(xpos) / len(xpos))
                c_label_hold.extend(xpos)
                xpos = [x + 3 for x in xpos]
                add_legend = False

                # xpos = [x + 2 for x in xpos]
                x_ticker.append(xpos[0] + 1)

            pk_min = -42
            pk_max = 52
            
            number_of_models = len(list(sorted(df_gwl.dataset.unique())))
            print(number_of_models)
            string_datasets = '{number_of_models}'.format(number_of_models=number_of_models)
            title = '{exp}({number_of_models}) GWL{gwl}$^\circ$C'.format(exp=exp[:3].upper()+exp[3]+'-'+exp[4]+'.'+exp[5], number_of_models=number_of_models, gwl=gwl)

            ax1.set_xticks(ax_x_pos)
            ax1.set_xticklabels(labels=region_labels, rotation='vertical', fontsize=region_fontsize)
            ax1.set_ylabel('T ($^\circ$C)', fontsize=y_axes_label_fontsize)
            # ax1.set_title(title, fontsize=title_fontsize)
            ax1.tick_params(axis='y', labelsize=y_tick_fontsize)
            ax1.grid(visible=True, which='major', axis='both', linestyle=':')
            ax1.set_ylim(pk_min, pk_max)
            

            ax2.axhline(0, color='grey', linewidth=0.8)
            ax2.set_ylabel('$\Delta T$ ($^{\circ}$C)', color=ax2_color, fontsize=y_axes_label_fontsize)
            ax2.xaxis.label.set_color('green')
            ax2.tick_params(axis='y', colors=ax2_color, labelsize=y_tick_fontsize)
            ax2.tick_params(axis='y', which='minor')
            ax2.set_ylim(-4.2, 5.2)
            # ax1.set_ylim(pk_min,pk_max)

            # Second X-axis
            # ax3 = ax1.twiny()
            # ax3.spines["bottom"].set_position(("axes", -0.20))
            # ax3.tick_params('both', length=0, width=0, which='minor')
            # ax3.tick_params('both', direction='in', which='major')
            # ax3.xaxis.set_ticks_position("bottom")
            # ax3.xaxis.set_label_position("bottom")

            # ax3.set_xticks(x_ticker)
            # ax3.xaxis.set_major_formatter(ticker.NullFormatter())
            # ax3.xaxis.set_minor_locator(ticker.FixedLocator(continent_x_pos))
            # ax3.xaxis.set_minor_formatter(ticker.FixedFormatter(continent_labels))
            # ax3.set_xbound(ax1.get_xbound())
            # ax3.tick_params(axis="x", which="both", rotation=90, labelsize=continent_fontsize)
            peak_plts = [hist_plt, future_plt, chng]
            change_label = "$\Delta T=\Delta \mu_{cold}^{future-historical}$-$\Delta \mu_{hot}^{future-historical}$"
            cold_diff_label = "$\Delta \mu_{cold}^{future-historical}$"
            hot_diff_label = "$\Delta \mu_{hot}^{future-historical}$"
            historical_cold_label = "$\mu_{cold}^{historical}$"
            historical_hot_label = "$\mu_{hot}^{historical}$"
            future_cold_label = "$\mu_{cold}^{future}$"
            future_hot_label = "$\mu_{hot}^{future}$"
            # ax1.legend(
            #     peak_plts,
            #     ["Historical", "Future", "$\Delta T_{cold}^{future-historical}$-$\Delta T_{hot}^{future-historical}$"],
            #     loc=2,
            #     ncol=3,
            #     fontsize=legend_fontsize
            # )
            ax1.legend(
                peak_plts,
                ["Historical", "Future", change_label],
                loc=4,
                ncol=1,
                fontsize=legend_fontsize
            )
            savefilename = '{exp}_{gwl}_multi_model_peak_change_COLD_ORDER_from_grids'.format(gwl=gwl, exp=exp)
            figure_saving(plt, output_path, savefilename)
            
            