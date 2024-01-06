# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:27:37 2024

@author: mayank
"""

import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.use('TkAgg')


def load_data(experiment, logs):
    # File paths
    los_path = os.path.join(os.getcwd(), experiment)
    file_paths = [os.path.join(los_path, f'{point}_{distance}m.csv') for point in ['start', 'ma', 'center', 'mf', 'end']
                      for distance in [3, 5, 7, 9]]

    # Load data from each file into a dictionary
    data = {}
    for file_path in file_paths:
        # Extracting the point and distance from the file name
        parts = file_path.split('/')[-1].split('.')[0].split('_')
        point, distance = parts[0], parts[1]
        point = point.split('\\')[-1]
        key = f"{point}_{distance}"

        # Reading the CSV file
        data[key] = pd.read_csv(file_path)

    # Check the structure of the first few dataframes to understand their format
    sample_data = {k: data[k].head() for k in list(data.keys())[:3]}
    if logs:
        print(sample_data)

    # Data Preprocessing

    # Dropping unnecessary columns and handling missing values
    for key in data:
        # Dropping unnecessary columns
        data[key] = data[key][['time', 'mean', 'rssi']]

        # Dropping rows with missing values in 'mean' and 'rssi' columns
        data[key].dropna(subset=['mean', 'rssi'], inplace=True)

    # Converting the 'time' column to a readable format (if necessary)
    # We will check the format of the 'time' column first
    time_format_sample = data['start_3m']['time'].iloc[0]
    if logs:
        print(time_format_sample)

    # Converting the 'time' column to datetime format
    for key in data:
        data[key]['time'] = pd.to_datetime(data[key]['time'], unit='ns')

    # Checking the updated format of the 'time' column in one of the dataframes
    if logs:
        print(data['start_3m'].head())

    return(data)


def kde(data, identity, logs):
    # Setting the style for the plots
    sns.set(style="whitegrid")

    # Plotting the distribution of RSSI values for a few sample datasets
    # plt.figure(figsize=(15, 10))

    axtitle = ["Start", "Approach", "Centre", "Depart", "End"]

    figure, axes = plt.subplots(3, 2, figsize=(15, 10))
    plt.xlabel('RSSI Value')
    plt.ylabel('Kernel Density')
    # We will plot the distribution for the 'start' point at different distances as a sample
    for i, point in enumerate(['start', 'ma', 'center', 'mf', 'end']):
        row, col = divmod(i, 2)  # Calculate the row and column for each subplot
        for distance in [3, 5, 7, 9]:
            sns.kdeplot(data[f'{point}_{distance}m']['rssi'], label=f'{point} {distance}m', ax=axes[row, col])
            axes[row, col].set_title(axtitle[i], fontweight='bold')
            axes[row, col].set_xlabel('RSSI')
            axes[row, col].legend(['3m', '5m', '7m', '9m'], loc='upper left')

    axes[2, 1].axis('off')
    plt.suptitle(f'Distribution of RSSI Values at Start Point for Different Distances (Kernel Density Estimation) - {identity}')

    # figure.legend(['3m', '5m', '7m', '9m'], loc='lower right')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    figure.savefig(identity+'.png')

    summary_stats = pd.DataFrame()

    for point in ['start', 'ma', 'center', 'mf', 'end']:
        for distance in [3, 5, 7, 9]:
            key = f'{point}_{distance}m'
            stats = data[key]['rssi'].describe()
            stats.name = key
            summary_stats = summary_stats._append(stats)

    # Rearranging the dataframe for better readability
    summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary_stats.reset_index(inplace=True)
    summary_stats.rename(columns={'index': 'Point_Distance'}, inplace=True)

    # print(summary_stats)
    if logs:
        print(summary_stats)


def anova_tukeys (data, identity, logs):
    anova_data = pd.DataFrame(columns=['RSSI', 'Distance', 'Condition'])
    for point in ['start', 'ma', 'center', 'mf', 'end']:
        for distance in [3, 5, 7, 9]:
            los_key = f'{point}_{distance}m'
            temp_data = pd.DataFrame()
            temp_data['RSSI'] = data[f'{point}_{distance}m']['rssi']
            temp_data['Distance'] = distance
            temp_data['Condition'] = identity
            anova_data = pd.concat([anova_data, temp_data])

    # Performing ANOVA
    anova_result = f_oneway(anova_data[anova_data['Distance'] == 3]['RSSI'],
                            anova_data[anova_data['Distance'] == 5]['RSSI'],
                            anova_data[anova_data['Distance'] == 7]['RSSI'],
                            anova_data[anova_data['Distance'] == 9]['RSSI'])
    if logs:
        print(anova_result)

    # Performing Tukey's HSD
    tukey_result = pairwise_tukeyhsd(endog=anova_data['RSSI'].astype('float'),
                                     groups=anova_data['Distance'].astype(str) + "_" + anova_data['Condition'],
                                     alpha=0.05)

    if logs:
        print(tukey_result)


def main():
    LOS_data = load_data("los", logs=False)
    nLOS_data = load_data("nlos", False)
    # print(LOS_data)
    # print(nLOS_data)

    kde(LOS_data, "LoS", False)
    kde(nLOS_data, "nLoS", False)
    anova_tukeys(LOS_data, "LoS", True)
    anova_tukeys(nLOS_data, "nLoS", True)


if __name__ == "__main__":
    main()
