# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:25:24 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


# %%
def data_transform(df, name, plot=False):
    
    df = df[[f'{name}']]
    
    ############### Log Transform ###############
    
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days

    def log_fit(x, a, b, c):
        
        return a * np.log(x + c) + b

    def get_log_trend(X, Y):
        
        initial_guess = [1, 1, 1]
        fitted_param, _ = curve_fit(log_fit, X, Y, p0=initial_guess)
        a, b, c = fitted_param
        
        trend = log_fit(X, a, b, c)
        return trend

    df[f'{name}_log_trend'] = get_log_trend(X, Y)
    df[f'{name}_residuals'] = df[f'{name}'] - df[f'{name}_log_trend']


    ############### Residual Scaling ###############
    
    rolling_window=30
    volatility = df[f'{name}_residuals'].rolling(rolling_window).std().dropna()

    Y = volatility.values.reshape(-1, 1)
    X = np.arange(len(volatility)).reshape(-1, 1)  # Time as feature

    def get_vol_trend(X, Y):
        
        regressor = LinearRegression()
        regressor.fit(X, Y)
        
        trend = regressor.predict(np.arange(len(df)).reshape(-1, 1))
        return trend

    df[f'{name}_vol_trend'] = get_vol_trend(X, Y)
    df[f'{name}_scaled_residuals'] = df[f'{name}_residuals'] / df[f'{name}_vol_trend']


    ############### Plotting ###############
    
    def plot_trend_residuals(df):
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].set_title("Original Data and Logarithmic Trend")
        axs[1].set_title("Residuals of Logarithmic Trend")
        axs[2].set_title('Scaled Residuals of Logarithmic Trend')

        axs[0].plot(df.index, df[f'{name}'], label='Original Data', color='blue')
        axs[0].plot(df.index, df[f'{name}_log_trend'], label='Logarithmic Trend', color='red', linestyle='--')
        axs[1].plot(df.index, df[f'{name}_residuals'], label='Residuals', color='green')
        axs[2].plot(df.index, df[f'{name}_scaled_residuals'], label='Scaled Residuals')

        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    if plot: plot_trend_residuals(df)
    return df


# %%
if __name__ == "__main__":
    
    from DataPreparation import data

    all_data = []
    for column in data.columns:
        all_data.append(data_transform(data, str(column), plot=True))
    all_data = pd.concat(all_data, axis=1)
    all_data.to_csv('residuals.csv')

