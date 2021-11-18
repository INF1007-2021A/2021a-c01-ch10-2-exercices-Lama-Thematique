#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse

# TODO: Définissez vos fonctions ici
def read_data(path: str) -> pd.DataFrame : 
    return pd.read_csv(path,sep=";")

def seperate_data(df:pd.DataFrame, ycol) -> Tuple[pd.DataFrame,pd.DataFrame] :
    return df.drop(ycol,axis=1),df[[ycol]]

def split(x,y):    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy().reshape(-1)
    y_test = y_test.to_numpy().reshape(-1)
    return x_train, x_test, y_train, y_test

def plotdiff(yp,yr):
    plt.plot(yr)
    
    plt.plot(yp)

    plt.show()

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    path = "data/winequality-white.csv"
    data = read_data(path=path)
    x,y = seperate_data(data,"quality")

    x_train, x_test, y_train, y_test = split(x,y)

    lreg = LR().fit(x_train,y_train)
    RFreg = RFR(n_estimators=100).fit(x_train,y_train)

    yl = lreg.predict(x_test)
    yRF = RFreg.predict(x_test)
    plotdiff(yl,y_test)
    plotdiff(yRF,y_test)

    msel = mse(y_test,yl)
    mseRF = mse(y_test,yRF)
    print(np.mean(msel))
    print(np.mean(mseRF))








    