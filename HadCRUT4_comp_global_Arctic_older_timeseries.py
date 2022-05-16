#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:44:13 2020

@author: jeb
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import statsmodels.api as sm
from numpy.polynomial.polynomial import polyfit
from scipy import stats
from scipy.interpolate import interp1d

fs=26

# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid'] = False
# plt.rcParams['grid.alpha'] = 0
plt.rcParams['grid.alpha'] = 1
co=0.9; plt.rcParams['grid.color'] = (co,co,co)
plt.rcParams["font.size"] = fs
plt.rcParams['legend.fontsize'] = fs*0.8
plt.rcParams['mathtext.default'] = 'regular'

# global
fn='/Users/jason/Dropbox/AMAP/Arctic-multi-indicators/data_multi_indicators/HadCRUT4/ihad4_krig_v2_0-360E_-90-90N_n_0p.dat.txt'
# useful? https://www-users.york.ac.uk/~kdc3/papers/coverage2013/series.html
# see https://climexp.knmi.nl/select.cgi?id=someone@somewhere&field=had4sst4_krig_v2
fn='/Users/jason/Dropbox/AMAP/Arctic-multi-indicators/data_multi_indicators/HadCRUT4/ihad4_krig_v2_0-360E_65-90N_n_0p.dat.txt'
fn='/Users/jason/Dropbox/AMAP/Arctic-multi-indicators/data_multi_indicators/HadCRUT4/ihad4_krig_v2_0-360E_65-90N_n_0p.dat_2021.txt'
df=pd.read_csv(fn,skiprows=21, delim_whitespace=True)
print(df.shape)
print(df)

#%%
z=np.asarray(df)

n=len(z)

for i in range(12):
    temp=z[:,i+1]
    v=[temp<-990]
    if np.sum(v):
        print(i,np.mean(temp[-5:-1]))#np.mean())
        z[-1,i+1]=np.mean(temp[-5:-1])
#%%
co=['b','r','k']
iyear=1851
fyear=2021
n_years=fyear-iyear+1

plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(14,10))

means=np.zeros((3,n_years))

for j,select_month in enumerate(range(0,3)):
    if j>=0:
    # if j==2:
        if select_month==0:
            select_months=[0,1,2,3,4,9,10,11]
            season_name='freeze up season Oct-May'
            season_name2='cold'
        if select_month==1:
            select_months=[5,6,7,8]
            season_name='melt season Jun-Sept'
            season_name2='warm'
        if select_month==2:
            select_months=[0,1,2,3,4,5,6,7,8,9,10,11]
            season_name='annual'
            season_name2='annual'
    
        select_months=np.asarray(select_months)
        select_months+=1
        
        print(z.shape)
        z[n_years-1,12]=np.mean(z[n_years-12:n_years-2:,12])
        for yy in range(0,n_years):
            # print()
            means[j,yy]=np.mean(z[yy,select_months])
            # print(yy+iyear,z[yy,select_months],len(z[yy,select_months]))
            # ss
            
        x=np.arange(n_years)+iyear
        y=means[j,:]
        plt.plot(x,y,'.',c=co[j],label=season_name)
    
        du_lowes=0
        if du_lowes:
            # lowess will return our "smoothed" data with a y value for at every x-value
            lowess = sm.nonparametric.lowess(y, x, frac=.3)
            
            # unpack the lowess smoothed points to their values
            lowess_x = list(zip(*lowess))[0]
            lowess_y = list(zip(*lowess))[1]
            
            # run scipy's interpolation. There is also extrapolation I believe
            f = interp1d(lowess_x, lowess_y, bounds_error=False)
            
            xnew = [i/10. for i in range(400)]
            
            # this this generate y values for our xvalues by our interpolator
            # it will MISS values outsite of the x window (less than 3, greater than 33)
            # There might be a better approach, but you can run a for loop
            #and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
            ynew = f(xnew)
            
            plt.plot(lowess_x, lowess_y, '-',c=co[j],linewidth=3)
            # plt.plot(xnew, ynew, '-',c=co[j])
        
        y=y[x>1970]
        x=x[x>1970]
        b, m = polyfit(x, y, 1)
        global_change=0.9494478105761218

        plt.plot(x, b + m * x, '--',c=co[j],linewidth=3)
        coefs = np.polyfit(x, y, 2)  # quadratic
        fit=coefs[0]*x**2+coefs[1]*x+coefs[2]
        coefs=stats.pearsonr(x,y)

        confidencex=str("%8.3f"%(1-coefs[1])).lstrip()
        print("change",m*50,"confidence",confidencex)
        print("change over global",m*50/global_change,"confidence",confidencex)
        
plt.axhline(y=0,c='gray',linewidth=0.5)

ylab='air temperature north of 65°N, ° C\nanomaly relative to 1981-2010'
plt.ylabel(ylab)
plt.legend(prop={'size': fs})
plt.show()
