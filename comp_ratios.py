#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:44:13 2020

@author: jeb

https://climexp.knmi.nl/select.cgi?id=someone@somewhere&field=had4sst4_krig_v2

!!!
select 1850-now anomalies: HadCRUT4 NOT HadCRUT4/HadSST4 filled-in by Cowtan and Way	
 
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
do_lowes_fit=0
if do_lowes_fit:import statsmodels.api as sm
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


if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/Arctic_vs_global_SAT/'
os.chdir(base_path)

co=['b','r','k']
iyear=1851
fyear=2021
n_years=fyear-iyear+1

do_time_series_plot=0
ly='p'

fn='./stats/HadCRUT4_CW2014_trend_ratio.csv'
df=pd.read_csv(fn)
print(df.columns)
# df.drop(df[df.mid_year < 1969].index, inplace=True)
# df = df.reset_index(drop=True)

# print(df)
# #%%
plt_ERA5=1
fn='./stats/ERA5_trend_ratio.csv'
dfERA5=pd.read_csv(fn)
print(dfERA5.columns)


if do_time_series_plot:
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(figsize=(14,10))


plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(14,10))

ax.plot(df.mid_year,df.ratio_HadCRUT4_CW2014,'-o',label='HadCRUT4 after CW2014')
if plt_ERA5:
    ax.plot(dfERA5.mid_year,dfERA5.ratio_ERA5,'-o',label='ERA5')

ax.set_ylabel('ratio')
# plt.xlabel('starting year of comparison ending in 2020')
# ax.set_xlabel('running middle year of 31-year comparison')
x0=1920
x0=1975
x1=2007
ax.set_xlim(x0,x1)
ax.set_ylim(-1,5)
# ax.set_xlim(1905,2007)

fig.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
for i,label in enumerate(labels):
    temp=str(int(label)-15)+'\nto\n'+str(int(label)+15)
    labels[i]=temp

ax.set_xticklabels(labels)
ax.hlines(y=0,xmin=x0,xmax=x1,color='grey',alpha=0.4)

plt.legend(prop={'size': fs},title='ratio of Arctic ÷ global warming rates')

if ly=='x':plt.show()
    
fig_path='./Figs/'
if ly == 'p':
    plt.savefig(fig_path+'ratio_change_2_data_sources_'+str(x0)+'_to_present.png', bbox_inches='tight', dpi=72)
    
# #%% output time series of ratio etc
# df2 = pd.DataFrame(columns = ['year','ratio_HadCRUT4_CW2014']) 
# df2.index.name = 'index'
# df2["year"]=pd.Series(i_year_regressions)
# df2["ratio_HadCRUT4_CW2014"]=pd.Series(Ratio_Arctic_vs_Global)
# df2.to_csv(ofile)

#%%

fn='./stats/HadCRUT4_CW2014_trend_ratio.csv'
df=pd.read_csv(fn)
print(df.columns)
df.drop(df[df.year0 < 1950].index, inplace=True)
df = df.reset_index(drop=True)


plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(14,10))

mean=(df.ratio_HadCRUT4_CW2014+dfERA5.ratio_ERA5)/2
err=abs(df.ratio_HadCRUT4_CW2014-dfERA5.ratio_ERA5)/2

x=df.mid_year
y=mean

ax.scatter(x, y, s=600, facecolor='w', edgecolors='k',linewidth=th,zorder=9)

plt.errorbar(x, y, yerr=[err, err],fmt='.',c='w', ecolor='k', capthick=30,capsize=10, 
            elinewidth=th,
            markeredgewidth=th)

ax.set_ylabel('ratio')
# plt.xlabel('starting year of comparison ending in 2020')
# ax.set_xlabel('running middle year of 31-year comparison')
ax.set_xlim(1975,2007.5)
ax.set_ylim(0,5)

ax.plot(df.mid_year,df.ratio_HadCRUT4_CW2014,'-o',label='HadCRUT4 after CW2014',c='r')
ax.plot(dfERA5.mid_year,dfERA5.ratio_ERA5,'-o',label='ERA5',c='b')
fig.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
for i,label in enumerate(labels):
    temp=str(int(label)-15)+'\nto\n'+str(int(label)+15)
    labels[i]=temp

ax.set_xticklabels(labels)

plt.legend(prop={'size': fs},title='Arctic to global warming rates')

ly='p'
if ly=='x':plt.show()
    
fig_path='./Figs/'
if ly == 'p':
    plt.savefig(fig_path+'ratio_change_2_data_sources_'+str(x0)+'_to_present_fancy.png', bbox_inches='tight', dpi=72)
    
