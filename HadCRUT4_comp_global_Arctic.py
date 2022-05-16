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
plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid'] = False
# plt.rcParams['grid.alpha'] = 0
plt.rcParams['grid.alpha'] = 0.3
co=0.9; plt.rcParams['grid.color'] = (co,co,co)
plt.rcParams["font.size"] = fs
plt.rcParams['legend.fontsize'] = fs*0.8
plt.rcParams['mathtext.default'] = 'regular'

darkmode=1
if darkmode:
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rcParams['axes.edgecolor'] = 'black'
else:
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'


if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/Arctic_vs_global_SAT/'
os.chdir(base_path)

co=['b','r','k']
iyear=1851
fyear=2021
n_years=fyear-iyear+1

do_time_series_plot=1
ly='x'

mid_years=[]
Ratio_Arctic_vs_Global=[]
Arctic=[]
Global=[]
changes=[]

def regressx(x,y,do_time_series_plot,color,label):
    if do_time_series_plot:
        plt.plot(x,y,'-',c=color,label=label)

    v=((x>=i_year_regression)&(x<=f_year_regression))
    n=np.sum(v)
    # print(i_year_regression,f_year_regression,n)
    y=y[v]
    x=x[v]

    b, m = polyfit(x, y, 1)
    # global_change=global_changes[j]

    if do_time_series_plot:
        plt.plot(x, b + m * x, '--',c=color,linewidth=3)
    # coefs = np.polyfit(x, y, 2)  # quadratic
    # fit=coefs[0]*x**2+coefs[1]*x+coefs[2]
    # coefs=stats.pearsonr(x,y)

    # confidencex=str("%8.3f"%(1-coefs[1])).lstrip()
    # print("change",m*n_years_regression,"confidence",confidencex)
    # print("change over global",m*50/global_change,"confidence",confidencex)
    return(m*n)    

    
# for i_year_regression in range(1951,1991):
# for i_year_regression in range(1961,1991):
# for ll in range(1851+15,fyear-14+1):
# for ll in range(fyear-15,fyear-14+1):
# for ll in range(1851+15,1851+15+1):
for ll in range(fyear-15,fyear-14):
    # print(ll)
    i_year_regression=ll-15
    f_year_regression=ll+14
    
    print(i_year_regression,f_year_regression)
    mid_years.append((i_year_regression+f_year_regression)/2)
    
    if do_time_series_plot:
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(figsize=(14,10))

    # global
    fn='./HadCRUT4/ihad4_krig_v2_0-360E_-90-90N_n_0p.dat.txt'
    df=pd.read_csv(fn,skiprows=21, delim_whitespace=True)
    # n=1 # drops incomplete 2021
    # df.drop(df.tail(n).index,inplace=True) # drop last n rows
        
    means=np.zeros((3,n_years))
    
    for j,select_month in enumerate(range(0,3)):
        # if j>=0:
        if j==2:
            if select_month==0:
                select_months=[0,1,2,3,4,9,10,11]
                season_name='freeze-up season Oct-May'
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
            
            z=np.asarray(df)
            # estimate missing months using avereage of pervious N years
            for i in range(12):
                temp=z[:,i+1]
                v=[temp<-990]
                if np.sum(v):
                    # print(i,np.mean(temp[-5:-1]))#np.mean())
                    z[-1,i+1]=np.mean(temp[-2:-1])
        
            # end estimate missing months using avereage of pervious N years
            z[n_years-1,12]=np.mean(z[n_years-12:n_years-2:,12])
            
            for yy in range(0,n_years):
                # print()
                means[j,yy]=np.mean(z[yy,select_months])
                # print(yy+iyear,z[yy,select_months],len(z[yy,select_months]))
                # ss

            x=np.arange(n_years)+iyear
            v=((x>=1961)&(x<=1990))
            y=means[j,:]-np.mean(means[j,v])

            Global_change=regressx(x,y,do_time_series_plot,'c','global')
            Global.append(Global_change)
    
    # ------------------------------------------- Arctic part
    
    # Arctic
    fn='./HadCRUT4/ihad4_krig_v2_0-360E_65-90N_n_0p.dat.txt' # comment this out to obtain global changes
    df=pd.read_csv(fn,skiprows=21, delim_whitespace=True)
    df=pd.read_csv(fn,skiprows=21, delim_whitespace=True)
    # n=1 # drops incomplete 2021
    # df.drop(df.tail(n).index,inplace=True) # drop last n rows
    
    
    means=np.zeros((3,n_years))
    
    for j,select_month in enumerate(range(0,3)):
        # if j>=0:
        if j==2:
            if select_month==0:
                select_months=[0,1,2,3,4,9,10,11]
                season_name='freeze-up season Oct-May'
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
            
            z=np.asarray(df)
            # estimate missing months using avereage of pervious N years
            for i in range(12):
                temp=z[:,i+1]
                v=[temp<-990]
                if np.sum(v):
                    # print(i,np.mean(temp[-5:-1]))#np.mean())
                    z[-1,i+1]=np.mean(temp[-2:-1])
                    z[n_years-1,12]=np.mean(z[n_years-12:n_years-2:,12])
            # end estimate missing months using avereage of pervious N years
            z[n_years-1,12]=np.mean(z[n_years-12:n_years-2:,12])
            for yy in range(0,n_years):
                # print()
                means[j,yy]=np.mean(z[yy,select_months])
                # print(yy+iyear,z[yy,select_months],len(z[yy,select_months]))
                # ss
                
            x=np.arange(n_years)+iyear
            v=((x>=1961)&(x<=1990))
            y=means[j,:]-np.mean(means[j,v])

            Arctic_change=regressx(x,y,do_time_series_plot,'r','Arctic')
            Arctic.append(Arctic_change)

            result=Arctic_change/Global_change
            if abs(result)>100:result=np.nan
            Ratio_Arctic_vs_Global.append(result)

            # changes.append(m*n_years_regression)
    
    if do_time_series_plot:
        ylab='air temperature anomaly °C,\nrelative to 1961-1990'
        plt.ylabel(ylab)
        plt.hlines(y = 0, xmin = iyear,xmax=fyear,color='lightgrey',zorder=1,alpha=0.3)
        # tit=str(i_year_regression)+' to 2020 comparison'
        tit=str(i_year_regression)+'-'+str(f_year_regression)
        plt.title(tit)#+'\n trends from HadCRUT4 after CW2014')
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.xaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ylab='air temperature north of 65°N, ° C\nanomaly relative to 1961-1990'
        plt.ylabel(ylab,color='w')

        # plt.legend(prop={'size': fs})
        # legend = plt.legend()
        leg = ax.legend()#title=str(i_year_regression)+' - '+str(f_year_regression))
        for color,text in zip(colors,leg.get_texts()):
            text.set_color(color)
        # plt.setp(legend.get_title(), color='w')

        if ly=='x':plt.show()
    
        fig_path='./Figs/'
        if ly == 'p':
            plt.savefig(fig_path+'HadCRUT4_CW2014_timeseries_'+tit+'.png', bbox_inches='tight', dpi=72, facecolor='k', edgecolor='k')

#%%
make_gif=0

if make_gif:
    msg='convert -loop 0 -delay 15 /Users/jason/Dropbox/Arctic_vs_global_SAT/Figs/HadCRUT4_CW2014_t*.png /Users/jason/Dropbox/Arctic_vs_global_SAT/HadCRUT4_CW2014_anim.gif'
    print('making gif')
    os.system(msg)

#%%

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(14,10))
Ratio_Arctic_vs_Global=np.array(Ratio_Arctic_vs_Global)
Ratio_Arctic_vs_Global[abs(Ratio_Arctic_vs_Global)>10]=np.nan
# v=np.where(Ratio_Arctic_vs_Global<10)
# v=v[0]
plt.plot(mid_years,Ratio_Arctic_vs_Global,'-o',label='warming rate of Arctic ÷ global\nHadCRUT4 after CW2014')

v=np.where(Ratio_Arctic_vs_Global>=4)
v=v[0]
for i in range(len(v)):
    print(Ratio_Arctic_vs_Global[v[i]],np.array(mid_years)[v[i]]-15,np.array(mid_years)[v[i]]+15)
plt.ylabel('ratio')
# plt.xlabel('starting year of comparison ending in 2020')
plt.xlabel('middle year of 30-year comparison')
plt.legend(prop={'size': fs})

if ly=='x':plt.show()
    
fig_path='./Figs/'
if ly == 'p':
    plt.savefig(fig_path+'ratio_change_HadCRUT4_CW2014.png', bbox_inches='tight', dpi=72)
    
#%% output time series of ratio etc
df2 = pd.DataFrame(columns = ['year0','year1','mid_year','Arctic','Global','ratio_HadCRUT4_CW2014']) 
df2.index.name = 'index'
df2["year0"]=pd.Series(np.array(mid_years)-14.5)
df2["year1"]=pd.Series(np.array(mid_years)+14.5)
df2["mid_year"]=pd.Series(mid_years)
df2["Arctic"]=pd.Series(Arctic)
df2["Global"]=pd.Series(Global)
df2["ratio_HadCRUT4_CW2014"]=pd.Series(Ratio_Arctic_vs_Global)
ofile='./stats/HadCRUT4_CW2014_trend_ratio.csv'
df2.to_csv(ofile)
os.system('open '+ofile)