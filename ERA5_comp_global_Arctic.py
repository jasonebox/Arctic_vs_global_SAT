#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Jason Box

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

th=1
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams["font.size"] = fs

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/Arctic_vs_global_SAT/'
os.chdir(base_path)

iyear=1950
fyear=2021
n_years=fyear-iyear+1

do_time_series_plot=1
ly='p'

mid_years=[]
Ratio_Arctic_vs_Global=[]
Arctic=[]
Global=[]
colors=['r','c']
fn='/Users/jason/Dropbox/Arctic_vs_global_SAT/ERA5/ERA5_t2m_1950-2020.csv'
fn='/Users/jason/Dropbox/ERA5/output/ERA5_t2m_annual_1950-2021.csv'
df=pd.read_csv(fn)

print(df.columns)

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


ranges=[]
mid_years=[]
# for ll in range(1961+15,1991-14):
# for ll in range(1950+15,2021-13):
for ll in range(2021-15,2021-13):
    
    i_year_regression=ll-15
    f_year_regression=ll+14
    
    print(i_year_regression,f_year_regression)
    mid_years.append((i_year_regression+f_year_regression)/2)

    
    if do_time_series_plot:
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(figsize=(14,10))
        # colors = []

    n_years_regression=fyear-i_year_regression+1
            
    changes=[]

    x=df.year.values
    y=df.Arctic.values
    y-=np.mean(y[((x>=1961)*(x<=1990))])

    Arctic_change=regressx(x,y,do_time_series_plot,'r','Arctic')
    Arctic.append(Arctic_change)

    x=df.year.values
    y=df.Global.values
    y-=np.mean(y[((x>=1961)*(x<=1990))])
    ranges.append(str(i_year_regression)+'\nto\n'+str(f_year_regression))
    
    Global_change=regressx(x,y,do_time_series_plot,'c','global')
    Global.append(Global_change)

    result=Arctic_change/Global_change
    if abs(result)>100:result=np.nan
    Ratio_Arctic_vs_Global.append(result)

    if do_time_series_plot:
        
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.xaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ylab='air temperature anomaly north of 65°N, ° C\nrelative to 1961-1990'
        plt.ylabel(ylab,color='w')
        plt.hlines(y = 0, xmin = iyear,xmax=fyear,color='lightgrey',zorder=1,alpha=0.3)

        # tit=str(i_year_regression)+' to 2020 comparison'
        tit='ERA5 '+str(i_year_regression)+'-'+str(f_year_regression)
        plt.title(tit)
        # leg = plt.legend(prop={'size': fs})
        # leg = ax.legend()
    
        # for color,text in zip(colors,leg.get_texts()):
        #     text.set_color(color)

        leg = ax.legend()#title=str(i_year_regression)+' - '+str(f_year_regression))
        for color,text in zip(colors,leg.get_texts()):
            text.set_color(color)

        # for text in leg.get_texts():
        #     text.set_color("w")
        # plt.legend()
        
        props = dict(boxstyle='round', facecolor='k',edgecolor='k',alpha=1)
        mult=1
        yy0=0.05
        # ax.text(0.78,0.04, '@Climate_Ice', transform=ax.transAxes,
        # ax.text(0.77,0.04, '@AMAP_Arctic', transform=ax.transAxes,
        c0=0.3
        ax.text(0.08,yy0, 'ERA5 data', transform=ax.transAxes,
                fontsize=font_size*mult,verticalalignment='top', bbox=props,color=[c0,c0,c0], rotation_mode="anchor")
        ax.text(0.77,yy0, '@climate_ice', transform=ax.transAxes,
                fontsize=font_size*mult,verticalalignment='top', bbox=props,color=[c0,c0,c0], rotation_mode="anchor")

        if ly=='x':plt.show()
    
        fig_path='./Figs/'
        if ly == 'p':
            fn=fig_path+'ERA5_timeseries_'+tit+'.png'
            fn=fn.replace(' ','_')
            plt.savefig(fn, bbox_inches='tight', dpi=72, facecolor='k', edgecolor='k')
            # os.system('open '+fn)

#%%
make_gif=0

#convert -loop 0 -delay 15 /Users/jason/Dropbox/Arctic_vs_global_SAT/Figs/ERA5_timeseries_ERA5_* /Users/jason/Dropbox/Arctic_vs_global_SAT/anim/ERA5_timeseries_anim_AR.gif 

if make_gif:
    msg='convert -loop 0 -delay 15 /Users/jason/Dropbox/Arctic_vs_global_SAT/Figs/ERA5_t*.png /Users/jason/Dropbox/Arctic_vs_global_SAT/ERA5_anim.gif'
    print('making gif')
    os.system(msg)


#%%
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(14,10))
Ratio_Arctic_vs_Global=np.array(Ratio_Arctic_vs_Global)
ly='x'
# v=np.where(Ratio_Arctic_vs_Global<10)
# v=v[0]
ax.plot(mid_years,Ratio_Arctic_vs_Global,'-o',label='warming rate of Arctic ÷ global\nERA5')

ax.set_ylabel('ratio')
# plt.xlabel('starting year of comparison ending in 2020')
# ax.set_xlabel('middle year of 30-year comparison')
ax.legend(prop={'size': fs})
# x=
# np.arange(1951+15-iyear-15,2021-14-iyear-15,10)
ax.set_xticks(mid_years[0::5])
ax.set_xticklabels(ranges[0::5])

if ly=='p':plt.show()
    
fig_path='./Figs/'
if ly == 'p':
    plt.savefig(fig_path+'ratio_change_ERA5.png', bbox_inches='tight', dpi=72)
    
#%% output time series of ratio etc
df2 = pd.DataFrame(columns = ['year0','year1','mid_year','Arctic','Global','ratio_ERA5']) 
df2.index.name = 'index'
df2["year0"]=pd.Series(np.array(mid_years)-14.5)
df2["year1"]=pd.Series(np.array(mid_years)+14.5)
df2["mid_year"]=pd.Series(mid_years)
df2["Arctic"]=pd.Series(Arctic)
df2["Global"]=pd.Series(Global)
df2["ratio_ERA5"]=pd.Series(Ratio_Arctic_vs_Global)
ofile='./stats/ERA5_trend_ratio.csv'
df2.to_csv(ofile)
os.system('open '+ofile)