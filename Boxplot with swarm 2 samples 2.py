#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:51:23 2022

@author: dimitardimitrov
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import ttest_rel
from scipy.stats import normaltest
from scipy.stats import iqr
from scipy.stats import mannwhitneyu
from statistics import median
from statistics import mean
import statsmodels.api as statsmodels
import matplotlib.transforms as transforms
import scikit_posthocs as sp



DataFile="Data files excel/331 SYN pff/331 SYNpff.xlsx"
Xlabel1='-'
Xlabel2='PFF'
Ylabel=" α-synuclein (norm.)"


#symbols :  α, β


SaveAs='Data files excel//331 SYN pff/331 SYNpff.png'
Columns=[0,1]
fig_dims = (0.8, 1.4)
yaxisrange=0,8
fontsz = 8
fontzszannot= 8
StatTest=mannwhitneyu


#Global parameters>>>>>>>>>>>>>>>>>>>>>>>>>>
sns.set(rc={"figure.dpi":300, 'savefig.dpi':1200})  
sns.set_theme(style="ticks")
fig, ax = plt.subplots(figsize=fig_dims)

# DATA files<<<<<<<<<<<<<<<<<<

df = pd.read_excel(DataFile, usecols=Columns) 
df=[df[col].dropna() for col in df]

def SaveFile(): 
    fig.savefig(SaveAs,
                format='png', bbox_inches='tight', pad_inches=0.07) 
    
#figure components arrangement
g=sns.boxplot(data=df, width=0.38, 
            boxprops={'zorder': 2, 'facecolor':'#999999', 'edgecolor': "k","linewidth": 1},
            medianprops={'zorder': 3, 'color': "k","linewidth": 1},
            whiskerprops={'zorder': 4, 'color': "k","linewidth": 1},
            capprops={'zorder': 5, 'color': "k","linewidth": 1},
            showfliers = False)



g=sns.stripplot(data=df, color='#003366', size=0.6, ax=g,zorder=3,  alpha=1)
offset = transforms.ScaledTranslation(5/72., 0, ax.figure.dpi_scale_trans)
trans = ax.collections[0].get_transform()
ax.collections[0].set_transform(trans + offset)
ax.collections[1].set_transform(trans + offset)


g.tick_params(labelsize=9, pad=0.2, length=2)
#g.axes.autoscale(axis='y', tight=False)
#ax.set_ylim(-0.1)
g.axes.set_ylim(yaxisrange)
g.axes.set_ylabel(Ylabel, fontsize = fontsz, labelpad=1)
sns.despine (top=True, right=True)
#g.axes.set_title(PlotTitle, fontsize= fontsz, pad=1) #SAVE AS PLOT TITLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

g.set_xticks([0,1])

nobs=str(df[0].value_counts().sum())
nobs2=str(df[1].value_counts().sum())


g.set_xticklabels([Xlabel1, Xlabel2], fontsize=fontsz)
#g.text(x=0, y=0.95, s=nobs, color='k', fontsize=7, horizontalalignment='center')
#g.text(x=1, y=0.95, s=nobs2,  color='k', fontsize=7, horizontalalignment='center')
#g.text(x=2, y=0.95, s=nobs3,  color='k', fontsize=7, horizontalalignment='center')

# Stats<<<<<<<<<<<<<<<<<<<<<<<<<<<<

pvalueStat= StatTest(df[0],df[1]).pvalue
dunn= sp.posthoc_dunn(df, p_adjust = 'hommel')


#‘bonferroni’ : one-step correction 
#‘sidak’ : one-step correction ‘holm-sidak’ : step-down method using Sidak adjustments 
#‘holm’ : step-down method using Bonferroni adjustments 
#‘simes-hochberg’ : step-up method (independent) 
#‘hommel’ : closed method based on Simes tests (non-negative) 
#‘fdr_bh’ : Benjamini/Hochberg (non-negative) 
#‘fdr_by’ : Benjamini/Yekutieli (negative) 
#‘fdr_tsbh’ : two stage fdr correction (non-negative) 
#‘fdr_tsbky’ : two stage fdr correction (non-negative)




if pvalueStat >= 0.05:
        symbol = "p=%.2f" % pvalueStat
if pvalueStat < 0.05:
            symbol = "p=%.2f" % pvalueStat
if pvalueStat < 0.01:
                symbol = "p=%.3f" % pvalueStat
if pvalueStat < 0.001:
        symbol = 'p<0.001'
if pvalueStat < 0.0001:
        symbol = 'p<0.0001'



g.axes.set_title(symbol, fontsize= fontsz, color="white", pad=1) #SAVE AS PLOT TITLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#plt.grid(which='major', axis='y') 

med1=median(df[0])
med2=median(df[1])


iqr1=iqr(df[0])
iqr2=iqr(df[1])


norm1=normaltest(df[0]).pvalue
norm2=normaltest(df[1]).pvalue


print(Ylabel)
print(Xlabel1)
print(Xlabel2)

print("N:")
print(nobs)
print(nobs2)

print("normality:")
print(norm1)
print(norm2)

print("median:")
print(med1)
print(med2)

print("IQR:")
print(iqr1)
print(iqr2)

print("Stats")
print(StatTest)
print(pvalueStat)
print("Dunn posthoc:")
print(dunn)

dfhead=pd.DataFrame(df)
print(dfhead.head(3))

plt.show(g)


Save = input('Save file? y/n')
if Save in ['y']:
    fig = g.figure 
    SaveFile()
    print ("Yay! Saved!")
else:
    print ("Not saved!")

