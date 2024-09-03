import numpy as np
import astropy
import sqlutil
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import dustmaps.sfd
import astropy.units as u
from astropy.modeling import models
import os
import csv
from os import path
import scipy
import m2fs_process as m2fs
import mycode
import crossmatcher
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
import dill as pickle
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor

with open('/hildafs/projects/phy200028p/mgwalker/m2fs/dtemp_files') as f:
    data=f.readlines()
dtemp_file=[]
for line in data:
    p=line.split()
    dtemp_file.append(p[0])
dtemp_file=np.array(dtemp_file)

ccd=[]
x=[]
y=[]
t=[]
temp=[]
n=[]
dwavdtemp=[]
dpixdtemp=[]

for i in range(0,len(dtemp_file)):
    ccd0,x0,y0,t0,temp0,n0,dwavdtemp0,dpixdtemp0=pickle.load(open(dtemp_file[i],'rb'))
    for j in range(0,len(x0)):
        ccd.append(ccd0)
        x.append(x0[j])
        y.append(y0[j])
        t.append(t0[j])
        temp.append(temp0[j])
        n.append(n0[j])
        dwavdtemp.append(dwavdtemp0[j])
        dpixdtemp.append(dpixdtemp0[j])

ccd=np.array(ccd)
x=np.array(x)
y=np.array(y)
t=np.array(t)
temp=np.array(temp)
n=np.array(n)
dwavdtemp=np.array(dwavdtemp)
dpixdtemp=np.array(dpixdtemp)

blue=np.where((ccd=='b')&(dwavdtemp==dwavdtemp)&(dpixdtemp==dpixdtemp)&(np.abs(dwavdtemp)<0.5))[0]
red=np.where((ccd=='r')&(dwavdtemp==dwavdtemp)&(dpixdtemp==dpixdtemp)&(np.abs(dwavdtemp)<0.5))[0]

xxx=np.column_stack((x[blue],y[blue],t[blue],temp[blue]))
yyy=dwavdtemp[blue]

permute=np.random.permutation(np.arange(len(xxx)))
x_train,x_test=xxx[permute[10000:]],xxx[permute[:10000]]
y_train,y_test=yyy[permute[10000:]],yyy[permute[:10000]]

blue_regr=linear_model.LinearRegression()
blue_regr.fit(xxx,yyy)




xxx=np.column_stack((x[red],y[red],t[red],temp[red]))
yyy=dwavdtemp[red]

permute=np.random.permutation(np.arange(len(xxx)))
x_train,x_test=xxx[permute[10000:]],xxx[permute[:10000]]
y_train,y_test=yyy[permute[10000:]],yyy[permute[:10000]]

red_regr=linear_model.LinearRegression()
red_regr.fit(xxx,yyy)


