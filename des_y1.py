import sqlutil
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
from astropy.io import fits 
import scipy as sp
import random
import mycode2
np.set_printoptions(threshold='nan')

#plt.rc('grid',alpha=0.7) # modify rcparams
#plt.rc('grid',color='white')
#plt.rc('grid',linestyle='-')
plt.rc('legend',frameon='True')
plt.rc('xtick.major',size=2)
plt.rc('ytick.major',size=2)
plt.rc('axes',axisbelow='True')
plt.rc('axes',grid='False')
plt.rc('axes',facecolor='white')
plt.rc('axes',linewidth=1)
plt.rc('xtick.major',width=1)
plt.rc('ytick.major',width=1)
plt.rc('xtick',direction='in')
plt.rc('ytick',direction='in')
plt.rc('xtick',labelsize='10')
plt.rc('ytick',labelsize='10')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='pdf')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
plt.rc('path',simplify='True')

with open('/nfs/nas-0-9/mgwalker.proj/m2fs/desy1_centers.dat') as f:
    data=f.readlines()[0:]
desy1_radeg=[]
desy1_decdeg=[]
for line in data: # fill arrays
    p=line.split()
    desy1_radeg.append(float(p[0]))
    desy1_decdeg.append(float(p[1]))
desy1_radeg=np.array(desy1_radeg)
desy1_decdeg=np.array(desy1_decdeg)

out='des_y1.dat'
g1=open(out,'w')

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>52.9256 and alpha_j2000<54.9256 and delta_j2000>-55.0492 and delta_j2000<-53.0492',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>55.0878 and alpha_j2000<57.0878 and delta_j2000>-44.5332 and delta_j2000<-42.5332',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>42.8820 and alpha_j2000<44.8820 and delta_j2000>-55.1188 and delta_j2000<-53.1188',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>69.9475 and alpha_j2000<71.9475 and delta_j2000>-51.2830 and delta_j2000<-53.2830',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>353.9975 and alpha_j2000<355.9975 and delta_j2000>-55.4060 and delta_j2000<-53.4060',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>316.2044 and alpha_j2000<318.2044 and delta_j2000>-52.1656 and delta_j2000<-50.1656',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>343.1765 and alpha_j2000<345.1765 and delta_j2000>-51.1633 and delta_j2000<-49.1633',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>34.6897 and alpha_j2000<36.6897 and delta_j2000>-53.2837 and delta_j2000<-51.2837',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

radeg,decdeg,gmag,siggmag,gspread,siggspread,rmag,sigrmag,rspread,sigrspread,imag,sigimag,ispread,sigispread,ebv=sqlutil.get('select alpha_j2000,delta_j2000,magmodel_g,magerrmodel_g,spread_g,spreaderr_g,magmodel_r,magerrmodel_r,spread_r,spreaderr_r,magmodel_i,magerrmodel_i,spread_i,spreaderr_i,ebv from des_yr1.main_griz_0614 where alpha_j2000>341.9796 and alpha_j2000<343.9796 and delta_j2000>-59.5689 and delta_j2000<-57.5689',db='wsdb',host='cappc127.ast.cam.ac.uk',user='matt_walker',password='5UN*ur8Y')
stars=(np.where(np.abs(rspread)<0.003))[0]
for j in range(0,len(stars)):
    string=str(round(radeg[stars[j]],7))+' '+str(round(decdeg[stars[j]],7))+' '+str(round(gmag[stars[j]],3))+' '+str(round(siggmag[stars[j]],3))+' '+str(round(gspread[stars[j]],3))+' '+str(round(rmag[stars[j]],3))+' '+str(round(sigrmag[stars[j]],3))+' '+str(round(rspread[stars[j]],3))+str(round(ebv[stars[j]],3))+' '+str(round(imag[stars[j]],3))+' '+str(round(sigimag[stars[j]],3))+' '+str(round(ispread[stars[j]],3))+' \n'
    g1.write(string)

g1.close()
