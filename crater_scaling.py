import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
#import mycode2
np.set_printoptions(threshold='nan')

#plt.rc('grid',alpha=0.7) # modify rcparams
#plt.rc('grid',color='white')
#plt.rc('grid',linestyle='-')
plt.rc('legend',frameon='True')
plt.rc('legend',framealpha=0.5)
plt.rc('legend',borderpad=1)
plt.rc('legend',borderaxespad=1)
plt.rc('legend',scatterpoints=1)
plt.rc('legend',numpoints=1)
plt.rc('xtick.major',size=2)
plt.rc('ytick.major',size=2)
plt.rc('xtick.minor',size=2)
plt.rc('ytick.minor',size=2)
plt.rc('axes',axisbelow='True')
plt.rc('axes',grid='False')
plt.rc('axes',facecolor='white')
plt.rc('axes',linewidth=0.5)
plt.rc('xtick.major',width=0.5)
plt.rc('ytick.major',width=0.5)
plt.rc('xtick.minor',width=0.5)
plt.rc('ytick.minor',width=0.5)
plt.rc('xtick',direction='in')
plt.rc('ytick',direction='in')
plt.rc('xtick',labelsize='6')
plt.rc('ytick',labelsize='6')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
data1='mcconnachie.dat'
data2='harris2.dat'
data3='harris3.dat'
data4='ret2_gradientpost_equal_weights.dat'
data5='cra_gradientpost_equal_weights.dat'
out1='ret2_scaling_newcommands.tex'
g1=open(out1,'w')


with open(data1) as f: # read data file
    data=f.readlines()

parent=[]
dsph=[]
rah=[]
ram=[]
ras=[]
chardecd=[]
decm=[]
decs=[]
glon=[]
glat=[]
distance=[]
sigdistance=[]
vsys=[]
sigvsys=[]
vmag=[]
sigvmag=[]
rhalfarcmin=[]
sigrhalfarcmin=[]
muv=[]
sigmuv=[]
ellip=[]
absvmag=[]
sigabsvmag=[]
rhalf=[]
sigrhalf=[]
vdisp=[]
sigvdisp=[]
feh=[]
sigfeh=[]
method=[]

for line in data: # fill arrays
    p=line.split()
    parent.append(str(p[0]))
    dsph.append(str(p[1]))
    rah.append(float(p[3]))
    ram.append(float(p[4]))
    ras.append(float(p[5]))
    chardecd.append(float(p[6]))
    decm.append(float(p[7]))
    decs.append(str(p[8]))
    glon.append(float(p[9]))
    glat.append(float(p[10]))
    distance.append(float(p[11]))
    sigdistance.append(float(p[12]))
    vsys.append(float(p[14]))
    sigvsys.append(float(p[15]))
    vmag.append(float(p[16]))
    sigvmag.append(float(p[17]))
    rhalfarcmin.append(float(p[19]))
    sigrhalfarcmin.append(float(p[20]))
    muv.append(float(p[22]))
    sigmuv.append(float(p[23]))
    ellip.append(float(p[25]))
    absvmag.append(float(p[26]))
    sigabsvmag.append(float(p[27]))
    rhalf.append(float(p[29]))
    sigrhalf.append(float(p[30]))
    vdisp.append(float(p[32]))
    sigvdisp.append(float(p[33]))
    feh.append(float(p[35]))
    sigfeh.append(float(p[36]))
    method.append(str(p[37]))

parent=np.array(parent)
dsph=np.array(dsph)
rah=np.array(rah)
ram=np.array(ram)
ras=np.array(ras)
chardecd=np.array(chardecd)
decm=np.array(decm)
decs=np.array(decs)
glon=np.array(glon)
glat=np.array(glat)
distance=np.array(distance)
sigdistance=np.array(sigdistance)
vsys=np.array(vsys)
sigvsys=np.array(sigvsys)
vmag=np.array(vmag)
sigvmag=np.array(sigvmag)
rhalfarcmin=np.array(rhalfarcmin)
sigrhalfarcmin=np.array(sigrhalfarcmin)
muv=np.array(muv)
sigmuv=np.array(sigmuv)
ellip=np.array(ellip)
absvmag=np.array(absvmag)
sigabsvmag=np.array(sigabsvmag)
rhalf=np.array(rhalf)
sigrhalf=np.array(sigrhalf)
vdisp=np.array(vdisp)
sigvdisp=np.array(sigvdisp)
feh=np.array(feh)
sigfeh=np.array(sigfeh)
method=np.array(method)

for i in range(0,len(rhalf)):
    if ellip[i] > 0.:
        rhalf[i]=rhalf[i]*np.sqrt(1.-ellip[i])########convert to geometric mean radius
        sigrhalf[i]=sigrhalf[i]*np.sqrt(1.-ellip[i])

with open(data2) as f: # read data file
    data=f.readlines()

gcname=[]
gcfeh=[]
crap=[]
gcred=[]
crap=[]
gcdistmodulus=[]
gcvmag=[]
gcabsvmag=[]

for line in data: # fill arrays
    p=line.split()

    gcname.append(str(p[0]))
    gcfeh.append(float(p[1]))
    gcred.append(float(p[3]))
    crap.append(float(p[4]))
    gcdistmodulus.append(float(p[5]))
    gcvmag.append(float(p[6]))
    gcabsvmag.append(float(p[7]))

gcname=np.array(gcname)
gcfeh=np.array(gcfeh)
gcred=np.array(gcred)
crap=np.array(crap)
gcdistmodulus=np.array(gcdistmodulus)
gcvmag=np.array(gcvmag)
gcabsvmag=np.array(gcabsvmag)

gcdist=10.**((gcdistmodulus+5.)/5.)

with open(data3) as f: # read data file
    data=f.readlines()

gcname2=[]
gcvr=[]
gcsigvr=[]
gcvlsr=[]
gcvdisp=[]
gcsigvdisp=[]
gcc=[]
gcrcore=[]
gcrhalf=[]

for line in data: # fill arrays
    p=line.split()
    gcname2.append(str(p[0]))
    gcvr.append(float(p[1]))
    gcsigvr.append(float(p[2]))
    gcvlsr.append(float(p[3]))
    gcvdisp.append(float(p[4]))
    gcsigvdisp.append(float(p[5]))
    gcc.append(float(p[6]))
    gcrcore.append(float(p[7]))
    gcrhalf.append(float(p[8]))

gcvr=np.array(gcvr)
gcsigvr=np.array(gcsigvr)
gcvlsr=np.array(gcvlsr)
gcvdisp=np.array(gcvdisp)
gcsigvdisp=np.array(gcsigvdisp)
gcc=np.array(gcc)
gcrcore=np.array(gcrcore)
gcrhalf=np.array(gcrhalf)

gcrhalf=gcdist*np.tan(gcrhalf/60.*np.pi/180.)

with open(data4) as f: # read data file
    data=f.readlines()

ret2vmean=[]
ret2vdisp=[]
ret2vgrad=[]
ret2vtheta=[]
ret2fehmean=[]
ret2fehdisp=[]
ret2fehgrad=[]
ret2like=[]

for line in data: # fill arrays
    p=line.split()
    ret2vmean.append(float(p[0]))
    ret2vdisp.append(float(p[1]))
    ret2vgrad.append(float(p[2]))
    ret2vtheta.append(float(p[3]))
    ret2fehmean.append(float(p[4]))
    ret2fehdisp.append(float(p[5]))
    ret2fehgrad.append(float(p[6]))
    ret2like.append(float(p[7]))

ret2vmean=np.array(ret2vmean)
ret2vdisp=np.array(ret2vdisp)
ret2vgrad=np.array(ret2vgrad)
ret2vtheta=np.array(ret2vtheta)
ret2fehmean=np.array(ret2fehmean)
ret2fehdisp=np.array(ret2fehdisp)
ret2fehgrad=np.array(ret2fehgrad)
ret2like=np.array(ret2like)

with open(data5) as f: # read data file
    data=f.readlines()

cravmean=[]
cravdisp=[]
cravgrad=[]
cravtheta=[]
crafehmean=[]
crafehdisp=[]
crafehgrad=[]
cralike=[]

for line in data: # fill arrays
    p=line.split()
    cravmean.append(float(p[0]))
    cravdisp.append(float(p[1]))
    cravgrad.append(float(p[2]))
    cravtheta.append(float(p[3]))
    crafehmean.append(float(p[4]))
    crafehdisp.append(float(p[5]))
    crafehgrad.append(float(p[6]))
    cralike.append(float(p[7]))

cravmean=np.array(cravmean)
cravdisp=np.array(cravdisp)
cravgrad=np.array(cravgrad)
cravtheta=np.array(cravtheta)
crafehmean=np.array(crafehmean)
crafehdisp=np.array(crafehdisp)
crafehgrad=np.array(crafehgrad)
cralike=np.array(cralike)


#ret2_vdisp=np.array([np.median(ret2vdisp)])
#ret2_sigvdisp=np.array([np.std(ret2vdisp)])
ret2_feh=np.array([np.median(ret2fehmean)])
ret2_sigfeh=np.array([np.std(ret2fehmean)])
ret2_rhalf=np.random.normal(loc=32.,scale=1.,size=len(ret2vdisp))
#ret2_sigrhalf=np.array([1.])
ret2_absvmag=np.random.normal(loc=-2.7,scale=0.1,size=len(ret2vdisp))
#ret2_sigabsvmag=np.array([0.1])

#cra_vdisp=np.array([np.median(cravdisp)])
#cra_sigvdisp=np.array([np.std(cravdisp)])
cra_feh=np.array([np.median(crafehmean)])
cra_sigfeh=np.array([np.std(crafehmean)])
cra_rhalf=np.random.normal(loc=32.,scale=1.,size=len(cravdisp))
#cra_sigrhalf=np.array([1.])
cra_absvmag=np.random.normal(loc=-5.5,scale=0.5,size=len(cravdisp))
#cra_sigabsvmag=np.array([0.1])

luminosity=10.**((absvmag-4.83)/(-2.5))
sigluminosity=np.log(10.)/2.5*10.**((absvmag-4.83)/(-2.5))*sigabsvmag
mrhalf=1./0.0043*rhalf*vdisp**2
sigmrhalf=np.sqrt(2.*rhalf/0.0043*vdisp*(sigvdisp**2)+((1./0.0043*(vdisp**2))**2)*sigrhalf**2)
mlratio=mrhalf/(luminosity)
sigmlratio=np.sqrt((sigmrhalf**2)/(luminosity**2)+((mrhalf/luminosity**2)**2)*sigluminosity**2)

ret2_luminosity=10.**((ret2_absvmag-4.83)/(-2.5))
#ret2_sigluminosity=np.log(10.)/2.5*10.**((ret2_absvmag-4.83)/(-2.5))*ret2_sigabsvmag
#ret2_mrhalf=1./0.0043*ret2_rhalf*ret2_vdisp**2
ret2_mrhalf=1./0.0043*ret2_rhalf*ret2vdisp**2
#ret2_sigmrhalf=np.sqrt(2.*ret2_rhalf/0.0043*ret2_vdisp*(ret2_sigvdisp**2)+((1./0.0043*(ret2_vdisp**2))**2)*ret2_sigrhalf**2)
ret2_mlratio=ret2_mrhalf/(ret2_luminosity)
#ret2_sigmlratio=np.sqrt((ret2_sigmrhalf**2)/(ret2_luminosity**2)+((ret2_mrhalf/ret2_luminosity**2)**2)*ret2_sigluminosity**2)

cra_luminosity=10.**((cra_absvmag-4.83)/(-2.5))
#cra_sigluminosity=np.log(10.)/2.5*10.**((cra_absvmag-4.83)/(-2.5))*cra_sigabsvmag
#cra_mrhalf=1./0.0043*cra_rhalf*cra_vdisp**2
cra_mrhalf=1./0.0043*cra_rhalf*cravdisp**2
#cra_sigmrhalf=np.sqrt(2.*cra_rhalf/0.0043*cra_vdisp*(cra_sigvdisp**2)+((1./0.0043*(cra_vdisp**2))**2)*cra_sigrhalf**2)
cra_mlratio=cra_mrhalf/(cra_luminosity)
#cra_sigmlratio=np.sqrt((cra_sigmrhalf**2)/(cra_luminosity**2)+((cra_mrhalf/cra_luminosity**2)**2)*cra_sigluminosity**2)

gcluminosity=10.**((gcabsvmag-4.83)/(-2.5))
#gcsigluminosity=np.log(10.)/2.5*10.**((gcabsvmag-4.83)/(-2.5))*gcsigabsvmag
gcmrhalf=1./0.0043*gcrhalf*gcvdisp**2
#gcsigmrhalf=np.sqrt(2.*gcrhalf*gcvdisp*(gcsigvdisp**2)+((1.*(gcvdisp**2))**2)*gcsigrhalf**2)
gcmlratio=gcmrhalf/(gcluminosity/2.)
#gcsigmlratio=np.sqrt((gcsigmrhalf**2)/(gcluminosity**2)+((gcmrhalf/gcluminosity**2)**2)*gcsigluminosity**2)

gs=plt.GridSpec(4,4) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size

ax1_1=fig.add_subplot(gs[1:2,0:2])
ax1_2=fig.add_subplot(gs[0:1,0:2])

#ax1_2.set_xlabel(r'$M_{\rm V}$ [mag]',fontsize=8,rotation=0)
ax1_2.set_ylabel(r'[Fe/H]',fontsize=13,rotation=90,labelpad=5)
ax1_2.set_xlim([0,-15])
ax1_2.set_ylim([-3.5,0.5])
ax1_2.set_xscale(u'linear')
ax1_2.set_yscale(u'linear')
ax1_2.set_xticks([0,-2.5,-5,-7.5,-10,-12.5,-15])
ax1_2.xaxis.set_major_formatter(plt.NullFormatter())
#ax1_2.set_xticklabels(vticks,rotation=0)
ax1_2.set_yticks([-3,-2,-1,0])
ax1_2.set_yticklabels([-3,-2,-1,0],rotation=0,fontsize=10)
ax1_2.scatter(gcabsvmag,gcfeh,s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax1_2.scatter(np.median(ret2_absvmag),ret2_feh,s=20,lw=0,edgecolor='none',alpha=0.99,marker='s',color='b',rasterized=True)
ax1_2.scatter(np.median(cra_absvmag),cra_feh,s=20,lw=0,edgecolor='none',alpha=0.99,marker='s',color='g',rasterized=True)
ax1_2.errorbar(absvmag[(parent == 'MW') & (sigfeh > 0)],feh[(parent == 'MW') & (sigfeh > 0)],xerr=sigabsvmag[(parent == 'MW') & (sigfeh > 0)],yerr=sigfeh[(parent == 'MW') & (sigfeh > 0)],elinewidth=0.25,fmt='o',capsize=0,alpha=1,color='b',rasterized=True)

ax1_2.errorbar(absvmag[(parent == 'M31') & (sigfeh >0)],feh[(parent == 'M31') & (sigfeh >0)],xerr=sigabsvmag[(parent == 'M31') & (sigfeh > 0)],yerr=sigfeh[(parent == 'M31') & (sigfeh > 0)],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
ax1_2.errorbar(np.median(ret2_absvmag),ret2_feh,xerr=np.std(ret2_absvmag),yerr=ret2_sigfeh,elinewidth=0.25,fmt='o',capsize=0,alpha=1,color='b',rasterized=True)
ax1_2.errorbar(np.median(cra_absvmag),cra_feh,xerr=np.std(cra_absvmag),yerr=cra_sigfeh,elinewidth=0.25,fmt='o',capsize=0,alpha=1,color='g',rasterized=True)
#ax1_2.text(np.median(ret2_absvmag)-0.15,ret2_feh[0]-0.4,'Ret2',fontsize=7,color='k')
#ax1_2.text(np.median(cra_absvmag)+0.55,cra_feh[0]+0.4,'Cra',fontsize=7,color='k')

ax1_1.set_xlabel(r'$M_{\rm V}$ [mag]',fontsize=13,rotation=0)
ax1_1.set_ylabel(r'$\frac{R_{h}\sigma^2_{v_{\rm los}}}{GL_{\rm V}}$  [M/L$_{\rm V}$]$_{\odot}$',fontsize=13,rotation=90,labelpad=3)
ax1_1.set_xlim([0,-15])
ax1_1.set_ylim([0.1,900])
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'log')
ax1_1.set_xticks([0,-2.5,-5,-7.5,-10,-12.5,-15])
ax1_1.set_yticks([0.1,1,10,100])
ax1_1.set_xticklabels([0,-2.5,-5,-7.5,-10,-12.5,-15],rotation=0,fontsize=10)
ax1_1.set_yticklabels([0.1,1,10,100],rotation=0,fontsize=10)
#ax1_1.set_yticks([1,10,100,1000])
#ax1_1.xaxis.set_minor_formatter(tick.FuncFormatter(showOnlySomeTicks))
ax1_1.scatter(gcabsvmag[gcvdisp > 0.],gcmlratio[gcvdisp > 0.],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True,label='GCs')

ax1_1.scatter(np.median(ret2_absvmag),np.median(ret2_mlratio),s=20,lw=0,edgecolor='none',alpha=0.99,marker='s',color='b',rasterized=True,label='Ret2')
ax1_1.scatter(np.median(cra_absvmag),np.median(cra_mlratio),s=20,lw=0,edgecolor='none',alpha=0.99,marker='s',color='g',rasterized=True,label='Cra')

ax1_1.errorbar(absvmag[(parent == 'MW') & (sigvdisp > 0)],mlratio[(parent == 'MW') & (sigvdisp > 0)],xerr=sigabsvmag[(parent == 'MW') & (sigvdisp > 0)],yerr=sigmlratio[(parent == 'MW') & (sigvdisp > 0)],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True,label='MW dSph')

ax1_1.errorbar(absvmag[(parent == 'M31') & (sigvdisp > 0)],mlratio[(parent == 'M31') & (sigvdisp > 0)],xerr=sigabsvmag[(parent == 'M31') & (sigvdisp > 0)],yerr=sigmlratio[(parent == 'M31') & (sigvdisp > 0)],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='M31 dSph')

lower_error=[np.percentile(ret2_mlratio,50)-np.percentile(ret2_mlratio,16)]
upper_error=[np.percentile(ret2_mlratio,84)-np.percentile(ret2_mlratio,50)]
ax1_1.errorbar(np.median(ret2_absvmag),np.median(ret2_mlratio),xerr=np.std(ret2_absvmag),yerr=[lower_error,upper_error],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True)
ax1_1.errorbar(np.median(cra_absvmag),np.median(cra_mlratio),xerr=np.std(cra_absvmag),yerr=[lower_error,upper_error],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='g',rasterized=True)
#ax1_1.text(ret2_absvmag[0]-0.15,ret2_mlratio[0]+45,'Ret2',fontsize=7,color='k')
ax1_1.legend(loc=1,fontsize=5,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)


g1.write(r'\newcommand{\mrhalf}{$'+str(round(np.median(2.5*ret2_mrhalf/1.e+5),1))+r'_{'+str(round(np.percentile(2.5*ret2_mrhalf/1.e+5,16)-np.percentile(2.5*ret2_mrhalf/1.e+5,50),1))+r'}^{+'+str(round(np.percentile(2.5*ret2_mrhalf/1.e+5,84)-np.percentile(2.5*ret2_mrhalf/1.e+5,50),1))+r'}\times 10^5$}'+'\n')
g1.write(r'\newcommand{\mlratio}{$'+str(int(np.median(2.*2.5*ret2_mlratio)))+r'_{'+str(int(np.percentile(2.*2.5*ret2_mlratio,16)-np.percentile(2.*2.5*ret2_mlratio,50)))+r'}^{+'+str(int(np.percentile(2.*2.5*ret2_mlratio,84)-np.percentile(2.*2.5*ret2_mlratio,50)))+r'}$}'+'\n')
g1.close()

plotfilename='crater_scaling.pdf'
plt.savefig(plotfilename,dpi=400)
#plt.show()
plt.close()
