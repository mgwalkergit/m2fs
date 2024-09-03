import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import m2fs_process as m2fs
import os
from isolate_model_result import Model
import scipy
from scipy.spatial import distance
import mycode
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
from pymultinest.solve import solve
from pymultinest import Analyzer
#matplotlib.use('pdf')

data_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_data/'
fits_list0='all_m2fshiresian_files'
fits_list='/hildafs/projects/phy200028p/mgwalker/m2fs/'+fits_list0

model_paras_files = "/hildafs/projects/phy200028p/mgwalker/m2fs/final_mask_MLP_clamped_76c0e105c350c062a69a_iter600.pt"
model = Model(model_paras_files)

lambdamin=5127.
lambdamax=5190.
liblambdamin0=5055.
liblambdamax0=5345.
dliblambda=0.05
nliblambda=np.long((liblambdamax0-liblambdamin0)/dliblambda)+1
liblambda=np.linspace(liblambdamin0,liblambdamax0,nliblambda)

input_paras=np.array([4321.,4.321,-3.21,-0.21])#pull dummy model so we can get wavelengths
specmodel=model(input_paras)
model_wav=model.wavelength
xa=[]
xb=[]
for i in range(0,len(model_wav)):
    xa.append((model_wav[i],))
    xb.append((model_wav[i],))
cdist2=distance.cdist(xa,xb,metric='sqeuclidean')

sigma0=np.arange(0.01,0.26,0.01)#smoothing bandwidth array for lookup table
weights=[]
sumweights=[]
for i in range(0,len(sigma0)):
    weights0=np.exp(-0.5*cdist2/sigma0[i]**2)
    weights.append(weights0)
    sumweights.append(np.dot(weights0,np.ones(len(model_wav))))
weights=np.array(weights)
sumweights=np.array(sumweights)



with open(fits_list) as f:
    data=f.readlines()
fitsfile=[]
obj=[]
for line in data:
    p=line.split()
    fitsfile.append(p[0])
    obj.append(p[1])
fitsfile=np.array(fitsfile)
obj=np.array(obj)
for i in range(0,len(fitsfile)):
    this=np.where(fitsfile==fitsfile[i])[0]
    if len(this)>1:
        print('ERROR: multiple listings of same observation in input file!!!')
        print(this)
        np.pause()
        
def getmodelspec(cube,wav,spec,varspec):
    from isolate_model_result import Model

    deltav=cube[0]
    teff=cube[1]
    logg=cube[2]
    feh=cube[3]
    alpha=cube[4]
    a0=cube[5]
    a1=cube[6]
    a2=cube[7]
    a3=cube[8]
    a4=cube[9]
    a5=cube[10]
    sigma=cube[11]
    v1=cube[12]
    v2=cube[13]
    phantom=10.**cube[14]
    phantom2=10.**cube[15]
    
    v=deltav
    v0=0.

    input_paras=np.array([teff,logg,feh,alpha])
    specmodel=model(input_paras)
    model_wav=model.wavelength

    dist=(sigma-sigma0)**2
    best=np.argsort(dist)[0]
    smoothed=np.dot(specmodel,weights[best])/sumweights[best]

    c=3.e+5
    lambdascale=0.5*(lambdamax-lambdamin)
    lambda0=lambdamin+lambdascale
    ilambdascale=1./lambdascale
    ic=1./c
    isigma2=1./sigma**2
    
    vmin=v-v0-v1-v2
    vmax=v+v0+v1+v2
    liblambdamin=lambdamin/(1.+vmax*ic)
    liblambdamax=lambdamax/(1.+vmin*ic)
    jmin=int((liblambdamin-liblambdamin0)/dliblambda)
    jmax=int((liblambdamax-liblambdamin0)/dliblambda)+1
        
    j0=np.arange(jmin,jmax,1,dtype='int')
    velocity=v+v0+v1*((liblambda[j0]-lambda0)*ilambdascale)+v2*((liblambda[j0]-lambda0)*ilambdascale)**2
    liblambdatwiddle=liblambda[j0]*(1.+velocity*ic)

    maxobscounts=np.max(spec[((varspec>0.)&(varspec<1.e+10))])
    polynomial=maxobscounts*a0+maxobscounts*(
        a1*((wav-lambda0)*ilambdascale)**1
        +a2*((wav-lambda0)*ilambdascale)**2
        +a3*((wav-lambda0)*ilambdascale)**3
        +a4*((wav-lambda0)*ilambdascale)**4
        +a5*((wav-lambda0)*ilambdascale)**5)

#    print(len(wav),len(liblambdatwiddle),len(smoothed),maxobscounts)
    interp=np.interp(wav,liblambdatwiddle,smoothed[jmin:jmax])

    return polynomial*interp

def run_nest(wav,spec,varspec,mask,prefix):

    from isolate_model_result import Model

    def myprior(cube):
        prior=[]
        prior.append([-500.,500.])#log of scaling constant that sets member fraction
        prior.append([3900.,7500.])
        prior.append([0.,5.])
        prior.append([-4.,0.5])
        prior.append([-0.8,1.])
        prior.append([-1.,1.])
        prior.append([-1.,1.])
        prior.append([-1.,1.])
        prior.append([-1.,1.])
        prior.append([-1.,1.])
        prior.append([-1.,1.])
        prior.append([0.06,0.120])
        prior.append([-10.,10.])
        prior.append([-10.,10.])
        prior.append([-1.,6.])
        prior.append([-2.,2.])
        prior=np.array(prior)
        x=np.array(cube)
        for i in range(0,len(x)):
            x[i]=prior[i][0]+(prior[i][1]-prior[i][0])*cube[i]
        return x

    def myloglike(cube):

        libsmooth=getmodelspec(cube,wav,spec,varspec)
#        deltav=cube[0]
        teff=cube[1]
        logg=cube[2]
        feh=cube[3]
        alpha=cube[4]
#        a0=cube[5]
#        a1=cube[6]
#        a2=cube[7]
#        a3=cube[8]
#        a4=cube[9]
#        a5=cube[10]
#        sigma=cube[11]
#        v1=cube[12]
#        v2=cube[13]
        phantom=10.**cube[14]
        phantom2=10.**cube[15]
    
#        v=deltav
#        v0=0.

#        input_paras=np.array([teff,logg,feh,alpha])
#        specmodel=model(input_paras)

#        model_wav=model.wavelength

#        weights=[]
#        for j in range(0,len(model_wav)): 
#            weights.append(np.sum(np.exp(-0.5*cdist2[j]/sigma**2)))
#        weights=np.array(weights)

#        smoothed=[]
#        for j in range(0,len(model_wav)):
#            smoothed.append(np.sum(specmodel*np.exp(-0.5*cdist2[j]/sigma**2))/weights[j])
#        smoothed=np.array(smoothed)

#        c=3.e+5
#        lambdascale=0.5*(lambdamax-lambdamin)
#        lambda0=lambdamin+lambdascale
#        ilambdascale=1./lambdascale
#        ic=1./c
#        isigma2=1./sigma**2
    
#        vmin=v-v0-v1-v2
#        vmax=v+v0+v1+v2
#        liblambdamin=lambdamin/(1.+vmax*ic)
#        liblambdamax=lambdamax/(1.+vmin*ic)
#        jmin=int((liblambdamin-liblambdamin0)/dliblambda)
#        jmax=int((liblambdamax-liblambdamin0)/dliblambda)+1
        
#        j0=np.arange(jmin,jmax,1,dtype='int')
#        velocity=v+v0+v1*((liblambda[j0]-lambda0)*ilambdascale)+v2*((liblambda[j0]-lambda0)*ilambdascale)**2
#        liblambdatwiddle=liblambda[j0]*(1.+velocity*ic)

#        maxobscounts=np.max(spec[((varspec>0.)&(varspec<1.e+10))])
#        polynomial=maxobscounts*a0+maxobscounts*(
#            a1*((wav-lambda0)*ilambdascale)**1
#            +a2*((wav-lambda0)*ilambdascale)**2
#            +a3*((wav-lambda0)*ilambdascale)**3
#            +a4*((wav-lambda0)*ilambdascale)**4
#            +a5*((wav-lambda0)*ilambdascale)**5)
    
#        interp=np.interp(wav,liblambdatwiddle,smoothed[jmin:jmax])
#        libsmooth=polynomial*interp

        keep=np.where(varspec<1.e+8)[0]
        sum1=-0.5*np.log(2.*np.pi)*int(len(keep))
        sum2=-0.5*np.sum(np.log(phantom2*varspec[keep]+phantom**2))
        sum3=-0.5*np.sum((spec[keep]-libsmooth[keep])**2/(phantom2*varspec[keep]+phantom**2))

        logl=sum1+sum2+sum3
        if ((feh<-2.5)&(logg>4.5)):
            logl=-1.e+30
        if ((teff>6000.)&(logg<1.)):
            logl=-1.e+30

        return logl

    parameters=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    n_params=len(parameters)
    result=solve(LogLikelihood=myloglike,Prior=myprior,n_dims=n_params,outputfiles_basename=prefix,verbose=True,resume=True,init_MPI=False,use_MPI=False)    
    return result

for i in range(0,len(fitsfile)):
    print(i,' of ',len(fitsfile))
    hdul=fits.open(fitsfile[i])
    fitsobject=m2fs.m2fs_getfromfits(hdul)
    filtername=hdul[0].header['filtername']
    m2fsrun=hdul[0].header['m2fsrun']
    field_name=hdul[0].header['field_name']
    temperature=hdul[0].header['dome_temp']
    hdul.close()
    
    root=[]
    root2=[]
    for j in range(0,len(fitsobject.obj)):
        root.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
        root2.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
    root=np.array(root)
    root2=np.array(root2)

    skies=np.where(fitsobject.obj=='SKY')[0]
    targets=np.where(fitsobject.icode>0)[0]

    if len(targets)==0:
        print('WARNING: NO TARGETS')
    for j in targets:
        out=data_directory+root[j]+'_skysub.dat'
        if ((len(np.where(fitsobject.mask[j]==False)[0])>0)&(fitsobject.filtername[j]=='HiRes')):#write skysub.dat file only if >100 good pixels in spectrum
            wav=[]
            spec=[]
            varspec=[]
            mask=[]
            for k in range(0,len(fitsobject.wav[j])):
                wav.append(fitsobject.wav[j][k])
                spec.append(fitsobject.spec[j][k])
                varspec.append(fitsobject.var[j][k])
                mask.append(fitsobject.mask[j][k])
            wav=np.array(wav)
            spec=np.array(spec)
            varspec=np.array(varspec)
            mask=np.array(mask)

            cube=[]
            cube.append(0.)#log of scaling constant that sets member fraction
            cube.append(5800.)
            cube.append(4.7)
            cube.append(0.)
            cube.append(0.)
            cube.append(0.1)
            cube.append(0.2)
            cube.append(0.3)
            cube.append(0.4)
            cube.append(0.5)
            cube.append(0.6)
            cube.append(0.09)
            cube.append(0.)
            cube.append(0.)
            cube.append(0.)
            cube.append(0.)
            cube=np.array(cube)
#            shite=getmodelspec(cube,wav,spec,varspec)
#            np.pause()
            prefix='/hildafs/projects/phy200028p/mgwalker/m2fs/chains/'+root2[j]

            shite=run_nest(wav,spec,varspec,mask,prefix)
#            a=Analyzer(n_params,outputfiles_basename=prefix)
#            bestfit=a.get_best_fit()
