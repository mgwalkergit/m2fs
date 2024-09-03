import numpy as np
import dill as pickle
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mycode as mc

data_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_data/'
template_library_filename='/hildafs/projects/phy200028p/mgwalker/hecto_data/ian_hires_lsf.pkl'
template_library_labels_filename='/hildafs/projects/phy200028p/mgwalker/hecto_data/ian_hires_lsf'

spec_filename='/hildafs/projects/phy200028p/mgwalker/m2fs_data/m2fs_ian_oct23_tuc2_tuc2_Mgb_HiRes_ra343.467564_dec-58.407529_hjd2460225.542_apB079_skysub.dat'
spec=pd.read_csv(spec_filename,delim_whitespace=True,names=['obslambda','na1','na2','obscounts','varobscounts'])
keep=np.where((spec.varobscounts<1.e+10)&(spec.varobscounts>0.))[0]
obslambda=np.array(spec.obslambda[keep])
obscounts=np.array(spec.obscounts[keep])
varobscounts=np.array(spec.varobscounts[keep])
maxobscounts=np.max(obscounts)

lambdamin=5127.
lambdamax=5190.
kernel=np.array([300.,0.5,0.25,0.2,0.03])
c=3.e+5
lambdascale=0.5*(lambdamax-lambdamin)
lambda0=lambdamin+lambdascale
ilambdascale=1./lambdascale
ic=1./c
ikernel2=1./kernel**2

template_library_exists=path.exists(template_library_filename)

if template_library_exists:
    
    template_library_labels,template_library_lambda,template_library_specnorm=pickle.load(open(template_library_filename,'rb'))

else:
    
    template_library_labels=pd.read_csv(template_library_labels_filename,delim_whitespace=True,names=['filename','teff','logg','feh','alpha','sigma'])
    
    template_library_lambda=pd.read_csv(template_library_labels.filename[0],delim_whitespace=True,usecols=[0],header=None).to_numpy().flatten()

    template_library_specnorm=np.zeros((len(template_library_labels.filename),len(template_library_lambda)))
    
    for i in range(0,len(template_library_labels.filename)):
        print('packing '+str(i)+' of '+str(len(template_library_labels.filename)))
        template_library_specnorm[i,:]=pd.read_csv(template_library_labels.filename[i],delim_whitespace=True,usecols=[1],header=None).to_numpy().flatten()

    pickle.dump((template_library_labels,template_library_lambda,template_library_specnorm),open(template_library_filename,'wb'))

dliblambda=(np.max(template_library_lambda)-np.min(template_library_lambda))/len(template_library_lambda)

label_min=np.array([np.min(template_library_labels.to_numpy().T[q]) for q in [1,2,3,4,5]])
label_max=np.array([np.max(template_library_labels.to_numpy().T[q]) for q in [1,2,3,4,5]])
label_avg=(label_min+label_max)/2.
label_ngrid=(label_max-label_min)/kernel
label_span=label_max-label_min

teff0=np.array((template_library_labels['teff']-label_avg[0])/label_span[0]*label_ngrid[0])
logg0=np.array((template_library_labels['logg']-label_avg[1])/label_span[1]*label_ngrid[1])
feh0=np.array((template_library_labels['feh']-label_avg[2])/label_span[2]*label_ngrid[2])
alpha0=np.array((template_library_labels['alpha']-label_avg[3])/label_span[3]*label_ngrid[3])
sigma0=np.array((template_library_labels['sigma']-label_avg[4])/label_span[4]*label_ngrid[4])

label_tree_arrays=np.c_[teff0,logg0,feh0,alpha0,sigma0]
label_tree=scipy.spatial.KDTree(label_tree_arrays)

fit_spec_prefix='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/testxxx'
fit_spec_prior=([-500.,500],[3900.,7500.],[0.,5.],[-4.,0.5],[-0.8,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[0.06,0.12],[-10.,10.],[-10.,10.],[-1.,6.],[-2.,2.])
fit_spec_parameters=['deltav','teff','logg','feh','alpha','a0','a1','a2','a3','a4','a5','sigma','v1','v2','phantom','phantom2']
fit_spec_resume=False

#shite=mc.fit_spec_loglike0([-500.,3900.,0,-4.,-0.08,-1.,-1.,-1.,-1.,-1.,-1.,0.06,-10.,-10.,-1.,-2.],obslambda,obscounts,varobscounts,template_library_labels,template_library_lambda,template_library_specnorm,label_tree,label_avg=label_avg,label_span=label_span,label_ngrid=label_ngrid,prior=fit_spec_prior,parameters=fit_spec_parameters,prefix=fit_spec_prefix,resume=fit_spec_resume,lambdamin=lambdamin,lambdamax=lambdamax,lambda0=lambda0,ilambdascale=ilambdascale,ikernel2=ikernel2,ic=ic,dliblambda=dliblambda,maxobscounts=maxobscounts)

#np.pause()

fit_spec_result,fit_spec_bestfit=mc.fit_spec(obslambda,obscounts,varobscounts,template_library_labels,template_library_lambda,template_library_specnorm,label_tree,label_avg=label_avg,label_span=label_span,label_ngrid=label_ngrid,prior=fit_spec_prior,parameters=fit_spec_parameters,prefix=fit_spec_prefix,resume=fit_spec_resume,lambdamin=lambdamin,lambdamax=lambdamax,lambda0=lambda0,ilambdascale=ilambdascale,ikernel2=ikernel2,ic=ic,dliblambda=dliblambda,maxobscounts=maxobscounts)
#fit_spec_result=mc.fit_spec(obslambda,obscounts,varobscounts,template_library_labels,template_library_lambda,template_library_specnorm,label_tree,label_avg=label_avg,label_span=label_span,label_ngrid=label_ngrid,prior=fit_spec_prior,parameters=fit_spec_parameters,prefix=fit_spec_prefix,resume=fit_spec_resume,lambdamin=lambdamin,lambdamax=lambdamax,lambda0=lambda0,ilambdascale=ilambdascale,ikernel2=ikernel2,ic=ic,dliblambda=dliblambda,maxobscounts=maxobscounts)

