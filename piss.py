doit=True
if doit:
    
    gs=plt.GridSpec(10,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:4,0:4])
    ax2=fig.add_subplot(gs[0:4,6:10])

    ax1.plot([-5,1],[-5,1],linestyle='--',color='k',lw=1)
    i=0
    ax1.errorbar(lowz_x[shetrone01],lowz_x[shetrone01]-lowz_y[shetrone01],xerr=lowz_sigx[shetrone01],yerr=np.sqrt((lowz_sigx[shetrone01])**2+lowz_sigy[shetrone01])**2),fmt='.',elinewidth=1,label='Shetrone+2001',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[shetrone03],lowz_x[shetrone03]-lowz_y[shetrone03],xerr=lowz_sigx[shetrone03],yerr=np.sqrt((lowz_sigx[shetrone03])**2+(lowz_sigy[shetrone03])**2),fmt='.',elinewidth=1,label='Shetrone+2003',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[fulbright],lowz_x[fulbright]-lowz_y[fulbright],xerr=lowz_sigx[fulbright],yerr=np.sqrt((lowz_sigx[fulbright])**2+(lowz_sigy[fulbright])**2),fmt='.',elinewidth=1,label='Fulbright+2004',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[sadakane],lowz_x[sadakane]-lowz_y[sadakane],xerr=lowz_sigx[sadakane],yerr=np.sqrt((lowz_sigx[sadakane])**2+(lowz_sigy[sadakane])**2),fmt='.',elinewidth=1,label='Sadakane+2004',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[aoki],lowz_x[aoki]-lowz_y[aoki],xerr=lowz_sigx[aoki],yerr=np.sqrt((lowz_sigx[aoki])**2+(lowz_sigy[aoki])**2),fmt='.',elinewidth=1,label='Aoki+2009',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[cohen],lowz_x[cohen]-lowz_y[cohen],xerr=lowz_sigx[cohen],yerr=np.sqrt((lowz_sigx[cohen])**2+(lowz_sigy[cohen])**2),fmt='.',elinewidth=1,label='Cohen+2009',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[taf],lowz_x[taf]-lowz_y[taf],xerr=lowz_sigx[taf],yerr=np.sqrt((lowz_sigx[taf])**2+(lowz_sigy[taf])**2),fmt='.',elinewidth=1,label='Tafelmeyer+2010',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[frebel10],lowz_x[frebel10]-lowz_y[frebel10],xerr=lowz_sigx[frebel10],yerr=np.sqrt((lowz_sigx[frebel10])**2+(lowz_sigy[frebel10])**2),fmt='.',elinewidth=1,label='Frebel+2010',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[norris],lowz_x[norris]-lowz_y[norris],xerr=lowz_sigx[norris],yerr=np.sqrt((lowz_sigx[norris])**2+(lowz_sigy[norris])**2),fmt='.',elinewidth=1,label='Norris+2010',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[stark],lowz_x[stark]-lowz_y[stark],xerr=lowz_sigx[stark],yerr=np.sqrt((lowz_sigx[stark])**2+(lowz_sigy[stark])**2),fmt='.',elinewidth=1,label='Starkenburg+2013',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[simon15],lowz_x[simon15]-lowz_y[simon15],xerr=lowz_sigx[simon15],yerr=np.sqrt((lowz_sigx[simon15])**2+(lowz_sigy[simon15])**2),fmt='.',elinewidth=1,label='Simon+2015',rasterized=True,markersize=3)
    i+=1
    ax1.errorbar(lowz_x[lucc],lowz_x[lucc]-lowz_y[lucc],xerr=lowz_sigx[lucc],yerr=np.sqrt((lowz_sigx[lucc])**2+(lowz_sigy[lucc])**2),fmt='.',elinewidth=1,label='Lucchesi+2020',rasterized=True,markersize=3)
#    ax1.errorbar(lowz_x[geisler],lowz_y[geisler],xerr=lowz_sigx[geisler],yerr=lowz_sigy[geisler],fmt='.',elinewidth=1,label='Geislerhesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone03],lowz_y[shetrone03],xerr=lowz_sigx[shetrone03],yerr=lowz_sigy[shetrone03],fmt='.',elinewidth=1,label='Shetrone03hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[letarte],lowz_y[letarte],xerr=lowz_sigx[letarte],yerr=lowz_sigy[letarte],fmt='.',elinewidth=1,label='Letartehesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone01],lowz_y[shetrone01],xerr=lowz_sigx[shetrone01],yerr=lowz_sigy[shetrone01],fmt='.',elinewidth=1,label='Shetrone01hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[simon10],lowz_y[simon10],xerr=lowz_sigx[simon10],yerr=lowz_sigy[simon10],fmt='.',elinewidth=1,label='Simon10hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone98],lowz_y[shetrone98],xerr=lowz_sigx[shetrone98],yerr=lowz_sigy[shetrone98],fmt='.',elinewidth=1,label='Shetrone98hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[koch],lowz_y[koch],xerr=lowz_sigx[koch],yerr=lowz_sigy[koch],fmt='.',elinewidth=1,label='Kochhesi+2020',rasterized=True)

    ax1.set_xlim([-4.5,1])
    ax1.set_ylim([-4.5,1])
    ax1.set_xticks([-4,-3,-2,-1,0,1])
    ax1.set_yticks([-4,-3,-2,-1,0,1])
    ax1.set_xlabel('[Fe/H], this work')
    ax1.set_ylabel('[Fe/H], previous')
    ax1.legend(loc=2,fontsize=4)
    plt.savefig('compare_lowmetallicity1_dev.pdf',dpi=300)
    plt.show()
    plt.close()
    
    gs=plt.GridSpec(10,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:4,0:4])
    ax2=fig.add_subplot(gs[0:4,6:10])

    ax2.plot([-5,1],[-5,1],linestyle='--',color='k',lw=1)
    i=0
    ax2.errorbar(lowz_x2[shetrone01],lowz_x2[shetrone01]-lowz_y2[shetrone01],xerr=lowz_sigx2[shetrone01],yerr=np.sqrt((lowz_sigx2[shetrone01)**2+(lowz_sigy2[shetrone01])**2),fmt='.',elinewidth=1,label='Shetrone+2001',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[shetrone03],lowz_x2[shetrone03]-lowz_y2[shetrone03],xerr=lowz_sigx2[shetrone03],yerr=np.sqrt((lowz_sigx2[shetrone03)**2+(lowz_sigy2[shetrone03])**2),fmt='.',elinewidth=1,label='Shetrone+2003',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[fulbright],lowz_x2[fulbright]-lowz_y2[fulbright],xerr=lowz_sigx2[fulbright],yerr=np.sqrt((lowz_sigx2[fulbright)**2+(lowz_sigy2[fulbright])**2),fmt='.',elinewidth=1,label='Fulbright+2004',rasterized=True,markersize=3)
    i+=1
#    ax2.errorbar(lowz_x2[sadakane],lowz_x2[sadakane]-lowz_y2[sadakane],xerr=lowz_sigx2[sadakane],yerr=np.sqrt((lowz_sigx2[sadakane)**2+(lowz_sigy2[sadakane])**2),fmt='.',elinewidth=1,label='Sadakane+2004',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[aoki],lowz_x2[aoki]-lowz_y2[aoki],xerr=lowz_sigx2[aoki],yerr=np.sqrt((lowz_sigx2[aoki)**2+(lowz_sigy2[aoki])**2),fmt='.',elinewidth=1,label='Aoki+2009',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[cohen],lowz_x2[cohen]-lowz_y2[cohen],xerr=lowz_sigx2[cohen],yerr=np.sqrt((lowz_sigx2[cohen)**2+(lowz_sigy2[cohen])**2),fmt='.',elinewidth=1,label='Cohen+2009',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[taf],lowz_x2[taf]-lowz_y2[taf],xerr=lowz_sigx2[taf],yerr=np.sqrt((lowz_sigx2[taf)**2+(lowz_sigy2[taf])**2),fmt='.',elinewidth=1,label='Tafelmeyer+2010',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[frebel10],lowz_x2[frebel10]-lowz_y2[frebel10],xerr=lowz_sigx2[frebel10],yerr=np.sqrt((lowz_sigx2[frebel10)**2+(lowz_sigy2[frebel10])**2),fmt='.',elinewidth=1,label='Frebel+2010',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[norris],lowz_x2[norris]-lowz_y2[norris],xerr=lowz_sigx2[norris],yerr=np.sqrt((lowz_sigx2[norris)**2+(lowz_sigy2[norris])**2),fmt='.',elinewidth=1,label='Norris+2010',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[stark],lowz_x2[stark]-lowz_y2[stark],xerr=lowz_sigx2[stark],yerr=np.sqrt((lowz_sigx2[stark)**2+(lowz_sigy2[stark])**2),fmt='.',elinewidth=1,label='Starkenburg+2013',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[simon15],lowz_x2[simon15]-lowz_y2[simon15],xerr=lowz_sigx2[simon15],yerr=np.sqrt((lowz_sigx2[simon15)**2+(lowz_sigy2[simon15])**2),fmt='.',elinewidth=1,label='Simon+2015',rasterized=True,markersize=3)
    i+=1
    ax2.errorbar(lowz_x2[lucc],lowz_x2[lucc]-lowz_y2[lucc],xerr=lowz_sigx2[lucc],yerr=np.sqrt((lowz_sigx2[lucc)**2+(lowz_sigy2[lucc])**2),fmt='.',elinewidth=1,label='Lucchesi+2020',rasterized=True,markersize=3)

    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.set_xlabel('[Mg/Fe], this work')
    ax2.set_ylabel('[Mg/Fe], previous')
#    ax2.legend(loc=2,fontsize=5)

    plt.savefig('compare_lowmetallicity2_dev.pdf',dpi=300)
    plt.show()
    plt.close()

