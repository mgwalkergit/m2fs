import numpy as np
import dill as pickle
import m2fs_process as m2fs

g1.close()
poo=True
if poo:
    g1=open('offsets_newcommands.tex','w')

    string='\\newcommand{\mtwofshireshectovn}{'+str(len(m2fshires_hecto_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireshthreevn}{'+str(len(m2fshires_h3_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireswalkervn}{'+str(len(m2fshires_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresapogeevn}{'+str(len(m2fshires_apogee_compare_v.T[0][m2fshires_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresssppvn}{'+str(len(m2fshires_sspp_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresmtwofsmedresvn}{'+str(len(m2fshires_m2fsmedres_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresgaiavn}{'+str(len(m2fshires_gaia_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreevn}{'+str(len(hecto_h3_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectowalkervn}{'+str(len(hecto_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeevn}{'+str(len(hecto_apogee_compare_v.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppvn}{'+str(len(hecto_sspp_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectomtwofsmedresvn}{'+str(len(hecto_m2fsmedres_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectogaiavn}{'+str(len(hecto_gaia_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreewalkervn}{'+str(len(h3_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeevn}{'+str(len(h3_apogee_compare_v.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreemtwofsmedresvn}{'+str(len(h3_m2fsmedres_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreegaiavn}{'+str(len(h3_gaia_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\walkerapogeevn}{'+str(len(walker_apogee_compare_v.T[0][walker_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\walkermtwofsmedresvn}{'+str(len(walker_m2fsmedres_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\walkergaiavn}{'+str(len(walker_gaia_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeevn}{'+str(len(m2fsmedres_apogee_compare_v.T[0][m2fsmedres_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgaiavn}{'+str(len(m2fsmedres_gaia_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\gaiaapogeevn}{'+str(len(gaia_apogee_compare_v.T[0][gaia_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofshireshectoteffn}{'+str(len(m2fshires_hecto_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireskirbyteffn}{'+str(len(m2fshires_kirby_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireshthreeteffn}{'+str(len(m2fshires_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresapogeeteffn}{'+str(len(m2fshires_apogee_compare_teff.T[0][m2fshires_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresssppteffn}{'+str(len(m2fshires_sspp_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresmtwofsmedresteffn}{'+str(len(m2fshires_m2fsmedres_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresgaiateffn}{'+str(len(m2fshires_gaia_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyteffn}{'+str(len(hecto_kirby_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreeteffn}{'+str(len(hecto_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeeteffn}{'+str(len(hecto_apogee_compare_teff.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppteffn}{'+str(len(hecto_sspp_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectomtwofsmedresteffn}{'+str(len(hecto_m2fsmedres_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectogaiateffn}{'+str(len(hecto_gaia_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreeteffn}{'+str(len(kirby_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeeteffn}{'+str(len(kirby_apogee_compare_teff.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
#    string='\\newcommand{\kirbymtwofsmedresteffn}{'+str(len(kirby_m2fsmedres_compare_teff.T[0]))+'} \n'
#    g1.write(string)
    string='\\newcommand{\kirbygaiateffn}{'+str(len(kirby_gaia_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeeteffn}{'+str(len(h3_apogee_compare_teff.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreemtwofsmedresteffn}{'+str(len(h3_m2fsmedres_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreegaiateffn}{'+str(len(h3_gaia_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeeteffn}{'+str(len(m2fsmedres_apogee_compare_teff.T[0][m2fsmedres_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgaiateffn}{'+str(len(m2fsmedres_gaia_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\gaiaapogeeteffn}{'+str(len(gaia_apogee_compare_teff.T[0][gaia_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofshireshectologgn}{'+str(len(m2fshires_hecto_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireskirbyloggn}{'+str(len(m2fshires_kirby_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireshthreeloggn}{'+str(len(m2fshires_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresapogeeloggn}{'+str(len(m2fshires_apogee_compare_logg.T[0][m2fshires_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiressspploggn}{'+str(len(m2fshires_sspp_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresmtwofsmedresloggn}{'+str(len(m2fshires_m2fsmedres_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresgaialoggn}{'+str(len(m2fsmedres_gaia_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyloggn}{'+str(len(hecto_kirby_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreeloggn}{'+str(len(hecto_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeeloggn}{'+str(len(hecto_apogee_compare_logg.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectosspploggn}{'+str(len(hecto_sspp_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectomtwofsmedresloggn}{'+str(len(hecto_m2fsmedres_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectogaialoggn}{'+str(len(hecto_gaia_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreeloggn}{'+str(len(kirby_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeeloggn}{'+str(len(kirby_apogee_compare_logg.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
#    string='\\newcommand{\kirbymtwofsmedresloggn}{'+str(len(kirby_m2fsmedres_compare_logg.T[0]))+'} \n'
#    g1.write(string)
    string='\\newcommand{\kirbygaialoggn}{'+str(len(kirby_gaia_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeeloggn}{'+str(len(h3_apogee_compare_logg.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreemtwofsmedresloggn}{'+str(len(h3_m2fsmedres_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreegaialoggn}{'+str(len(h3_gaia_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeeloggn}{'+str(len(m2fsmedres_apogee_compare_logg.T[0][m2fsmedres_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsgaialoggn}{'+str(len(m2fsmedres_gaia_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\gaiaapogeeloggn}{'+str(len(gaia_apogee_compare_logg.T[0][gaia_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofshireshectofehn}{'+str(len(m2fshires_hecto_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireskirbyfehn}{'+str(len(m2fshires_kirby_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireshthreefehn}{'+str(len(m2fshires_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresapogeefehn}{'+str(len(m2fshires_apogee_compare_z.T[0][m2fshires_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresssppfehn}{'+str(len(m2fshires_sspp_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresmtwofsmedresfehn}{'+str(len(m2fshires_m2fsmedres_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresgaiafehn}{'+str(len(m2fshires_gaia_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyfehn}{'+str(len(hecto_kirby_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreefehn}{'+str(len(hecto_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeefehn}{'+str(len(hecto_apogee_compare_z.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppfehn}{'+str(len(hecto_sspp_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectomtwofsmedresfehn}{'+str(len(hecto_m2fsmedres_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectogaiafehn}{'+str(len(hecto_gaia_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreefehn}{'+str(len(kirby_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeefehn}{'+str(len(kirby_apogee_compare_z.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
#    string='\\newcommand{\kirbymtwofsmedresfehn}{'+str(len(kirby_m2fsmedres_compare_z.T[0]))+'} \n'
#    g1.write(string)
    string='\\newcommand{\kirbygaiafehn}{'+str(len(kirby_gaia_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeefehn}{'+str(len(h3_apogee_compare_z.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreemtwofsmedresfehn}{'+str(len(h3_m2fsmedres_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreegaiafehn}{'+str(len(h3_gaia_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeefehn}{'+str(len(m2fsmedres_apogee_compare_z.T[0][m2fsmedres_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsgaiafehn}{'+str(len(m2fsmedres_gaia_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\gaiaapogeefehn}{'+str(len(gaia_apogee_compare_z.T[0][gaia_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofshireshectoalphan}{'+str(len(m2fshires_hecto_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireskirbyalphan}{'+str(len(m2fshires_kirby_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshireshthreealphan}{'+str(len(m2fshires_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresapogeealphan}{'+str(len(m2fshires_apogee_compare_alpha.T[0][m2fshires_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofshiresmtwofsmedresalphan}{'+str(len(m2fshires_m2fsmedres_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyalphan}{'+str(len(hecto_kirby_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreealphan}{'+str(len(hecto_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeealphan}{'+str(len(hecto_apogee_compare_alpha.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectomtwofsmedresalphan}{'+str(len(hecto_m2fsmedres_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreealphan}{'+str(len(kirby_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeealphan}{'+str(len(kirby_apogee_compare_alpha.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
#    string='\\newcommand{\kirbymtwofsmedresalphan}{'+str(len(kirby_m2fsmedres_compare_alpha.T[0]))+'} \n'
#    g1.write(string)
    string='\\newcommand{\hthreeapogeealphan}{'+str(len(h3_apogee_compare_alpha.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreemtwofsmedresalphan}{'+str(len(h3_m2fsmedres_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeealphan}{'+str(len(m2fsmedres_apogee_compare_alpha.T[0][m2fsmedres_apogee_keep1]))+'} \n'
    g1.write(string)
    
    string='\\newcommand{\mtwofshiresvoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectovoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyvoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreevoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkervoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[4])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresvoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[5])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[5])))+'$} \n'
    g1.write(string)
#    string='\\newcommand{\gaiavoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[7])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[7])))+'$} \n'
#    g1.write(string)

    string='\\newcommand{\mtwofshiresteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[0])))+'\pm '+str(int(np.std(teff_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[1])))+'\pm '+str(int(np.std(teff_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[2])))+'\pm '+str(int(np.std(teff_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreeteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[3])))+'\pm '+str(int(np.std(teff_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[4])))+'\pm '+str(int(np.std(teff_result['samples'].T[4])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[5])))+'\pm '+str(int(np.std(teff_result['samples'].T[5])))+'$} \n'
    g1.write(string)
#    string='\\newcommand{\gaiateffoffset}{$'+str(int(np.mean(teff_result['samples'].T[7])))+'\pm '+str(int(np.std(teff_result['samples'].T[7])))+'$} \n'
#    g1.write(string)

    string='\\newcommand{\mtwofshiresloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectologgoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreeloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[4])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[5])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[5])))+'$} \n'
    g1.write(string)
#    string='\\newcommand{\gaialoggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[7])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[7])))+'$} \n'
#    g1.write(string)

    string='\\newcommand{\mtwofshiresfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectofehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreefehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[4])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[5])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[5])))+'$} \n'
    g1.write(string)
#    string='\\newcommand{\gaiafehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[7])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[7])))+'$} \n'
#    g1.write(string)

    string='\\newcommand{\mtwofshiresalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreealphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkeralphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[4])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[5])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[5])))+'$} \n'
    g1.write(string)

    g1.close()

    gs=plt.GridSpec(17,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax11=fig.add_subplot(gs[0:3,0:6])
    ax21=fig.add_subplot(gs[5:8,0:6])
    ax31=fig.add_subplot(gs[8:11,0:6])
    ax41=fig.add_subplot(gs[11:14,0:6])
#    ax51=fig.add_subplot(gs[14:17,0:6])

    m2fshires_keep=np.where((m2fshires_hecto_compare_v.T[1]<5.)&(m2fshires_hecto_compare_v.T[3]<5.))[0]
    m2fsmedres_keep=np.where((m2fshires_m2fsmedres_compare_v.T[1]<5.)&(m2fshires_m2fsmedres_compare_v.T[3]<5.))[0]
    ax11.errorbar(m2fshires_hecto_compare_v.T[2][m2fshires_keep],m2fshires_hecto_compare_v.T[2][m2fshires_keep]-m2fshires_hecto_compare_v.T[0][m2fshires_keep],xerr=m2fshires_hecto_compare_v.T[1][m2fshires_keep],yerr=np.sqrt((m2fshires_hecto_compare_v.T[1][m2fshires_keep])**2+(m2fshires_hecto_compare_v.T[3][m2fshires_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True,label='Hecto')
    ax11.errorbar(m2fshires_m2fsmedres_compare_v.T[2][m2fsmedres_keep],m2fshires_m2fsmedres_compare_v.T[2][m2fsmedres_keep]-m2fshires_m2fsmedres_compare_v.T[0][m2fsmedres_keep],xerr=m2fshires_m2fsmedres_compare_v.T[3][m2fsmedres_keep],yerr=np.sqrt((m2fshires_m2fsmedres_compare_v.T[3][m2fsmedres_keep])**2+(m2fshires_m2fsmedres_compare_v.T[1][m2fsmedres_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True,label='M2FS MedRes')
    m2fshires_offset=np.mean(vlos_result['samples'].T[1])-np.mean(vlos_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-x0-m2fshires_offset
    ax11.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax11.set_xlim([-400,400])
    ax11.set_ylim([-10,10])
    ax11.set_yticks([-10,-5,0,5,10])
    ax11.set_xticks([-400,-200,0,200,400])
    ax11.set_xticklabels([-400,-200,0,200,400],fontsize=7)
    ax11.set_yticklabels([-10,-5,0,5,10],fontsize=7)
    ax11.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)
    ax11.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,\, M2FS\, HiRes}$ [km/s]',fontsize=7)
    ax11.legend(loc=2,fontsize=5,borderaxespad=0)
    
    m2fshires_keep=np.where((m2fshires_walker_compare_v.T[1]<5.)&(m2fshires_walker_compare_v.T[3]<5.))[0]
    m2fsmedres_keep=np.where((walker_m2fsmedres_compare_v.T[1]<5.)&(walker_m2fsmedres_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_walker_compare_v.T[1]<5.)&(hecto_walker_compare_v.T[3]<5.))[0]
    ax21.errorbar(m2fshires_walker_compare_v.T[0][m2fshires_keep],m2fshires_walker_compare_v.T[0][m2fshires_keep]-m2fshires_walker_compare_v.T[2][m2fshires_keep],xerr=m2fshires_walker_compare_v.T[1][m2fshires_keep],yerr=np.sqrt((m2fshires_walker_compare_v.T[1][m2fshires_keep])**2+(m2fshires_walker_compare_v.T[3][m2fshires_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS HiRes',ms=1,rasterized=True)
    ax21.errorbar(walker_m2fsmedres_compare_v.T[2][m2fsmedres_keep],walker_m2fsmedres_compare_v.T[2][m2fsmedres_keep]-walker_m2fsmedres_compare_v.T[0][m2fsmedres_keep],xerr=walker_m2fsmedres_compare_v.T[3][m2fsmedres_keep],yerr=np.sqrt((walker_m2fsmedres_compare_v.T[3][m2fsmedres_keep])**2+(walker_m2fsmedres_compare_v.T[1][m2fsmedres_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS MedRes',ms=1,rasterized=True)
    ax21.errorbar(hecto_walker_compare_v.T[0][hecto_keep],hecto_walker_compare_v.T[0][hecto_keep]-hecto_walker_compare_v.T[2][hecto_keep],xerr=hecto_walker_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_walker_compare_v.T[1][hecto_keep])**2+(hecto_walker_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(vlos_result['samples'].T[4])-np.mean(vlos_result['samples'].T[0])
    hecto_offset=np.mean(vlos_result['samples'].T[4])-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-x0-m2fshires_offset
    hecto_y0=x0-x0-hecto_offset
    ax21.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax21.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax21.set_xlim([-400,400])
    ax21.set_ylim([-9.9,10])
    ax21.set_yticks([-5,0,5,10])
    ax21.set_yticklabels([-5,0,5,10],fontsize=7)
#    ax21.set_xticklabels([])
    ax21.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,W09}$ [km/s]',fontsize=7,labelpad=-2)
    ax21.legend(loc=2,fontsize=5,borderaxespad=0)
    
    m2fshires_keep=np.where((m2fshires_apogee_compare_v.T[1]<5.)&(m2fshires_apogee_compare_v.T[3]<5.))[0]
    m2fsmedres_keep=np.where((m2fsmedres_apogee_compare_v.T[1]<5.)&(m2fsmedres_apogee_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_apogee_compare_v.T[1]<5.)&(hecto_apogee_compare_v.T[3]<5.))[0]
    ax31.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax31.errorbar(m2fshires_apogee_compare_v.T[0][m2fshires_keep],m2fshires_apogee_compare_v.T[0][m2fshires_keep]-m2fshires_apogee_compare_v.T[2][m2fshires_keep],xerr=m2fshires_apogee_compare_v.T[1][m2fshires_keep],yerr=np.sqrt((m2fshires_apogee_compare_v.T[1][m2fshires_keep])**2+(m2fshires_apogee_compare_v.T[3][m2fshires_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax31.errorbar(hecto_apogee_compare_v.T[0][hecto_keep],hecto_apogee_compare_v.T[0][hecto_keep]-hecto_apogee_compare_v.T[2][hecto_keep],xerr=hecto_apogee_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_apogee_compare_v.T[1][hecto_keep])**2+(hecto_apogee_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax31.errorbar(m2fsmedres_apogee_compare_v.T[0][m2fsmedres_keep],m2fsmedres_apogee_compare_v.T[0][m2fsmedres_keep]-m2fsmedres_apogee_compare_v.T[2][m2fsmedres_keep],xerr=m2fsmedres_apogee_compare_v.T[1][m2fsmedres_keep],yerr=np.sqrt((m2fsmedres_apogee_compare_v.T[1][m2fsmedres_keep])**2+(m2fsmedres_apogee_compare_v.T[3][m2fsmedres_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
    m2fshires_offset=0.-np.mean(vlos_result['samples'].T[0])
    hecto_offset=0.-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-x0-m2fshires_offset
    hecto_y0=x0-x0-hecto_offset
    ax31.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax31.set_xlim([-400,400])
    ax31.set_ylim([-9.9,9.9])
    ax31.set_yticks([-5,0,5])
    ax31.set_yticklabels([-5,0,5],fontsize=7)
    ax31.set_xticklabels([])
    ax31.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,Apogee}$ [km/s]',fontsize=7,labelpad=12)

    m2fshires_keep=np.where((m2fshires_h3_compare_v.T[1]<5.)&(m2fshires_h3_compare_v.T[3]<5.))[0]
    m2fsmedres_keep=np.where((h3_m2fsmedres_compare_v.T[1]<5.)&(h3_m2fsmedres_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_h3_compare_v.T[1]<5.)&(hecto_h3_compare_v.T[3]<5.))[0]
    ax41.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=True,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax41.errorbar(m2fshires_h3_compare_v.T[0][m2fshires_keep],m2fshires_h3_compare_v.T[0][m2fshires_keep]-m2fshires_h3_compare_v.T[2][m2fshires_keep],xerr=m2fshires_h3_compare_v.T[1][m2fshires_keep],yerr=np.sqrt((m2fshires_h3_compare_v.T[1][m2fshires_keep])**2+(m2fshires_h3_compare_v.T[3][m2fshires_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax41.errorbar(hecto_h3_compare_v.T[0][hecto_keep],hecto_h3_compare_v.T[0][hecto_keep]-hecto_h3_compare_v.T[2][hecto_keep],xerr=hecto_h3_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_h3_compare_v.T[1][hecto_keep])**2+(hecto_h3_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax41.errorbar(h3_m2fsmedres_compare_v.T[2][m2fsmedres_keep],h3_m2fsmedres_compare_v.T[2][m2fsmedres_keep]-h3_m2fsmedres_compare_v.T[0][m2fsmedres_keep],xerr=h3_m2fsmedres_compare_v.T[3][m2fsmedres_keep],yerr=np.sqrt((h3_m2fsmedres_compare_v.T[3][m2fsmedres_keep])**2+(h3_m2fsmedres_compare_v.T[1][m2fsmedres_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
    m2fshires_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[0])
    hecto_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-x0-m2fshires_offset
    hecto_y0=x0-x0-hecto_offset
    ax41.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax41.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,H3}$ [km/s]',fontsize=7,labelpad=-2)
    ax41.set_xlim([-400,400])
    ax41.set_ylim([-10,9.9])
    ax41.set_yticks([-10,-5,0,5])
    ax41.set_yticklabels([-10,-5,0,5],fontsize=7)
#    ax41.set_xticklabels([])
    ax41.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)    

#    m2fshires_keep=np.where((m2fshires_gaia_compare_v.T[1]<5.)&(m2fshires_gaia_compare_v.T[3]<5.))[0]
#    m2fsmedres_keep=np.where((m2fsmedres_gaia_compare_v.T[1]<5.)&(m2fsmedres_gaia_compare_v.T[3]<5.))[0]
#    hecto_keep=np.where((hecto_gaia_compare_v.T[1]<5.)&(hecto_gaia_compare_v.T[3]<5.))[0]
#    ax51.errorbar(m2fshires_gaia_compare_v.T[0][m2fshires_keep],m2fshires_gaia_compare_v.T[0][m2fshires_keep]-m2fshires_gaia_compare_v.T[2][m2fshires_keep],xerr=m2fshires_gaia_compare_v.T[1][m2fshires_keep],yerr=np.sqrt((m2fshires_gaia_compare_v.T[1][m2fshires_keep])**2+(m2fshires_gaia_compare_v.T[3][m2fshires_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
#    ax51.errorbar(hecto_gaia_compare_v.T[0][hecto_keep],hecto_gaia_compare_v.T[0][hecto_keep]-hecto_gaia_compare_v.T[2][hecto_keep],xerr=hecto_gaia_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_gaia_compare_v.T[1][hecto_keep])**2+(hecto_gaia_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
#    ax51.errorbar(m2fsmedres_gaia_compare_v.T[0][m2fsmedres_keep],m2fsmedres_gaia_compare_v.T[0][m2fsmedres_keep]-m2fsmedres_gaia_compare_v.T[2][m2fsmedres_keep],xerr=m2fsmedres_gaia_compare_v.T[1][m2fsmedres_keep],yerr=np.sqrt((m2fsmedres_gaia_compare_v.T[1][m2fsmedres_keep])**2+(m2fsmedres_gaia_compare_v.T[3][m2fsmedres_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
#    m2fshires_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[0])
#    hecto_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[1])
#    print(offset)
#    x0=np.linspace(-500,10000,100)
#    m2fshires_y0=x0-x0-m2fshires_offset
#    hecto_y0=x0-x0-hecto_offset
#    ax51.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
#    ax51.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,Gaia}$ [km/s]',fontsize=7,labelpad=12)
#    ax51.set_xlim([-400,400])
#    ax51.set_ylim([-10,9.9])
#    ax51.set_yticks([-10,-5,0,5])
#    ax51.set_yticklabels([-10,-5,0,5],fontsize=7)
#    ax51.set_xticks([-400,-200,0,200,400])
#    ax51.set_xticklabels([-400,-200,0,200,400],fontsize=7)
#    ax51.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)    
    
    plt.savefig('v_offset.pdf',dpi=270)
    plt.show()
    plt.close()
    
    gs=plt.GridSpec(18,18)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax11=fig.add_subplot(gs[0:3,0:3])
    ax21=fig.add_subplot(gs[5:8,0:3])
    ax31=fig.add_subplot(gs[8:11,0:3])
    ax41=fig.add_subplot(gs[11:14,0:3])
#    ax51=fig.add_subplot(gs[14:17,0:3])

    ax12=fig.add_subplot(gs[0:3,5:8])
    ax22=fig.add_subplot(gs[5:8,5:8])
    ax32=fig.add_subplot(gs[8:11,5:8])
    ax42=fig.add_subplot(gs[11:14,5:8])
#    ax52=fig.add_subplot(gs[14:17,5:8])

    ax13=fig.add_subplot(gs[0:3,10:13])
    ax23=fig.add_subplot(gs[5:8,10:13])
    ax33=fig.add_subplot(gs[8:11,10:13])
    ax43=fig.add_subplot(gs[11:14,10:13])
#    ax53=fig.add_subplot(gs[14:17,10:13])
    
    ax14=fig.add_subplot(gs[0:3,15:18])
    ax24=fig.add_subplot(gs[5:8,15:18])
    ax34=fig.add_subplot(gs[8:11,15:18])
    ax44=fig.add_subplot(gs[11:14,15:18])
    
    m2fshires_keep=np.where((m2fshires_hecto_compare_v.T[1]<5.)&(m2fshires_hecto_compare_v.T[3]<5.)&(m2fshires_hecto_compare_logg.T[1]<0.5)&(m2fshires_hecto_compare_logg.T[3]<0.5)&(m2fshires_hecto_compare_z.T[1]<0.5)&(m2fshires_hecto_compare_z.T[3]<0.5)&(m2fshires_hecto_compare_alpha.T[1]<0.5)&(m2fshires_hecto_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fshires_m2fsmedres_compare_v.T[1]<5.)&(m2fshires_m2fsmedres_compare_v.T[3]<5.)&(m2fshires_m2fsmedres_compare_logg.T[1]<0.5)&(m2fshires_m2fsmedres_compare_logg.T[3]<0.5)&(m2fshires_m2fsmedres_compare_z.T[1]<0.5)&(m2fshires_m2fsmedres_compare_z.T[3]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[1]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    ax11.errorbar(m2fshires_hecto_compare_teff.T[2][m2fshires_keep]/1000,m2fshires_hecto_compare_teff.T[0][m2fshires_keep]/1000,xerr=m2fshires_hecto_compare_teff.T[3][m2fshires_keep]/1000,yerr=m2fshires_hecto_compare_teff.T[1][m2fshires_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True,label='Hecto')
    ax11.errorbar(m2fshires_m2fsmedres_compare_teff.T[2][m2fsmedres_keep]/1000,m2fshires_m2fsmedres_compare_teff.T[0][m2fsmedres_keep]/1000,xerr=m2fshires_m2fsmedres_compare_teff.T[3][m2fsmedres_keep]/1000,yerr=m2fshires_m2fsmedres_compare_teff.T[1][m2fsmedres_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True,label='M2FS MedRes')

    m2fshires_offset=np.mean(teff_result['samples'].T[1])-np.mean(teff_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset/1000
    ax11.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax11.set_xlim([3.5,8])
    ax11.set_ylim([3.5,8])
    ax11.set_xticks([4,5,6,7,8])
    ax11.set_yticks([4,5,6,7,8])
    ax11.set_xticklabels([4,5,6,7,8],fontsize=6)
    ax11.set_yticklabels([4,5,6,7,8],fontsize=6)
    ax11.set_xlabel(r'$T_{\rm eff}$ [$10^3$ K]',fontsize=6)
    ax11.set_ylabel(r'$T_{\rm eff,\, M2FS\, HiRes}$ [$10^3$ K]',fontsize=6)
    ax11.legend(loc=2,fontsize=5)

    m2fshires_keep=np.where((m2fshires_kirby_compare_logg.T[1]<0.5)&(m2fshires_kirby_compare_logg.T[3]<0.5)&(m2fshires_kirby_compare_z.T[1]<0.5)&(m2fshires_kirby_compare_z.T[3]<0.5)&(m2fshires_kirby_compare_alpha.T[1]<0.5)&(m2fshires_kirby_compare_alpha.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((kirby_m2fsmedres_compare_logg.T[1]<0.5)&(kirby_m2fsmedres_compare_logg.T[3]<0.5)&(kirby_m2fsmedres_compare_z.T[1]<0.5)&(kirby_m2fsmedres_compare_z.T[3]<0.5)&(kirby_m2fsmedres_compare_alpha.T[1]<0.5)&(kirby_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax21.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax21.errorbar(m2fshires_kirby_compare_teff.T[0][m2fshires_keep]/1000,m2fshires_kirby_compare_teff.T[2][m2fshires_keep]/1000,xerr=m2fshires_kirby_compare_teff.T[1][m2fshires_keep]/1000,yerr=m2fshires_kirby_compare_teff.T[3][m2fshires_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS HiRes',ms=1,rasterized=True)
#    ax21.errorbar(kirby_m2fsmedres_compare_teff.T[2][m2fsmedres_keep]/1000,kirby_m2fsmedres_compare_teff.T[0][m2fsmedres_keep]/1000,xerr=kirby_m2fsmedres_compare_teff.T[3][m2fsmedres_keep]/1000,yerr=m2fsmedres_kirby_compare_teff.T[1][m2fsmedres_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS MedRes',ms=1,rasterized=True)
    ax21.errorbar([-10],[-10],fmt='.',elinewidth=0.5,color='cyan',alpha=0.3,ms=1,label='M2FS MedRes',rasterized=True)
    ax21.errorbar(hecto_kirby_compare_teff.T[0][hecto_keep]/1000,hecto_kirby_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_kirby_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_kirby_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(teff_result['samples'].T[2])-np.mean(teff_result['samples'].T[0])
    hecto_offset=np.mean(teff_result['samples'].T[2])-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax21.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax21.set_xlim([3.5,8])
    ax21.set_ylim([3.5,7.99])
    ax21.set_yticks([4,5,6,7])
    ax21.set_yticklabels([4,5,6,7],fontsize=6)
    ax21.set_xticklabels([])
    ax21.set_ylabel(r'$T_{\rm eff,K10}$ [K]',fontsize=6)
    ax21.legend(loc=2,fontsize=5,borderaxespad=0)

    m2fshires_keep=np.where((m2fshires_apogee_compare_logg.T[1]<0.5)&(m2fshires_apogee_compare_logg.T[3]<0.5)&(m2fshires_apogee_compare_z.T[1]<0.5)&(m2fshires_apogee_compare_z.T[3]<0.5)&(m2fshires_apogee_compare_alpha.T[1]<0.5)&(m2fshires_apogee_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fsmedres_apogee_compare_logg.T[1]<0.5)&(m2fsmedres_apogee_compare_logg.T[3]<0.5)&(m2fsmedres_apogee_compare_z.T[1]<0.5)&(m2fsmedres_apogee_compare_z.T[3]<0.5)&(m2fsmedres_apogee_compare_alpha.T[1]<0.5)&(m2fsmedres_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax31.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax31.errorbar(m2fshires_apogee_compare_teff.T[0][m2fshires_keep]/1000,m2fshires_apogee_compare_teff.T[2][m2fshires_keep]/1000,xerr=m2fshires_apogee_compare_teff.T[1][m2fshires_keep]/1000,yerr=m2fshires_apogee_compare_teff.T[3][m2fshires_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS HiRes',ms=1,rasterized=True)
    ax31.errorbar(m2fsmedres_apogee_compare_teff.T[0][m2fsmedres_keep]/1000,m2fsmedres_apogee_compare_teff.T[2][m2fsmedres_keep]/1000,xerr=m2fsmedres_apogee_compare_teff.T[1][m2fsmedres_keep]/1000,yerr=m2fsmedres_apogee_compare_teff.T[3][m2fsmedres_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS MedRes',ms=1,rasterized=True)
    ax31.errorbar(hecto_apogee_compare_teff.T[0][hecto_keep]/1000,hecto_apogee_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_apogee_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_apogee_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=0.-np.mean(teff_result['samples'].T[0])
    hecto_offset=0.-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax31.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax31.set_xlim([3.5,8])
    ax31.set_ylim([3.5,7.99])
    ax31.set_yticks([4,5,6,7])
    ax31.set_yticklabels([4,5,6,7],fontsize=6)
    ax31.set_xticklabels([])
    ax31.set_ylabel(r'$T_{\rm eff,Apogee}$ [K]',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_h3_compare_logg.T[1]<0.5)&(m2fshires_h3_compare_logg.T[3]<0.5)&(m2fshires_h3_compare_z.T[1]<0.5)&(m2fshires_h3_compare_z.T[3]<0.5)&(m2fshires_h3_compare_alpha.T[1]<0.5)&(m2fshires_h3_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((h3_m2fsmedres_compare_logg.T[1]<0.5)&(h3_m2fsmedres_compare_logg.T[3]<0.5)&(h3_m2fsmedres_compare_z.T[1]<0.5)&(h3_m2fsmedres_compare_z.T[3]<0.5)&(h3_m2fsmedres_compare_alpha.T[1]<0.5)&(h3_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax41.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=True,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax41.errorbar(m2fshires_h3_compare_teff.T[0][m2fshires_keep]/1000,m2fshires_h3_compare_teff.T[2][m2fshires_keep]/1000,xerr=m2fshires_h3_compare_teff.T[1][m2fshires_keep]/1000,yerr=m2fshires_h3_compare_teff.T[3][m2fshires_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax41.errorbar(h3_m2fsmedres_compare_teff.T[0][m2fsmedres_keep]/1000,h3_m2fsmedres_compare_teff.T[2][m2fsmedres_keep]/1000,xerr=h3_m2fsmedres_compare_teff.T[1][m2fsmedres_keep]/1000,yerr=h3_m2fsmedres_compare_teff.T[3][m2fsmedres_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax41.errorbar(hecto_h3_compare_teff.T[0][hecto_keep]/1000,hecto_h3_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_h3_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_h3_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[0])
    hecto_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax41.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax41.set_xlim([3.5,8])
    ax41.set_ylim([3.5,7.99])
    ax41.set_yticks([4,5,6,7])
    ax41.set_yticklabels([4,5,6,7],fontsize=6)
    ax41.set_xticks([4,5,6,7,8])
    ax41.set_xticklabels([4,5,6,7,8],fontsize=6)
    ax41.set_ylabel(r'$T_{\rm eff,H3}$ [K]',fontsize=6)
    ax41.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=6)

#    m2fshires_keep=np.where((m2fshires_gaia_compare_logg.T[1]<0.5)&(m2fshires_gaia_compare_logg.T[3]<0.5)&(m2fshires_gaia_compare_z.T[1]<0.5)&(m2fshires_gaia_compare_z.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((m2fsmedres_gaia_compare_logg.T[1]<0.5)&(m2fsmedres_gaia_compare_logg.T[3]<0.5)&(m2fsmedres_gaia_compare_z.T[1]<0.5)&(m2fsmedres_gaia_compare_z.T[3]<0.5))[0]
#    hecto_keep=np.where((hecto_gaia_compare_logg.T[1]<0.5)&(hecto_gaia_compare_logg.T[3]<0.5)&(hecto_gaia_compare_z.T[1]<0.5)&(hecto_gaia_compare_z.T[3]<0.5))[0]
#    ax51.errorbar(m2fshires_gaia_compare_teff.T[0][m2fshires_keep]/1000,m2fshires_gaia_compare_teff.T[2][m2fshires_keep]/1000,xerr=m2fshires_gaia_compare_teff.T[1][m2fshires_keep]/1000,yerr=m2fshires_gaia_compare_teff.T[3][m2fshires_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax51.errorbar(m2fsmedres_gaia_compare_teff.T[0][m2fsmedres_keep]/1000,m2fsmedres_gaia_compare_teff.T[2][m2fsmedres_keep]/1000,xerr=m2fsmedres_gaia_compare_teff.T[1][m2fsmedres_keep]/1000,yerr=m2fsmedres_gaia_compare_teff.T[3][m2fsmedres_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
#    ax51.errorbar(hecto_gaia_compare_teff.T[0][hecto_keep]/1000,hecto_gaia_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_gaia_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_gaia_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
#    m2fshires_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[0])
#    hecto_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[1])
#    x0=np.linspace(-500,10000,100)
#    m2fshires_y0=x0-m2fshires_offset/1000
#    hecto_y0=x0-hecto_offset/1000
#    ax51.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
#    ax51.set_xlim([3.5,8])
#    ax51.set_ylim([3.5,7.99])
#    ax51.set_xticks([4,5,6,7,8])
#    ax51.set_xticklabels([4,5,6,7,8],fontsize=6)
#    ax51.set_yticks([4,5,6,7])
#    ax51.set_yticklabels([4,5,6,7],fontsize=6)
#    ax51.set_ylabel(r'$T_{\rm eff,Gaia}$ [K]',fontsize=6)
#    ax51.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_hecto_compare_v.T[1]<5.)&(m2fshires_hecto_compare_v.T[3]<5.)&(m2fshires_hecto_compare_logg.T[1]<0.5)&(m2fshires_hecto_compare_logg.T[3]<0.5)&(m2fshires_hecto_compare_z.T[1]<0.5)&(m2fshires_hecto_compare_z.T[3]<0.5)&(m2fshires_hecto_compare_alpha.T[1]<0.5)&(m2fshires_hecto_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fshires_m2fsmedres_compare_v.T[1]<5.)&(m2fshires_m2fsmedres_compare_v.T[3]<5.)&(m2fshires_m2fsmedres_compare_logg.T[1]<0.5)&(m2fshires_m2fsmedres_compare_logg.T[3]<0.5)&(m2fshires_m2fsmedres_compare_z.T[1]<0.5)&(m2fshires_m2fsmedres_compare_z.T[3]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[1]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[3]<0.5))[0]

    ax12.errorbar(m2fshires_hecto_compare_logg.T[2][m2fshires_keep],m2fshires_hecto_compare_logg.T[0][m2fshires_keep],xerr=m2fshires_hecto_compare_logg.T[3][m2fshires_keep],yerr=m2fshires_hecto_compare_logg.T[1][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax12.errorbar(m2fshires_m2fsmedres_compare_logg.T[2][m2fsmedres_keep],m2fshires_m2fsmedres_compare_logg.T[0][m2fsmedres_keep],xerr=m2fshires_m2fsmedres_compare_logg.T[3][m2fsmedres_keep],yerr=m2fshires_m2fsmedres_compare_logg.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
    m2fshires_offset=np.mean(logg_result['samples'].T[1])-np.mean(logg_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    ax12.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax12.set_xlim([0,5])
    ax12.set_ylim([0.01,5])
    ax12.set_xticks([0,1,2,3,4,5])
    ax12.set_yticks([0,1,2,3,4,5])
    ax12.set_xticklabels([0,1,2,3,4,5],fontsize=6)
    ax12.set_yticklabels([0,1,2,3,4,5],fontsize=6)
    ax12.set_xlabel(r'$\log g$',fontsize=6)
    ax12.set_ylabel(r'$\log g_{\rm M2FS\, HiRes}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_kirby_compare_logg.T[1]<0.5)&(m2fshires_kirby_compare_logg.T[3]<0.5)&(m2fshires_kirby_compare_z.T[1]<0.5)&(m2fshires_kirby_compare_z.T[3]<0.5)&(m2fshires_kirby_compare_alpha.T[1]<0.5)&(m2fshires_kirby_compare_alpha.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((kirby_m2fsmedres_compare_logg.T[1]<0.5)&(kirby_m2fsmedres_compare_logg.T[3]<0.5)&(kirby_m2fsmedres_compare_z.T[1]<0.5)&(kirby_m2fsmedres_compare_z.T[3]<0.5)&(kirby_m2fsmedres_compare_alpha.T[1]<0.5)&(kirby_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax22.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax22.errorbar(m2fshires_kirby_compare_logg.T[0][m2fshires_keep],m2fshires_kirby_compare_logg.T[2][m2fshires_keep],xerr=m2fshires_kirby_compare_logg.T[1][m2fshires_keep],yerr=m2fshires_kirby_compare_logg.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax22.errorbar(kirby_m2fsmedres_compare_logg.T[2][m2fsmedres_keep],kirby_m2fsmedres_compare_logg.T[0][m2fsmedres_keep],xerr=kirby_m2fsmedres_compare_logg.T[3][m2fsmedres_keep],yerr=kirby_m2fsmedres_compare_logg.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax22.errorbar(hecto_kirby_compare_logg.T[0][hecto_keep],hecto_kirby_compare_logg.T[2][hecto_keep],xerr=hecto_kirby_compare_logg.T[1][hecto_keep],yerr=hecto_kirby_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(logg_result['samples'].T[2])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[2])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax22.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax22.set_xlim([0,5])
    ax22.set_ylim([0.01,4.99])
    ax22.set_yticks([1,2,3,4])
    ax22.set_yticklabels([1,2,3,4],fontsize=6)
    ax22.set_xticklabels([])
    ax22.set_ylabel(r'$\log g_{\rm K10}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_apogee_compare_logg.T[1]<0.5)&(m2fshires_apogee_compare_logg.T[3]<0.5)&(m2fshires_apogee_compare_z.T[1]<0.5)&(m2fshires_apogee_compare_z.T[3]<0.5)&(m2fshires_apogee_compare_alpha.T[1]<0.5)&(m2fshires_apogee_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fsmedres_apogee_compare_logg.T[1]<0.5)&(m2fsmedres_apogee_compare_logg.T[3]<0.5)&(m2fsmedres_apogee_compare_z.T[1]<0.5)&(m2fsmedres_apogee_compare_z.T[3]<0.5)&(m2fsmedres_apogee_compare_alpha.T[1]<0.5)&(m2fsmedres_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax32.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax32.errorbar(m2fshires_apogee_compare_logg.T[0][m2fshires_keep],m2fshires_apogee_compare_logg.T[2][m2fshires_keep],xerr=m2fshires_apogee_compare_logg.T[1][m2fshires_keep],yerr=m2fshires_apogee_compare_logg.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax32.errorbar(m2fsmedres_apogee_compare_logg.T[0][m2fsmedres_keep],m2fsmedres_apogee_compare_logg.T[2][m2fsmedres_keep],xerr=m2fsmedres_apogee_compare_logg.T[1][m2fsmedres_keep],yerr=m2fsmedres_apogee_compare_logg.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax32.errorbar(hecto_apogee_compare_logg.T[0][hecto_keep],hecto_apogee_compare_logg.T[2][hecto_keep],xerr=hecto_apogee_compare_logg.T[1][hecto_keep],yerr=hecto_apogee_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax32.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax32.set_xlim([0,5])
    ax32.set_ylim([0.01,4.99])
    ax32.set_yticks([1,2,3,4])
    ax32.set_yticklabels([1,2,3,4],fontsize=6)
    ax32.set_xticklabels([])
    ax32.set_ylabel(r'$\log g_{\rm Apo}$',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_h3_compare_logg.T[1]<0.5)&(m2fshires_h3_compare_logg.T[3]<0.5)&(m2fshires_h3_compare_z.T[1]<0.5)&(m2fshires_h3_compare_z.T[3]<0.5)&(m2fshires_h3_compare_alpha.T[1]<0.5)&(m2fshires_h3_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((h3_m2fsmedres_compare_logg.T[1]<0.5)&(h3_m2fsmedres_compare_logg.T[3]<0.5)&(h3_m2fsmedres_compare_z.T[1]<0.5)&(h3_m2fsmedres_compare_z.T[3]<0.5)&(h3_m2fsmedres_compare_alpha.T[1]<0.5)&(h3_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax42.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=True,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax42.errorbar(m2fshires_h3_compare_logg.T[0][m2fshires_keep],m2fshires_h3_compare_logg.T[2][m2fshires_keep],xerr=m2fshires_h3_compare_logg.T[1][m2fshires_keep],yerr=m2fshires_h3_compare_logg.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax42.errorbar(h3_m2fsmedres_compare_logg.T[0][m2fsmedres_keep],h3_m2fsmedres_compare_logg.T[2][m2fsmedres_keep],xerr=h3_m2fsmedres_compare_logg.T[1][m2fsmedres_keep],yerr=h3_m2fsmedres_compare_logg.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax42.errorbar(hecto_h3_compare_logg.T[0][hecto_keep],hecto_h3_compare_logg.T[2][hecto_keep],xerr=hecto_h3_compare_logg.T[1][hecto_keep],yerr=hecto_h3_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax42.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax42.set_xlim([0,5])
    ax42.set_ylim([0,4.99])
    ax42.set_yticks([0,1,2,3,4])
    ax42.set_yticklabels([0,1,2,3,4],fontsize=6)
    ax42.set_xticks([0,1,2,3,4,5])
    ax42.set_xticklabels([0,1,2,3,4,5],fontsize=6)
    ax42.set_ylabel(r'$\log g_{\rm H3}$',fontsize=6)
    ax42.set_xlabel(r'$\log g$',fontsize=6)

#    m2fshires_keep=np.where((m2fshires_gaia_compare_logg.T[1]<0.5)&(m2fshires_gaia_compare_logg.T[3]<0.5)&(m2fshires_gaia_compare_z.T[1]<0.5)&(m2fshires_gaia_compare_z.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((m2fsmedres_gaia_compare_logg.T[1]<0.5)&(m2fsmedres_gaia_compare_logg.T[3]<0.5)&(m2fsmedres_gaia_compare_z.T[1]<0.5)&(m2fsmedres_gaia_compare_z.T[3]<0.5))[0]
#    hecto_keep=np.where((hecto_gaia_compare_logg.T[1]<0.5)&(hecto_gaia_compare_logg.T[3]<0.5)&(hecto_gaia_compare_z.T[1]<0.5)&(hecto_gaia_compare_z.T[3]<0.5))[0]
#    ax52.errorbar(m2fshires_gaia_compare_logg.T[0][m2fshires_keep],m2fshires_gaia_compare_logg.T[2][m2fshires_keep],xerr=m2fshires_gaia_compare_logg.T[1][m2fshires_keep],yerr=m2fshires_gaia_compare_logg.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax52.errorbar(m2fsmedres_gaia_compare_logg.T[0][m2fsmedres_keep],m2fsmedres_gaia_compare_logg.T[2][m2fsmedres_keep],xerr=m2fsmedres_gaia_compare_logg.T[1][m2fsmedres_keep],yerr=m2fsmedres_gaia_compare_logg.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
#    ax52.errorbar(hecto_gaia_compare_logg.T[0][hecto_keep],hecto_gaia_compare_logg.T[2][hecto_keep],xerr=hecto_gaia_compare_logg.T[1][hecto_keep],yerr=hecto_h3_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
#    m2fshires_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[0])
#    hecto_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[1])
#    x0=np.linspace(-500,10000,100)
#    m2fshires_y0=x0-m2fshires_offset
#    hecto_y0=x0-hecto_offset
#    ax52.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
#    ax52.set_xlim([0,5])
#    ax52.set_ylim([0,4.99])
#    ax52.set_xticks([0,1,2,3,4,5])
#    ax52.set_xticklabels([0,1,2,3,4,5],fontsize=6)
#    ax52.set_yticks([0,1,2,3,4])
#    ax52.set_yticklabels([0,1,2,3,4],fontsize=6)
#    ax52.set_ylabel(r'$\log g_{\rm Gaia}$',fontsize=6)
#    ax52.set_xlabel(r'$\log g$',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_hecto_compare_v.T[1]<5.)&(m2fshires_hecto_compare_v.T[3]<5.)&(m2fshires_hecto_compare_logg.T[1]<0.5)&(m2fshires_hecto_compare_logg.T[3]<0.5)&(m2fshires_hecto_compare_z.T[1]<0.5)&(m2fshires_hecto_compare_z.T[3]<0.5)&(m2fshires_hecto_compare_alpha.T[1]<0.5)&(m2fshires_hecto_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fshires_m2fsmedres_compare_v.T[1]<5.)&(m2fshires_m2fsmedres_compare_v.T[3]<5.)&(m2fshires_m2fsmedres_compare_logg.T[1]<0.5)&(m2fshires_m2fsmedres_compare_logg.T[3]<0.5)&(m2fshires_m2fsmedres_compare_z.T[1]<0.5)&(m2fshires_m2fsmedres_compare_z.T[3]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[1]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[3]<0.5))[0]

    ax13.errorbar(m2fshires_hecto_compare_z.T[2][m2fshires_keep],m2fshires_hecto_compare_z.T[0][m2fshires_keep],xerr=m2fshires_hecto_compare_z.T[3][m2fshires_keep],yerr=m2fshires_hecto_compare_z.T[1][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax13.errorbar(m2fshires_m2fsmedres_compare_z.T[2][m2fsmedres_keep],m2fshires_m2fsmedres_compare_z.T[0][m2fsmedres_keep],xerr=m2fshires_m2fsmedres_compare_z.T[3][m2fsmedres_keep],yerr=m2fshires_m2fsmedres_compare_z.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
    m2fshires_offset=np.mean(z_result['samples'].T[1])-np.mean(z_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    ax13.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax13.set_xlim([-4,1])
    ax13.set_ylim([-4,1])
    ax13.set_xticks([-4,-3,-2,-1,0,1])
    ax13.set_yticks([-4,-3,-2,-1,0,1])
    ax13.set_xticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax13.set_yticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax13.set_xlabel(r'[Fe/H]',fontsize=6)
    ax13.set_ylabel(r'[Fe/H]$_{\rm M2FS\, HiRes}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_kirby_compare_logg.T[1]<0.5)&(m2fshires_kirby_compare_logg.T[3]<0.5)&(m2fshires_kirby_compare_z.T[1]<0.5)&(m2fshires_kirby_compare_z.T[3]<0.5)&(m2fshires_kirby_compare_alpha.T[1]<0.5)&(m2fshires_kirby_compare_alpha.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((kirby_m2fsmedres_compare_logg.T[1]<0.5)&(kirby_m2fsmedres_compare_logg.T[3]<0.5)&(kirby_m2fsmedres_compare_z.T[1]<0.5)&(kirby_m2fsmedres_compare_z.T[3]<0.5)&(kirby_m2fsmedres_compare_alpha.T[1]<0.5)&(kirby_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax23.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax23.errorbar(m2fshires_kirby_compare_z.T[0][m2fshires_keep],m2fshires_kirby_compare_z.T[2][m2fshires_keep],xerr=m2fshires_kirby_compare_z.T[1][m2fshires_keep],yerr=m2fshires_kirby_compare_z.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax23.errorbar(kirby_m2fsmedres_compare_z.T[2][m2fsmedres_keep],kirby_m2fsmedres_compare_z.T[0][m2fsmedres_keep],xerr=kirby_m2fsmedres_compare_z.T[3][m2fsmedres_keep],yerr=kirby_m2fsmedres_compare_z.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax23.errorbar(hecto_kirby_compare_z.T[0][hecto_keep],hecto_kirby_compare_z.T[2][hecto_keep],xerr=hecto_kirby_compare_z.T[1][hecto_keep],yerr=hecto_kirby_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(z_result['samples'].T[2])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[2])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax23.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax23.set_xlim([-4,1])
    ax23.set_ylim([-4,1])
    ax23.set_yticks([-3,-2,-1,0,1])
    ax23.set_yticklabels([-3,-2,-1,0,1],fontsize=6)
    ax23.set_xticklabels([])
    ax23.set_ylabel(r'[Fe/H]$_{\rm K10}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_apogee_compare_logg.T[1]<0.5)&(m2fshires_apogee_compare_logg.T[3]<0.5)&(m2fshires_apogee_compare_z.T[1]<0.5)&(m2fshires_apogee_compare_z.T[3]<0.5)&(m2fshires_apogee_compare_alpha.T[1]<0.5)&(m2fshires_apogee_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fsmedres_apogee_compare_logg.T[1]<0.5)&(m2fsmedres_apogee_compare_logg.T[3]<0.5)&(m2fsmedres_apogee_compare_z.T[1]<0.5)&(m2fsmedres_apogee_compare_z.T[3]<0.5)&(m2fsmedres_apogee_compare_alpha.T[1]<0.5)&(m2fsmedres_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax33.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax33.errorbar(m2fshires_apogee_compare_z.T[0][m2fshires_keep],m2fshires_apogee_compare_z.T[2][m2fshires_keep],xerr=m2fshires_apogee_compare_z.T[1][m2fshires_keep],yerr=m2fshires_apogee_compare_z.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax33.errorbar(m2fsmedres_apogee_compare_z.T[0][m2fsmedres_keep],m2fsmedres_apogee_compare_z.T[2][m2fsmedres_keep],xerr=m2fsmedres_apogee_compare_z.T[1][m2fsmedres_keep],yerr=m2fsmedres_apogee_compare_z.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax33.errorbar(hecto_apogee_compare_z.T[0][hecto_keep],hecto_apogee_compare_z.T[2][hecto_keep],xerr=hecto_apogee_compare_z.T[1][hecto_keep],yerr=hecto_apogee_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax33.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax33.set_xlim([-4,1])
    ax33.set_ylim([-4,1])
    ax33.set_yticks([-3,-2,-1,0])
    ax33.set_yticklabels([-3,-2,-1,0],fontsize=6)
    ax33.set_xticklabels([])
    ax33.set_ylabel(r'[Fe/H]$_{\rm Apo}$',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_h3_compare_logg.T[1]<0.5)&(m2fshires_h3_compare_logg.T[3]<0.5)&(m2fshires_h3_compare_z.T[1]<0.5)&(m2fshires_h3_compare_z.T[3]<0.5)&(m2fshires_h3_compare_alpha.T[1]<0.5)&(m2fshires_h3_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((h3_m2fsmedres_compare_logg.T[1]<0.5)&(h3_m2fsmedres_compare_logg.T[3]<0.5)&(h3_m2fsmedres_compare_z.T[1]<0.5)&(h3_m2fsmedres_compare_z.T[3]<0.5)&(h3_m2fsmedres_compare_alpha.T[1]<0.5)&(h3_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax43.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=True,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax43.errorbar(m2fshires_h3_compare_z.T[0][m2fshires_keep],m2fshires_h3_compare_z.T[2][m2fshires_keep],xerr=m2fshires_h3_compare_z.T[1][m2fshires_keep],yerr=m2fshires_h3_compare_z.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax43.errorbar(h3_m2fsmedres_compare_z.T[2][m2fsmedres_keep],h3_m2fsmedres_compare_z.T[0][m2fsmedres_keep],xerr=h3_m2fsmedres_compare_z.T[3][m2fsmedres_keep],yerr=h3_m2fsmedres_compare_z.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax43.errorbar(hecto_h3_compare_z.T[0][hecto_keep],hecto_h3_compare_z.T[2][hecto_keep],xerr=hecto_h3_compare_z.T[1][hecto_keep],yerr=hecto_h3_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax43.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax43.set_xlim([-4,1])
    ax43.set_ylim([-4,1])
    ax43.set_yticks([-4,-3,-2,-1,0])
    ax43.set_yticklabels([-4,-3,-2,-1,0],fontsize=6)
    ax43.set_xticks([-4,-3,-2,-1,0,1])
    ax43.set_xticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax43.set_ylabel(r'[Fe/H]$_{\rm H3}$',fontsize=6)
    ax43.set_xlabel(r'[Fe/H]',fontsize=6)

#    m2fshires_keep=np.where((m2fshires_gaia_compare_logg.T[1]<0.5)&(m2fshires_gaia_compare_logg.T[3]<0.5)&(m2fshires_gaia_compare_z.T[1]<0.5)&(m2fshires_gaia_compare_z.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((m2fsmedres_gaia_compare_logg.T[1]<0.5)&(m2fsmedres_gaia_compare_logg.T[3]<0.5)&(m2fsmedres_gaia_compare_z.T[1]<0.5)&(m2fsmedres_gaia_compare_z.T[3]<0.5))[0]
#    hecto_keep=np.where((hecto_gaia_compare_logg.T[1]<0.5)&(hecto_gaia_compare_logg.T[3]<0.5)&(hecto_gaia_compare_z.T[1]<0.5)&(hecto_gaia_compare_z.T[3]<0.5))[0]
#    ax53.errorbar(m2fshires_gaia_compare_z.T[0][m2fshires_keep],m2fshires_gaia_compare_z.T[2][m2fshires_keep],xerr=m2fshires_gaia_compare_z.T[1][m2fshires_keep],yerr=m2fshires_gaia_compare_z.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax53.errorbar(m2fsmedres_gaia_compare_z.T[0][m2fsmedres_keep],m2fsmedres_gaia_compare_z.T[2][m2fsmedres_keep],xerr=m2fsmedres_gaia_compare_z.T[1][m2fsmedres_keep],yerr=m2fsmedres_gaia_compare_z.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
#    ax53.errorbar(hecto_gaia_compare_z.T[0][hecto_keep],hecto_gaia_compare_z.T[2][hecto_keep],xerr=hecto_gaia_compare_z.T[1][hecto_keep],yerr=hecto_gaia_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
#    m2fshires_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[0])
#    hecto_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[1])
#    x0=np.linspace(-500,10000,100)
#    m2fshires_y0=x0-m2fshires_offset
#    hecto_y0=x0-hecto_offset
#    ax53.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
#    ax53.set_xlim([-4,1])
#    ax53.set_ylim([-4,1])
#    ax53.set_yticks([-4,-3,-2,-1,0])
#    ax53.set_yticklabels([-4,-3,-2,-1,0],fontsize=6)
#    ax53.set_xticklabels([-4,-3,-2,-1,0,1],fontsize=6)
#    ax53.set_ylabel(r'[Fe/H]$_{\rm Gaia}$',fontsize=6)
#    ax53.set_xlabel(r'[Fe/H]',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_hecto_compare_v.T[1]<5.)&(m2fshires_hecto_compare_v.T[3]<5.)&(m2fshires_hecto_compare_logg.T[1]<0.5)&(m2fshires_hecto_compare_logg.T[3]<0.5)&(m2fshires_hecto_compare_z.T[1]<0.5)&(m2fshires_hecto_compare_z.T[3]<0.5)&(m2fshires_hecto_compare_alpha.T[1]<0.5)&(m2fshires_hecto_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fshires_m2fsmedres_compare_v.T[1]<5.)&(m2fshires_m2fsmedres_compare_v.T[3]<5.)&(m2fshires_m2fsmedres_compare_logg.T[1]<0.5)&(m2fshires_m2fsmedres_compare_logg.T[3]<0.5)&(m2fshires_m2fsmedres_compare_z.T[1]<0.5)&(m2fshires_m2fsmedres_compare_z.T[3]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[1]<0.5)&(m2fshires_m2fsmedres_compare_alpha.T[3]<0.5))[0]

    ax14.errorbar(m2fshires_hecto_compare_alpha.T[2][m2fshires_keep],m2fshires_hecto_compare_alpha.T[0][m2fshires_keep],xerr=m2fshires_hecto_compare_alpha.T[3][m2fshires_keep],yerr=m2fshires_hecto_compare_alpha.T[1][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax14.errorbar(m2fshires_m2fsmedres_compare_alpha.T[2][m2fsmedres_keep],m2fshires_m2fsmedres_compare_alpha.T[0][m2fsmedres_keep],xerr=m2fshires_m2fsmedres_compare_alpha.T[3][m2fsmedres_keep],yerr=m2fshires_m2fsmedres_compare_alpha.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',ms=1,rasterized=True)
    m2fshires_offset=np.mean(alpha_result['samples'].T[1])-np.mean(alpha_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    ax14.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax14.set_xlim([-1,1])
    ax14.set_ylim([-1,1])
    ax14.set_xticks([-1,-0.5,0,0.5,1])
    ax14.set_yticks([-1,-0.5,0,0.5,1])
    ax14.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax14.set_yticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax14.set_xlabel(r'[Mg/Fe]',fontsize=6)
    ax14.set_ylabel(r'[Mg/Fe]$_{\rm M2FS\, HiRes}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_kirby_compare_logg.T[1]<0.5)&(m2fshires_kirby_compare_logg.T[3]<0.5)&(m2fshires_kirby_compare_z.T[1]<0.5)&(m2fshires_kirby_compare_z.T[3]<0.5)&(m2fshires_kirby_compare_alpha.T[1]<0.5)&(m2fshires_kirby_compare_alpha.T[3]<0.5))[0]
#    m2fsmedres_keep=np.where((kirby_m2fsmedres_compare_logg.T[1]<0.5)&(kirby_m2fsmedres_compare_logg.T[3]<0.5)&(kirby_m2fsmedres_compare_z.T[1]<0.5)&(kirby_m2fsmedres_compare_z.T[3]<0.5)&(kirby_m2fsmedres_compare_alpha.T[1]<0.5)&(kirby_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax24.tick_params(right=False,top=False,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax24.errorbar(m2fshires_kirby_compare_alpha.T[0][m2fshires_keep],m2fshires_kirby_compare_alpha.T[2][m2fshires_keep],xerr=m2fshires_kirby_compare_alpha.T[1][m2fshires_keep],yerr=m2fshires_kirby_compare_alpha.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
#    ax24.errorbar(kirby_m2fsmedres_compare_alpha.T[2][m2fsmedres_keep],kirby_m2fsmedres_compare_alpha.T[0][m2fsmedres_keep],xerr=kirby_m2fsmedres_compare_alpha.T[3][m2fsmedres_keep],yerr=kirby_m2fsmedres_compare_alpha.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax24.errorbar(hecto_kirby_compare_alpha.T[0][hecto_keep],hecto_kirby_compare_alpha.T[2][hecto_keep],xerr=hecto_kirby_compare_alpha.T[1][hecto_keep],yerr=hecto_kirby_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(alpha_result['samples'].T[2])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[2])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax24.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax24.set_xlim([-1,1])
    ax24.set_ylim([-1,1])
    ax24.set_yticks([-0.5,0,0.5])
    ax24.set_yticklabels([-0.5,0,0.5],fontsize=6)
    ax24.set_xticklabels([])
    ax24.set_ylabel(r'[Mg/Fe]$_{\rm K10}$',fontsize=6)

    m2fshires_keep=np.where((m2fshires_apogee_compare_logg.T[1]<0.5)&(m2fshires_apogee_compare_logg.T[3]<0.5)&(m2fshires_apogee_compare_z.T[1]<0.5)&(m2fshires_apogee_compare_z.T[3]<0.5)&(m2fshires_apogee_compare_alpha.T[1]<0.5)&(m2fshires_apogee_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((m2fsmedres_apogee_compare_logg.T[1]<0.5)&(m2fsmedres_apogee_compare_logg.T[3]<0.5)&(m2fsmedres_apogee_compare_z.T[1]<0.5)&(m2fsmedres_apogee_compare_z.T[3]<0.5)&(m2fsmedres_apogee_compare_alpha.T[1]<0.5)&(m2fsmedres_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax34.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax34.errorbar(m2fshires_apogee_compare_alpha.T[0][m2fshires_keep],m2fshires_apogee_compare_alpha.T[2][m2fshires_keep],xerr=m2fshires_apogee_compare_alpha.T[1][m2fshires_keep],yerr=m2fshires_apogee_compare_alpha.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax34.errorbar(m2fsmedres_apogee_compare_alpha.T[0][m2fsmedres_keep],m2fsmedres_apogee_compare_alpha.T[2][m2fsmedres_keep],xerr=m2fsmedres_apogee_compare_alpha.T[1][m2fsmedres_keep],yerr=m2fsmedres_apogee_compare_alpha.T[3][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax34.errorbar(hecto_apogee_compare_alpha.T[0][hecto_keep],hecto_apogee_compare_alpha.T[2][hecto_keep],xerr=hecto_apogee_compare_alpha.T[1][hecto_keep],yerr=hecto_apogee_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax34.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax34.set_xlim([-1,1])
    ax34.set_ylim([-1,1])
    ax34.set_yticks([-0.5,0,0.5])
    ax34.set_yticklabels([-0.5,0,0.5],fontsize=6)
    ax34.set_xticklabels([])
    ax34.set_ylabel(r'[Mg/Fe]$_{\rm Apo}$',fontsize=6)
    
    m2fshires_keep=np.where((m2fshires_h3_compare_logg.T[1]<0.5)&(m2fshires_h3_compare_logg.T[3]<0.5)&(m2fshires_h3_compare_z.T[1]<0.5)&(m2fshires_h3_compare_z.T[3]<0.5)&(m2fshires_h3_compare_alpha.T[1]<0.5)&(m2fshires_h3_compare_alpha.T[3]<0.5))[0]
    m2fsmedres_keep=np.where((h3_m2fsmedres_compare_logg.T[1]<0.5)&(h3_m2fsmedres_compare_logg.T[3]<0.5)&(h3_m2fsmedres_compare_z.T[1]<0.5)&(h3_m2fsmedres_compare_z.T[3]<0.5)&(h3_m2fsmedres_compare_alpha.T[1]<0.5)&(h3_m2fsmedres_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax44.tick_params(right=False,top=True,left=False,bottom=True,labelbottom=True,labelleft=True,labeltop=False,labelright=False,direction='inout',length=5)
    ax44.errorbar(m2fshires_h3_compare_alpha.T[0][m2fshires_keep],m2fshires_h3_compare_alpha.T[2][m2fshires_keep],xerr=m2fshires_h3_compare_alpha.T[1][m2fshires_keep],yerr=m2fshires_h3_compare_alpha.T[3][m2fshires_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax44.errorbar(h3_m2fsmedres_compare_alpha.T[2][m2fsmedres_keep],h3_m2fsmedres_compare_alpha.T[0][m2fsmedres_keep],xerr=h3_m2fsmedres_compare_alpha.T[3][m2fsmedres_keep],yerr=h3_m2fsmedres_compare_alpha.T[1][m2fsmedres_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='cyan',label='M2FS',ms=1,rasterized=True)
    ax44.errorbar(hecto_h3_compare_alpha.T[0][hecto_keep],hecto_h3_compare_alpha.T[2][hecto_keep],xerr=hecto_h3_compare_alpha.T[1][hecto_keep],yerr=hecto_h3_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fshires_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fshires_y0=x0-m2fshires_offset
    hecto_y0=x0-hecto_offset
    ax44.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax44.set_xlim([-1,1])
    ax44.set_ylim([-1,1])
    ax44.set_xticks([-1,-0.5,0,0.5,1])
    ax44.set_yticks([-1,-0.5,0,0.5])
    ax44.set_yticklabels([-1,-0.5,0,0.5],fontsize=6)
    ax44.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax44.set_ylabel(r'[Mg/Fe]$_{\rm H3}$',fontsize=6)
    ax44.set_xlabel(r'[Mg/Fe]',fontsize=6)
    
    plt.savefig('atm_offset.pdf',dpi=270)
    plt.show()
    plt.close()        
