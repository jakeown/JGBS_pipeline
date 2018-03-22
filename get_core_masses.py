import numpy
from pylab import *
from scipy.optimize import curve_fit
from scipy.stats import mstats
import sys
import matplotlib.pyplot as plt
from astropy import wcs
from astropy import coordinates
from astropy import units as u
from astroquery.simbad import Simbad
from astroquery.ned import Ned
import astroquery
import astropy

import good_cores_getsources
#import make_catalog
import make_CMF
#import make_DS9_region
#import make_plots

def core_mass_fits(region_name = 'Aquila', T_Dust=15.0, T_Dust_err=3.0, distance = 260., Dunham_YSOs_file = 'Dunham_YSOs.dat'):

	getsources_core_catalog = '/mnt/scratch-lustre/jkeown/GS_Extractions/GS-Extractions/' + region_name + '/' + region_name + '.sw.final.reliable.ok.cat'
	SED_figure_directory = '/mnt/scratch-lustre/jkeown/JCMT_GBS_Jybeam/JCMT_pipeline/Figures/' + region_name + '/'
	# These are the values in each column of "cores_array" and "protostar_array"
	#NO,    XCO_P,   YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02

	# These are the values in each column of the "additional" "cores_array" and "protostar_array"
	# NO    XCO_P   YCO_P PEAK_SRC01 PEAK_BGF01 CONV_SRC01 CONV_BGF01 PEAK_SRC02 PEAK_BGF02 CONV_SRC02 CONV_BGF02

	# Import the raw getsources core catalog (includes "bad" sources that don't pass HGBS selection criteria)
	cores_array1 = numpy.loadtxt(getsources_core_catalog,comments='!')
	# Find the indices of the "good" cores that pass HGBS selection criteria
	good_core_indices = good_cores_getsources.get_good_cores(region_name)
	# Create new array of only the "good" cores to be used for our analysis
	cores_array = cores_array1[numpy.array(good_core_indices)]

	if region_name=='PipeE1':
		cores_array = numpy.array(cores_array1)
		print cores_array
		z = numpy.zeros(len(cores_array))
		cores_array = numpy.stack((cores_array, z))
		print cores_array

	# Calculate the deconvolved core radii
	AFWH05 = cores_array[:,23]
	BFWH05 = cores_array[:,24]
	A = numpy.float64(((((AFWH05)/60.)/60.)*numpy.pi)/180.) #radians
	A1 = numpy.float64(numpy.tan(A/2.)*2.*distance) #pc
	B = numpy.float64(((((BFWH05)/60.)/60.)*numpy.pi)/180.) #radians
	B1 = numpy.float64(numpy.tan(B/2.)*2.*distance) #pc
	FWHM_mean = mstats.gmean([A1,B1])
	HPBW = numpy.float64(((((14.1)/60.)/60.)*numpy.pi)/180.) #radians
	HPBW1 = numpy.float64(numpy.tan(HPBW/2.)*2.*distance) #pc
	R_deconv = ((FWHM_mean**2.0) - (HPBW1**2.0))**0.5 #pc

	R_deconv = numpy.where(((FWHM_mean**2.0) - (HPBW1**2.0))<=0., FWHM_mean, R_deconv)

	resolved = numpy.where(((FWHM_mean**2.0) - (HPBW1**2.0))<=0., 0, 1)

	# Calculate the Bonnor-Ebert masses of each core based on their R_deconvolved  
	c_s = 0.2 #km/s
	G = 4.302*10**-3 #pc/M_solar (km/s)^2
	M_BE = (2.4*R_deconv*(c_s**2.))/G #M_solar
	
	#M_BE = numpy.where(((FWHM_mean**2.0) - (HPBW1**2.0))<0, 9999., M_BE)

	# Define a function that produces a Flux given wavelength, Temp, and Mass
	# We will input wavelength then find T and M using least squares minimization below
	def core_mass(wavelength, T, M):
		#wavelength input in microns, Temp in Kelvin, Mass in M_solar
		#returns S_v (i.e., Flux) in units of Jy  
		D = distance #parsecs to cloud
		wavelength_mm = numpy.array(wavelength)*10.**-3.
		exponent = 1.439*(wavelength_mm**-1)*((T/10.)**-1)
		aaa = (0.12*(numpy.exp(exponent)-1.0))**-1.0
		bbb = (0.1*((numpy.array(wavelength)/300.)**-2.0))/0.01
		ccc = (D/100.)**-2
		ddd = wavelength_mm**-3.
		return M*aaa*bbb*ccc*ddd

	# Define another function that calculates Mass directly from wavelength, Temp, and Flux
	# This will be used to find the Mass of cores that don't have reliable least-squares fits
	def core_mass_from_flux(wavelength, T, S_v):
		#wavelength input in microns, Temp in Kelvin, Mass in M_solar
		#returns S_v (i.e., Flux) in units of Jy  
		D = distance #parsecs to cloud
		wavelength_mm = wavelength*10.**-3.
		exponent = 1.439*(wavelength_mm**-1)*((T/10.)**-1)
		aaa = 0.12*(numpy.exp(exponent)-1.0)
		bbb = ((0.1*((wavelength/300.)**-2.0))/0.01)**-1.0
		ccc = (D/100.)**2.0
		ddd = wavelength_mm**3.0
		return S_v*aaa*bbb*ccc*ddd

	# Define another function that calculates Mass uncertainty due to temp
	def core_mass_err_dT(wavelength, T, S_v, dT):
		#wavelength input in microns, Temp in Kelvin, Mass in M_solar
		#returns S_v (i.e., Flux) in units of Jy  
		D = distance #parsecs to cloud
		wavelength_mm = wavelength*10.**-3.
		exponent = 1.439*(wavelength_mm**-1)*((T/10.)**-1)
		aaa = 0.12*(numpy.exp(exponent))
		bbb = ((0.1*((wavelength/300.)**-2.0))/0.01)**-1.0
		ccc = (D/100.)**2.0
		ddd = wavelength_mm**3.0
		eee = 1.439*10*(wavelength_mm**-1)*(T**-2.)
		return S_v*aaa*bbb*ccc*ddd*dT*eee

	# Define another function that calculates Mass uncertainty due to flux
	def core_mass_err_dS_v(wavelength, T, S_v, dS_v):
		#wavelength input in microns, Temp in Kelvin, Mass in M_solar
		#returns S_v (i.e., Flux) in units of Jy  
		D = distance #parsecs to cloud
		wavelength_mm = wavelength*10.**-3.
		exponent = 1.439*(wavelength_mm**-1)*((T/10.)**-1)
		aaa = 0.12*(numpy.exp(exponent)-1.0)
		bbb = ((0.1*((wavelength/300.)**-2.0))/0.01)**-1.0
		ccc = (D/100.)**2.0
		ddd = wavelength_mm**3.0
		return aaa*bbb*ccc*ddd*dS_v

	# Create some empty arrays to which we will append accepted values
	Masses = []
	Temps = []
	Masses_err = []
	Temps_err = []
	counter=0
	
	# Loop through all the "good" cores
	for NO,    XCO_P,   YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02 in cores_array:
		
		#flux_err_run1 = [FXT_BEST01/SIG_MONO01, FXT_BEST02/SIG_MONO02] ## Should these be FXP_BEST?
		
		# Find the longest significant wavelength and corresponding flux
		wave = 850.
		flux_fit = FXT_BEST02 # Fluxes are in mJy, so multiple by 10^3 to get Jy
		flux_fit_err = FXT_ERRO02 # Fluxes are in mJy, so multiple by 10^3 to get Jy
		# Find the mass corresponding to that flux measurement
		# ***This uses the median of the best-fit Temps from the cores with 
		# reliable SED fits (i.e., those that pass the test above)
		Mass_fit = core_mass_from_flux(wave, T_Dust, flux_fit)
		# Can add more uncertainties (e.g., calibration, etc.) below
		Mass_error = (core_mass_err_dT(wave, T_Dust, flux_fit, T_Dust_err) + 
					core_mass_err_dS_v(wave, T_Dust, flux_fit, flux_fit_err)**2.0)**0.5
		# Store the Mass with uncertainties
		# Need to perform a more in-depth error analysis 
		Masses.append(Mass_fit)
		Masses_err.append(Mass_error)
		Temps.append(T_Dust)
		Temps_err.append(T_Dust_err)
		
	#Replace nans if they exist in the M_BE array
	if len(cores_array)>1:
		where_are_nans = numpy.isnan(M_BE)
		M_BE[where_are_nans] = 9999

	# Calculate the alpha_BE ratio to determine prestellar cores
	alpha_BE = numpy.array(M_BE)/numpy.array(Masses)

	#alpha_BE = numpy.where(((FWHM_mean**2.0) - (HPBW1**2.0))<0, 9999., alpha_BE)
	
	# Create an array indicating a core as candidate(1)/robust(2) prestellar
	candidate_array = numpy.where(alpha_BE<=5.0, 1, 0)
	#candidate_array = numpy.where(alpha_BE<=alpha_factor, 1, 0)
	robust_candidate_array = numpy.where(alpha_BE<=2.0, 2, candidate_array)

	# Remove protostars from the alpha_BE array and find the indices of the remaining 
	# candidate/robust prestellar cores
	robust_prestellar_indices = numpy.where(alpha_BE<=2.0)
	candidate_prestellar_indices = numpy.where(alpha_BE<=5.0)

	Masses2=numpy.array(Masses)

	# Find the final list of prestellar candidate/robust Masses with protostars removed
	prestellar_candidates = Masses2[numpy.array(candidate_prestellar_indices[0])]
	prestellar_robust = Masses2[numpy.array(robust_prestellar_indices[0])]
	print 'prestellar candidates: ' + str(len(prestellar_candidates))
	print 'robust prestellar candidates: ' + str(len(prestellar_robust))
	
	# Plot Mass versus Radius and save the figure
	fig = plt.figure()
	plt.scatter(R_deconv,Masses, label='starless')
	if len(Masses)>1:
		plt.scatter(R_deconv[numpy.array(candidate_prestellar_indices[0])],prestellar_candidates, color='red', label='candidate')
		plt.scatter(R_deconv[numpy.array(robust_prestellar_indices[0])],prestellar_robust, color='green', label='robust')
	plt.yscale('log')
	plt.xscale('log')
	#plt.legend()
	plt.title(region_name + ' Cores')
	plt.ylabel("Mass, M (M$_\odot$)")
	plt.xlabel("Deconvolved FWHM size, R (pc)")
	#plt.xlim([10**-3, 2*10**-1])
	#plt.ylim([10**-3, 10**2])
	fig.savefig(SED_figure_directory + 'mass_vs_radius_' + region_name + '_Feb2018.png')

	Cloud,Name,Av,alpha,T_bol,L_bol,alphaPrime,TbolPrime,LbolPrime,likelyAGB,Dunham_RA,Dunham_DEC,Class = numpy.loadtxt(Dunham_YSOs_file, delimiter=',', unpack=True, dtype=[('Cloud','S30'),('Name','S40'), ('Av',float),('alpha',float), ('T_bol',float),('L_bol',float), ('alphaPrime',float),('TbolPrime',float), ('LbolPrime',float),('likelyAGB','S1'), ('Dunham_RA',float),('Dunham_DEC',float),('Class','S10') ])
	#Dunham_indices = numpy.where(Cloud==cloud_name)
	Spitzer_YSOs_RA = Dunham_RA
	Spitzer_YSOs_DEC = Dunham_DEC
	Spitzer_YSOs_Name = Name

	potential_matches = []
	YSO_matches = []
	count = 0
	for line in cores_array:
		match_counter=0
		YSO_index = 0
		for RA,DEC in zip(Spitzer_YSOs_RA, Spitzer_YSOs_DEC):
			distance = ((line[3]-RA)**2 + (line[4]-DEC)**2)**0.5
			if distance < 6.0/3600. and match_counter==0:
				# matched_counter prevents counting indices twice 
				# if two YSO candidates fall within getsources ellipse
				potential_matches.append(count)
				match_counter+=1
				YSO_matches.append(YSO_index)
			YSO_index+=1
		count += 1

	Spitzer_column = numpy.zeros(len(cores_array[:,0]), dtype='S50')
	Spitzer_column[numpy.arange(0,len(cores_array[:,0]))] = 'None'
	if len(potential_matches)>0:
		Spitzer_column[numpy.array(potential_matches)] = Spitzer_YSOs_Name[numpy.array(YSO_matches)]

	# Save a text file with RA and Dec of the "good" cores
	# This is needed for the SIMBAD cross-match
	numpy.savetxt(SED_figure_directory + region_name +'_SIMBAD_RA_DEC.dat', zip(cores_array[:,3],cores_array[:,4]))

	# Cross-match cores with SIMBAD catalog
	print "Cross-matching SIMBAD catalog:"
	RA, Dec = numpy.loadtxt(SED_figure_directory + region_name +'_SIMBAD_RA_DEC.dat', unpack=True)
	Simbad.ROW_LIMIT = 1
	results = []

	if len(cores_array)==1:
		result_table = Simbad.query_region(astropy.coordinates.SkyCoord(ra=RA, dec=Dec, unit=(u.deg, u.deg)), radius=6. * u.arcsec)
		if result_table != None:
			results.append(result_table['MAIN_ID'][0].replace(" ", "_"))
		else:
			results.append('None')

		# Cross-match cores with NED catalog
		print "Cross-matching NED catalog:"
		Ned.ROW_LIMIT = 1
		results2 = []
		result_table_value='Yes'
		try:
			result_table = Ned.query_region(astropy.coordinates.SkyCoord(ra=RA, dec=Dec, unit=(u.deg, u.deg)), radius=6. * u.arcsec)
		except astroquery.exceptions.RemoteServiceError: 
			result_table_value=None
		if result_table_value != None:
			results2.append(result_table['Object Name'][0].replace(" ", "_"))
		else:
			results2.append('None')	
	else:
		for i,j in zip(RA,Dec):
			result_table = Simbad.query_region(astropy.coordinates.SkyCoord(ra=i, dec=j, unit=(u.deg, u.deg)), radius=6. * u.arcsec)
			if result_table != None:
				results.append(result_table['MAIN_ID'][0].replace(" ", "_"))
			else:
				results.append('None')

		# Cross-match cores with NED catalog
		print "Cross-matching NED catalog:"
		Ned.ROW_LIMIT = 1
		results2 = []
		for i,j in zip(RA,Dec):
			result_table_value='Yes'
			try:
				result_table = Ned.query_region(astropy.coordinates.SkyCoord(ra=i, dec=j, unit=(u.deg, u.deg)), radius=6. * u.arcsec)
			except astroquery.exceptions.RemoteServiceError: 
				result_table_value=None
			if result_table_value != None:
				results2.append(result_table['Object Name'][0].replace(" ", "_"))
			else:
				results2.append('None')

	running_number = numpy.arange(len(cores_array[:,0]))+1

	header1 = 'running_NO, getsources_NO, XCO_P, YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02, R_deconv, R_fwhm_mean, resolved, Mass, M_err, Temp, T_err, alpha_BE, SIMBAD_match, NED_match, Spitzer_match'

	# Append the Radius, Mass, Temperature, alpha_BE, etc. arrays as columns 
	# onto the "good cores" array and save as a .dat file
	numpy.savetxt(SED_figure_directory + region_name +'_good_sources_Feb2018.dat', numpy.column_stack((running_number, cores_array,numpy.array(R_deconv),numpy.array(FWHM_mean),numpy.array(resolved), numpy.array(Masses), numpy.array(Masses_err),numpy.array(Temps), numpy.array(Temps_err),numpy.array(alpha_BE),numpy.array(results), numpy.array(results2), Spitzer_column)), fmt='%s', header=header1)

	
	# Create the catalog of good cores; includes flux measurments, positions, etc.
	#make_catalog.make_catalog(region_name=region_name, cloud_name=cloud_name, distance=distance, additional_cores_array=additional_cores_array, good_cores_array = cores_array, cross_matched_core_indices=cross_matched_core_indices, cross_matched_proto_indices=cross_matched_proto_indices, alpha_BE=alpha_BE, getsources_core_catalog = getsources_core_catalog, R_deconv=R_deconv, FWHM_mean = FWHM_mean, Masses=Masses, Masses_err = Masses_err, Temps=Temps, Temps_err=Temps_err, not_accepted_counter = not_accepted_counter, CSAR_catalog = CSAR_catalog, high_res_coldens_image = high_res_coldens_image, SED_figure_directory=SED_figure_directory, Dunham_YSOs_file=Dunham_YSOs_file)

	#make_CMF.CMF_plotter(region=region_name, Masses_minus_protos=Masses2, prestellar_candidates=prestellar_candidates, prestellar_robust=prestellar_robust, SED_figure_directory=SED_figure_directory)

	# Create plot of column density PDF
	#make_plots.coldense_vs_cores(region_name=region_name, SED_figure_directory=SED_figure_directory, high_res_coldens_image=high_res_coldens_image)

	# Create histogram of background column densities for the prestellar cores 
	#make_plots.bg_coldense_plotter(region_name=region_name, SED_figure_directory=SED_figure_directory)

	# Create histogram of core dust temperatures from the cores with reliable SED fits
	#make_plots.core_temp_plotter(region_name=region_name, SED_figure_directory=SED_figure_directory)

	# Create histogram of core radii for the starless core population
	#make_plots.core_size_plotter(region_name=region_name, SED_figure_directory=SED_figure_directory)

	# Create DS9 region files for the good core and proto catalogs at all wavelengths
	#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='all_cores', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
	#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='proto', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
	#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='prestellar_candidates', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
	#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='prestellar_robust', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
	#if len(cross_matched_core_indices)>0:
		#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='proto_cores', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
		#make_DS9_region.make_DS9_regions_good_cores(getsources_core_catalog=getsources_core_catalog, YSO_catalog = YSO_catalog, DS9_region_directory = SED_figure_directory, catalog_type='starless_cores', cross_matched_core_indices=cross_matched_core_indices, candidate_prestellar_indices=candidate_prestellar_indices, robust_prestellar_indices=robust_prestellar_indices, visual_checks=visual_checks)
	#make_DS9_region.make_DS9_regions_CSAR(CSAR_catalog = CSAR_catalog, DS9_region_directory=SED_figure_directory)
	
	#plt.show()

##core_mass_fits(region_name = 'Aquila', distance=436.)
#core_mass_fits(region_name = 'Auriga', distance=450.)
#core_mass_fits(region_name = 'CepheusL1228', distance=200.)
#core_mass_fits(region_name = 'CepheusL1251', distance=300.)
#core_mass_fits(region_name = 'CepheusSouth', distance=288.)
#core_mass_fits(region_name = 'CrA', distance=130.)
#core_mass_fits(region_name = 'IC5146', distance=950.)
#core_mass_fits(region_name = 'Lupus', distance=140.)
#core_mass_fits(region_name = 'OphL1688', distance=139.)
#core_mass_fits(region_name = 'OphL1689_1709_12', distance=139.)
#core_mass_fits(region_name = 'OphScoN2', distance=139.)
#core_mass_fits(region_name = 'OphScoN3', distance=139.) ## No sources pass selection criteria 
#core_mass_fits(region_name = 'OrionA', distance=450.)
#core_mass_fits(region_name = 'OrionB_L1622', distance=415.)
#core_mass_fits(region_name = 'OrionB_N2023', distance=415.)
#core_mass_fits(region_name = 'OrionB_N2068', distance=415.)
#core_mass_fits(region_name = 'PerseusIC348', distance=250.)
core_mass_fits(region_name = 'PerseusWest', distance=250.)
#core_mass_fits(region_name = 'PipeB59', distance=145.)
#core_mass_fits(region_name = 'PipeE1', distance=145.)
#core_mass_fits(region_name = 'SerpensE', distance=436.)
#core_mass_fits(region_name = 'SerpensMain', distance=436.)
#core_mass_fits(region_name = 'SerpensMWC297', distance=250.)
#core_mass_fits(region_name = 'SerpensN', distance=436.)
#core_mass_fits(region_name = 'TaurusB18_E', distance=140.)
#core_mass_fits(region_name = 'TaurusB18_W', distance=140.)
#core_mass_fits(region_name = 'TaurusL1495', distance=140.)
#core_mass_fits(region_name = 'TaurusTMC', distance=140.)
