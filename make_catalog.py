import numpy
import astropy
from astropy import wcs
from astropy import coordinates
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.stats import mstats
from astroquery.simbad import Simbad
from astroquery.ned import Ned
import astroquery

import CSAR_core_cross_match

def make_catalog(region_name, cloud_name, distance, good_cores_array, additional_cores_array, cross_matched_core_indices, cross_matched_proto_indices, alpha_BE, getsources_core_catalog, R_deconv, FWHM_mean, Masses, Masses_err, Temps, Temps_err, not_accepted_counter, CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1157/CSAR/CEPl1157_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1157/080615/cep1157_255_mu.image.resamp.fits', SED_figure_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/L1157/L1157_core_SED/', Dunham_YSOs_file = 'Dunham_YSOs.dat'):

	# These are the values in each column of "good_cores_array"
	#NO,    XCO_P,   YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02, SIG_MONO03, FM03, FXP_BEST03, FXP_ERRO03, FXT_BEST03, FXT_ERRO03, AFWH03, BFWH03, THEP03, SIG_MONO04, FM04, FXP_BEST04, FXP_ERRO04, FXT_BEST04, FXT_ERRO04, AFWH04, BFWH04, THEP04, SIG_MONO05, FM05, FXP_BEST05, FXP_ERRO05, FXT_BEST05, FXT_ERRO05, AFWH05, BFWH05, THEP05, SIG_MONO06, FM06, FXP_BEST06, FXP_ERRO06, FXT_BEST06, FXT_ERRO06, AFWH06, BFWH06, THEP06, SIG_MONO07, FM07, FXP_BEST07, FXP_ERRO07, FXT_BEST07, FXT_ERRO07, AFWH07, BFWH07, THEP07

	# These are the values in each column of the "additional" "cores_array" and "protostar_array"
	# NO    XCO_P   YCO_P PEAK_SRC01 PEAK_BGF01 CONV_SRC01 CONV_BGF01 PEAK_SRC02 PEAK_BGF02 CONV_SRC02 CONV_BGF02 PEAK_SRC03 PEAK_BGF03 CONV_SRC03 CONV_BGF03 PEAK_SRC04 PEAK_BGF04 CONV_SRC04 CONV_BGF04 PEAK_SRC05 PEAK_BGF05 CONV_SRC05 CONV_BGF05 PEAK_SRC06 PEAK_BGF06 CONV_SRC06 CONV_BGF06 PEAK_SRC07 PEAK_BGF07 CONV_SRC07 CONV_BGF07
	
	# Make a column of 1's identifying protostellar cores
	protostellar_catalog_column = numpy.empty(len(good_cores_array[:,0]), dtype='object')
	if len(cross_matched_core_indices)>0:
		protostellar_catalog_column[numpy.array(cross_matched_core_indices)]='1'
		protostellar_catalog_column = numpy.array(protostellar_catalog_column, dtype='S12')

	# Make a column indicating core type: starless, prestellar, or protostellar 
	core_type_column = numpy.where(alpha_BE<=5.0, "prestellar", "starless")
	core_type_column = numpy.array(core_type_column, dtype='S12')
	core_type_column2 = numpy.where(protostellar_catalog_column=='1', "protostellar", core_type_column)

	# Make S_peak/S_background column at each wavelength
	S_peak_bg_070 = additional_cores_array[:,3]/additional_cores_array[:,4]
	S_peak_bg_160 = additional_cores_array[:,7]/additional_cores_array[:,8]
	S_peak_bg_165 = additional_cores_array[:,11]/additional_cores_array[:,12]
	S_peak_bg_250 = additional_cores_array[:,15]/additional_cores_array[:,16]
	S_peak_bg_255 = additional_cores_array[:,19]/additional_cores_array[:,20]
	S_peak_bg_350 = additional_cores_array[:,23]/additional_cores_array[:,24]
	S_peak_bg_500 = additional_cores_array[:,27]/additional_cores_array[:,28]

	# Make S_conv column at each wavelength (convert MJy/str to Jy/beam, then to H2/cm**2)
	# Prepareobs scales down the column density image by a factor of 1e20
	# The final units in the catalog will be off by a factor of 1e20  
	S_conv_070 = numpy.array(additional_cores_array[:,5]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2)) 
	S_conv_160 = numpy.array(additional_cores_array[:,9]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))
	S_conv_165 = numpy.array(additional_cores_array[:,13]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))
	S_conv_250 = numpy.array(additional_cores_array[:,17]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))
	S_conv_255 = numpy.array(additional_cores_array[:,21]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))
	S_conv_350 = numpy.array(additional_cores_array[:,25]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))
	S_conv_500 = numpy.array(additional_cores_array[:,29]*(10**6)*((numpy.pi/180.0/3600.0)**2)*1.13309*(36.3**2))

	N_H2_bg = numpy.array(additional_cores_array[:,20])

	# Define a function that produces a Flux/beam given wavelength, Temp, and ColDense
	# We will input wavelength then find T and M using least squares minimization below
	def col_dense(wavelength, T, N_H2):
		#wavelength input in microns, Temp in Kelvin, N_H2 in cm**-2
		#returns S_v in units of Jy/beam  
		wavelength_mm = numpy.array(wavelength)*10.**-3.
		exponent = 1.439*(wavelength_mm**-1)*((T/10.)**-1)
		aaa = ((2.02*10**20)*(numpy.exp(exponent)-1.0))**-1.0
		bbb = (0.1*((numpy.array(wavelength)/300.)**-2.0))/0.01
		ccc = (36.3/10.)**2.
		ddd = wavelength_mm**-3.
		return N_H2*aaa*bbb*ccc*ddd*(10**-3)

	guess = [10.0, 1.0*10.**21.]
	N_H2_peak = []
	counter = 0
	for S_160, S_250, S_350, S_500 in zip(S_conv_160, S_conv_250, S_conv_350, S_conv_500):
		#print 'Fitting S_peak for Core ' + str(counter) + ' of ' + str(int(len(good_cores_array[:,0])))
		wavelengths = [160.,250.,350.,500.]
		fluxes = [S_160, S_250, S_350, S_500]
		flux_err = [S_160*0.2, S_250*0.1, S_350*0.1, S_500*0.1]
		try:
			popt,pcov = curve_fit(col_dense, wavelengths, fluxes, p0=guess, sigma=flux_err)
		except RuntimeError:
			popt = [-9999., -9999.]
		N_H2_peak.append(popt[1])
		counter+=1

	# Calculate the FWHM_mean at 500 microns
	AFWH07 = good_cores_array[:,68]
	BFWH07 = good_cores_array[:,69]
	A = numpy.float64(((((AFWH07)/60.)/60.)*numpy.pi)/180.) #radians
	A1 = numpy.float64(numpy.tan(A/2.)*2.*distance*(3.086e18)) #cm
	B = numpy.float64(((((BFWH07)/60.)/60.)*numpy.pi)/180.) #radians
	B1 = numpy.float64(numpy.tan(B/2.)*2.*distance*(3.086e18)) #cm
	FWHM_mean_500 = mstats.gmean([A1,B1])
	
	Vol_dense_peak = (((4.0*numpy.log(2.0))/numpy.pi)**0.5)*(numpy.array(N_H2_peak)/FWHM_mean_500)

	# Import CSAR-matched core indices
	print "Cross-matching getsources and CSAR Catalogs:"
	CSAR_matched_cores_indices = CSAR_core_cross_match.cross_match_CSAR(getsources_core_catalog, CSAR_catalog, high_res_coldens_image)

	# Get a cloumn of 1's identifying CSAR cross-matched cores
	CSAR_catalog_column = numpy.zeros(len(good_cores_array[:,0]), dtype='int')
	CSAR_catalog_column[numpy.array(CSAR_matched_cores_indices)]+=1

	# Make a column indicating the number of significant Herschel bands
	N_SED = []
	for line in good_cores_array:
		counter = 0
		if line[8]>5 and line[12]>0:
			counter+=1
		# Statement below uses the 160micron map, not the temp-corrected map
		if line[17]>5 and line[21]>0:
			counter+=1
		if line[35]>5 and line[39]>0:
			counter+=1
		if line[53]>5 and line[57]>0:
			counter+=1
		if line[62]>5 and line[66]>0:
			counter+=1
		N_SED.append(counter)
	
	# Convert the decimal degrees coordinates of getsources into hh:mm:ss and dd:mm:ss
	RA_array = []
	Dec_array = []
	HGBS_name_array = []
	for line in good_cores_array:		
		RA = astropy.coordinates.Angle(line[3], u.degree)
		DEC = astropy.coordinates.Angle(line[4], u.degree)
		RA_hours = str('{:.0f}'.format(round(RA.hms[0],2)).zfill(2))
		RA_minutes = str('{:.0f}'.format(round(RA.hms[1],2)).zfill(2))
		RA_seconds = str('{:.2f}'.format(round(RA.hms[2],2)).zfill(5))
		if DEC.hms[0] > 0:
			DEC_degs = str('{:.0f}'.format(round(DEC.dms[0],2)).zfill(2))
			DEC_minutes = str('{:.0f}'.format(round(DEC.dms[1],2)).zfill(2))
			DEC_seconds = str('{:.2f}'.format(round(DEC.dms[2],2)).zfill(5))
			name_sign = '+'
			HGBS_name = RA_hours+RA_minutes+RA_seconds[0:4]+name_sign+DEC_degs+DEC_minutes+DEC_seconds[0:2]
		else:
			DEC_degs = str('{:.0f}'.format(round(DEC.dms[0],2)).zfill(3))
			DEC_minutes = str('{:.0f}'.format(round(DEC.dms[0]*-1,2)).zfill(2))
			DEC_seconds = str('{:.2f}'.format(round(DEC.dms[2]*-1,2)).zfill(5))
			HGB_name = RA_hours+RA_minutes+RA_seconds[0:4]+DEC_degs+DEC_minutes+DEC_seconds[0:2]
			
		RA_array.append(RA_hours + ':' + RA_minutes + ':' + RA_seconds)
		Dec_array.append(DEC_degs + ':' + DEC_minutes + ':' + DEC_seconds)

		HGBS_name_array.append("HGBS_J"+HGBS_name)
	
	core_number = numpy.arange(len(good_cores_array[:,0]))+1
	
	catalog_array_70_160 = good_cores_array[:,8:26]
	catalog_array_70_160 = numpy.delete(catalog_array_70_160, (1,10), 1)

	catalog_array_250 = good_cores_array[:,35:44]
	catalog_array_250 = numpy.delete(catalog_array_250, 1, 1)

	catalog_array_coldense = good_cores_array[:,44:53]
	catalog_array_coldense = numpy.delete(catalog_array_coldense, 1, 1)

	catalog_array_350_500 = good_cores_array[:,53:71]
	catalog_array_350_500 = numpy.delete(catalog_array_350_500, (1,10), 1)

	Cloud,Name,Av,alpha,T_bol,L_bol,alphaPrime,TbolPrime,LbolPrime,likelyAGB,Dunham_RA,Dunham_DEC,Class = numpy.loadtxt(Dunham_YSOs_file, delimiter=',', unpack=True, dtype=[('Cloud','S30'),('Name','S40'), ('Av',float),('alpha',float), ('T_bol',float),('L_bol',float), ('alphaPrime',float),('TbolPrime',float), ('LbolPrime',float),('likelyAGB','S1'), ('Dunham_RA',float),('Dunham_DEC',float),('Class','S10') ])
	Dunham_indices = numpy.where(Cloud==cloud_name)
	Spitzer_YSOs_RA = Dunham_RA[Dunham_indices]
	Spitzer_YSOs_DEC = Dunham_DEC[Dunham_indices]
	Spitzer_YSOs_Name = Name[Dunham_indices]

	potential_matches = []
	YSO_matches = []
	count = 0
	for line in good_cores_array:
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

	Spitzer_column = numpy.zeros(len(good_cores_array[:,0]), dtype='S40')
	Spitzer_column[numpy.arange(0,len(good_cores_array[:,0]))] = 'None'
	if len(potential_matches)>0:
		Spitzer_column[numpy.array(potential_matches)] = Spitzer_YSOs_Name[numpy.array(YSO_matches)]

	# Cross-match cores with SIMBAD catalog
	print "Cross-matching SIMBAD catalog:"
	RA, Dec = numpy.loadtxt(SED_figure_directory + region_name +'_SIMBAD_RA_DEC.dat', unpack=True)
	Simbad.ROW_LIMIT = 1
	results = []
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
	
	zipped_array = zip(core_number, HGBS_name_array, RA_array, Dec_array, catalog_array_70_160[:,0], catalog_array_70_160[:,1], catalog_array_70_160[:,2], S_peak_bg_070, S_conv_070, catalog_array_70_160[:,3], catalog_array_70_160[:,4], catalog_array_70_160[:,5], catalog_array_70_160[:,6], catalog_array_70_160[:,7], catalog_array_70_160[:,8], catalog_array_70_160[:,9], catalog_array_70_160[:,10], S_peak_bg_160, S_conv_160, catalog_array_70_160[:,11], catalog_array_70_160[:,12], catalog_array_70_160[:,13], catalog_array_70_160[:,14], catalog_array_70_160[:,15], catalog_array_250[:,0], catalog_array_250[:,1], catalog_array_250[:,2], S_peak_bg_250, S_conv_250, catalog_array_250[:,3], catalog_array_250[:,4], catalog_array_250[:,5], catalog_array_250[:,6], catalog_array_250[:,7], catalog_array_350_500[:,0], catalog_array_350_500[:,1], catalog_array_350_500[:,2], S_peak_bg_350, S_conv_350, catalog_array_350_500[:,3], catalog_array_350_500[:,4], catalog_array_350_500[:,5], catalog_array_350_500[:,6], catalog_array_350_500[:,7], catalog_array_350_500[:,8], catalog_array_350_500[:,9], catalog_array_350_500[:,10], S_peak_bg_500, catalog_array_350_500[:,11], catalog_array_350_500[:,12], catalog_array_350_500[:,13], catalog_array_350_500[:,14], catalog_array_350_500[:,15], catalog_array_coldense[:,0], additional_cores_array[:,19], S_peak_bg_255, S_conv_255, N_H2_bg, catalog_array_coldense[:,5], catalog_array_coldense[:,6], catalog_array_coldense[:,7], N_SED, CSAR_catalog_column, core_type_column2, results, results2, Spitzer_column)
	
	catalog1 = numpy.array(zipped_array, dtype=[('core_number',int),('HGBS_name_array','S30'),('RA_array','S16'),('Dec_array','S16'),('catalog_array_70_160_1',float),('catalog_array_70_160_2',float), ('catalog_array_70_160_3',float),('S_peak_bg_070',float),('S_conv_070',float), ('catalog_array_70_160_4',float),('catalog_array_70_160_5',float),('catalog_array_70_160_6',float), ('catalog_array_70_160_7',float),('catalog_array_70_160_8',float), ('catalog_array_70_160_9',float),('catalog_array_70_160_10',float), ('catalog_array_70_160_11',float),('S_peak_bg_160',float),('S_conv_160',float),('catalog_array_70_160_12',float),('catalog_array_70_160_13',float),('catalog_array_70_160_14',float), ('catalog_array_70_160_15',float),('catalog_array_70_160_16',float), ('catalog_array_250_1',float),('catalog_array_250_2',float), ('catalog_array_250_3',float),('S_peak_bg_250',float),('S_conv_250',float),('catalog_array_250_4',float),('catalog_array_250_5',float),('catalog_array_250_6',float), ('catalog_array_250_7',float),('catalog_array_250_8',float),('catalog_array_350_500_1',float),('catalog_array_350_500_2',float), ('catalog_array_350_500_3',float),('S_peak_bg_350',float),('S_conv_350',float),('catalog_array_350_500_4',float),('catalog_array_350_500_5',float),('catalog_array_350_500_6',float), ('catalog_array_350_500_7',float),('catalog_array_350_500_8',float), ('catalog_array_350_500_9',float),('catalog_array_350_500_10',float), ('catalog_array_350_500_11',float),('S_peak_bg_500',float),('catalog_array_350_500_12',float),('catalog_array_350_500_13',float),('catalog_array_350_500_14',float), ('catalog_array_350_500_15',float),('catalog_array_350_500_16',float), ('catalog_array_coldense_1',float),('catalog_array_coldense_2',float),('S_peak_bg_255',float),('S_conv_255',float),('additional_cores_array_28',float),('catalog_array_coldense_6',float), ('catalog_array_coldense_7',float),('catalog_array_coldense_8',float),('N_SED',int),('CSAR_catalog_column',int),('core_type_column','S16'), ('SIMBAD_column','S60'), ('NED_column','S60'), ('Spitzer_column','S40')])

	header1 = 'core_number, core_name, RA_hms, DEC_dms, sig_070, peak_flux_070, peak_flux_err_070, peak_flux_over_bg_070, peak_070_conv_500, total_flux_070, total_flux_err_070, AFWHM_070, BFWHM_070, PA_070, sig_160, peak_flux_160, peak_flux_err_160, peak_flux_over_bg_160, peak_160_conv_500, total_flux_160, total_flux_err_160, AFWHM_160, BFWHM_160, PA_160, sig_250, peak_flux_250, peak_flux_err_250, peak_flux_over_bg_250, peak_250_conv_500, total_flux_250, total_flux_err_250, AFWHM_250, BFWHM_250, PA_250, sig_350, peak_flux_350, peak_flux_err_350, peak_flux_over_bg_350, peak_350_conv_500, total_flux_350, total_flux_err_350, AFWHM_350, BFWHM_350, PA_350, sig_500, peak_flux_500, peak_flux_err_500, peak_flux_over_bg_500, total_flux_500, total_flux_err_500, AFWHM_500, BFWHM_500, PA_500, sig_coldens, peak_flux_coldens, peak_flux_over_bg_coldens, peak_coldens_conv_500, peak_bg_coldens, AFWHM_coldens, BFWHM_coldens, PA_coldens, N_SED, CSAR, core_type, SIMBAD_match, NED_match, Spitzer_match' 
	
	numpy.savetxt(SED_figure_directory + region_name + '_core_catalog1.dat', catalog1, fmt="%i %s %s %s %3.1f %1.2e %1.1e %3.3f %1.2e %1.2e %1.1e %3.1f %3.1f %3.1f %3.1f %1.2e %1.1e %3.3f %1.2e %1.2e %1.1e %3.1f %3.1f %3.1f %3.1f %1.2e %1.1e %3.3f %1.2e %1.2e %1.1e %3.1f %3.1f %3.1f %3.1f %1.2e %1.1e %3.3f %1.2e %1.2e %1.1e %3.1f %3.1f %3.1f %3.1f %1.2e %1.1e %3.3f %1.2e %1.1e %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %i %i %s %s %s %s", header=header1)

	mu = 2.8 # mean molecular weight 
	mass_H = 1.67372e-24 # (grams) mass of neutral Hydrogen atom
	solar_mass = 1.989e33 # (grams)
	mass_H_solar_masses = mass_H / solar_mass
	parsec = 3.086e18 # cm
	R_deconv_cm = numpy.array(R_deconv)*parsec
	FWHM_mean_cm = numpy.array(FWHM_mean)*parsec
	N_H2_avg_1 = (numpy.array(Masses)/(numpy.pi*(R_deconv_cm**2.))) * (1/(mu*mass_H_solar_masses))
	N_H2_avg_2 = (numpy.array(Masses)/(numpy.pi*(FWHM_mean_cm**2.))) * (1/(mu*mass_H_solar_masses))
	avg_Volume_dens_1 = (numpy.array(Masses)/(numpy.pi*(4./3.)*(R_deconv_cm**3.))) * (1/(mu*mass_H_solar_masses))
	avg_Volume_dens_2 = (numpy.array(Masses)/(numpy.pi*(4./3.)*(FWHM_mean_cm**3.))) * (1/(mu*mass_H_solar_masses))
	
	catalog2 = numpy.array(zip(core_number, HGBS_name_array, RA_array, Dec_array, R_deconv, FWHM_mean, Masses, Masses_err, Temps, Temps_err, N_H2_peak, N_H2_avg_1, N_H2_avg_2, Vol_dense_peak, avg_Volume_dens_1, avg_Volume_dens_2, alpha_BE, core_type_column2, not_accepted_counter), dtype=[('core_number',int),('HGBS_name_array','S30'),('RA_array','S16'),('Dec_array','S16'),('R_deconv',float),('FWHM_mean',float),('Masses',float), ('Masses_err',float),('Temps',float),('Temps_err',float),('N_H2_peak',float),('N_H2_avg_1',float),('N_H2_avg_2',float),('Vol_dense_peak',float),('avg_Volume_dens1',float),('avg_Volume_dens_2',float),('alpha_BE',float),('core_type_column2','S16'),('not_accepted_counter','S16')])

	header2 = 'core_number, core_name, RA_hms, DEC_dms, R_deconv, FWHM_mean, Mass, Mass_err, Temp_dust, Temp_dust_err, N_H2_peak, N_H2_avg_1, N_H2_avg_2, Vol_dense_peak, avg_Volume_dens_1, avg_Volume_dens_2, alpha_BE, core_type_column2, not_accepted_counter' 
	
	numpy.savetxt(SED_figure_directory + region_name +'_core_catalog2.dat', catalog2, fmt="%i %s %s %s %1.1e %1.1e %1.3f %1.2f %2.1f %2.1f %1.2e %1.2e %1.2e %1.2e %1.2e %1.2e %2.1f %s %s", header=header2)
	
	
#make_catalog()

