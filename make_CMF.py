import numpy
from pylab import *
from scipy.optimize import curve_fit
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import astropy
from astropy.io import fits
import pyregion
from numpy import genfromtxt
from astropy import wcs
from astropy import coordinates
import matplotlib.patheffects as Patheffects
from astropy import units as u
import aplpy

#### Lognormal Distribution

def func(x, mu, sig, A):
	aaa = A/(sig*((2*numpy.pi)**0.5))
	bbb = -((x-mu)**2)/(2*(sig**2))
	ccc = numpy.exp(bbb)
	return aaa*ccc

#### Power Law
def power_law(Mass, constant1, power):
	y1=numpy.log10(constant1) + power*log10(Mass)
	return y1

#### Chabrier IMF
def chabrier(x):
	aaa = 0.076
	bbb = -((x-numpy.log10(0.25))**2)/(2*(0.55**2))
	ccc = numpy.exp(bbb)
	return aaa*ccc

#### Plot the Core Mass Function
def CMF_plotter(region, Masses_minus_protos, prestellar_candidates, prestellar_robust, SED_figure_directory):

	# Lognormal fit guess parameters
	guess = [-1, 0.4, 10.]

	#### Kroupa IMF
	M1 = np.array(np.arange(0.5,10,0.001))
	y1 = (400/7.0)*M1**-1.3

	M2 = np.array(np.arange(0.08,0.5,0.001))
	y2 = (400/3.5)*M2**-0.3

	M3 = np.array(np.arange(0.01,0.08,0.001))
	y3 = (400/0.28)*M3**0.7

	#### Chabrier IMF
	chab = 10**(numpy.log10(chabrier(numpy.linspace(-2.0,0.0,100)))+numpy.log10(1500))
	# Power-law tail of Chabrier IMF
	chab2 = np.array(np.arange(1.0,10,0.001))
	y_chab2 = 10**(numpy.log10(0.041*chab2**-1.35)+numpy.log10(1500))
	
	# Plot the mass function for the entire starless core population
	fig = plt.figure()
	histogram, bins = numpy.histogram(Masses_minus_protos, bins=numpy.logspace(-2.0,1.0,14))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', markerfacecolor='none', marker='^')
	plt.plot(centers, histogram, color='green', drawstyle='steps-mid')
	#plt.hist(total_masses, bins=numpy.logspace(-2.0,1.0,11))
	plt.annotate(str(len(Masses_minus_protos)) + " Total Cores", xy=(0.7,0.9), xycoords=('axes fraction'), color="green")
	plt.annotate(str(len(prestellar_candidates)) + " Prestellar Cores", xy=(0.7,0.85), xycoords=('axes fraction'), color="blue")
	plt.annotate(str(len(prestellar_robust)) + " Robust Prestellar", xy=(0.7,0.8), xycoords=('axes fraction'), color="blue")

	# Plot the Kroupa, then Chabrier IMF
	plt.plot(M1,y1, color='black', linestyle='dashdot')
	plt.plot(M2,y2, color='black', linestyle='dashdot')
	plt.plot(M3,y3, color='black', linestyle='dashdot')
	plt.plot(chab2,y_chab2, color='black')
	plt.plot(10**numpy.linspace(-2.0,0.0,100),chab, color="black")

	# Perform a lognormal fit to the CMF
	#popt, pcov = curve_fit(func, numpy.log10(centers), histogram, p0=guess)
	#plt.plot(10**numpy.linspace(-2.0,1.0,100),func(numpy.linspace(-2.0,1.0,100), popt[0], popt[1], popt[2]), color="green")
	
	plt.semilogy()
	plt.semilogx()
	plt.title(region + ' CMF')
	plt.xlabel("Mass, M (M$_\odot$)")
	plt.ylabel("Number of objects per mass bin: $\Delta$N/$\Delta$logM")
	
	# Plot prestellar candidates CMF
	histogram, bins = numpy.histogram(prestellar_candidates, bins=numpy.logspace(-2.0,1.0,14))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers, histogram, color='blue', yerr = root_N_err, linestyle='None', markerfacecolor='none', marker='^')
	plt.plot(centers, histogram, color='blue', drawstyle='steps-mid')
	#plt.hist(prestellar_cand, bins=numpy.logspace(-2.0,1.0,11), color='blue')
	popt, pcov = curve_fit(func, numpy.log10(centers), histogram, p0=guess)
	plt.plot(10**numpy.linspace(-2.0,1.0,100),func(numpy.linspace(-2.0,1.0,100), popt[0], popt[1], popt[2]), color="brown", linestyle='dashdot')
	plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$", xy=(0.7,0.2), xycoords=('axes fraction'), color="brown")
	
	# Plot robust prestellar candidate CMF
	histogram, bins = numpy.histogram(prestellar_robust, bins=numpy.logspace(-2.0,1.0,14))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers, histogram, yerr = root_N_err, linestyle='None', color='blue', marker='^')
	plt.plot(centers, histogram, color='blue', drawstyle='steps-mid')
	#plt.hist(prestellar_robust, bins=numpy.logspace(-2.0,1.0,11), color='red')
	popt, pcov = curve_fit(func, numpy.log10(centers), histogram, p0=guess)
	plt.plot(10**numpy.linspace(-2.0,1.0,100),func(numpy.linspace(-2.0,1.0,100), popt[0], popt[1], popt[2]), color="red")
	plt.xlim([10**-2, 3*10**1])
	plt.ylim([10**-2, 10**3])
	plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$", xy=(0.7,0.15), xycoords=('axes fraction'), color="red")
	fig.savefig(SED_figure_directory + region + "_CMF.png")
	#plt.show()
	plt.close()

#### Below are my personal functions used for the Cepheus analysis.
#### These are currently specialized for Cepheus, but could be adapted to work with other regions

#### Plot the Core Mass Function of the combined Cepheus regions
def Cepheus_CMF_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_radius, L1157_mass, L1157_alpha, L1157_type = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1172_radius, L1172_mass, L1172_alpha, L1172_type = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1228_radius, L1228_mass, L1228_alpha, L1228_type = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1241_radius, L1241_mass, L1241_alpha, L1241_type = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1251_radius, L1251_mass, L1251_alpha, L1251_type = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)

	Aquila_mass, Aquila_type, Aquila_alpha = numpy.loadtxt('tablea2.dat', usecols=(10,21, 20), dtype=[('bg',float),('type','S20'),('bg2',float)], unpack=True)
	f = open('tablea2.dat', "r")
	lines = numpy.array(f.readlines())
	f.close()

	starless_indices_Aquila = numpy.where(Aquila_type!='protostellar')

	prestellar_indices_Aquila = numpy.where(Aquila_type=='prestellar')

	robust_indices_Aquila = numpy.where(Aquila_alpha<=2.0)

	count = 0
	indices = []
	for line in lines[prestellar_indices_Aquila]:
		if "high-V_LSR" in line:
			indices.append(count)
		count+=1

	Aquila_prestellar = numpy.delete(Aquila_mass[prestellar_indices_Aquila], numpy.array(indices))

	count = 0
	indices = []
	for line in lines[robust_indices_Aquila]:
		if "high-V_LSR" in line or 'protostellar' in line:
			indices.append(count)
		count+=1

	Aquila_robust = numpy.delete(Aquila_mass[robust_indices_Aquila], numpy.array(indices))
	print len(Aquila_robust)

	
	Cepheus_mass = numpy.concatenate([L1157_mass, L1172_mass, L1228_mass, L1241_mass, L1251_mass])
	Cepheus_alpha = numpy.concatenate([L1157_alpha, L1172_alpha, L1228_alpha, L1241_alpha, L1251_alpha])
	Cepheus_type = numpy.concatenate([L1157_type, L1172_type, L1228_type, L1241_type, L1251_type])
	Cepheus_prestellar_masses = []
	for g,j,k in zip(Cepheus_mass, Cepheus_type, Cepheus_alpha):
		if j!='protostellar' and k<=5.0:
			Cepheus_prestellar_masses.append(g)

	Cepheus_robust_masses = []
	for g,j,k in zip(Cepheus_mass, Cepheus_type, Cepheus_alpha):
		if j!='protostellar' and k<=2.0:
			Cepheus_robust_masses.append(g)

	L1157_robust_masses = []
	for g,j,k in zip(L1157_mass, L1157_type, L1157_alpha):
		if j!='protostellar' and k<=2.0:
			L1157_robust_masses.append(g)

	L1172_robust_masses = []
	for g,j,k in zip(L1172_mass, L1172_type, L1172_alpha):
		if j!='protostellar' and k<=2.0:
			L1172_robust_masses.append(g)

	L1228_robust_masses = []
	for g,j,k in zip(L1228_mass, L1228_type, L1228_alpha):
		if j!='protostellar' and k<=2.0:
			L1228_robust_masses.append(g)

	L1241_robust_masses = []
	for g,j,k in zip(L1241_mass, L1241_type, L1241_alpha):
		if j!='protostellar' and k<=2.0:
			L1241_robust_masses.append(g)

	L1251_robust_masses = []
	for g,j,k in zip(L1251_mass, L1251_type, L1251_alpha):
		if j!='protostellar' and k<=2.0:
			L1251_robust_masses.append(g)

	robust_masses = [L1157_robust_masses, L1172_robust_masses, L1228_robust_masses, L1241_robust_masses, L1251_robust_masses]

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')

	robust_indices_L1157 = numpy.where(L1157_alpha<=2.0)
	robust_indices_L1172 = numpy.where(L1172_alpha<=2.0)
	robust_indices_L1228 = numpy.where(L1228_alpha<=2.0)
	robust_indices_L1241 = numpy.where(L1241_alpha<=2.0)
	robust_indices_L1251 = numpy.where(L1251_alpha<=2.0)

	starless_masses_L1157 = numpy.array(L1157_mass)[starless_indices_L1157]
	starless_masses_L1172 = numpy.array(L1172_mass)[starless_indices_L1172]
	starless_masses_L1228 = numpy.array(L1228_mass)[starless_indices_L1228]
	starless_masses_L1241 = numpy.array(L1241_mass)[starless_indices_L1241]
	starless_masses_L1251 = numpy.array(L1251_mass)[starless_indices_L1251]
	Masses_minus_protos = numpy.concatenate([starless_masses_L1157, starless_masses_L1172, starless_masses_L1228, starless_masses_L1241, starless_masses_L1251])

	prestellar_masses_L1157 = numpy.array(L1157_mass)[prestellar_indices_L1157]
	prestellar_masses_L1172 = numpy.array(L1172_mass)[prestellar_indices_L1172]
	prestellar_masses_L1228 = numpy.array(L1228_mass)[prestellar_indices_L1228]
	prestellar_masses_L1241 = numpy.array(L1241_mass)[prestellar_indices_L1241]
	prestellar_masses_L1251 = numpy.array(L1251_mass)[prestellar_indices_L1251]
	prestellar_candidates = numpy.concatenate([prestellar_masses_L1157, prestellar_masses_L1172, prestellar_masses_L1228, prestellar_masses_L1241, prestellar_masses_L1251])

	robust_masses_L1157 = numpy.array(L1157_mass)[robust_indices_L1157]
	robust_masses_L1172 = numpy.array(L1172_mass)[robust_indices_L1172]
	robust_masses_L1228 = numpy.array(L1228_mass)[robust_indices_L1228]
	robust_masses_L1241 = numpy.array(L1241_mass)[robust_indices_L1241]
	robust_masses_L1251 = numpy.array(L1251_mass)[robust_indices_L1251]
	prestellar_robust = numpy.concatenate([robust_masses_L1157, robust_masses_L1172, robust_masses_L1228, robust_masses_L1241, robust_masses_L1251])	

	radii_L1157 = numpy.array(L1157_radius)[starless_indices_L1157]
	radii_L1172 = numpy.array(L1172_radius)[starless_indices_L1172]
	radii_L1228 = numpy.array(L1228_radius)[starless_indices_L1228]
	radii_L1241 = numpy.array(L1241_radius)[starless_indices_L1241]
	radii_L1251 = numpy.array(L1251_radius)[starless_indices_L1251]
	starless_radii = numpy.concatenate([radii_L1157, radii_L1172, radii_L1228, radii_L1241, radii_L1251])

	radii_L1157 = numpy.array(L1157_radius)[prestellar_indices_L1157]
	radii_L1172 = numpy.array(L1172_radius)[prestellar_indices_L1172]
	radii_L1228 = numpy.array(L1228_radius)[prestellar_indices_L1228]
	radii_L1241 = numpy.array(L1241_radius)[prestellar_indices_L1241]
	radii_L1251 = numpy.array(L1251_radius)[prestellar_indices_L1251]
	prestellar_radii = numpy.concatenate([radii_L1157, radii_L1172, radii_L1228, radii_L1241, radii_L1251])

	radii_L1157 = numpy.array(L1157_radius)[robust_indices_L1157]
	radii_L1172 = numpy.array(L1172_radius)[robust_indices_L1172]
	radii_L1228 = numpy.array(L1228_radius)[robust_indices_L1228]
	radii_L1241 = numpy.array(L1241_radius)[robust_indices_L1241]
	radii_L1251 = numpy.array(L1251_radius)[robust_indices_L1251]
	robust_radii = numpy.concatenate([radii_L1157, radii_L1172, radii_L1228, radii_L1241, radii_L1251])

	# Plot Mass versus Radius and save the figure
	fig = plt.figure()
	plt.scatter(starless_radii,Masses_minus_protos, label='starless')
	plt.scatter(prestellar_radii,prestellar_candidates, color='red', label='candidate')
	plt.scatter(robust_radii,prestellar_robust, color='green', label='robust')
	plt.yscale('log')
	plt.xscale('log')
	#plt.legend()
	plt.title('Cepheus Cores')
	plt.ylabel("Mass, M (M$_\odot$)")
	plt.xlabel("Deconvolved FWHM size, R (pc)")
	plt.xlim([10**-3, 2*10**-1])
	plt.ylim([10**-3, 10**2])
	fig.savefig('Cepheus_mass_vs_radius.png')

	# Lognormal fit guess parameters
	guess = [-1, 0.4, 10.]

	#### Kroupa IMF
	M1 = np.array(np.arange(0.5,10,0.001))
	y1 = (400/7.0)*M1**-1.3

	M2 = np.array(np.arange(0.08,0.5,0.001))
	y2 = (400/3.5)*M2**-0.3

	M3 = np.array(np.arange(0.01,0.08,0.001))
	y3 = (400/0.28)*M3**0.7

	#### Chabrier IMF
	chab = 10**(numpy.log10(chabrier(numpy.linspace(-2.0,0.0,100)))+numpy.log10(1500))
	# Power-law tail of Chabrier IMF
	chab2 = np.array(np.arange(1.0,10,0.001))
	y_chab2 = 10**(numpy.log10(0.041*chab2**-1.35)+numpy.log10(1500))

	fig1 = plt.figure()
	# Plot the Kroupa, then Chabrier IMF
	#plt.plot(M1,y1, color='black', linestyle='dashdot')
	#plt.plot(M2,y2, color='black', linestyle='dashdot')
	#plt.plot(M3,y3, color='black', linestyle='dashdot')
	#plt.plot(chab2,y_chab2, color='black')
	#plt.plot(10**numpy.linspace(-2.0,0.0,100),chab, color="black")
	
	plt.semilogy()
	plt.semilogx()
	#plt.title('Cepheus CMF')
	plt.xlabel("Mass, M (M$_\odot$)", size=20)
	plt.ylabel("Cores per mass bin: $\Delta$N/$\Delta$logM", size=20)
	colors = ['red', 'blue', 'green', 'brown', 'black']
	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251']
	ypos = [0.8,0.75,0.7,0.65,0.6]
	factor = [1., 10., 100., 1000., 10000.]
	counter=0
	for i in robust_masses:
		histogram, bins = numpy.histogram(i, bins=numpy.logspace(-2.0,1.0,15))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		print histogram
		print centers
		erry = root_N_err[numpy.where(centers>0.1)]*factor[counter]
		erry2 = numpy.where(erry==factor[counter], 0, erry)
		plt.errorbar(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)]*factor[counter], yerr = erry2, linestyle='None', marker='^', color=colors[counter])
		plt.plot(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)]*factor[counter], drawstyle='steps-mid', label=regions[counter],color=colors[counter], linewidth=2.0)
		#popt, pcov = curve_fit(func, numpy.log10(centers[numpy.where(centers>0.2)]), histogram[numpy.where(centers>0.2)], p0=guess, sigma=root_N_err[numpy.where(centers>0.2)])
		#plt.plot(10**numpy.linspace(-0.9,1.0,100),func(numpy.linspace(-0.9,1.0,100), popt[0], popt[1], popt[2]), color=colors[counter])
		#plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$ $\sigma$ = " + str("{0:.2f}".format(round(popt[1],2))), xy=(0.65,ypos[counter]), xycoords=('axes fraction'),color=colors[counter])
		counter+=1
		print counter
	
	plt.xlim([10**-2, 3*10**1])
	plt.ylim([.5, 4*10**5])
	plt.legend(loc='upper left', prop={'size':18})
	plt.tick_params(axis='both', labelsize=20)
	fig1.savefig("Cepheus_CMF_2.pdf", bbox_inches='tight')
	plt.show()
	
	# Plot the mass function for the entire starless core population
	fig = plt.figure()
	#histogram, bins = numpy.histogram(Masses_minus_protos, bins=numpy.logspace(-2.0,1.0,11))
	#root_N_err = histogram**0.5
	#centers = (bins[:-1]+bins[1:])/2
	#plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', markerfacecolor='none', marker='^')
	#plt.plot(centers, histogram, color='green', drawstyle='steps-mid')
	#plt.hist(total_masses, bins=numpy.logspace(-2.0,1.0,11))
	#plt.annotate(str(len(Masses_minus_protos)) + " Total Cores", xy=(0.7,0.9), xycoords=('axes fraction'), color="green")
	#plt.annotate(str(len(Cepheus_prestellar_masses)) + " Prestellar Cores", xy=(0.7,0.85), xycoords=('axes fraction'), color="blue")
	#plt.annotate(str(len(Cepheus_robust_masses)) + " Robust Prestellar", xy=(0.7,0.8), xycoords=('axes fraction'), color="blue")

	# Plot the Kroupa, then Chabrier IMF
	plt.plot(M1,y1, color='black', linestyle='dashdot')
	plt.plot(M2,y2, color='black', linestyle='dashdot')
	plt.plot(M3,y3, color='black', linestyle='dashdot')
	plt.plot(chab2,y_chab2, color='black')
	plt.plot(10**numpy.linspace(-2.0,0.0,100),chab, color="black")

	# Perform a lognormal fit to the CMF
	#popt, pcov = curve_fit(func, numpy.log10(centers), histogram, p0=guess)
	#plt.plot(10**numpy.linspace(-2.0,1.0,100),func(numpy.linspace(-2.0,1.0,100), popt[0], popt[1], popt[2]), color="green")
	
	plt.semilogy()
	plt.semilogx()
	#plt.title('Cepheus CMF')
	plt.xlabel("Mass, M (M$_\odot$)", size=20)
	plt.ylabel("Cores per mass bin: $\Delta$N/$\Delta$logM", size=20)
	plt.tick_params(axis='both', labelsize=20)
	
	# Plot prestellar candidates CMF
	#histogram, bins = numpy.histogram(Cepheus_prestellar_masses, bins=numpy.logspace(-2.0,1.0,11))
	#root_N_err = histogram**0.5
	#centers = (bins[:-1]+bins[1:])/2
	#plt.errorbar(centers[numpy.where(histogram>1)], histogram[numpy.where(histogram>1)], color='blue', yerr = root_N_err[numpy.where(histogram>1)], linestyle='None', markerfacecolor='none', marker='^')
	#plt.plot(centers[numpy.where(histogram>1)], histogram[numpy.where(histogram>1)], color='blue', drawstyle='steps-mid')
	#plt.hist(prestellar_cand, bins=numpy.logspace(-2.0,1.0,11), color='blue')
	#popt, pcov = curve_fit(func, numpy.log10(centers[numpy.where(histogram>1)]), histogram[numpy.where(histogram>1)], p0=guess)
	#plt.plot(10**numpy.linspace(-2.0,1.0,100),func(numpy.linspace(-2.0,1.0,100), popt[0], popt[1], popt[2]), color="brown", linestyle='dashdot')
	#plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$", xy=(0.7,0.7), xycoords=('axes fraction'), color="brown")

	#guess2 = [10.,-1.35]
	#print centers[numpy.where(centers>0.9)]
	#popt2, pcov2 = curve_fit(power_law, centers[numpy.where(centers>0.9)], histogram[numpy.where(centers>0.9)], p0=guess2)
	#print popt2
	#print pcov2
	
	# Plot Aquila prestellar robust CMF
	histogram, bins = numpy.histogram(Aquila_robust, bins=numpy.logspace(-2.0,1.0,15))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)], color='blue', yerr = root_N_err[numpy.where(centers>0.1)], linestyle='None', markerfacecolor='none', marker='^')
	plt.plot(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)], color='blue', drawstyle='steps-mid', label="Aquila", linewidth=2.0)
	popt, pcov = curve_fit(func, numpy.log10(centers[numpy.where(centers>0.2)]), histogram[numpy.where(centers>0.2)], p0=guess, sigma=root_N_err[numpy.where(centers>0.2)])
	a = abs(10**(popt[0]+(pcov[0][0]**0.5)) - 10**popt[0])
	b = abs(10**(popt[0]-(pcov[0][0]**0.5)) - 10**popt[0])
	c = [a,b]
	plt.plot(10**numpy.linspace(-0.9,1.0,100),func(numpy.linspace(-0.9,1.0,100), popt[0], popt[1], popt[2]), color="red")
	plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$ $\sigma$ = " + str("{0:.2f}".format(round(popt[1],2))), xy=(0.58,0.75), xycoords=('axes fraction'), color="red", size=14)

	guess2 = [10.,-1.35]
	print centers[numpy.where(centers>0.9)]
	popt2, pcov2 = curve_fit(power_law, centers[numpy.where(centers>0.9)], numpy.log10(histogram[numpy.where(centers>0.9)]), p0=guess2)
	print popt2
	print pcov2

	# Plot robust prestellar candidate CMF
	print len(Cepheus_robust_masses)
	print len(Masses_minus_protos)
	histogram, bins = numpy.histogram(Cepheus_robust_masses, bins=numpy.logspace(-2.0,1.0,15))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)], yerr = root_N_err[numpy.where(centers>0.1)], linestyle='None', color='green', marker='^')
	plt.plot(centers[numpy.where(centers>0.1)], histogram[numpy.where(centers>0.1)], color='green', drawstyle='steps-mid', label="Cepheus", linewidth=2.0)
	#plt.hist(prestellar_robust, bins=numpy.logspace(-2.0,1.0,11), color='red')
	popt, pcov = curve_fit(func, numpy.log10(centers[numpy.where(centers>0.2)]), histogram[numpy.where(centers>0.2)], p0=guess, sigma=root_N_err[numpy.where(centers>0.2)])
	a = abs(10**(popt[0]+(pcov[0][0]**0.5)) - 10**popt[0])
	b = abs(10**(popt[0]-(pcov[0][0]**0.5)) - 10**popt[0])
	c = [a,b]
	plt.plot(10**numpy.linspace(-0.9,1.0,100),func(numpy.linspace(-0.9,1.0,100), popt[0], popt[1], popt[2]), color="brown")
	guess2 = [10.,-1.35]
	print centers[numpy.where(centers>0.9)]
	popt2, pcov2 = curve_fit(power_law, centers[numpy.where(centers>0.9)], numpy.log10(histogram[numpy.where(centers>0.9)]), p0=guess2)
	print popt2
	print pcov2
	plt.xlim([10**-2, 3*10**1])
	plt.ylim([1, 5*10**2])
	plt.annotate('Peak = ' + str("{0:.2f}".format(round(10**popt[0],2))) + " M$_\odot$ $\sigma$ = " + str("{0:.2f}".format(round(popt[1],2))), xy=(0.58,0.7), xycoords=('axes fraction'), color="brown", size=14)
	plt.annotate('Kroupa IMF', xy=(0.04,0.86), xycoords=('axes fraction'), color="black", size=16, rotation=35)
	plt.annotate('Chabrier IMF', xy=(0.04,0.66), xycoords=('axes fraction'), color="black", size=16, rotation=45)
	plt.legend(prop={'size':18})
	fig.savefig("Cepheus_CMF.pdf", bbox_inches='tight')
	plt.show()
	plt.close()

def coldense_vs_cores(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/', L1157_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1157/080615/cep1157_255_mu.fits', L1172_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1172/082315/cep1172_255_mu.fits', L1228_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1228/082315/cep1228_255_mu.fits', L1241_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1241/071415/cep1241_255_mu.fits', L1251_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1251/082315/cep1251_255_mu.fits'):
	L1157_radius, L1157_mass, L1157_alpha, L1157_type = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1172_radius, L1172_mass, L1172_alpha, L1172_type = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1228_radius, L1228_mass, L1228_alpha, L1228_type = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1241_radius, L1241_mass, L1241_alpha, L1241_type = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)
	L1251_radius, L1251_mass, L1251_alpha, L1251_type = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(4,6,16,17), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20')], unpack=True)

	wavelength = [L1157_coldense_image, L1172_coldense_image, L1228_coldense_image, L1241_coldense_image, L1251_coldense_image]

	region_mask = ["""polygon(20:42:59.746,+68:24:54.66,20:41:44.062,+68:22:37.19,20:40:18.798,+68:21:43.51,20:35:57.488,+67:59:58.31,20:34:41.481,+67:52:04.50,20:30:57.485,+67:29:08.83,20:31:57.431,+67:24:31.78,20:32:53.018,+67:20:14.92,20:34:47.128,+67:11:15.86,20:38:12.161,+66:53:23.99,20:38:31.292,+66:51:11.34,20:38:54.138,+66:48:58.70,20:39:47.051,+66:47:09.01,20:40:54.917,+66:48:38.77,20:41:32.726,+66:50:52.04,20:42:37.288,+66:54:55.14,20:46:33.655,+67:16:28.94,20:48:08.565,+67:24:47.13,20:49:48.036,+67:31:54.15,20:50:11.937,+67:33:18.24,20:51:25.071,+67:40:07.06)""", """polygon(21:06:03.933,+68:38:47.74,20:52:05.954,+68:26:58.96,20:56:20.756,+66:59:38.65,20:56:58.835,+66:59:25.69,20:57:19.127,+66:53:54.12,21:11:09.618,+67:04:37.18,21:08:30.942,+68:29:19.55,21:07:37.597,+68:32:13.38,21:07:09.514,+68:31:47.53,21:07:07.795,+68:34:29.47)""", """polygon(21:07:34.103,+76:49:45.04,21:08:25.824,+77:16:15.57,21:08:31.482,+77:57:09.73,21:08:57.651,+78:10:43.67,21:08:19.820,+78:15:10.26,21:01:13.894,+78:15:21.52,20:58:47.831,+78:15:10.47,20:56:27.754,+78:09:20.46,20:52:56.539,+78:09:35.24,20:51:00.082,+78:13:49.03,20:49:46.778,+78:14:39.32,20:45:52.884,+78:10:49.68,20:46:23.573,+77:40:13.41,20:47:29.671,+76:54:30.83,20:48:31.529,+76:51:32.20,20:49:27.504,+76:48:32.14)""", """polygon(22:11:15.122,+77:38:08.14,21:56:17.258,+77:42:30.14,21:55:49.751,+77:41:06.28,21:55:18.678,+77:42:48.60,21:44:10.790,+77:45:41.75,21:43:28.822,+77:45:33.79,21:43:55.952,+77:17:08.91,21:45:24.699,+77:16:37.02,21:45:28.601,+77:14:38.40,21:44:48.960,+77:13:42.93,21:43:55.955,+77:14:43.42,21:44:59.967,+75:56:51.42,21:46:43.846,+75:50:06.25,21:58:08.840,+75:45:15.54,22:10:45.599,+75:40:22.51,22:11:43.128,+75:40:10.36,22:14:43.175,+77:25:20.96)
""", """polygon(22:30:02.398,+75:43:27.20,22:17:51.515,+75:51:18.32,22:17:15.995,+75:44:06.40,22:16:10.988,+75:42:44.32,22:17:18.087,+75:29:36.11,22:14:24.915,+75:25:09.46,22:16:41.945,+75:00:25.94,22:18:17.939,+74:49:00.15,22:19:22.504,+74:48:10.96,22:19:45.244,+74:45:16.13,22:19:53.787,+74:35:05.60,22:21:11.718,+74:35:42.29,22:26:34.567,+74:41:47.59,22:30:14.152,+74:40:38.47,22:36:40.338,+74:39:50.34,22:40:07.563,+74:34:39.84,22:41:54.559,+74:28:16.04,22:43:04.560,+74:48:07.93,22:45:24.651,+74:52:28.66,22:44:28.759,+75:14:15.62,22:44:57.653,+75:20:28.98,22:43:12.009,+75:28:16.52,22:41:08.117,+75:47:35.07,22:39:24.250,+75:45:57.68,22:37:36.747,+75:49:07.72)"""]
	
	total_masses = []
	total_masses_Av_1 = []
	total_masses_Av_5 = []
	total_areas = []
	region_names = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251']
	Cepheus_combined_coldense = []
	for i in range(len(region_mask)):
		print '\n' + region_names[i] + ': \n'
		f = fits.open(wavelength[i])
		data = fits.getdata(wavelength[i])
		r = pyregion.parse(region_mask[i])
		mymask = r.get_mask(hdu=f[0])
		new_mask = numpy.where(mymask==0, 0, data)
	
		fig2_1157 = plt.figure()
		histogram, bins = numpy.histogram(new_mask, bins=numpy.logspace(numpy.log10(3.e20),numpy.log10(100.e21),51))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='blue', linestyle='None', marker='None')
		plt.plot(centers, histogram, color='blue', drawstyle='steps-mid')
		plt.semilogy()
		plt.semilogx()
		fig2_1157.savefig(region_names[i]+"_coldens_hist.png")
		Cepheus_combined_coldense.append(new_mask.flatten())
		fig2_1157 = plt.close()
	fig2_1157 = plt.figure()

	data = fits.getdata('aql_coldens_high250_edgeMask.fits')
	bins = numpy.logspace(numpy.log10(3.e20),numpy.log10(100.e21),51)
	histogram, bins = numpy.histogram(data, bins=bins, range=(bins.min(), bins.max()))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers, histogram, yerr = root_N_err, color='blue', linestyle='None', marker='None')
	plt.plot(centers, histogram, color='blue', drawstyle='steps-mid', label='Aquila', linewidth=2.0)
	#Aquila_pdf_1, Aquila_pdf_2 = numpy.loadtxt("Aquila_pdf.dat", unpack=True)
	#plt.errorbar(Aquila_pdf_1, Aquila_pdf_2, yerr = Aquila_pdf_2**0.5, color='blue', linestyle='None', marker='None')
	#plt.plot(Aquila_pdf_1, Aquila_pdf_2, color='blue', drawstyle='steps-mid', label='Aquila')

	Cepheus_combined_coldense = numpy.hstack(Cepheus_combined_coldense)
	bins = numpy.logspace(numpy.log10(3.e20),numpy.log10(100.e21),51)
	histogram, bins = numpy.histogram(Cepheus_combined_coldense, bins=bins, range=(bins.min(), bins.max()))
	root_N_err = histogram**0.5
	centers = (bins[:-1]+bins[1:])/2
	plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', marker='None')
	plt.plot(centers, histogram, color='green', drawstyle='steps-mid', label='Cepheus', linewidth=2.0)
	plt.xlim([3.e20, 100.e21])
	plt.ylim([10.,10.**7])
	plt.ylabel("Pixels per bin: $\Delta$N/$\Delta$logN$_{H_{2}}$", size=20)	
	plt.xlabel("Column Density, N$_{H_{2}}$ [cm$^{-2}$]", size=20)
	plt.semilogy()
	plt.semilogx()
	plt.legend(prop={'size':18})
	plt.tick_params(axis='both', labelsize=20)
	fig2_1157.savefig("Cep_coldens_hist.pdf", bbox_inches='tight')
	
	protostellar_indices_L1157 = numpy.where(L1157_type=='protostellar')
	protostellar_indices_L1172 = numpy.where(L1172_type=='protostellar')
	protostellar_indices_L1228 = numpy.where(L1228_type=='protostellar')
	protostellar_indices_L1241 = numpy.where(L1241_type=='protostellar')
	protostellar_indices_L1251 = numpy.where(L1251_type=='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')

	robust_indices_L1157 = numpy.where(L1157_alpha<=2.0)
	robust_indices_L1172 = numpy.where(L1172_alpha<=2.0)
	robust_indices_L1228 = numpy.where(L1228_alpha<=2.0)
	robust_indices_L1241 = numpy.where(L1241_alpha<=2.0)
	robust_indices_L1251 = numpy.where(L1251_alpha<=2.0)

	# Make a plot of mean column density versus number of cores
	data = fits.getdata(L1157_coldense_image)
	newmask=numpy.where(data>5.e21)
	mean_coldens = numpy.mean(data[newmask])
	std_coldens = numpy.std(data[newmask])
	print std_coldens
	fig = plt.figure()

	plt.errorbar(mean_coldens, len(prestellar_indices_L1157[0]), xerr=std_coldens, linestyle='None', color='blue', marker='.')
	plt.errorbar(mean_coldens, len(robust_indices_L1157[0]), xerr=std_coldens, linestyle='None', color='blue', marker='^')
	plt.errorbar(mean_coldens, len(protostellar_indices_L1157[0]), xerr=std_coldens, linestyle='None', color='blue', marker='*')
	
	data = fits.getdata(L1172_coldense_image)
	newmask=numpy.where(data>5.e21)
	mean_coldens = numpy.mean(data[newmask])
	std_coldens = numpy.std(data[newmask])
	print std_coldens
	
	plt.errorbar(mean_coldens, len(prestellar_indices_L1172[0]), xerr=std_coldens, linestyle='None', color='green', marker='.')
	plt.errorbar(mean_coldens, len(robust_indices_L1172[0]), xerr=std_coldens, linestyle='None', color='green', marker='^')
	plt.errorbar(mean_coldens, len(protostellar_indices_L1172[0]), xerr=std_coldens, linestyle='None', color='green', marker='*')

	data = fits.getdata(L1228_coldense_image)
	newmask=numpy.where(data>5.e21)
	mean_coldens = numpy.mean(data[newmask])
	std_coldens = numpy.std(data[newmask])
	print std_coldens

	plt.errorbar(mean_coldens, len(prestellar_indices_L1228[0]), xerr=std_coldens, linestyle='None', color='yellow', marker='.')
	plt.errorbar(mean_coldens, len(robust_indices_L1228[0]), xerr=std_coldens, linestyle='None', color='yellow', marker='^')
	plt.errorbar(mean_coldens, len(protostellar_indices_L1228[0]), xerr=std_coldens, linestyle='None', color='yellow', marker='*')

	data = fits.getdata(L1241_coldense_image)
	newmask=numpy.where(data>5.e21)
	mean_coldens = numpy.mean(data[newmask])
	std_coldens = numpy.std(data[newmask])
	print std_coldens

	plt.errorbar(mean_coldens, len(prestellar_indices_L1241[0]), xerr=std_coldens, linestyle='None', color='orange', marker='.')
	plt.errorbar(mean_coldens, len(robust_indices_L1241[0]), xerr=std_coldens, linestyle='None', color='orange', marker='^')
	plt.errorbar(mean_coldens, len(protostellar_indices_L1241[0]), xerr=std_coldens, linestyle='None', color='orange', marker='*')

	data = fits.getdata(L1251_coldense_image)
	newmask=numpy.where(data>5.e21)
	mean_coldens = numpy.mean(data[newmask])
	std_coldens = numpy.std(data[newmask])
	print std_coldens

	plt.errorbar(mean_coldens, len(prestellar_indices_L1251[0]), xerr=std_coldens, linestyle='None', color='red', marker='.')
	plt.errorbar(mean_coldens, len(robust_indices_L1251[0]), xerr=std_coldens, linestyle='None', color='red', marker='^')
	plt.errorbar(mean_coldens, len(protostellar_indices_L1251[0]), xerr=std_coldens, linestyle='None', color='red', marker='*')

	fig.savefig("Cepheus_mean_coldens_vs_cores.png")

	


def beam_avg_vs_core_number_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_radius, L1157_mass, L1157_alpha, L1157_type, L1157_NH2_peak, L1157_voldens = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(4,6,16,17,10,13), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float)], unpack=True)
	L1172_radius, L1172_mass, L1172_alpha, L1172_type, L1172_NH2_peak, L1172_voldens = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(4,6,16,17,10,13), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float)], unpack=True)
	L1228_radius, L1228_mass, L1228_alpha, L1228_type, L1228_NH2_peak, L1228_voldens = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(4,6,16,17,10,13), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float)], unpack=True)
	L1241_radius, L1241_mass, L1241_alpha, L1241_type, L1241_NH2_peak, L1241_voldens = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(4,6,16,17,10,13), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float)], unpack=True)
	L1251_radius, L1251_mass, L1251_alpha, L1251_type, L1251_NH2_peak, L1251_voldens = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(4,6,16,17,10,13), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float)], unpack=True)

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')

	robust_indices_L1157 = numpy.where(L1157_alpha<=2.0)
	robust_indices_L1172 = numpy.where(L1172_alpha<=2.0)
	robust_indices_L1228 = numpy.where(L1228_alpha<=2.0)
	robust_indices_L1241 = numpy.where(L1241_alpha<=2.0)
	robust_indices_L1251 = numpy.where(L1251_alpha<=2.0)

	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251', 'Cepheus']
	Cepheus_starless = numpy.hstack((L1157_NH2_peak[starless_indices_L1157], L1172_NH2_peak[starless_indices_L1172], L1228_NH2_peak[starless_indices_L1228], L1241_NH2_peak[starless_indices_L1241], L1251_NH2_peak[starless_indices_L1251]))	
	starless_cores = [L1157_NH2_peak[starless_indices_L1157], L1172_NH2_peak[starless_indices_L1172], L1228_NH2_peak[starless_indices_L1228], L1241_NH2_peak[starless_indices_L1241], L1251_NH2_peak[starless_indices_L1251], Cepheus_starless]
	Cepheus_prestellar = numpy.hstack((L1157_NH2_peak[prestellar_indices_L1157], L1172_NH2_peak[prestellar_indices_L1172], L1228_NH2_peak[prestellar_indices_L1228], L1241_NH2_peak[prestellar_indices_L1241], L1251_NH2_peak[prestellar_indices_L1251]))	
	prestellar_cores = [L1157_NH2_peak[prestellar_indices_L1157], L1172_NH2_peak[prestellar_indices_L1172], L1228_NH2_peak[prestellar_indices_L1228], L1241_NH2_peak[prestellar_indices_L1241], L1251_NH2_peak[prestellar_indices_L1251], Cepheus_prestellar]

	Cepheus_starless_vol = numpy.hstack((L1157_voldens[starless_indices_L1157], L1172_voldens[starless_indices_L1172], L1228_voldens[starless_indices_L1228], L1241_voldens[starless_indices_L1241], L1251_voldens[starless_indices_L1251]))	
	starless_cores_vol = [L1157_voldens[starless_indices_L1157], L1172_voldens[starless_indices_L1172], L1228_voldens[starless_indices_L1228], L1241_voldens[starless_indices_L1241], L1251_voldens[starless_indices_L1251], Cepheus_starless_vol]
	Cepheus_prestellar_vol = numpy.hstack((L1157_voldens[prestellar_indices_L1157], L1172_voldens[prestellar_indices_L1172], L1228_voldens[prestellar_indices_L1228], L1241_voldens[prestellar_indices_L1241], L1251_voldens[prestellar_indices_L1251]))	
	prestellar_cores_vol = [L1157_voldens[prestellar_indices_L1157], L1172_voldens[prestellar_indices_L1172], L1228_voldens[prestellar_indices_L1228], L1241_voldens[prestellar_indices_L1241], L1251_voldens[prestellar_indices_L1251], Cepheus_prestellar_vol]
	for i in range(len(starless_cores)):
		fig_region=plt.figure()
		histogram, bins = numpy.histogram(starless_cores[i], bins=numpy.logspace(numpy.log10(3.e20),numpy.log10(100.e21),8))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='black', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='black', linestyle='--')
		histogram, bins = numpy.histogram(prestellar_cores[i], bins=numpy.logspace(numpy.log10(3.e20),numpy.log10(100.e21),8))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='black', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='black', drawstyle='--')
		plt.semilogy()
		plt.semilogx()
		plt.ylabel("Number of cores per logarithmic bin")
		plt.xlabel("Beam-averaged Peak Column Density (500 $\mu$m), N$_{H2}$[cm$^{-2}$]")
		fig_region.savefig(regions[i] + "_coldense_vs_core_number.png")

		fig_region_vol=plt.figure()
		histogram, bins = numpy.histogram(starless_cores_vol[i], bins=numpy.logspace(numpy.log10(1.e3),numpy.log10(1.e6),8))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='black', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='black', linestyle='--')
		histogram, bins = numpy.histogram(prestellar_cores_vol[i], bins=numpy.logspace(numpy.log10(1.e3),numpy.log10(1.e6),8))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='black', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='black', drawstyle='--')
		plt.semilogy()
		plt.semilogx()
		plt.ylabel("Number of cores per logarithmic bin")
		plt.xlabel("Beam-averaged Peak Volume Density (500 $\mu$m), n$_{H2}$[cm$^{-3}$]")
		fig_region_vol.savefig(regions[i] + "_voldense_vs_core_number.png")

def voldens_vs_mass_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_radius, L1157_mass, L1157_alpha, L1157_type, L1157_NH2_peak, L1157_voldens, L1157_voldens_deconv = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(4,6,16,17,10,13,14), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float), ('voldens2',float)], unpack=True)
	L1172_radius, L1172_mass, L1172_alpha, L1172_type, L1172_NH2_peak, L1172_voldens, L1172_voldens_deconv = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(4,6,16,17,10,13,14), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float), ('voldens2',float)], unpack=True)
	L1228_radius, L1228_mass, L1228_alpha, L1228_type, L1228_NH2_peak, L1228_voldens, L1228_voldens_deconv = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(4,6,16,17,10,13,14), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float), ('voldens2',float)], unpack=True)
	L1241_radius, L1241_mass, L1241_alpha, L1241_type, L1241_NH2_peak, L1241_voldens, L1241_voldens_deconv = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(4,6,16,17,10,13,14), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float), ('voldens2',float)], unpack=True)
	L1251_radius, L1251_mass, L1251_alpha, L1251_type, L1251_NH2_peak, L1251_voldens, L1251_voldens_deconv = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(4,6,16,17,10,13,14), dtype=[('Radius',float),('Mass',float),('alpha',float),('type','S20'),('NH2_peak',float),('voldens',float), ('voldens2',float)], unpack=True)

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')

	robust_indices_L1157 = numpy.where(L1157_alpha<=2.0)
	robust_indices_L1172 = numpy.where(L1172_alpha<=2.0)
	robust_indices_L1228 = numpy.where(L1228_alpha<=2.0)
	robust_indices_L1241 = numpy.where(L1241_alpha<=2.0)
	robust_indices_L1251 = numpy.where(L1251_alpha<=2.0)

	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251', 'Cepheus']
	Cepheus_prestellar_vol = numpy.hstack((L1157_voldens_deconv[prestellar_indices_L1157], L1172_voldens_deconv[prestellar_indices_L1172], L1228_voldens_deconv[prestellar_indices_L1228], L1241_voldens_deconv[prestellar_indices_L1241], L1251_voldens_deconv[prestellar_indices_L1251]))	
	prestellar_cores_vol = [L1157_voldens_deconv[prestellar_indices_L1157], L1172_voldens_deconv[prestellar_indices_L1172], L1228_voldens_deconv[prestellar_indices_L1228], L1241_voldens_deconv[prestellar_indices_L1241], L1251_voldens_deconv[prestellar_indices_L1251], Cepheus_prestellar_vol]
	Cepheus_prestellar_mass = numpy.hstack((L1157_mass[prestellar_indices_L1157], L1172_mass[prestellar_indices_L1172], L1228_mass[prestellar_indices_L1228], L1241_mass[prestellar_indices_L1241], L1251_mass[prestellar_indices_L1251]))	
	prestellar_cores_mass = [L1157_mass[prestellar_indices_L1157], L1172_mass[prestellar_indices_L1172], L1228_mass[prestellar_indices_L1228], L1241_mass[prestellar_indices_L1241], L1251_mass[prestellar_indices_L1251], Cepheus_prestellar_mass]
	for i in range(len(regions)):
		if i>4:
			fig = plt.figure()
			plt.scatter(prestellar_cores_mass[i], prestellar_cores_vol[i], alpha=0.50)
			plt.semilogy()
			plt.semilogx()
			bins = numpy.logspace(numpy.log10(0.1),numpy.log10(10.),8)
			vols = prestellar_cores_vol[i]
			infs = numpy.where(vols==inf)
			vols = numpy.delete(vols, infs)
			masses = prestellar_cores_mass[i]
			masses = numpy.delete(masses, infs)
			digitized = numpy.digitize(masses, bins)
			bin_medians = [numpy.median(vols[digitized==i]) for i in range(1,len(bins))]
			bin_lengths = [len(vols[digitized==i]) for i in range(1,len(bins))]
			bin_medians_upper_err = [numpy.percentile(vols[digitized==i], 75) for i in range(1,len(bins))]
			bin_medians_lower_err = [numpy.percentile(vols[digitized==i], 25) for i in range(1,len(bins))]
			centers = (bins[:-1]+bins[1:])/2
			plt.errorbar(centers, numpy.array(bin_medians),yerr=[bin_medians_lower_err,bin_medians_upper_err], color='red', marker='^')
			plt.ylabel("Deconvolved Average Volume Density, n$_{H2}$ [cm$^{-3}$]")
			plt.xlabel("Derived Core Mass, M [M$_\odot$]")
			#plt.show()
			fig.savefig("Cepheus_voldens_vs_mass.png")

def mass_within_contour(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/', L1157_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1157/080615/cep1157_255_mu.fits', L1172_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1172/082315/cep1172_255_mu.fits', L1228_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1228/082315/cep1228_255_mu.fits', L1241_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1241/071415/cep1241_255_mu.fits', L1251_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1251/082315/cep1251_255_mu.fits'):

	distances = [325., 288., 200., 300., 300.]	

	wavelength = [L1157_coldense_image, L1172_coldense_image, L1228_coldense_image, L1241_coldense_image, L1251_coldense_image]

	region_mask = ["""polygon(20:42:59.746,+68:24:54.66,20:41:44.062,+68:22:37.19,20:40:18.798,+68:21:43.51,20:35:57.488,+67:59:58.31,20:34:41.481,+67:52:04.50,20:30:57.485,+67:29:08.83,20:31:57.431,+67:24:31.78,20:32:53.018,+67:20:14.92,20:34:47.128,+67:11:15.86,20:38:12.161,+66:53:23.99,20:38:31.292,+66:51:11.34,20:38:54.138,+66:48:58.70,20:39:47.051,+66:47:09.01,20:40:54.917,+66:48:38.77,20:41:32.726,+66:50:52.04,20:42:37.288,+66:54:55.14,20:46:33.655,+67:16:28.94,20:48:08.565,+67:24:47.13,20:49:48.036,+67:31:54.15,20:50:11.937,+67:33:18.24,20:51:25.071,+67:40:07.06)""", """polygon(21:06:03.933,+68:38:47.74,20:52:05.954,+68:26:58.96,20:56:20.756,+66:59:38.65,20:56:58.835,+66:59:25.69,20:57:19.127,+66:53:54.12,21:11:09.618,+67:04:37.18,21:08:30.942,+68:29:19.55,21:07:37.597,+68:32:13.38,21:07:09.514,+68:31:47.53,21:07:07.795,+68:34:29.47)""", """polygon(21:07:34.103,+76:49:45.04,21:08:25.824,+77:16:15.57,21:08:31.482,+77:57:09.73,21:08:57.651,+78:10:43.67,21:08:19.820,+78:15:10.26,21:01:13.894,+78:15:21.52,20:58:47.831,+78:15:10.47,20:56:27.754,+78:09:20.46,20:52:56.539,+78:09:35.24,20:51:00.082,+78:13:49.03,20:49:46.778,+78:14:39.32,20:45:52.884,+78:10:49.68,20:46:23.573,+77:40:13.41,20:47:29.671,+76:54:30.83,20:48:31.529,+76:51:32.20,20:49:27.504,+76:48:32.14)""", """polygon(22:11:15.122,+77:38:08.14,21:56:17.258,+77:42:30.14,21:55:49.751,+77:41:06.28,21:55:18.678,+77:42:48.60,21:44:10.790,+77:45:41.75,21:43:28.822,+77:45:33.79,21:43:55.952,+77:17:08.91,21:45:24.699,+77:16:37.02,21:45:28.601,+77:14:38.40,21:44:48.960,+77:13:42.93,21:43:55.955,+77:14:43.42,21:44:59.967,+75:56:51.42,21:46:43.846,+75:50:06.25,21:58:08.840,+75:45:15.54,22:10:45.599,+75:40:22.51,22:11:43.128,+75:40:10.36,22:14:43.175,+77:25:20.96)
""", """polygon(22:30:02.398,+75:43:27.20,22:17:51.515,+75:51:18.32,22:17:15.995,+75:44:06.40,22:16:10.988,+75:42:44.32,22:17:18.087,+75:29:36.11,22:14:24.915,+75:25:09.46,22:16:41.945,+75:00:25.94,22:18:17.939,+74:49:00.15,22:19:22.504,+74:48:10.96,22:19:45.244,+74:45:16.13,22:19:53.787,+74:35:05.60,22:21:11.718,+74:35:42.29,22:26:34.567,+74:41:47.59,22:30:14.152,+74:40:38.47,22:36:40.338,+74:39:50.34,22:40:07.563,+74:34:39.84,22:41:54.559,+74:28:16.04,22:43:04.560,+74:48:07.93,22:45:24.651,+74:52:28.66,22:44:28.759,+75:14:15.62,22:44:57.653,+75:20:28.98,22:43:12.009,+75:28:16.52,22:41:08.117,+75:47:35.07,22:39:24.250,+75:45:57.68,22:37:36.747,+75:49:07.72)"""]
	
	mu = 2.8 # mean molecular weight 
	mass_H = 1.67372e-24 # (grams) mass of neutral Hydrogen atom
	solar_mass = 1.989e33 # (grams)
	mass_H_solar_masses = mass_H / solar_mass
	
	total_masses = []
	total_masses_Av_1 = []
	total_masses_Av_5 = []
	total_areas = []
	region_names = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251']
	for i in range(len(region_mask)):
		print '\n' + region_names[i] + ': \n'
		f = fits.open(wavelength[i])
		data = fits.getdata(wavelength[i])
		r = pyregion.parse(region_mask[i])
		mymask = r.get_mask(hdu=f[0])
		new_mask = numpy.where(mymask==0, 0, data)
	
		distance=distances[i]
		Av_contour = numpy.where(new_mask>(0))
		Av_contour_coldense = new_mask[Av_contour]
		total_H2 = numpy.sum(Av_contour_coldense.flatten())
		num_pixels = len(Av_contour_coldense.flatten())
		area_per_pixel = (2*(distance*numpy.tan(numpy.radians(0.5*(3.0/3600.)))))**2.
		total_area = area_per_pixel*num_pixels
		total_particles = total_H2*(1./(3.24078e-19)**2.)*area_per_pixel
		total_mass = mu*mass_H_solar_masses*total_particles
		print "Total Mass (M_sun): " + str(round(total_mass, 1))
		print "Total Area (pc^2): " + str(round(total_area, 1))
		total_masses.append(total_mass)
		total_areas.append(total_area)

		Av_contour = numpy.where(new_mask>(1.*0.94e21))
		Av_contour_coldense = new_mask[Av_contour]
		total_H2 = numpy.sum(Av_contour_coldense.flatten())
		num_pixels = len(Av_contour_coldense.flatten())
		area_per_pixel = (2*(distance*numpy.tan(numpy.radians(0.5*(3.0/3600.)))))**2.
		total_particles = total_H2*(1./(3.24078e-19)**2.)*area_per_pixel
		total_mass = mu*mass_H_solar_masses*total_particles
		print "Mass in Av > 1: " + str(round(total_mass, 1))
		total_masses_Av_1.append(total_mass)

		Av_contour = numpy.where(new_mask>(5.*0.94e21))
		Av_contour_coldense = new_mask[Av_contour]
		total_H2 = numpy.sum(Av_contour_coldense.flatten())
		num_pixels = len(Av_contour_coldense.flatten())
		area_per_pixel = (2*(distance*numpy.tan(numpy.radians(0.5*(3.0/3600.)))))**2.
		total_particles = total_H2*(1./(3.24078e-19)**2.)*area_per_pixel
		total_mass = mu*mass_H_solar_masses*total_particles
		print "Mass in Av > 5: " + str(round(total_mass, 1))
		total_masses_Av_5.append(total_mass)

	print '\n' + 'Cepheus Total' + ': \n'
	print "Total Mass (M_sun): " + str(round(numpy.sum(total_masses),1))
	print "Total Area (pc^2): " + str(round(numpy.sum(total_areas),1))
	print "Mass in Av > 1: " + str(round(numpy.sum(total_masses_Av_1), 1))
	print "Mass in Av > 5: " + str(round(numpy.sum(total_masses_Av_5), 1))
	
	bar_width = 0.35
	opacity = 0.4
	figure3 = plt.figure()
	plt.bar(numpy.arange(len(total_masses)), numpy.array(total_masses)/numpy.array(total_areas), bar_width, color='blue', alpha=opacity, label='Total')
	plt.bar(numpy.arange(len(total_masses_Av_1)), numpy.array(total_masses_Av_1)/numpy.array(total_areas), bar_width, color='red', alpha=opacity, label='Av>1')
	plt.bar(numpy.arange(len(total_masses_Av_5)), numpy.array(total_masses_Av_5)/numpy.array(total_areas), bar_width, color='green', alpha=opacity, label='Av>5')
	plt.xticks(numpy.arange(len(total_masses))+bar_width*0.5, region_names)
	plt.ylabel("Surface Mass Density (M$_\odot$ pc$^{-2}$)")
	plt.xlabel("Region")
	#plt.legend()
	plt.show()
	figure3.savefig("mass_per_area_per_region.png")

	figure3 = plt.figure()
	plt.bar(numpy.arange(len(total_masses)), numpy.array(total_masses), bar_width, color='blue', alpha=opacity, label='Total')
	plt.bar(numpy.arange(len(total_masses_Av_1)), numpy.array(total_masses_Av_1), bar_width, color='red', alpha=opacity, label='Av>1')
	plt.bar(numpy.arange(len(total_masses_Av_5)), numpy.array(total_masses_Av_5), bar_width, color='green', alpha=opacity, label='Av>5')
	plt.xticks(numpy.arange(len(total_masses))+bar_width*0.5, region_names)
	plt.ylabel("Total Mass (M$_\odot$)")
	plt.xlabel("Region")
	plt.legend()
	plt.show()
	figure3.savefig("mass_per_region.png")

def bg_coldense_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_bg, L1157_type = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog1.dat', usecols=(57,63), dtype=[('bg',float),('type','S20')], unpack=True)
	L1172_bg, L1172_type = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog1.dat', usecols=(57,63), dtype=[('bg',float),('type','S20')], unpack=True)
	L1228_bg, L1228_type = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog1.dat', usecols=(57,63), dtype=[('bg',float),('type','S20')], unpack=True)
	L1241_bg, L1241_type = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog1.dat', usecols=(57,63), dtype=[('bg',float),('type','S20')], unpack=True)
	L1251_bg, L1251_type = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog1.dat', usecols=(57,63), dtype=[('bg',float),('type','S20')], unpack=True)
	Aquila_bg, Aquila_type = numpy.loadtxt('tablea1.dat', usecols=(61,67), dtype=[('bg',float),('type','S20')], unpack=True)

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')
	prestellar_indices_Aquila = numpy.where(Aquila_type=='prestellar')

	prestellar_indices_Aquila_high_CO = numpy.where(prestellar_indices_Aquila[0]>711)
	prestellar_indices_Aquila_new = numpy.delete(prestellar_indices_Aquila, prestellar_indices_Aquila_high_CO)
	prestellar_indices_Aquila_high_CO_2 = numpy.where(prestellar_indices_Aquila_new==708)
	prestellar_indices_Aquila_new_2 = numpy.delete(prestellar_indices_Aquila_new, prestellar_indices_Aquila_high_CO_2)

	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251', 'Cepheus']
	Cepheus_prestellar_bg = numpy.hstack((L1157_bg[prestellar_indices_L1157], L1172_bg[prestellar_indices_L1172], L1228_bg[prestellar_indices_L1228], L1241_bg[prestellar_indices_L1241], L1251_bg[prestellar_indices_L1251]))
	print len(Cepheus_prestellar_bg)
	print Cepheus_prestellar_bg	
	prestellar_cores_bg = [L1157_bg[prestellar_indices_L1157], L1172_bg[prestellar_indices_L1172], L1228_bg[prestellar_indices_L1228], L1241_bg[prestellar_indices_L1241], L1251_bg[prestellar_indices_L1251], Cepheus_prestellar_bg]
	counter=0
	for i in prestellar_cores_bg:
		fig = plt.figure()
		histogram, bins = numpy.histogram(Aquila_bg[prestellar_indices_Aquila_new_2], bins=numpy.linspace(0., 30., 20))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='blue', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='blue', drawstyle='steps-mid', label='Aquila', linewidth=2.0)

		histogram, bins = numpy.histogram(i/10., bins=numpy.linspace(0., 30., 20))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', markerfacecolor='none', marker='^')
		plt.plot(centers, histogram, color='green', drawstyle='steps-mid', label='Cepheus', linewidth=2.0)

		
		Av_range = numpy.arange(0.,100.)
		Av_7 = numpy.zeros(len(Av_range))+(7.*0.94)
		plt.plot(Av_7, Av_range, color='black', linestyle='--')		

		plt.ylabel("Cores per bin: $\Delta$N/$\Delta$N$_{H_2}$", size=20)
		plt.xlabel("Background Column Density, N$_{H_2}$ [10$^{21}$ cm$^{-2}$]", size=20)
		plt.legend(prop={'size':18})
		plt.tick_params(axis='both', labelsize=20)
		fig.savefig(regions[counter] + '_cores_vs_bg_coldens.pdf', bbox_inches='tight')
		if counter==5:
			plt.show()
		counter+=1
		plt.close()
		print float(len(numpy.where(i/10.>7.*0.94)[0]))/float(len(i))

	print float(len(numpy.where(Aquila_bg[prestellar_indices_Aquila_new_2]>7.*0.94)[0]))/float(len(Aquila_bg[prestellar_indices_Aquila_new_2]))

def cross_match_filaments(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/', filament_images1=["/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1157/120115_flat/255/+interface/L1157.255.obs.filaments.0072as.fits", "/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1172/120115_flat/255/+interface/L1172.255.obs.filaments.0072as.fits", "/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1228/120115_flat/255/+interface/L1228.255.obs.filaments.0072as.fits", "/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1241/120115_flat/255/+interface/L1241.255.obs.filaments.0072as.fits", "/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1251/120115_flat/255/+interface/L1251.255.obs.filaments.0072as.fits"]):
	
	L1157_bg, L1157_type, L1157_RA, L1157_Dec = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog1.dat', usecols=(57,63,2,3), dtype=[('bg',float),('type','S20'), ('type1','S20'), ('type2','S20')], unpack=True)
	L1172_bg, L1172_type, L1172_RA, L1172_Dec = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog1.dat', usecols=(57,63,2,3), dtype=[('bg',float),('type','S20'), ('type1','S20'), ('type2','S20')], unpack=True)
	L1228_bg, L1228_type, L1228_RA, L1228_Dec = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog1.dat', usecols=(57,63,2,3), dtype=[('bg',float),('type','S20'), ('type1','S20'), ('type2','S20')], unpack=True)
	L1241_bg, L1241_type, L1241_RA, L1241_Dec = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog1.dat', usecols=(57,63,2,3), dtype=[('bg',float),('type','S20'), ('type1','S20'), ('type2','S20')], unpack=True)
	L1251_bg, L1251_type, L1251_RA, L1251_Dec = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog1.dat', usecols=(57,63,2,3), dtype=[('bg',float),('type','S20'), ('type1','S20'), ('type2','S20')], unpack=True)

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')


	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251']

	RA_centres = ['20h40m49.0s','21h01m34.0s', '20h58m01.0s', '21h58m05.0s', '22h30m36.0s']
	Dec_centres = ['+67d33m55.0s','+67d51m12.0s','+77d32m21.0s','+76d40m22.0s','+75d13m22.0s']
	distance = [325., 288., 200., 300., 300.]
	radius = numpy.zeros(len(distance))+0.9

	
	for i in range(len(regions)):
		fig_filaments=plt.figure()
		l1157 = aplpy.FITSFigure(filament_images1[i], figure=fig_filaments)
		#l1157.show_contour('/Users/jkeown/Documents/Research/cepheus/cep1251_filt+conv/colMapbetaFixed_HerOnly.fits', colors='white')
		l1157.show_regions('/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/' + regions[i] + '/' + regions[i] + '_core_SED/255_prestellar_candidates.reg')
		l1157.show_regions('/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/' + regions[i] + '/' + regions[i] + '_core_SED/255_prestellar_robust.reg')
		#l1157.show_regions('/mnt/scratch-lustre/jkeown/DS9_regions/L' + name + '/YSO_sources.reg')

		c1 = astropy.coordinates.Angle(RA_centres[i])
		c2 = astropy.coordinates.Angle(Dec_centres[i])

		l1157.recenter(c1.degrees,c2.degrees,radius[i])

		l1157.show_colorscale(cmap='Greys', vmin=0., vmax=.001)

		#l1157.add_colorbar()
		#l1157.colorbar.set_axis_label_text('H$_{2}$ Column Density (10$^{20}$ cm$^{-2}$)')
	
		scale = str('{:.2f}'.format(2*(distance[i]*numpy.tan(numpy.radians(0.5*(10.0/60.))))))

		l1157.add_scalebar((10./60.), path_effects=[Patheffects.withStroke(linewidth=3, foreground='white')])
		l1157.scalebar.show(10./60.)  # length in degrees (ten arcminutes)
		l1157.scalebar.set_corner('top left')
		l1157.scalebar.set_label('10' + "'" + ' = ' + scale + ' pc')

		l1157.add_label(0.8,0.93,'col. density',relative=True, path_effects=[Patheffects.withStroke(linewidth=3, foreground='white')])

		l1157.tick_labels.set_xformat('hh:mm:ss')
		l1157.tick_labels.set_yformat('dd:mm:ss')
		fig_filaments.savefig(regions[i] + "_filaments_prestellar_cores.pdf")
		plt.close()
		l1157.close()

	Cepheus_prestellar_RA = numpy.hstack((L1157_RA[prestellar_indices_L1157], L1172_RA[prestellar_indices_L1172], L1228_RA[prestellar_indices_L1228], L1241_RA[prestellar_indices_L1241], L1251_RA[prestellar_indices_L1251]))	
	prestellar_cores_RA = [L1157_RA[prestellar_indices_L1157], L1172_RA[prestellar_indices_L1172], L1228_RA[prestellar_indices_L1228], L1241_RA[prestellar_indices_L1241], L1251_RA[prestellar_indices_L1251], Cepheus_prestellar_RA]
	Cepheus_prestellar_Dec = numpy.hstack((L1157_Dec[prestellar_indices_L1157], L1172_Dec[prestellar_indices_L1172], L1228_Dec[prestellar_indices_L1228], L1241_Dec[prestellar_indices_L1241], L1251_Dec[prestellar_indices_L1251]))	
	prestellar_cores_Dec = [L1157_Dec[prestellar_indices_L1157], L1172_Dec[prestellar_indices_L1172], L1228_Dec[prestellar_indices_L1228], L1241_Dec[prestellar_indices_L1241], L1251_Dec[prestellar_indices_L1251], Cepheus_prestellar_Dec]

	Cepheus_starless_RA = numpy.hstack((L1157_RA[starless_indices_L1157], L1172_RA[starless_indices_L1172], L1228_RA[starless_indices_L1228], L1241_RA[starless_indices_L1241], L1251_RA[starless_indices_L1251]))	
	starless_cores_RA = [L1157_RA[starless_indices_L1157], L1172_RA[starless_indices_L1172], L1228_RA[starless_indices_L1228], L1241_RA[starless_indices_L1241], L1251_RA[starless_indices_L1251], Cepheus_starless_RA]
	Cepheus_starless_Dec = numpy.hstack((L1157_Dec[starless_indices_L1157], L1172_Dec[starless_indices_L1172], L1228_Dec[starless_indices_L1228], L1241_Dec[starless_indices_L1241], L1251_Dec[starless_indices_L1251]))	
	starless_cores_Dec = [L1157_Dec[starless_indices_L1157], L1172_Dec[starless_indices_L1172], L1228_Dec[starless_indices_L1228], L1241_Dec[starless_indices_L1241], L1251_Dec[starless_indices_L1251], Cepheus_starless_Dec]

	def func(RA=starless_cores_RA, Dec=starless_cores_Dec, filament_images=filament_images1):
		total_matched = 0
		starless_percentages=[]
		for index in range(len(regions)):
			RA_values = []
			Dec_values = []
			for i,j in zip(RA[index], Dec[index]):
		
				c1 = astropy.coordinates.Angle(i, u.h)
				c2 = astropy.coordinates.Angle(j, u.degree)
				RA_values.append(c1.degrees)
				Dec_values.append(c2.degrees)
	
			L1157_positions = numpy.column_stack((RA_values, Dec_values))

			f=fits.getdata(filament_images[index])
			w = wcs.WCS(filament_images[index])
			pos_pix = w.wcs_world2pix(L1157_positions, 1)
	
			mask_shape = numpy.shape(f)
			matched_cores=[]
			counter=0
			for i in pos_pix:
				ypos = int(round(i[1],0))-1
				xpos = int(round(i[0],0))-1
				if ypos<=mask_shape[0] and xpos<=mask_shape[1]:
					if f[ypos][xpos]!=0:
						matched_cores.append(counter)
				counter+=1
			total_matched = total_matched+len(matched_cores)
			percentage = float(len(matched_cores)) / float(len(RA[index]))
			#print float(len(matched_cores)) / float(len(RA[index]))
			starless_percentages.append(percentage)
		#print float(total_matched) / len(RA[index+1])
		starless_percentages.append(float(total_matched) / len(RA[index+1]))
		return starless_percentages

	bar_width = 0.35
	opacity1 = 0.6
	opacity2 = 0.6		
	figure3 = plt.figure()
	
	starless_percentages = func()
	starless_percentages.append(0.75)
	plt.bar(numpy.arange(len(starless_percentages)), starless_percentages, bar_width, color='green', alpha=opacity1, label='starless')

	starless_percentages = func(RA=prestellar_cores_RA, Dec=prestellar_cores_Dec)
	starless_percentages.append(0.85)
	plt.bar(numpy.arange(len(starless_percentages))+bar_width, starless_percentages, bar_width, color='blue', alpha=opacity2, label='prestellar')
	
	regions.append("Cepheus")
	regions.append("Aquila")
	plt.xticks(numpy.arange(len(starless_percentages))+bar_width, regions)
	plt.ylabel("Core Fraction on Filaments")
	plt.xlabel("Region")
	plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102), loc=3, ncol=2, mode='expand', borderaxespad=0.)
	plt.show()
	figure3.savefig("cores_on_filaments.png")

def core_temp_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_temp, L1157_type, L1157_type2 = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(8,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1172_temp, L1172_type, L1172_type2 = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(8,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1228_temp, L1228_type, L1228_type2 = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(8,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1241_temp, L1241_type, L1241_type2 = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(8,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1251_temp, L1251_type, L1251_type2 = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(8,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	Aquila_temp, Aquila_type = numpy.loadtxt('tablea2.dat', usecols=(12,21), dtype=[('bg',float),('type','S20')], unpack=True)
	f = open('tablea2.dat', "r")
	lines = numpy.array(f.readlines())
	f.close()

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')
	starless_indices_Aquila = numpy.where(Aquila_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')
	prestellar_indices_Aquila = numpy.where(Aquila_type=='prestellar')

	count = 0
	indices = []
	for line in lines[starless_indices_Aquila]:
		if "SED" in line or "high-V_LSR" in line:
			indices.append(count)
		count+=1

	Aquila = numpy.delete(Aquila_temp[starless_indices_Aquila], numpy.array(indices))
	print Aquila

	L1157=[]
	for i,j in zip(L1157_temp[starless_indices_L1157], L1157_type2[starless_indices_L1157]):
		if j!="no_SED_fit":
			L1157.append(i)

	L1172=[]
	for i,j in zip(L1172_temp[starless_indices_L1172], L1172_type2[starless_indices_L1172]):
		if j!="no_SED_fit":
			L1172.append(i)

	L1228=[]
	for i,j in zip(L1228_temp[starless_indices_L1228], L1228_type2[starless_indices_L1228]):
		if j!="no_SED_fit":
			L1228.append(i)

	L1241=[]
	for i,j in zip(L1241_temp[starless_indices_L1241], L1241_type2[starless_indices_L1241]):
		if j!="no_SED_fit":
			L1241.append(i)

	L1251=[]
	for i,j in zip(L1251_temp[starless_indices_L1251], L1251_type2[starless_indices_L1251]):
		if j!="no_SED_fit":
			L1251.append(i)

	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251', 'Cepheus']
	
	Cepheus_starless_temp = numpy.hstack((L1157, L1172, L1228, L1241, L1251))	
	starless_cores_temp = [L1157, L1172, L1228, L1241, L1251, Cepheus_starless_temp]
	counter=0
	for i in starless_cores_temp:
		fig = plt.figure()
		histogram, bins = numpy.histogram(Aquila, bins=numpy.linspace(7., 30., 30))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='blue', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='blue', drawstyle='steps-mid', label='Aquila', linewidth=2.0)

		histogram, bins = numpy.histogram(i, bins=numpy.linspace(7., 30., 30))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', markerfacecolor='none', marker='^')
		plt.plot(centers, histogram, color='green', drawstyle='steps-mid', label='Cepheus', linewidth=2.0)

		plt.ylabel("Cores per bin: $\Delta$N/$\Delta$T", size=20)
		plt.xlabel("Dust Temperature [K]", size=20)
		plt.legend(prop={'size':18})
		plt.tick_params(axis='both', labelsize=20)
		fig.savefig(regions[counter] + '_cores_vs_temp.pdf', bbox_inches='tight')
		if counter==5:
			plt.show()
		counter+=1
		plt.close()

def core_size_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/'):
	
	L1157_radius, L1157_type, L1157_type2 = numpy.loadtxt(catalog_directory+'/L1157/L1157_core_SED/'+'L1157_core_catalog2.dat', usecols=(4,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1172_radius, L1172_type, L1172_type2 = numpy.loadtxt(catalog_directory+'/L1172/L1172_core_SED/'+'L1172_core_catalog2.dat', usecols=(4,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1228_radius, L1228_type, L1228_type2 = numpy.loadtxt(catalog_directory+'/L1228/L1228_core_SED/'+'L1228_core_catalog2.dat', usecols=(4,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1241_radius, L1241_type, L1241_type2 = numpy.loadtxt(catalog_directory+'/L1241/L1241_core_SED/'+'L1241_core_catalog2.dat', usecols=(4,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	L1251_radius, L1251_type, L1251_type2 = numpy.loadtxt(catalog_directory+'/L1251/L1251_core_SED/'+'L1251_core_catalog2.dat', usecols=(4,17,18), dtype=[('bg',float),('type','S20'),('type2','S20')], unpack=True)
	Aquila_radius, Aquila_type = numpy.loadtxt('tablea2.dat', usecols=(8,21), dtype=[('bg',float),('type','S20')], unpack=True)
	f = open('tablea2.dat', "r")
	lines = numpy.array(f.readlines())
	f.close()

	starless_indices_L1157 = numpy.where(L1157_type!='protostellar')
	starless_indices_L1172 = numpy.where(L1172_type!='protostellar')
	starless_indices_L1228 = numpy.where(L1228_type!='protostellar')
	starless_indices_L1241 = numpy.where(L1241_type!='protostellar')
	starless_indices_L1251 = numpy.where(L1251_type!='protostellar')
	starless_indices_Aquila = numpy.where(Aquila_type!='protostellar')

	prestellar_indices_L1157 = numpy.where(L1157_type=='prestellar')
	prestellar_indices_L1172 = numpy.where(L1172_type=='prestellar')
	prestellar_indices_L1228 = numpy.where(L1228_type=='prestellar')
	prestellar_indices_L1241 = numpy.where(L1241_type=='prestellar')
	prestellar_indices_L1251 = numpy.where(L1251_type=='prestellar')
	prestellar_indices_Aquila = numpy.where(Aquila_type=='prestellar')

	count = 0
	indices = []
	for line in lines[starless_indices_Aquila]:
		if "SED" in line or "high-V_LSR" in line:
			indices.append(count)
		count+=1

	Aquila = numpy.delete(Aquila_radius[starless_indices_Aquila], numpy.array(indices))
	print Aquila

	L1157=[]
	for i,j in zip(L1157_radius[starless_indices_L1157], L1157_type2[starless_indices_L1157]):
		if j!="no_SED_fit":
			L1157.append(i)

	L1172=[]
	for i,j in zip(L1172_radius[starless_indices_L1172], L1172_type2[starless_indices_L1172]):
		if j!="no_SED_fit":
			L1172.append(i)

	L1228=[]
	for i,j in zip(L1228_radius[starless_indices_L1228], L1228_type2[starless_indices_L1228]):
		if j!="no_SED_fit":
			L1228.append(i)

	L1241=[]
	for i,j in zip(L1241_radius[starless_indices_L1241], L1241_type2[starless_indices_L1241]):
		if j!="no_SED_fit":
			L1241.append(i)

	L1251=[]
	for i,j in zip(L1251_radius[starless_indices_L1251], L1251_type2[starless_indices_L1251]):
		if j!="no_SED_fit":
			L1251.append(i)

	regions = ['L1157', 'L1172', 'L1228', 'L1241', 'L1251', 'Cepheus']
	
	Cepheus_starless_radius = numpy.hstack((L1157, L1172, L1228, L1241, L1251))	
	starless_cores_radius = [L1157, L1172, L1228, L1241, L1251, Cepheus_starless_radius]
	counter=0
	for i in starless_cores_radius:
		fig = plt.figure()
		histogram, bins = numpy.histogram(i, bins=numpy.linspace(0., 0.1, 30))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='green', linestyle='None', markerfacecolor='none', marker='^')
		plt.plot(centers, histogram, color='green', drawstyle='steps-mid', label='Cepheus')

		histogram, bins = numpy.histogram(Aquila, bins=numpy.linspace(0., 0.1, 30))
		root_N_err = histogram**0.5
		centers = (bins[:-1]+bins[1:])/2
		plt.errorbar(centers, histogram, yerr = root_N_err, color='blue', linestyle='None', marker='^')
		plt.plot(centers, histogram, color='blue', drawstyle='steps-mid', label='Aquila')

		plt.ylabel("Number of cores per bin: $\Delta$N/$\Delta$R")
		plt.xlabel("Core Radius [pc]")
		plt.legend()
		fig.savefig(regions[counter] + '_cores_vs_radius.png')
		if counter==5:
			plt.show()
		counter+=1
		plt.close()

def core_lifetime_plotter(catalog_directory = '/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/', L1157_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1157/080615/cep1157_255_mu.fits', L1172_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1172/082315/cep1172_255_mu.fits', L1228_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1228/082315/cep1228_255_mu.fits', L1241_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1241/071415/cep1241_255_mu.fits', L1251_coldense_image='/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1251/082315/cep1251_255_mu.fits', Dunham_YSOs_file = "Dunham_YSOs.dat"):
	Cloud,Name,Av,alpha,T_bol,L_bol,alphaPrime,TbolPrime,LbolPrime,likelyAGB,Dunham_RA,Dunham_DEC,Class = numpy.loadtxt(Dunham_YSOs_file, delimiter=',', unpack=True, dtype=[('Cloud','S30'),('Name','S40'), ('Av',float),('alpha',float), ('T_bol',float),('L_bol',float), ('alphaPrime',float),('TbolPrime',float), ('LbolPrime',float),('likelyAGB','S1'), ('Dunham_RA',float),('Dunham_DEC',float),('Class','S10')])
	indices = []
	count=0
	for i,j in zip(Cloud, Class):
		if i=="Cepheus" and j=="II":
			indices.append(count)
		count+=1
	#print len(indices)
	
	RA_values=[]
	Dec_values=[]
	for i,j in zip(Dunham_RA[numpy.array(indices)], Dunham_DEC[numpy.array(indices)]):
		c1 = astropy.coordinates.Angle(i, u.degree)
		c2 = astropy.coordinates.Angle(j, u.degree)
		RA_values.append(c1.degrees)
		Dec_values.append(c2.degrees)

	header = 'Region file format: DS9 version 4.1 \nglobal color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \nfk5'
	#new_line = ' #color=red width=2\n'

	ellipse = []
	for i in RA_values:
		ellipse.append('point')

	data = numpy.column_stack((ellipse, Dunham_RA[numpy.array(indices)], Dunham_DEC[numpy.array(indices)]))
	numpy.savetxt('Dunham_YSOs_Cepheus_class2.reg', data, delimiter=' ', fmt = '%s', header=header)
	
	L1157_positions = numpy.column_stack((Dunham_RA[numpy.array(indices)], Dunham_DEC[numpy.array(indices)]))

	images_array = [L1157_coldense_image, L1172_coldense_image, L1228_coldense_image, L1241_coldense_image, L1251_coldense_image]

	def func(L1157_positions=L1157_positions, images_array=images_array):
		total_matched = 0
		starless_percentages=[]
		for index in range(len(images_array)):
			f=fits.getdata(images_array[index])
			w = wcs.WCS(images_array[index])
			pos_pix = w.wcs_world2pix(L1157_positions, 1)
	
			mask_shape = numpy.shape(f)
			matched_cores=[]
			counter=0
			for i in pos_pix:
				ypos = int(round(i[1],0))-1
				xpos = int(round(i[0],0))-1
				if ypos<=mask_shape[0] and ypos>=0 and xpos<=mask_shape[1] and xpos>=0:
					if f[ypos][xpos]!=0 and f[ypos][xpos]!=numpy.nan:
						matched_cores.append(counter)
				counter+=1
			print len(matched_cores)
			total_matched = total_matched+len(matched_cores)
			starless_percentages.append(float(len(matched_cores)))
		return starless_percentages
	starless_percentages = func()
	print numpy.sum(starless_percentages)

#print (2*(260.*numpy.tan(numpy.radians(0.5*(11.**0.5)))))**2.

#core_lifetime_plotter()
#core_temp_plotter()
#core_size_plotter()	
#cross_match_filaments()
#bg_coldense_plotter()
#voldens_vs_mass_plotter()
#mass_within_contour()
#beam_avg_vs_core_number_plotter()
#Cepheus_CMF_plotter()
#coldense_vs_cores()
