
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.special import lambertw

from solcore.light_source import LightSource
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver
from solcore.structure import Junction

# Define fundamental physical constants
q = 1.60217662E-19  # electronic charge [C]
k = 1.38064852E-23/q   # Boltzmann constant [eV/K]
h = 6.62607004E-34/q  # Planck constant expressed in [eV.s]
c = 299792458  # Speed of light [m.s^-1]

# Load the AM1.5G solar spectrum
wl = np.linspace(300, 4000, 4000) * 1e-9    #wl contains the x-ordinate in wavelength
am15g = LightSource(source_type='standard', x=wl, version='AM1.5g')

#############################################
# Step 1 : SolCore IV curve implementation
# Take a GaAs solar cell with a 1.42eV band-gap and plot the IV curve

eg=1.42
V = np.linspace(0, 1.3, 500)
db_junction = Junction(kind='DB', T=298, Eg=eg, A=1, R_shunt=np.inf, n=1)
my_solar_cell = SolarCell([db_junction], T=298, R_series=0)

solar_cell_solver(my_solar_cell, 'iv',
                      user_options={'T_ambient': 298, 'db_mode': 'top_hat', 'voltages': V, 'light_iv': True,
                                    'internal_voltages': np.linspace(0, 1.3, 400), 'wavelength': wl,
                                    'mpp': True, 'light_source': am15g})

plt.figure(1)
plt.plot(V, my_solar_cell.iv.IV[1], 'k')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc {my_solar_cell.iv.Isc:.2f}')
plt.text(0.1,280,f'Voc {my_solar_cell.iv.Voc:.2f}')
plt.text(0.1,260,f'Pmax {my_solar_cell.iv.Pmpp:.2f}')
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.show()

# Unclear to NED how to access the internal J01 value!  Will work this out from Isc Voc
j01solcore=my_solar_cell.iv.Isc/np.exp(my_solar_cell.iv.Voc/(k*298))
print(f'SolCore J01= {j01solcore:.3e}')

#############################################
# Step 2 : Perform the same calculation using the analytical expressions of Pusch et al.

# Need to calculate the limit to Jsc
# Transform the AM1.5G spectrum to photon flux to enable quick limiting Jsc calculation
solarflux = LightSource(source_type='standard', version='AM1.5g', x=wl*1e9,
                        output_units='photon_flux_per_nm')
# Establish an interpolation function to allow integration over arbitrary limits
solarfluxInterpolate = InterpolatedUnivariateSpline(solarflux.spectrum()[0], solarflux.spectrum()[1], k=1)


# Analytical expressions to find IMax and VMax using LambertW function
# Find Jsc [limits are expressed in eV]
def getJsc(lowlim,upplim) :
    return q*solarfluxInterpolate.integral(1240/upplim, 1240/lowlim)
# Find J01 assuming abrupt junction & Boltzmann approximation
def getJ01(eg,t) :
   return ((2*np.pi* q )/(h**3 * c**2))* k*t * (eg**2 + 2*eg*(k*t) + 2*(k*t)**2)*np.exp(-(eg)/(k*t))
# Find Vmax
def getVmax(eg,emax,t) :
    return (k*t*(lambertw(np.exp(1)*(getJsc(eg,emax)/getJ01(eg,t)))-1)).real
# Find Imax
def getImax(eg,emax,t) :
    return getJsc(eg,emax) - getJ01(eg,t)*np.exp((getVmax(eg,emax,t))/(k*t))


# Calculate PV parameters
jsc=getJsc(eg,10)
j01=getJ01(eg,298)
voc=k*298*np.log(jsc/j01)
vmax=getVmax(eg,10,298)
imax=getImax(eg,10,298)
print(f'Analytical J01= {j01:.3e}')
# Calculate IV curve
I=np.array([jsc-j01*np.exp(vi/(k*298)) for vi in V])

plt.figure(2)
plt.plot(V, I, 'k')
plt.plot([0,voc,vmax],[jsc,0,imax],'o',color='red')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc {jsc:.2f}')
plt.text(0.1,280,f'Voc {voc:.2f}')
plt.text(0.1,260,f'Pmax {vmax*imax:.2f}')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.show()


###

# Calculating the multi-junction solar cell efficiency and thermal generation
#Uses the analytical expressions from [Pusch et al., JPV (2019)](doi.org/10.1109/JPHOTOV.2019.2903180)

import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import lambertw
from solcore.light_source import LightSource

# Define fundamental physical constants
q=1.60217662E-19  # electronic charge [C]
k=1.38064852E-23/q   # Boltzmann constant [eV/K]
h=6.62607004E-34/q  # Planck constant expressed in [eV.s]
c=299792458  # Speed of light [m.s^-1]

# Load the AM1.5G solar spectrum
wl = np.linspace(300, 4000, 4000) * 1e-9    #wl contains the x-ordinate in wavelength
am15g = LightSource(source_type='standard', x=wl*1e9, version='AM1.5g')
# Establish an interpolation fuction to allow integration over arbitary limits
solarpowerInterpolate = InterpolatedUnivariateSpline(am15g.spectrum()[0], am15g.spectrum()[1], k=1)

# Convert the AM1.5G spectrum to photon flux
solarflux = LightSource(source_type='standard', version='AM1.5g', x=wl*1e9, output_units='photon_flux_per_nm')
# Establish an interpolation function to allow integration over arbitary limits
solarfluxInterpolate = InterpolatedUnivariateSpline(solarflux.spectrum()[0], solarflux.spectrum()[1], k=1)

# Routine to find Jsc [limits are expressed in eV]
def getJsc(lowlim,upplim) :
    return q*solarfluxInterpolate.integral(1240/upplim, 1240/lowlim)
# Routine to find J01
def getJ01(eg,t) :
   return ((2*math.pi* q )/(h**3 * c**2))* k*t * (eg**2 + 2*eg*(k*t) + 2*(k*t)**2)*math.exp(-(eg)/(k*t))
# Routine to find Vmax
def getVmax(eg,emax,t) :
    return (k*t*(lambertw(math.exp(1)*(getJsc(eg,emax)/getJ01(eg,t)))-1)).real
# Routine to find Imax
def getImax(eg,emax,t) :
    return getJsc(eg,emax) - getJ01(eg,t)*math.exp((getVmax(eg,emax,t))/(k*t))
# Routine to fine absorbed power
def getAbsorbedPower(lowlim,upplim) :
    return solarpowerInterpolate.integral(1240/upplim, 1240/lowlim)

## Plotting Efficiency Contour map for 2-terminal series constrained tandem overlayed with a thermal generation map

# Plot a 2 junction, 2 terminal tandem map overlayed with a thermal generation map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getPmax(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        upperlimit=egs[i]
#    Find the minimum Imaxs
    minImax=np.amin(Imaxs)

#   Find tandem voltage
    vTandem=0
    for i in range(0,len(egs),1) :
        vsubcell=k*298*math.log((jscs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

    return vTandem*minImax

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getPmax([yy,xx])/1000

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

f3=plt.figure(3)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45])
plt.clabel(contours,inline=True,fontsize=10)
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')

# Plot a 2 junction, 2 terminal Q tandem map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getQ(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        upperlimit=egs[i]
#    Find the minimum Imaxs
        minImax=np.amin(Imaxs)

#   Find tandem voltage
    vTandem=0
    for i in range(0,len(egs),1) :
        vsubcell=k*298*math.log((jscs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

# Tandem power
    pTandem=vTandem*minImax
    pAbsorbed=getAbsorbedPower(np.amin(egs),10)
    return pAbsorbed-pTandem

@np.vectorize
def calcQ(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getQ([yy,xx])

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=calcQ(X,Y)

#plt.figure(5)
contours=plt.contourf(X,Y,Z,cmap='coolwarm',levels=[250,300,350,400,450,500,550,600,650,700,750])
plt.colorbar()
plt.show()

## Plotting Efficiency Contour map for 2-terminal series constrained tandem overlayed with a thermal generation map with complete absorption

# Plot a 2 junction, 2 terminal tandem map overlayed with a thermal generation map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getPmax(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        upperlimit=egs[i]
#    Find the minimum Imaxs
        minImax=np.amin(Imaxs)

#   Find tandem voltage
    vTandem=0
    for i in range(0,len(egs),1) :
        vsubcell=k*298*math.log((jscs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

    return vTandem*minImax

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getPmax([yy,xx])/1000

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

f4=plt.figure(4)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45])
plt.clabel(contours,inline=True,fontsize=10)
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')

# Plot a 2 junction, 2 terminal Q tandem map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getQ(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        upperlimit=egs[i]
#    Find the minimum Imaxs
        minImax=np.amin(Imaxs)

#   Find tandem voltage
    vTandem=0
    for i in range(0,len(egs),1) :
        vsubcell=k*298*math.log((jscs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

# Tandem power
    pTandem=vTandem*minImax
    pAbsorbed=1000
    return pAbsorbed-pTandem

@np.vectorize
def calcQ(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getQ([yy,xx])

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=calcQ(X,Y)

#plt.figure(5)
contours=plt.contourf(X,Y,Z,cmap='coolwarm',levels=[500,550,600,650,700,750,800,850,900,950,1000])
plt.colorbar()
plt.show()

## Plotting Contour map for 4-terminal tandem overlayed with thermal generation Q
# Plot a 2 junction, 4 terminal tandem map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getPmax(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    pTandem=0

    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        pTandem=pTandem+k*298*math.log((jscs[i]-Imaxs[i])/j01s[i])*Imaxs[i]
        upperlimit=egs[i]

    return pTandem

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getPmax([yy,xx])/1000

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

f5=plt.figure(5)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45,0.46])
plt.clabel(contours,inline=True,fontsize=10)
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')

def getQ(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    pTandem=0

    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        pTandem=pTandem+k*298*math.log((jscs[i]-Imaxs[i])/j01s[i])*Imaxs[i]
        upperlimit=egs[i]
# Tandem power
    pAbsorbed=getAbsorbedPower(np.amin(egs),10)
    return pAbsorbed-pTandem

@np.vectorize
def calcQ(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getQ([yy,xx])

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
ZZ=calcQ(X,Y)

contours=plt.contourf(X,Y,ZZ,cmap='coolwarm',levels=[250,300,350,400,450,500,550,600])
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')
plt.colorbar()
plt.show()

## Plotting Contour map for 4-terminal tandem overlayed with thermal generation Q total absorption
# Plot a 2 junction, 4 terminal tandem map

#Function to return pmax for n J cell
#Argument is a list of bandgaps in decending order.
def getPmax(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    pTandem=0

    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        pTandem=pTandem+k*298*math.log((jscs[i]-Imaxs[i])/j01s[i])*Imaxs[i]
        upperlimit=egs[i]

    return pTandem

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getPmax([yy,xx])/1000

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

f6=plt.figure(6)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45,0.46])
plt.clabel(contours,inline=True,fontsize=10)
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')

def getQ(egs) :
    # Since we need previous eg info have to iterate the Jsc array
    jscs=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10
    pTandem=0

    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        pTandem=pTandem+k*298*math.log((jscs[i]-Imaxs[i])/j01s[i])*Imaxs[i]
        upperlimit=egs[i]
# Tandem power
    return 1000-pTandem

@np.vectorize
def calcQ(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getQ([yy,xx])

x=np.linspace(0.7, 1.28, 70)
y=np.linspace(1.3, 1.9, 70)
X,Y=np.meshgrid(x,y)
ZZ=calcQ(X,Y)

contours=plt.contourf(X,Y,ZZ,cmap='coolwarm')
plt.ylabel('Top cell band-gap energy / eV')
plt.xlabel('Bottom cell band-gap energy / eV')
plt.title('4-T tandem + thermal generation Q total absorption')
plt.colorbar()
plt.show()
