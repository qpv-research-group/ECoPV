import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

interval=0.02 # used for setting correct number of data points
wl=np.arange(380,780+interval,interval) # visible wavelength in nm, 380nm to 780nm, 1nm intervals

# spectrum generation (1 dip parameters to spectrum)
def gen_spectrum_1dip(center,width,peak,base=0): # center and width in nm
    spectrum=np.array(np.zeros(len(wl)))
    wl1=center-width/2
    wl2=center+width/2
    for idx in np.arange(0, len(wl), 1):
        if (idx>=(wl1-380)/interval) and (idx<=(wl2-380)/interval):
            spectrum[idx]=peak
        else:
            spectrum[idx]=base
    return spectrum

# spectrum generation (1 gaussian parameters to spectrum)
def gen_spectrum_1gauss(center,width,peak,base=0): # center and width in nm
    spectrum=np.array(np.zeros(len(wl)))
    k=peak-base
    for idx in np.arange(0, len(wl), 1):
        spectrum[idx]=base+k*np.exp(-((wl[idx]-center)/width)**2)
    return spectrum

# spectrum generation (2 dip parameters to spectrum)
def gen_spectrum_2dip(center1,width1,center2,width2,peak=1,base=0): # center and width in nm
    spectrum=np.array(np.zeros(len(wl)))
    dip1_wl1=center1-width1/2
    dip1_wl2=center1+width1/2
    dip2_wl1=center2-width2/2
    dip2_wl2=center2+width2/2
    for idx in np.arange(0, len(wl), 1):
        if ((idx>=(dip1_wl1-380)/interval) and (idx<=(dip1_wl2-380)/interval)) or ((idx>=(dip2_wl1-380)/interval) and (idx<=(dip2_wl2-380)/interval)):
            spectrum[idx]=peak
        else:
            spectrum[idx]=base
    return spectrum

# spectrum generation (2 gaussian parameters to spectrum)
def gen_spectrum_2gauss(center1,width1,center2,width2,peak=1,base=0): # center and width in nm
    spectrum=np.array(np.zeros(len(wl)))
    k=peak-base
    for idx in np.arange(0, len(wl), 1):
        spectrum[idx]=base+k*np.exp(-((wl[idx]-center1)/width1)**2)+k*np.exp(-((wl[idx]-center2)/width2)**2)
    spectrum=spectrum/spectrum.max()
    return spectrum

#Ref https://scipython.com/blog/converting-a-spectrum-to-a-colour/
def spec_to_xyz(spec):
    cmf = np.loadtxt('cmf.txt', usecols=(1,2,3))        
    cmf_new=np.zeros((len(wl),3)) # Interpolating cmf data to be in 1nm intervals
           
    intfunc = interp1d(np.arange(380,781,1),cmf[:,0],fill_value="extrapolate")
    cmf_new[:,0] = intfunc(wl)
    intfunc = interp1d(np.arange(380,781,1),cmf[:,1],fill_value="extrapolate")
    cmf_new[:,1] = intfunc(wl)
    intfunc = interp1d(np.arange(380,781,1),cmf[:,2],fill_value="extrapolate")
    cmf_new[:,2] = intfunc(wl)
    cmf=cmf_new
    
    """Convert a spectrum to an xyz point.
    The spectrum must be on the same grid of points as the colour-matching
    function, self.cmf: 380-780 nm in 1 nm steps.
    """
    #Trying to get Python cooridnates to match Matlab
    #Insert complete path to the excel file and index of the worksheet
    df = pd.read_excel("ASTMG173_split.xlsx", sheet_name=0)
    # insert the name of the column as a string in brackets
    AM1_5G_wl  = list(df['A']) 
    AM1_5G_Spec = list(df['C'])

    intfunc = interp1d(AM1_5G_wl,AM1_5G_Spec,fill_value="extrapolate")
    AM1_5G = intfunc(wl)
    AM1_5G= np.array(AM1_5G)

    X = 0
    Y = 0
    Z = 0
    Ymax = 0

    for i in range(0, len(wl)):
        Ymax = Ymax + interval * cmf[i,1] * AM1_5G[i]
        X = X + interval * cmf[i,0] * AM1_5G[i] * spec[i]
        Y = Y + interval * cmf[i,1] * AM1_5G[i] * spec[i]
        Z = Z + interval * cmf[i,2] * AM1_5G[i] * spec[i]
    
    if Ymax == 0:
        return (X,Y,Z)

    X = X/Ymax
    Y = Y/Ymax
    Z = Z/Ymax
    XYZ = (X,Y,Z)

    return XYZ

def delta_E_CIE2000(Lab1, Lab2, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using CIE 2000 recommendation.
    Parameters
    ----------
    Lab1 : array_like, (3,)        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)        *CIE Lab* *array_like* colour 2.
    Returns
    -------
    numeric:        Colour difference :math:`\Delta E_{ab}`.
    Ref: Lindbloom, B. (2009). Delta E (CIE 2000). Retrieved February 24,
            2014, from http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html
    """
    
    L1, a1, b1 = np.ravel(Lab1)
    L2, a2, b2 = np.ravel(Lab2)

    kL = 1
    kC = 1
    kH = 1

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = (np.arctan2(b1, a1_prime) * 180) / np.pi
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = (np.arctan2(b2, a2_prime) * 180) / np.pi
    if h2_prime < 0.0:
        h2_prime += 360

    h_bar_prime = (0.5 * (h1_prime + h2_prime + 360)
                   if np.fabs(h1_prime - h2_prime) > 180 else
                   0.5 * (h1_prime + h2_prime))

    t = (1 - 0.17 * np.cos(np.pi * (h_bar_prime - 30) / 180) +
         0.24 * np.cos(np.pi * (2 * h_bar_prime) / 180) +
         0.32 * np.cos(np.pi * (3 * h_bar_prime + 6) / 180) -
         0.20 * np.cos(np.pi * (4 * h_bar_prime - 63) / 180))

    if np.fabs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    else:
        delta_h_prime = (h2_prime - h1_prime + 360
                         if h2_prime <= h1_prime else
                         h2_prime - h1_prime - 360)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * np.sqrt(c1_prime * c2_prime) *
                     np.sin(np.pi * (0.5 * delta_h_prime) / 180))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * np.sin(np.pi * (2 * delta_theta) / 180)

    return np.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)
