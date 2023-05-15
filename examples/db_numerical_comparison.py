import matplotlib.pyplot as plt
import numpy as np
from ecopv.optimization_functions import db_cell_calculation_noR, getPmax, getIVmax
from solcore.solar_cell import SolarCell, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.light_source import LightSource
from solcore.constants import kb, q

k = kb / q
T = 300
kbT = k * T

from solcore import si

interval = 0.1
wavelengths = si(np.arange(300, 4000, interval), "nm")
wl_cell = np.arange(
    300, 4000, interval
)

black_cell_Eg = [
    [1.34],
    [0.96, 1.63],
    [0.93, 1.37, 1.90],
    [0.72, 1.11, 1.49, 2.00],
    [0.70, 1.01, 1.33, 1.67, 2.14],
    [0.70, 0.96, 1.20, 1.47, 1.79, 2.24],
]

opts = {'light_iv': True, 'wavelength': wavelengths, 'mpp': True, 'T': 300}

light_source = LightSource(
    source_type="standard",
    version= "AM1.5g",
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

for Egs in black_cell_Eg:

    V = np.arange(0, np.sum(Egs), 0.001)
    opts['voltages'] = V

    db_junctions = [Junction(kind='DB', Eg=Eg, A=1, R_shunt=1e30, n=1) for
                    Eg in
                    Egs[::-1]]
    solar_cell_db_A1 = SolarCell(db_junctions)

    solar_cell_solver(solar_cell_db_A1, 'iv', user_options=opts)

    analytic_eta = getPmax(Egs[::-1],
        photon_flux_cell[1],
        photon_flux_cell[0],
        interval,
        None,
        1,
        4.43,
        "no_R",
        2)/10

    j01s, jscs, Vmaxs, Imaxs = db_cell_calculation_noR(Egs[::-1],
                                                       photon_flux_cell[1],
                                                       photon_flux_cell[0],
                                                       interval
                                                       )

    minImax = np.amin(Imaxs)
    vsubcell = kbT * np.log((jscs - minImax) / j01s)

    vTandem = np.sum(vsubcell)

    numerical_eta = solar_cell_db_A1.iv["Eta"]*100
    print(analytic_eta, numerical_eta)
    print(100*(analytic_eta - numerical_eta)/analytic_eta)
    print("V", vTandem, solar_cell_db_A1.iv["Vmpp"])
    print("I", minImax/10, solar_cell_db_A1.iv["Impp"]/10)

    plt.figure()
    plt.plot(solar_cell_db_A1.iv["IV"][0], solar_cell_db_A1.iv["IV"][1]/10)
    plt.xlim(0, 1.1*solar_cell_db_A1.iv["Voc"])
    plt.ylim(0, 1.1*np.max(solar_cell_db_A1.iv["IV"][1]/10))
    plt.axhline(y=minImax/10)
    plt.axvline(x=vTandem)
    plt.plot(solar_cell_db_A1.iv["Vmpp"], solar_cell_db_A1.iv["Impp"]/10, 'o')
    plt.plot()
    plt.show()


