from Spectrum_Functions import spec_to_xyz, spec_to_xyz_old, \
    gen_spectrum_1dip, gen_spectrum_1dip_old, \
    gen_spectrum_1gauss, gen_spectrum_1gauss_old, \
    gen_spectrum_2dip, gen_spectrum_2dip_old, \
    gen_spectrum_2gauss, gen_spectrum_2gauss_old, \
    wl
import matplotlib.pyplot as plt
import pandas as pd
from time import time

start = time()
spec = gen_spectrum_1dip_old(500, 80, 0.8, 0.2)
print(time() - start)

start = time()
spec_new = gen_spectrum_1dip(500, 80, 0.8, 0.2)
print(time() - start)

plt.figure()
plt.plot(wl, spec)
plt.plot(wl, spec_new, '--')
plt.show()

start = time()
spec = gen_spectrum_1gauss_old(500, 80, 0.8, 0.2)
print(time() - start)

start = time()
spec_new = gen_spectrum_1gauss(500, 80, 0.8, 0.2)
print(time() - start)

plt.figure()
plt.plot(wl, spec)
plt.plot(wl, spec_new, '--')
plt.show()

start = time()
spec = gen_spectrum_2dip_old(500, 80, 700, 20, 0.8, 0.2)
print(time() - start)

start = time()
spec_new = gen_spectrum_2dip(500, 80, 700, 20, 0.8, 0.2)
print(time() - start)

plt.figure()
plt.plot(wl, spec)
plt.plot(wl, spec_new, '--')
plt.show()

start = time()
spec = gen_spectrum_2gauss_old(500, 80, 550,40, 0.8, 0.2)
print(time() - start)

start = time()
spec_new = gen_spectrum_2gauss(500, 80, 550, 40, 0.8, 0.2)
print(time() - start)

plt.figure()
plt.plot(wl, spec)
plt.plot(wl, spec_new, '--')
plt.show()


XYZ_old = spec_to_xyz_old(spec)
print(XYZ_old)

df = pd.read_excel("ASTMG173_split.xlsx", sheet_name=0)

XYZ = spec_to_xyz(spec, df)
print(XYZ)

