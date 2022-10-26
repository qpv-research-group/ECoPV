import numpy as np
from Spectrum_Functions_de import spec_to_xyz, delta_E_CIE2000, gen_spectrum_1dip, gen_spectrum_2dip, gen_spectrum_1gauss, gen_spectrum_2gauss

import matplotlib.pyplot as plt
from solcore.optimization import DE, PDE
from solcore.light_source import LightSource
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver
from solcore.structure import Junction

from time import time

import sys
from colormath.color_objects import LabColor, XYZColor,  sRGBColor
from colormath.color_conversions import convert_color
import pandas as pd
from skimage import io

interval=0.2 # interval between each two wavelength points, 0.02 needed for low dE values
wl=np.arange(380,780+interval,interval)

wl_cell=np.arange(300, 4000, 0.5)

df = pd.read_excel("ASTMG173_split.xlsx", sheet_name=0)

light = LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
)

photon_flux =  LightSource(
    source_type="standard", version="AM1.5g", x=wl, output_units="photon_flux_per_nm"
)

photon_flux_norm = photon_flux.spectrum(wl)[1]/max(photon_flux.spectrum(wl)[1])

photon_flux_cell = LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell)[1]

solar_spec_col = LightSource(
    source_type="standard", version="AM1.5g", x=wl, output_units="photon_flux_per_nm"
).spectrum(wl)[1]

class Optimize:
    def __init__(self, colors_tuple, color_name):
        '''
        Send color in from op_send_sq.py script
        colors_tuple is (a,b,c) LAB coordinates
        color_name is the name of the color from 2005 ColorChecker, used to save the filename
        
        Function calculates uses scipy minimize to find the optimum deltaE value.
        Reads in values from excel sheets generated from matlab
        Saves data out to excel sheet

        '''
        print(color_name)
        colors_tuple=tuple(colors_tuple)
        self.lab = LabColor(*colors_tuple)
        self.target_color_LAB = colors_tuple
        self.xyz = convert_color(self.lab, XYZColor)

        self.dE_threshold1d = 1
        self.dE_threshold = 1

        self.peak_ = 1
        self.base_ = 0

        self.color_2005 = color_name #name from ColorChecker 2005
        # define ranges for optimize
        self.vis_bound = [380,780]
        self.width_max = [0.0,300]
        self.width2_max = [0.0,150]
        self.peak_range = [0,1]
        
        self.ranges=[self.vis_bound,self.width_max,self.peak_range]
        self.ranges2d = [self.vis_bound,self.width2_max,self.peak_range,self.vis_bound,self.width2_max, self.peak_range]

        self.ranges_MJ= [self.vis_bound,self.width_max, self.peak_range, [1.05, 1.4]]
        self.ranges2d_MJ = [self.vis_bound, self.width2_max, self.vis_bound, self.width2_max]


        print("Optimization for 1 top hat dip (2)")

        color_1d = eff_colour_1dip(self.target_color_LAB, self.dE_threshold1d)

        PDE_obj_1dip = DE(color_1d.evaluate, mutation=(0.5, 2),
                                     bounds=self.ranges, maxiters=1000)


        self.result1d_Minimize = PDE_obj_1dip.solve()
        res_color = self.result1d_Minimize
        best_pop_1d = res_color[0]

        if res_color[1] < 0:
            print("Met colour threshold")
            self.one_peak_eff = 1

        else:
            self.one_peak_eff = 0

        R_spec = gen_spectrum_1dip(best_pop_1d[0], best_pop_1d[1], best_pop_1d[2], wl=wl_cell)
        Eg_guess = 1240/1.13
        fraction_lost = np.sum((R_spec * photon_flux_cell)[wl_cell<Eg_guess]) / np.sum(photon_flux_cell[wl_cell<Eg_guess])
        print('lost photon fraction', fraction_lost)

        # stdout_backup = sys.stdout
        #
        # with open('Log', 'a') as f:
        #     sys.stdout = f
        #     self.one_peak_eff, _, _, _ = elec_calc_1dip(best_pop_1d)
        #
        # sys.stdout = stdout_backup

        #print("Jsc/Voc:", self.Jsc, self.Voc)

        #elec_2d.plot(best_pop_2d)

        print('parameters for best result:', best_pop_1d, '\n')

        print("Optimization for 2 top hat dip (2)")
        # self.result1d_Minimize = minimize(self.op_delta_E_1dip,self.x0,method= method2,
        #                                   bounds=self.ranges,tol = tolerance,
        #                                   args=df)#,options={'return_all': True,'fatol': 0.1,'adaptive': True})#'initial_tr_radius': 0.5})

        color_2d = eff_colour_2dip(self.target_color_LAB, self.dE_threshold)

        start = time()

        PDE_obj_2dip = DE(color_2d.evaluate, mutation=(0.5, 2),
                                     bounds=self.ranges2d_MJ, maxiters=1000)

        print('Optimization in ', time() - start, ' s')


        self.result2d_Minimize = PDE_obj_2dip.solve()
        res_color = self.result2d_Minimize
        best_pop_2d = res_color[0]

        if res_color[1] < 0:
            print("Met colour threshold")

        R_spec = gen_spectrum_2dip(best_pop_2d[0], best_pop_2d[1], 1,
                                   best_pop_2d[2], best_pop_2d[3], 1, wl=wl_cell)
        Eg_guess = 1240/1.13
        fraction_lost = np.sum((R_spec * photon_flux_cell)[wl_cell<Eg_guess]) / np.sum(photon_flux_cell[wl_cell<Eg_guess])
        print('lost photon fraction', fraction_lost)

        stdout_backup = sys.stdout

        with open('Log', 'a') as f:
            sys.stdout = f
            self.two_peak_eff, self.Eg, self.Jsc, self.Voc, Eg, eff = elec_calc_2dip(best_pop_2d)

            plt.figure()
            plt.plot(Eg, 100*eff)
            plt.title(color_name)
            plt.plot(self.Eg, self.two_peak_eff, 'or')
            plt.show()

        sys.stdout = stdout_backup

        #print("Jsc/Voc:", self.Jsc, self.Voc)

        #elec_2d.plot(best_pop_2d)

        print('parameters for best result:', best_pop_2d, '\n', 'optimized eff:', self.two_peak_eff,
              'optimized Eg:', self.Eg)


        # print("Optimization for 2 Gaussian dips")
        # # self.result2gauss_Minimize = minimize(self.op_delta_E_2gauss,self.x02d,method=method2,
        # #                                       bounds=self.ranges2d,tol = tolerance,
        # #                                       args=df)
        #
        # PDE_obj_2gauss = DE(self.op_delta_E_2gauss, mutation=(0.5, 2),
        #                     bounds=self.ranges2d, maxiters=300)
        #
        # self.result2gauss_Minimize = PDE_obj_2gauss.solve()
        # res = self.result2gauss_Minimize
        # best_pop_2g = res[0]
        #
        # print('parameters for best result:', best_pop_2g, '\n', 'optimized value:', res[1])
        #
        # best_pop_evo = res[2]
        # best_fitn_evo = res[3]
        # mean_fitn_evo = res[4]
        # final_fitness = res[1]

        # plot evolution of the fitness of the best population per iteration

        # plt.figure()
        # plt.plot(best_fitn_evo, '-k', label='Best')
        # plt.plot(mean_fitn_evo, '--r', label='Mean')
        # plt.xlabel('iteration')
        # plt.ylabel('delta_E')
        # plt.title('2 gaussian')
        # plt.show()

        best_pop1d = best_pop_1d[:3]
        best_pop2d = best_pop_2d[:2].tolist() +  [1] + best_pop_2d[2:4].tolist() + [1]

        plt.figure()
        plt.plot(wl, gen_spectrum_1dip(*best_pop1d))
#        plt.plot(wl, gen_spectrum_1gauss(*best_pop_1g))
        plt.plot(wl, gen_spectrum_2dip(*best_pop2d), '--')
 #       plt.plot(wl, gen_spectrum_2gauss(*best_pop_2g))
        plt.title(color_name)
        plt.show()

        rgb_target = convert_color(self.xyz, sRGBColor)
        rgb_target_list = [rgb_target.rgb_r*255, rgb_target.rgb_g*255, rgb_target.rgb_b*255]

        XYZ=spec_to_xyz(gen_spectrum_1dip(*best_pop1d), solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        rgb_d1 = convert_color(XYZ, sRGBColor)
        rgb_d1_list = [rgb_d1.rgb_r*255, rgb_d1.rgb_g*255, rgb_d1.rgb_b*255]

        XYZ=spec_to_xyz(gen_spectrum_2dip(*best_pop2d), solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        rgb_d2 = convert_color(XYZ, sRGBColor)
        rgb_d2_list = [rgb_d2.rgb_r*255, rgb_d2.rgb_g*255, rgb_d2.rgb_b*255]

        palette = np.array([[rgb_target_list, rgb_target_list],
                            [rgb_d1_list, rgb_d2_list]], dtype=int)

        io.imshow(palette)
        plt.show()

        return


    def op_delta_E_1dip(self,x, df=df):
    
        center = x[0]
        width = x[1]
        peak = x[2]
        #base = x[3]
        spectrum = gen_spectrum_1dip(center,width,peak,base=0)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)
        if delta_E < self.dE_threshold1d:
            mean_R = np.mean(spectrum)
            scaled_E = mean_R*delta_E # penalize high average R
        else:
            scaled_E = delta_E
        #print('delta_E 1dip is ',delta_E)
        # if delta_E < self.dE_threshold1d:
        #     for i in range(2): # Trying to avoid excel writing error
        #         #print('Delta E Below Threshold')
        #         #WRITE EXCEL FUNCTION
        #         xcel_file = self.color_2005+'_D1_'+'dE_'+str(delta_E).replace('.','_')#color_name
        #         cwd = os.getcwd()
        #         filepath=os.path.join(cwd, r"color_data",xcel_file+'.xlsx')
        #         cp_list =([center, width, peak, spectrum,Lab,filepath]) # color parameter list to be stored in delta_E dict for op send to try different color parameters
        #         self.d1_cp[delta_E] = cp_list
        #
        #         workbook = xlsxwriter.Workbook(filepath)
        #         worksheet = workbook.add_worksheet()
        #         worksheet.write('A1', 'Efficiency')
        #         worksheet.write('B1', 'dE_D1')
        #         worksheet.write('B2', delta_E)
        #         worksheet.write('C1', 'RGB color')
        #         worksheet.write('C2',self.color)
        #         worksheet.write('D1','LAB color')
        #         worksheet.write('D2', self.color_lab)
        #         worksheet.write('E1', 'Eg')
        #         worksheet.write('F1', 'eff_CM')
        #
        #         worksheet.write('G1', 'Color')
        #         worksheet.write('G2', self.color_2005)
        #         worksheet.write('H1', 'Method')
        #         worksheet.write('H2', 'D1')
        #
        #
        #         worksheet.write('I1', 'Center')
        #         worksheet.write('I2', center)
        #         worksheet.write('J1', 'Width')
        #         worksheet.write('J2', width)
        #         worksheet.write('K1', 'Peak')
        #         worksheet.write('K2', peak)
        #
        #         workbook.close()
        # return delta_E
        return scaled_E
    
    def op_delta_E_1gauss(self,x, df=df):


        center = x[0]
        width = x[1]
        peak = x[2]
        spectrum = gen_spectrum_1gauss(center,width,peak,base=0)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)

        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)

        if delta_E < self.dE_threshold1d:
            mean_R = np.mean(spectrum)
            scaled_E = mean_R*delta_E # penalize high average R
        else:
            scaled_E = delta_E
        #print('delta_E 1gauss is ',delta_E)
        # if delta_E < self.dE_threshold1d:
        #     for i in range(2):
        #         #print('Delta E Below Threshold')
        #         xcel_file = self.color_2005+'_G1_'+'dE_'+str(delta_E).replace('.','_')#color_name
        #         cwd = os.getcwd()
        #         filepath=os.path.join(cwd, r"color_data",xcel_file+'.xlsx')
        #         cp_list =([center, width, peak, spectrum,Lab,filepath])
        #         self.g1_cp[delta_E] = cp_list
        #
        #         workbook = xlsxwriter.Workbook(filepath)
        #         worksheet = workbook.add_worksheet()
        #         worksheet.write('A1', 'Efficiency')
        #         worksheet.write('B1', 'dE_G1')
        #         worksheet.write('B2', delta_E)
        #         worksheet.write('C1', 'RGB color')
        #         worksheet.write('C2',self.color)
        #         worksheet.write('D1','LAB color')
        #         worksheet.write('D2', self.color_lab)
        #         worksheet.write('E1', 'Eg')
        #         worksheet.write('F1', 'eff_CM')
        #
        #         worksheet.write('G1', 'Color')
        #         worksheet.write('G2', self.color_2005)
        #         worksheet.write('H1', 'Method')
        #         worksheet.write('H2', 'G1')
        #
        #
        #         worksheet.write('I1', 'Center')
        #         worksheet.write('I2', center)
        #         worksheet.write('J1', 'Width')
        #         worksheet.write('J2', width)
        #         worksheet.write('K1', 'Peak')
        #         worksheet.write('K2', peak)
        #         workbook.close()
        # return delta_E
        return scaled_E
    
    
    def op_delta_E_2dip(self,x, df=df):
        center = x[0]
        width = x[1]
        peak1 = x[2]
        center2 = x[3]
        width2 = x[4]
        peak2 = x[5]
        spectrum = gen_spectrum_2dip(center,width,peak1,center2,width2,peak2)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)

        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)
        if delta_E < self.dE_threshold:
            mean_R = np.mean(spectrum)
            scaled_E = mean_R*delta_E # penalize high average R
        else:
            scaled_E = delta_E
        #print('delta_E 2dip is ',delta_E)
        # if delta_E < self.dE_threshold:
        #     for i in range(2):
        #         xcel_file = self.color_2005+'_D2_'+'dE_'+str(delta_E).replace('.','_')#color_name
        #         cwd = os.getcwd()
        #         filepath=os.path.join(cwd, r"color_data",xcel_file+'.xlsx')
        #
        #         #print('Delta E Below Threshold')
        #         cp_list = ([center, width, center2, width2,spectrum,Lab,filepath])
        #         self.d2_cp[delta_E] = cp_list
        #
        #
        #         workbook = xlsxwriter.Workbook(filepath)
        #         worksheet = workbook.add_worksheet()
        #         worksheet.write('A1', 'Efficiency')
        #         worksheet.write('B1', 'dE_D2')
        #         worksheet.write('B2', delta_E)
        #         worksheet.write('C1', 'RGB color')
        #         worksheet.write('C2',self.color)
        #         worksheet.write('D1','LAB color')
        #         worksheet.write('D2', self.color_lab)
        #         worksheet.write('E1', 'Eg')
        #         worksheet.write('F1', 'eff_CM')
        #
        #         worksheet.write('G1', 'Color')
        #         worksheet.write('G2', self.color_2005)
        #         worksheet.write('H1', 'Method')
        #         worksheet.write('H2', 'D2')
        #
        #         worksheet.write('I1', 'Center')
        #         worksheet.write('I2', center)
        #         worksheet.write('J1', 'Width')
        #         worksheet.write('J2', width)
        #
        #         worksheet.write('K1', 'Center2')
        #         worksheet.write('K2', center2)
        #         worksheet.write('L1', 'Width2')
        #         worksheet.write('L2', width2)
        #
        #         workbook.close()
        #return delta_E
        return scaled_E

    
    def op_delta_E_2gauss(self,x, df=df):
        center = x[0]
        width = x[1]
        peak1 = x[2]
        center2 = x[3]
        width2 = x[4]
        peak2 = x[5]
        spectrum = gen_spectrum_2gauss(center,width,peak1,center2,width2,peak2)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)

        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)
        if delta_E < self.dE_threshold:
            mean_R = np.mean(spectrum)
            scaled_E = mean_R*delta_E # penalize high average R
        else:
            scaled_E = delta_E
        #print('delta_E 2gauss is ',delta_E)
        # if delta_E < self.dE_threshold:
        #     for i in range(2):
        #         xcel_file = self.color_2005+'_G2_'+'dE_'+str(delta_E).replace('.','_')#color_name
        #         cwd = os.getcwd()
        #         filepath=os.path.join(cwd, r"color_data",xcel_file+'.xlsx')
        #
        #         #print('Delta E Below Threshold')
        #         cp_list = ([center, width, center2, width2,spectrum,Lab,filepath])
        #         self.g2_cp[delta_E] = cp_list
        #
        #
        #         workbook = xlsxwriter.Workbook(filepath)
        #         worksheet = workbook.add_worksheet()
        #         worksheet.write('A1', 'Efficiency')
        #         worksheet.write('B1', 'dE_G2')
        #         worksheet.write('B2', delta_E)
        #         worksheet.write('C1', 'RGB color')
        #         worksheet.write('C2',self.color)
        #         worksheet.write('D1','LAB color')
        #         worksheet.write('D2', self.color_lab)
        #         worksheet.write('E1', 'Eg')
        #         worksheet.write('F1', 'eff_CM')
        #
        #         worksheet.write('G1', 'Color')
        #         worksheet.write('G2', self.color_2005)
        #         worksheet.write('H1', 'Method')
        #         worksheet.write('H2', 'G2')
        #
        #         worksheet.write('I1', 'Center')
        #         worksheet.write('I2', center)
        #         worksheet.write('J1', 'Width')
        #         worksheet.write('J2', width)
        #
        #         worksheet.write('K1', 'Center2')
        #         worksheet.write('K2', center2)
        #         worksheet.write('L1', 'Width2')
        #         worksheet.write('L2', width2)
        #
        #         workbook.close()
        # return delta_E
        return scaled_E


class op_1dip_MJ():

    def __init__(self, tg, th):

        self.target_color_LAB = tg
        self.dE_threshold1d = th

    def evaluate(self,x, df=df):

        center = x[0]
        width = x[1]
        peak = x[2]
        Eg1 = x[3]

        db_junction0 = Junction(kind='DB', T=293, Eg=Eg1, A=1, R_shunt=np.inf, n=1)

        solar_cell = SolarCell([db_junction0], T=293, R_series=0)

        #base = x[3]
        spectrum = gen_spectrum_1dip(center,width,peak,base=0)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)

        if delta_E < self.dE_threshold1d:

            spec_cell = 1 - gen_spectrum_1dip(center, width, peak, wl_cell)

            light_for_cells = LightSource(source_type='custom', x_data=wl_cell,
                                          y_data=light.spectrum(wl_cell)[1] * spec_cell,
                                          input_units='photon_flux_per_nm', output_units='photon_flux_per_nm')

            # plt.figure()
            # plt.plot(wl, initial_light_source.spectrum(wl)[1] * (1-spectrum))
            # plt.title(str(Eg1) + str(Eg2))
            # plt.show()

            V = np.linspace(0, 5, 500)

            solar_cell_solver(solar_cell, 'iv',
                              user_options={'T_ambient': 293, 'db_mode': 'top_hat',
                                            'voltages': V,
                                            'light_iv': True,
                                            'internal_voltages': np.linspace(-1, 5, 1100),
                                            'wavelength': wl_cell * 1e-9,
                                            'mpp': True,
                                            'light_source': light_for_cells})
            eta = solar_cell.iv['Eta']
            print(eta*100)

            if eta > 0.45:
                print('WEIRD EFFICIENCY', x)

            return -eta

        else:
            return delta_E

    def plot(self, x, df=df):

        center = x[0]
        width = x[1]
        peak = x[2]
        Eg1 = x[3]

        db_junction0 = Junction(kind='DB', T=293, Eg=Eg1, A=1, R_shunt=np.inf, n=1)

        solar_cell = SolarCell([db_junction0], T=293, R_series=0)

        #base = x[3]
        spectrum = gen_spectrum_1dip(center,width,peak,base=0)
        XYZ=spec_to_xyz(spectrum, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)

        spec_cell = 1 - gen_spectrum_1dip(center, width, peak, wl_cell)

        light_for_cells = LightSource(source_type='custom', x_data=wl_cell,
                                      y_data=light.spectrum(wl_cell)[1] * spec_cell,
                                      input_units='photon_flux_per_nm', output_units='photon_flux_per_nm')

        # plt.figure()
        # plt.plot(wl, initial_light_source.spectrum(wl)[1] * (1-spectrum))
        # plt.title(str(Eg1) + str(Eg2))
        # plt.show()

        V = np.linspace(0, 5, 500)

        solar_cell_solver(solar_cell, 'iv',
                          user_options={'T_ambient': 293, 'db_mode': 'top_hat',
                                        'voltages': V,
                                        'light_iv': True,
                                        'internal_voltages': np.linspace(-1, 5, 1100),
                                        'wavelength': wl_cell * 1e-9,
                                        'mpp': True,
                                        'light_source': light_for_cells})

        eta = solar_cell.iv['Eta']

        plt.figure()
        plt.plot(V, solar_cell.iv.IV[1], 'k', linewidth=4, label='Total')
        # plt.plot(V, -solar_cell[0].iv(V), 'r', label='Bottom')
        # plt.plot(V, -solar_cell[1].iv(V), 'g', label='Middle')
        plt.ylim(0, 400)
        plt.xlim(0, 3.75)
        plt.legend()
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A/m$^2$)')
        plt.text(3, 150, str(np.round(eta*100, 2)))
        plt.show()

        # plt.figure()
        # plt.plot(wl_cell, light.spectrum(wl_cell)[1] * spec_cell)
        # plt.show()


class eff_colour_1dip():

    def __init__(self, tg, th):

        self.target_color_LAB = tg
        self.dE_threshold = th

    def calculate(self, x):

        center = x[0]
        width = x[1]
        peak = x[2]

        R_spec = gen_spectrum_1dip(center,width,peak,base=0)
        XYZ=spec_to_xyz(R_spec, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)

        return delta_E, R_spec

    def evaluate(self,x):

        delta_E, R_spec = self.calculate(x)

        if delta_E < self.dE_threshold: # meets colour threshold; now try to optimize how efficiently we can make this colour

            #print("below dE threshold")
            n_available = np.sum((1-R_spec) * photon_flux_norm)

            #print(-n_available)

            return -n_available

        else:
            return delta_E

    def plot(self, x, df=df):

        delta_E, R_spec = self.calculate(x, df)

        plt.figure()
        plt.plot(wl, R_spec)
        plt.title(str(delta_E))
        plt.show()

def elec_calc_1dip(x):

    center = x[0]
    width = x[1]
    peak = x[2]

    V = np.arange(0, 1.5, 0.0005)

    R_spec = gen_spectrum_1dip(center, width, peak, wl=wl_cell,base=0)

    light_for_cells = LightSource(source_type='custom', x_data=wl_cell,
                                  y_data=light.spectrum(wl_cell)[1] * (1-R_spec),
                                  input_units='photon_flux_per_nm', output_units='photon_flux_per_nm')

    Eg = np.arange(1.1, 1.35, 0.0005)
    eta = np.empty_like(Eg)
    Isc = np.empty_like(Eg)
    Voc = np.empty_like(Eg)

    for j1, bg in enumerate(Eg):
        db_junction0 = Junction(kind='DB', T=298.15, Eg=bg, A=1, n=1)

        solar_cell = SolarCell([db_junction0], T=298.15, R_series=0)

        solar_cell_solver(solar_cell, 'iv',
                          user_options={'T_ambient': 298.15, 'voltages': V,
                                        'light_iv': True,
                                        'internal_voltages': np.arange(0, 1.5, 0.0005),
                                        'wavelength': wl_cell * 1e-9,
                                        'mpp': True,
                                        'light_source': light_for_cells
                                        })

        eta[j1] = solar_cell.iv['Pmpp']/1000
        Isc[j1] = solar_cell.iv['Isc']
        Voc[j1] = solar_cell.iv['Voc']

    return np.max(eta)*100, Eg[np.argmax(eta)], Isc[np.argmax(eta)]/10, Voc[np.argmax(eta)]



class eff_colour_2dip():

    def __init__(self, tg, th):

        self.target_color_LAB = tg
        self.dE_threshold = th

    def calculate(self, x):

        center = x[0]
        width = x[1]
        center2 = x[2]
        width2 = x[3]

        R_spec = gen_spectrum_2dip(center,width,1,center2, width2,1,base=0)
        XYZ=spec_to_xyz(R_spec, solar_spec_col)
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta_E=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color_LAB)

        return delta_E, R_spec

    def evaluate(self,x):

        delta_E, R_spec = self.calculate(x)

        if delta_E < self.dE_threshold: # meets colour threshold; now try to optimize how efficiently we can make this colour

            #print("below dE threshold")
            n_available = np.sum((1-R_spec) * photon_flux_norm)

            #print(-n_available)

            return -n_available

        else:
            return delta_E

    def plot(self, x, df=df):

        delta_E, R_spec = self.calculate(x, df)

        plt.figure()
        plt.plot(wl, R_spec)
        plt.title(str(delta_E))
        plt.show()


def elec_calc_2dip(x):

    center = x[0]
    width = x[1]
    center2 = x[2]
    width2 = x[3]

    V = np.arange(0, 1.5, 0.0005)

    R_spec = gen_spectrum_2dip(center, width, 1, center2, width2, 1, wl=wl_cell,base=0)

    light_for_cells = LightSource(source_type='custom', x_data=wl_cell,
                                  y_data=light.spectrum(wl_cell)[1] * (1-R_spec),
                                  input_units='photon_flux_per_nm', output_units='photon_flux_per_nm')

    # plt.figure()
    # plt.plot(wl_cell, light_for_cells.spectrum(wl_cell)[1])
    # plt.xlim(300, 2500)
    # plt.show()


    Eg = np.arange(1.1, 1.35, 0.0005)
    eta = np.empty_like(Eg)
    Isc = np.empty_like(Eg)
    Voc = np.empty_like(Eg)

    for j1, bg in enumerate(Eg):
        db_junction0 = Junction(kind='DB', T=298.15, Eg=bg, A=1, n=1)

        solar_cell = SolarCell([db_junction0], T=298.15, R_series=0)

        solar_cell_solver(solar_cell, 'iv',
                          user_options={'T_ambient': 298.15, 'voltages': V,
                                        'light_iv': True,
                                        'internal_voltages': np.arange(0, 1.5, 0.0005),
                                        'wavelength': wl_cell * 1e-9,
                                        'mpp': True,
                                        'light_source': light_for_cells
                                        })

        eta[j1] = solar_cell.iv['Pmpp']/1000
        Isc[j1] = solar_cell.iv['Isc']
        Voc[j1] = solar_cell.iv['Voc']

    # plt.figure()
    # plt.plot(Eg, eta)
    # plt.show()
    #
    # db_junction0 = Junction(kind='DB', T=298.15, Eg=Eg[np.argmax(eta)], A=1, n=1)
    #
    # solar_cell = SolarCell([db_junction0], T=298.15, R_series=0)
    #
    # solar_cell_solver(solar_cell, 'iv',
    #                   user_options={'T_ambient': 298.15, 'voltages': V,
    #                                 'light_iv': True,
    #                                 'internal_voltages': np.arange(0, 2, 0.0005),
    #                                 'wavelength': wl_cell * 1e-9,
    #                                 'mpp': True,
    #                                 'light_source': light_for_cells
    #                                 })
    #
    # plt.figure()
    # plt.plot(V, solar_cell.iv.IV[1], 'k', linewidth=4, label='Total')
    # plt.ylim(0, 500)
    # plt.xlim(0, )
    # plt.legend()
    # plt.xlabel('Voltage (V)')
    # plt.ylabel('Current (A/m$^2$)')
    # plt.show()


    return np.max(eta)*100, Eg[np.argmax(eta)], Isc[np.argmax(eta)]/10, Voc[np.argmax(eta)], Eg, eta
