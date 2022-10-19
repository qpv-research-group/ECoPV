import optimize_pipeline as op
import SQlimit as SQl
import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from time import time

# color_lab = ([37.9900000000000,13.5600000000000,14.0600000000000],
# [65.7100000000000,18.1300000000000,17.8100000000000],
# [49.9300000000000,-4.88000000000000,-21.9300000000000],
# [43.1400000000000,-13.1000000000000,21.9100000000000],
# [55.1100000000000,8.84000000000000,-25.4000000000000],
# [70.7200000000000,-33.4000000000000,-0.199000000000000],
# [62.6600000000000,36.0700000000000,57.1000000000000],
# [40.0200000000000,10.4100000000000,-45.9600000000000],
# [51.1200000000000,48.2400000000000,16.2500000000000],
# [30.3300000000000,22.9800000000000,-21.5900000000000],
# [72.5300000000000,-23.7100000000000,57.2600000000000],
# [71.9400000000000,19.3600000000000,67.8600000000000],
# [28.7800000000000,14.1800000000000,-50.3000000000000],
# [55.2600000000000,-38.3400000000000,31.3700000000000],
# [42.1000000000000,53.3800000000000,28.1900000000000],
# [81.7300000000000,4.04000000000000,79.8200000000000],
# [51.9400000000000,49.9900000000000,-14.5700000000000],
# [51.0400000000000,-28.6300000000000,-28.6400000000000],
# [96.5400000000000,-0.425000000000000,1.18600000000000],
# [81.2600000000000,-0.638000000000000,-0.335000000000000],
# [66.7700000000000,-0.734000000000000,-0.504000000000000],
# [50.8700000000000,-0.153000000000000,-0.270000000000000],
# [35.6600000000000,-0.421000000000000,-1.23100000000000],
# [20.4600000000000,-0.0790000000000000,-0.973000000000000])

# color_names =("DarkSkin",
#     "LightSkin",
#     "BlueSky",
#     "Foliage",
#     "BlueFlower",
#     "BluishGreen",
#     "Orange",
#     "PurplishBlue",
#     "ModerateRed",
#     "Purple",
#     "YellowGreen",
#     "OrangeYellow",
#     "Blue",
#     "Green",
#     "Red",
#     "Yellow",
#     "Magenta",
#     "Cyan",
#     "White-9-5",
#     "Neutral-8",
#     "Neutral-6-5",
#     "Neutral-5",
#     "Neutral-3-5",
#     "Black-2"
#     )

if __name__ == '__main__':
    start = time()

    color_lab = ([55.2600000000000,-38.3400000000000,31.3700000000000],
                 #[20.4600000000000,-0.0790000000000000,-0.973000000000000]
                    [62.6600000000000,36.0700000000000,57.1000000000000],
    [51.9400000000000,49.9900000000000,-14.5700000000000]
                 )

    color_names =("Green",
                  #"Black-2"
                  "Orange",
                  "Magenta"
                  )

    Eg = [1.12,1.68]

    #%%
    color_d = [] #color data spectrum, center
    for i in range(len(color_lab)):
        print('Color names: ',color_names[i])
        color_d.append(op.Optimize(color_lab[i],color_names[i])) # CALLING OPTIMIZE_PIPELINE

    total_list = []
    wcde = {}

    #%%



    print(time()-start)

    '''
    color_d[0]#which color name set tu_cp[0] 0: dp1, 1: dp2, 2: g1, 3: g2
    Next entry is delta_E which is a dictionary key to a list of color data 
    1d cases[0] 0: center 1: width,2: peak 3:spectrum  [center, width, peak, spectrum,lab,filepath]
    For 2D cases 0:center 1:width 2:center2 3:width2 4:spectrum [center, width, center2, width2,spectrum,lab,filepath]
    res = list(test_dict.keys())[0]
      
    # printing initial key
    print("The first key of dictionary is : " + str(res))
    '''
    efficiency = {}


    #TRYING TO MAKE DATA STRUCTURE FOR NESTED DICTIONARIES MORE CLEAR
    # >> d = {}
    # d['dict1'] = {}
    # >>> d['dict1']['innerkey'] = 'value'
    # >>> d['dict1']['innerkey2'] = 'value2'
    # >>> d
    # {'dict1': {'innerkey': 'value', 'innerkey2': 'value2'}


    # for c in range(len(color_lab)): #for each color
    #     efficiency[color_names[c]] = []
    #     print('color name is ',color_names[c])
    #     efficiency[color_names[c]].append(color_lab[c]) # target color
    #
    #     s_type = {}
    #     #Access member function here to store JCE data
    #     #color_name_d = color_d[c] #which color data we are using, ex. blue, green, dark skin
    #     for s in range(len(color_d[c].tu_cp)): #For each spectrum generation type
    #         s_gen = color_d[c].tu_cp[s]# which way the spectrum is being generated, d1, d2, g1, g2
    #         Stype = '' # spectrum type
    #         if s==0:
    #             Stype = 'd1_'
    #         elif s==1:
    #             Stype = 'd2_'
    #         elif s==2:
    #             Stype = 'g1_'
    #         elif s==3:
    #             Stype = 'g2'
    #
    #         s_gen_key = list(s_gen) # makes list of keys. Each key representing a dE value
    #
    #         s_type[Stype] = []
    #         for dE in s_gen_key: # for each delta E stored below a tolerance
    #             if len(s_gen[dE]) == 6:
    #                 spectrum = 1-s_gen[dE][3]
    #                 LAB_produced= s_gen[dE][4] #handles 1Dip,1Gauss/2Dip, 2Gauss arrays different sizes
    #             elif len(s_gen[dE]) == 7: # Which LAB IS PRODUCED ex. d1
    #                 spectrum = 1-s_gen[dE][4]
    #                 LAB_produced = s_gen[dE][5]#sGen for specific deltaE
    #             # wl_color_spec = np.arange(380,780.02,0.02)
    #             # plt.plot(wl_color_spec,spectrum)
    #             print('dE: ',dE)
    #
    #             figure_title = str(color_names[c])+'_'+Stype+'_'+str(dE).replace('.','_')
    #             SQ1 = SQl.SQlim(spectrum,Eg, figure_title) # EFFICIENCY
    #
    #             E_subcell, PCEsubcell, eff, eff_CM = SQ1.available_E(Eg) # storing deltaE key as efficiency
    #
    #             s_type[Stype].append([dE,eff,eff_CM,LAB_produced])
    #
    #             converted_list = [str(element) for element in Eg]
    #             Eg_String= "_".join(converted_list)
    #
    #             path = s_gen[dE][-1]
    #             data = pd.read_excel(path, header=None)
    #             row_pos = 1
    #             col_pos =0
    #             data.loc[row_pos, col_pos] = eff
    #             row_pos = 1
    #             col_pos =4
    #             data.loc[row_pos, col_pos] = Eg_String
    #             row_pos = 1
    #             col_pos =5
    #             data.loc[row_pos, col_pos] = eff_CM
    #             data.to_excel(path)
    #              #nested dictionary to store effieciency data for each
    #     efficiency[color_names[c]].append(s_type)


    # #%%

    # #Run through dictionary to compute maximum values

    # #RUN THROUGH CLASS HERE SINCE DATA IS STILL ALL STORED OUT here


    # c = list(efficiency)
    # print(c)
    # for color in  c:
    #    print('color is ',c)
    #    print('Target color is',efficiency[color][0])
    #    print('List of D1,D2,G1,G2 dictionaries ', efficiency[color][1])
    #    skey = list(efficiency[color][1])
    #    max_value = []
    #    for st in skey: #spectrum type st in skey
    #    #efficiency[which_color][spec_type]

    #        print('For Spectrum ',st)
    #        print('LENGTH IS ',len(efficiency[color][1][st]))
    #        for i in range(len(efficiency[color][1][st])):
    #            mv = [efficiency[color][1][st][i][0],efficiency[color][1][st][i][1],efficiency[color][1][st][i][2]]
    #            max_value.append(mv)
    #            #print('whole data is ',efficiency[color][1][st]) # USEFUL EXAMPLE OF HOW TO ACCESS THIS DATA
    #            print('dE is ',efficiency[color][1][st][i][0],\
    #                  'eff is ',efficiency[color][1][st][i][1],\
    #                  'color produced is ',efficiency[color][1][st][i][2])
    #    #print('max value is ',max(max_value))
    #        i_mv = [] # index of max value
    #        max_indices = [x[1] for x in max_value]

    #        if len(max_indices)==0:
    #            print("list is empty")
    #        elif len((max_indices))!=0:
    #            a = max(max_indices)
    #            i_mv.append(max_indices.index(a)) # IN CASE MULTIPLE COLORS RETURN THE SAME EFF VALUE

    #        #max_value = np.asarray(max_value)
    #        #print(max_value)
    #            print('max value is   ',a,'for ',st,'with dE of ',max_value[i_mv[0]][0],\
    #                  ' and color generated ',max_value[i_mv[0]][2],'compared to target color ',efficiency[color][0])
    #        #print(i_mv)
    #        #print(max_value[i_mv[0]])

