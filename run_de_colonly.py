import optim_de_onlycol as op
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, xyYColor, XYZColor
import SQlimit as SQl
import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from joblib import Parallel, delayed

calc = False

def convert_xyY_to_Lab(xyY_list):

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()

def convert_xyY_to_XYZ(xyY_list):
    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, XYZColor)
    return lab.get_value_tuple()

def internal_run(i):

    loop_res = op.Optimize(color_lab[i], color_names[i])
    # eff_result[i, :4] = loop_res.params
    # eff_result[i, 4] = loop_res.final_dE
    # eff_result[i, 5] = loop_res.final_n

    return np.hstack((loop_res.params, loop_res.final_dE, loop_res.final_n))


single_J_result = pd.read_csv("paper_colours.csv")

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

color_lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

color_names =(
    "DarkSkin",
    "LightSkin",
    "BlueSky",
    "Foliage",
    "BlueFlower",
    "BluishGreen",
    "Orange",
    "PurplishBlue",
    "ModerateRed",
    "Purple",
    "YellowGreen",
    "OrangeYellow",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Magenta",
    "Cyan",
    "White-9-5",
    "Neutral-8",
    "Neutral-6-5",
    "Neutral-5",
    "Neutral-3-5",
    "Black-2"
    )

# if __name__ == '__main__':
start = time()

# color_lab = ([55.2600000000000,-38.3400000000000,31.3700000000000],
#              #[20.4600000000000,-0.0790000000000000,-0.973000000000000]
#                 [62.6600000000000,36.0700000000000,57.1000000000000],
# [51.9400000000000,49.9900000000000,-14.5700000000000]
#              )
#
# color_names =("Green",
#               #"Black-2"
#               "Orange",
#               "Magenta"
#               )

#Eg = [1.12,1.68]
n_trials = 20
eff_result_all = np.empty((n_trials, len(color_names), 6))

for j1 in range(n_trials):

    if calc:

        #eff_result = np.empty((len(color_names), 6))

        eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
                                             (i1) for i1 in range(len(color_lab)))

        eff_result_all[j1] = np.vstack(eff_result_par)

        np.savetxt("fixed_peak_result_peaklocs_" + str(j1) + ".txt", eff_result_all[j1])
        # eff_result = pd.DataFrame(data=eff_result)
        #
        # eff_result.to_csv("fixed_peak_result_peaklocs_" + str(j1) + ".csv")



    else:

        eff_result = np.loadtxt("fixed_peak_result_peaklocs_" + str(j1) + ".txt")
        eff_result_all[j1] = eff_result
    # # print(eff_result)

print('TIME TAKEN:', time()-start)

plt.figure()
plt.plot(color_names, eff_result_all[:,:,0].T, 'o')
plt.show()

plt.figure()
plt.plot(color_names, eff_result_all[:,:,1].T, 'o')
plt.show()

plt.figure()
plt.plot(color_names, eff_result_all[:,:,2].T, 'o')
plt.show()

plt.figure()
plt.plot(color_names, eff_result_all[:,:,3].T, 'o')
plt.show()

plt.figure()
plt.plot(color_names, eff_result_all[:,:,4].T, 'o')
plt.show()

plt.figure()
plt.plot(color_names, eff_result_all[:,:,5].T, 'o')
plt.ylim(-1700, -700)
plt.show()
#
#
# # Position of bars on x-axis
# ind = np.arange(len(color_names))
# plt.figure(figsize=(10,5))
# width = 0.3
#
# plt.bar(ind, single_J_result['Eg'] , width, label='Paper result')
# plt.bar(ind + width, eff_result['Eg'], width, label='Python DE')
# plt.ylabel(r'$E_g$')
# #plt.title('Here goes title of the plot')
#
# plt.xticks(ind + width / 2, color_names, rotation=45)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
#
# plt.figure(figsize=(10,5))
# width = 0.3
#
# plt.bar(ind, single_J_result['eta'] , width, label='Paper result')
# plt.bar(ind + width, eff_result['D2'], width, label='Python DE')
# plt.ylabel(r'$Eff$')
# #plt.title('Here goes title of the plot')
#
# plt.xticks(ind + width / 2, color_names, rotation=45)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10,5))
# width = 0.3
#
# plt.bar(ind, single_J_result['Voc'] , width, label='Paper result')
# plt.bar(ind + width, eff_result['Voc'], width, label='Python DE')
# plt.ylabel(r'$V_{oc}$')
# #plt.title('Here goes title of the plot')
#
# plt.xticks(ind + width / 2, color_names, rotation=45)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10,5))
# width = 0.3
#
# plt.bar(ind, single_J_result['Jsc'] , width, label='Paper result')
# plt.bar(ind + width, eff_result['Jsc'], width, label='Python DE')
# plt.ylabel(r'$J_{sc}$')
# #plt.title('Here goes title of the plot')
#
# plt.xticks(ind + width / 2, color_names, rotation=45)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


# efficiency = {}


# TRYING TO MAKE DATA STRUCTURE FOR NESTED DICTIONARIES MORE CLEAR
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

