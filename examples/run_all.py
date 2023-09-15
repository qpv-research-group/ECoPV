import subprocess

import subprocess

# program_list = ['gamut_luminance_colors.py',
#                 'gamut_luminance_fixedEg_ERE.py',
#                 'ESI_Si_based_tandem.py',
#                 'GaInP_GaAs_Ge_3J.py',
#                 'optimize_across_luminance.py']

program_list = ['optimize_across_luminance.py',
                'optimize_all_BB.py',
]

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
