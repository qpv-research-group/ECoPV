import subprocess

import subprocess

program_list = ['gamut_luminance_colors.py',
                'gamut_luminance_fixedEg_ERE.py',
                'run_cell_color_optim_loop_fixEg_ERE.py',
                'run_cell_color_optim_loop_fixEg_3J.py',
                'run_cell_color_optim_loop_luminance.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
