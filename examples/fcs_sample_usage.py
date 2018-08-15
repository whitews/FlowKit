from flowkit import Sample
import os

fcs_dir = "/media/sf_vbox_share/cliburn_projects/eqapol_2018-07/EQAPOL_normal"

#fcs_file_path = "test_comp_example.fcs"
#comp_file_path = "comp_complete_example.csv"

#fcs_file_path = "test_data_2d_01.fcs"

fcs_file_path = os.path.join(fcs_dir, "AMJ_5L_CMV pp65.fcs")
comp_file_path = os.path.join(fcs_dir, "CompMatrixDenny06Nov09")

sample = Sample(
    fcs_path_or_data=fcs_file_path,
    compensation=comp_file_path,
    subsample_count=None
)
# sample.apply_logicle_transform()
sample.apply_asinh_transform()
sample.plot_scatter(
    18,
    3,
    y_min=60000,
    y_max=70000,
    source='xform',
    subsample=True,
    contours=False
)