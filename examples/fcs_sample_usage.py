import flowkit as fk
from bokeh.plotting import show

fcs_file_path = "test_comp_example.fcs"
comp_file_path = "comp_complete_example.csv"

sample = fk.Sample(
    fcs_path_or_data=fcs_file_path,
    compensation=comp_file_path,
    subsample_count=50000,
    filter_negative_scatter=True,
    filter_anomalous_events=False
)

xform = fk.transforms.LogicleTransform(
    'logicle',
    param_t=262144,
    param_w=0.5,
    param_m=4.5,
    param_a=0
)
sample.apply_transform(xform)

fig = sample.plot_scatter(
    3,
    6,
    source='xform',
    subsample=True
)

show(fig)
