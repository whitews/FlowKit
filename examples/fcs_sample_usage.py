import flowkit as fk
from bokeh.plotting import show

# paths to an FCS file and compensation matrix (saved as a simple CSV file)
fcs_file_path = "data/test_comp_example.fcs"
comp_file_path = "data/comp_complete_example.csv"

# create a Sample instance and give the optional comp matrix
# this file is slightly non-standard with a common off-by-one data offset,
# so we force reading it by setting ignore_offset_error to True.
sample = fk.Sample(
    fcs_path_or_data=fcs_file_path,
    compensation=comp_file_path,
    ignore_offset_error=True  # only needed b/c FCS has off-by-one data offset issue
)

# sub-sample events to 50k for better performance when plotting
# the events are not deleted, and any analysis will be performed on all events.
sample.subsample_events(50000)

# create a LogicleTransform instance (one of many transform types in FlowKit)
xform = fk.transforms.LogicleTransform(
    'logicle',
    param_t=262144,
    param_w=0.5,
    param_m=4.5,
    param_a=0
)

# apply our transform to the sample
# This will apply post-compensation if a comp matrix has already been loaded.
sample.apply_transform(xform)

# create a scatter plot using 2 channels (can use a PnN number or label for selecting channels)
fig = sample.plot_scatter(
    3,
    6,
    source='xform',
    subsample=True
)

show(fig)
