"""
FCS keyword strings
"""

# FCS 3.1 reserves certain keywords as being part of the FCS standard. Some
# of these are required, and others are optional. However, all of these
# keywords shall be prefixed by the '$' character. No other keywords shall
# begin with the '$' character. All keywords are case-insensitive, however
# most cytometers use all uppercase for keyword strings. FlowKit follows
# the convention used in FlowIO and internally stores and references all
# FCS keywords as lowercase for more convenient typing by developers.
# noinspection SpellCheckingInspection
FCS_STANDARD_KEYWORDS = [
    'beginanalysis',
    'begindata',
    'beginstext',
    'byteord',
    'datatype',
    'endanalysis',
    'enddata',
    'endstext',
    'mode',
    'nextdata',
    'par',
    'tot',
    # start optional standard keywords
    'abrt',
    'btim',
    'cells',
    'com',
    'csmode',
    'csvbits',
    'cyt',
    'cytsn',
    'date',
    'etim',
    'exp',
    'fil',
    'gate',
    'inst',
    'last_modified',
    'last_modifier',
    'lost',
    'op',
    'originality',
    'plateid',
    'platename',
    'proj',
    'smno',
    'spillover',
    'src',
    'sys',
    'timestep',
    'tr',
    'vol',
    'wellid'
]
