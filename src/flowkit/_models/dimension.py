"""
Module defining the Dimension, RatioDimension, and QuadrantDivider classes
"""


class Dimension(object):
    """
    Represents a single dimension of an array of FCS data.

    :param dimension_id: A string identifying the dimension, typically matching the PnN label of a channel in an
        FCS sample
    :param compensation_ref: A string referencing the ID of a Matrix instance
    :param transformation_ref: A string referencing the ID of an instance of a Transform subclass
    :param range_min: For use in defining the boundaries of a RectangleGate. A float defining the minimum boundary
        for the dimension. If None, the minimum is unbounded.
    :param range_max: For use in defining the boundaries of a RectangleGate. A float defining the maximum boundary
        for the dimension. If None, the maximum is unbounded.
    """
    def __init__(
            self,
            dimension_id,
            compensation_ref='uncompensated',
            transformation_ref=None,
            range_min=None,
            range_max=None
    ):
        # a compensation reference is required, with the default value being
        # the string 'uncompensated' for non-compensated dimensions. Use 'FCS'
        # for specifying the embedded spill in the FCS file. Otherwise, it is a
        # reference to a Matrix in the GatingStrategy
        self.compensation_ref = compensation_ref

        # ID is required
        self.id = dimension_id

        # transformation is optional, but if present must be a string
        if transformation_ref is not None and not isinstance(transformation_ref, str):
            raise TypeError("Transformation reference must be a text string or None")

        self.transformation_ref = transformation_ref

        if range_min is not None:
            self.min = float(range_min)
        else:
            self.min = range_min
        if range_max is not None:
            self.max = float(range_max)
        else:
            self.max = range_max

    def __repr__(self):
        return f'{self.__class__.__name__}(id: {self.id})'


class RatioDimension(object):
    """
    Represents a ratio of two FCS dimensions (specified by a RatioTransform).

    :param ratio_ref: A string referencing the ID of a RatioTransform instance
    :param compensation_ref: A string referencing the ID of a Matrix instance
    :param transformation_ref: A string referencing the ID of an instance of a Transform subclass
    :param range_min: For use in defining the boundaries of a RectangleGate. A float defining the minimum boundary
        for the dimension. If None, the minimum is unbounded.
    :param range_max: For use in defining the boundaries of a RectangleGate. A float defining the maximum boundary
        for the dimension. If None, the maximum is unbounded.
    """
    def __init__(
            self,
            ratio_ref,
            compensation_ref,
            transformation_ref=None,
            range_min=None,
            range_max=None
    ):
        # ratio dimension has no label, but does have a reference to a
        # RatioTransform
        self.ratio_ref = ratio_ref

        # a compensation reference is required, although the value can be
        # the string 'uncompensated' for non-compensated dimensions, or 'FCS'
        # for using the embedded spill in the FCS file. Otherwise, it is a
        # reference to a Matrix in the GatingStrategy
        self.compensation_ref = compensation_ref

        # transformation is optional
        self.transformation_ref = transformation_ref

        if range_min is not None:
            self.min = float(range_min)
        if range_max is not None:
            self.max = float(range_max)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'ratio_reference: {self.ratio_ref})'
        )


class QuadrantDivider(object):
    """
    Represents a divider for a single Dimension, used as part of a QuadrantGate definition.

    :param divider_id: A string identifying the divider
    :param dimension_ref: A string identifying the dimension, typically matching the PnN label
        of a channel in an FCS sample
    :param compensation_ref: A string referencing the ID of a Matrix instance
    :param values: One or more values used for partitioning the given Dimension
    :param transformation_ref: A string referencing the ID of an instance of a Transform subclass
    """
    def __init__(
            self,
            divider_id,
            dimension_ref,
            compensation_ref,
            values,
            transformation_ref=None
    ):
        self.id = divider_id
        self.dimension_ref = dimension_ref

        # a compensation reference is required, although the value can be
        # the string 'uncompensated' for non-compensated dimensions, or 'FCS'
        # for using the embedded spill in the FCS file. Otherwise, it is a
        # reference to a Matrix in the GatingStrategy
        self.compensation_ref = compensation_ref

        # transformation is optional
        self.transformation_ref = transformation_ref

        self.values = values

    def __repr__(self):
        return f'{self.__class__.__name__}(id: {self.id}, dim_ref: {self.dimension_ref})'
