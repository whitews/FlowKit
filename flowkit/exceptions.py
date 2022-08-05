"""
flowkit.exceptions
~~~~~~~~~~~~~~~~~~
This module contains the set of FlowKit exceptions.
"""


class GateTreeError(Exception):
    """An error modifying a GatingStrategy gate tree occurred."""
    pass


class GateReferenceError(Exception):
    """An error referencing a Gate instance occurred."""
    pass


class QuadrantReferenceError(Exception):
    """An error when requesting a single Quadrant as a gate in a GatingStrategy."""
    pass


class FlowJoWSPParsingError(Exception):
    """An error parsing a FlowJo .wsp file"""
    pass
