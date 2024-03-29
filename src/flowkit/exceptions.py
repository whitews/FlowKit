"""
flowkit.exceptions
~~~~~~~~~~~~~~~~~~
This module contains the set of FlowKit exceptions.
"""
# import FlowIO exceptions
from flowio.exceptions import FCSParsingError, DataOffsetDiscrepancyError  # noqa


class FlowKitWarning(Warning):
    """A generic FlowKit warning"""
    pass


class FlowKitException(Exception):
    """A generic FlowKit exception"""
    pass


class GateTreeError(FlowKitException):
    """An error modifying a GatingStrategy gate tree occurred."""
    pass


class GateReferenceError(FlowKitException):
    """An error referencing a Gate instance occurred."""
    pass


class QuadrantReferenceError(FlowKitException):
    """An error when requesting a single Quadrant as a gate in a GatingStrategy."""
    pass


class FlowJoWSPParsingError(FlowKitException):
    """An error parsing a FlowJo .wsp file"""
    pass


class FlowJoWSPParsingWarning(FlowKitWarning):
    """A warning when parsing a FlowJo .wsp file"""
    pass
