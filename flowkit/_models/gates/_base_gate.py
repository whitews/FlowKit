"""
Module for the Gate abstract base class
"""
from abc import ABC, abstractmethod


class Gate(ABC):
    """
    Represents a single flow cytometry gate
    """
    def __init__(
            self,
            gate_name,
            dimensions
    ):
        self.gate_name = gate_name
        if dimensions is None:
            self.dimensions = []
        else:
            self.dimensions = dimensions
        self.gate_type = None

    def get_dimension(self, dim_id):
        """
        Retrieve the Dimension instance given the dimension ID

        :param dim_id: Dimension ID
        :return: Dimension instance
        """
        for dim in self.dimensions:
            if dim_id == dim.id:
                return dim

    def get_dimension_ids(self):
        """
        Retrieve all gate Dimension IDs in order

        :return: list of Dimension ID strings
        """
        return [dim.id for dim in self.dimensions]

    @abstractmethod
    def apply(self, df_events):
        """
        Abstract method to apply gate to given DataFrame of events
        :param df_events: pandas DataFrame containing event data
        :return:
        """
        pass
