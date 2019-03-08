import numpy as np
from flowkit import gml_utils


class Matrix(object):
    def __init__(
        self,
        matrix_element,
        xform_namespace,
        data_type_namespace
    ):
        self.id = gml_utils.find_attribute_value(matrix_element, xform_namespace, 'id')
        self.fluorochomes = []
        self.detectors = []
        self.matrix = []

        fluoro_el = matrix_element.find(
            '%s:fluorochromes' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        fcs_dim_els = fluoro_el.findall(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=matrix_element.nsmap
        )

        for dim_el in fcs_dim_els:
            label = gml_utils.find_attribute_value(dim_el, data_type_namespace, 'name')

            if label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            self.fluorochomes.append(label)

        detectors_el = matrix_element.find(
            '%s:detectors' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        fcs_dim_els = detectors_el.findall(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=matrix_element.nsmap
        )

        for dim_el in fcs_dim_els:
            label = gml_utils.find_attribute_value(dim_el, data_type_namespace, 'name')

            if label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            self.detectors.append(label)

        spectrum_els = matrix_element.findall(
            '%s:spectrum' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        for spectrum_el in spectrum_els:
            matrix_row = []

            coefficient_els = spectrum_el.findall(
                '%s:coefficient' % xform_namespace,
                namespaces=matrix_element.nsmap
            )

            for co_el in coefficient_els:
                value = gml_utils.find_attribute_value(co_el, xform_namespace, 'value')
                if value is None:
                    raise ValueError(
                        'Matrix coefficient must have only 1 value (line %d)' % co_el.sourceline
                    )

                matrix_row.append(float(value))

            self.matrix.append(matrix_row)

        self.matrix = np.array(self.matrix)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, dims: {len(self.fluorochomes)})'
        )

    def apply(self, sample):
        pass
