#include <Python.h>
#include <numpy/arrayobject.h>
#include "utils.h"

static PyObject *wrap_calc_wind_count(PyObject *self, PyObject *args) {
    double point_x, point_y;
    int vert_count;
    PyObject *poly_vertices;

    // parse the input args tuple
    if (!PyArg_ParseTuple(args, "ddiO!", &point_x, &point_y, &vert_count, &PyArray_Type, &poly_vertices)) {
        return NULL;
    }

    PyObject *poly_vert_array = PyArray_FROM_OTF(poly_vertices, NPY_DOUBLE, NPY_IN_ARRAY);
    double *poly_vertices_c = (double *) PyArray_DATA(poly_vert_array);

    // now we can call our function!
    int wind_count = calc_wind_count(point_x, point_y, vert_count, poly_vertices_c);

    Py_DECREF(poly_vert_array);

    return Py_BuildValue("i", wind_count);
}

static PyObject *wrap_points_in_polygon(PyObject *self, PyObject *args) {
    PyObject *poly_vertices;
    PyObject *points;
    int vert_count;
    int point_count;

    // parse the input args tuple
    if (!PyArg_ParseTuple(args, "OiOi", &poly_vertices, &vert_count, &points, &point_count)) {
        return NULL;
    }

    PyObject *poly_vert_array = PyArray_FROM_OTF(poly_vertices, NPY_DOUBLE, NPY_IN_ARRAY);
    double *poly_vertices_c = (double *) PyArray_DATA(poly_vert_array);

    PyObject *points_array = PyArray_FROM_OTF(points, NPY_DOUBLE, NPY_IN_ARRAY);
    double *points_c = (double *) PyArray_DATA(points_array);

    // now we can call our function!
    int *is_in_polygon = malloc(point_count * sizeof(int));

    is_in_polygon = points_in_polygon(poly_vertices_c, vert_count, points_c, point_count);

    Py_DECREF(poly_vert_array);
    Py_DECREF(points_array);

    long int dims[1];
    dims[0] = point_count;

    return PyArray_SimpleNewFromData(1, dims, NPY_INT32, is_in_polygon);
}

static PyMethodDef module_methods[] = {
    {"calc_wind_count", wrap_calc_wind_count, METH_VARARGS, NULL},
    {"points_in_polygon", wrap_points_in_polygon, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef utilsdef = {
        PyModuleDef_HEAD_INIT,
        "utils_c",
        NULL,
        -1,
        module_methods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_utils_c(void) {
    PyObject *m = PyModule_Create(&utilsdef);
#else
PyMODINIT_FUNC initutils_c(void) {
    PyObject *m = Py_InitModule3("utils_c", module_methods, NULL);
#endif

    if (m == NULL) {
        return NULL;
    }
    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}