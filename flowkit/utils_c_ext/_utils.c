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

    //double poly_vertices_c = *(double *) poly_vertices->data;
//    double poly_vertices_c[vert_count][2];
//    memcpy(poly_vertices_c, &poly_vertices, vert_count);

    PyObject *poly_vert_array = PyArray_FROM_OTF(poly_vertices, NPY_DOUBLE, NPY_IN_ARRAY);
    double *poly_vertices_c = (double *) PyArray_DATA(poly_vert_array);

    // now we can call our function!
    int wind_count = calc_wind_count(point_x, point_y, vert_count, poly_vertices_c);

    Py_DECREF(poly_vert_array);

    return Py_BuildValue("i", wind_count);
}

static PyMethodDef module_methods[] = {
    {"calc_wind_count", wrap_calc_wind_count, METH_VARARGS, NULL},
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