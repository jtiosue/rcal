EDIT!!! TO DO!!


#include "Python.h"


/*This file creates the ``rcal._c_code`` module with the ``c_generate_matrix`` function.
*/

// Module info
static char _c_code_name[] = "_c_code";

static char _c_code_docstring[] =
    "``rcal._c_code`` module for generating the calibration matrix";


// Define module functions; wrap the source code.

static char c_generate_matrix_docstring[] =
    "c_generate_matrix.\n\n"
    "            \n";


static PyObject* c_generate_matrix(PyObject* self, PyObject* args) {
    /*
    This is the function that we call from python with
    ``rcal._c_code.c_generate_matrix``. See the docstring above
    for details on what ``args`` should be.
    */
    
    return ;
}


// Create the module.

static PyMethodDef CCodeMethods[] = {
    {
        "c_generate_matrix",
        c_generate_matrix,
        METH_VARARGS,
        c_generate_matrix_docstring
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef CCodeModule = {
    PyModuleDef_HEAD_INIT,
    _c_code_name,  // name of module
    _c_code_docstring,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module,
         // or -1 if the module keeps state in global variables.
    CCodeMethods
};


PyMODINIT_FUNC PyInit__c_code(void) {
    // MUST BE PyInit_modulename.
    return PyModule_Create(&CCodeModule);
}
