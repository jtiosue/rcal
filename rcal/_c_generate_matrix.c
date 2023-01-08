// Copyright 2023 Joseph T. Iosue

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Python.h"


/*This file creates the ``rcal._c_generate_matrix`` module with the ``c_generate_matrix`` function.
*/


PyObject* two_tuple(PyObject* one, PyObject* two) {
    return Py_BuildValue("OO", one, two);
}

long alpha_index(PyObject* indices, PyObject* p) {
    PyObject* t = two_tuple(PyUnicode_FromString("alpha"), p);
    return PyLong_AsLong(PyDict_GetItem(indices, t));
}

long a_index(PyObject* indices, PyObject* r) {
    PyObject* t = two_tuple(PyUnicode_FromString("a"), r);
    return PyLong_AsLong(PyDict_GetItem(indices, t));
}

long b_index(PyObject* indices, PyObject* r) {
    PyObject* t = two_tuple(PyUnicode_FromString("b"), r);
    return PyLong_AsLong(PyDict_GetItem(indices, t));
}

double get_rating(PyObject* data, PyObject* r, PyObject* p, PyObject* d) {
    PyObject* t = Py_BuildValue("OOO", r, p, d);
    return PyFloat_AsDouble(PyDict_GetItem(data, t));
}


// Module info
static char _c_generate_matrix_name[] = "_c_generate_matrix";

static char _c_generate_matrix_docstring[] =
    "``rcal._c_generate_matrix`` module for generating the calibration matrix";


// Define module functions; wrap the source code.

static char c_generate_matrix_docstring[] =
    "c_generate_matrix.\n\n"
    "See the docstring of the rcal.py_generate_matrix function.\n"
    "The input and output for this function are the same as for that one.\n";


static PyObject* c_generate_matrix(PyObject* self, PyObject* args) {
    /*
    This is the function that we call from python with
    ``rcal._c_generate_matrix.c_generate_matrix``. See the docstring above
    for details on what ``args`` should be.
    (data, indices, rating_delta, lam)

    indices is a dictionary mapping parameters to unique integer indices. keys are
    ('a', r), ('b', r), and ('alpha', p), for reviewers r and people p.
    It must be that indices[('a', r)] takes values 0 through num_reviewers - 1,
    indices[('b', r)] takes values num_reviewers through 2 num_reviewers - 1,
    and indices[('alpha', p)] takes values 2 num_reviewers and above.
    */
    PyObject *data, *indices, *py_rating_delta, *py_lam;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &data, &indices, &py_rating_delta, &py_lam)) {
        return NULL;
    }


    double rating_delta = PyFloat_AsDouble(py_rating_delta);
    double lam = PyFloat_AsDouble(py_lam);

    PyObject* R = PySet_New(PyList_New(0));
    PyObject* P = PyDict_New();

    PyObject* keys = PyDict_Keys(data);
    long num_keys = (long)PyList_Size(keys);
    PyObject *key;

    // Create R and P. R is a set of reviewers.
    // P is a dictionary mapping people to list of tuples (r, d)
    for (long i=0; i<num_keys; i++){
        key = PyList_GetItem(keys, i);
        PySet_Add(R, PyTuple_GetItem(key, 0));

        PyList_Append(
            PyDict_SetDefault(P, PyTuple_GetItem(key, 1), PyList_New(0)), 
            two_tuple(
                PyTuple_GetItem(key, 0),
                PyTuple_GetItem(key, 2)   
            )
        );
    }

    long num_reviewers = (long)PySet_Size(R);

    // Figure out N
    PyObject* P_keys = PyDict_Keys(P);
    long num_P = (long)PyList_Size(P_keys);
    double N = 0.;
    double size;
    for (long i=0; i<num_P; i++) {
        size = (double)PyList_Size(PyDict_GetItem(P, PyList_GetItem(P_keys, i)));
        N += size * size;
    }

    // Make the A matrix and C vector all zeros.
    long num_params = (long)PyDict_Size(indices);
    double *C = (double*)malloc(num_params * sizeof(double));
    double **A = (double**)malloc(num_params * sizeof(double*));
    for(long i=0; i<num_params; i++) {
        C[i] = 0.;
        A[i] = (double*)malloc(num_params * sizeof(double));
        for(long j=0; j<num_params; j++) {
            A[i][j] = 0.;
        }
    }


    // derivatives wrt ar
    PyObject* r;
    long ar;
    double val = (2. * lam / ((double)num_reviewers)) * rating_delta;
    for(long i=0; i<num_reviewers; i++) {
        r = PySet_Pop(R);
        ar = a_index(indices, r);
        A[ar][ar] += val * rating_delta;
        C[ar] += val;
    }


    PyObject *p, *r1, *d1, *r2, *d2, *t, *l;
    double deltad, rating1, rating2;
    long lsize, alphap, ar1, ar2, br1, br2;
    for(long i=0; i<num_P; i++) {

        p = PyList_GetItem(P_keys, i);
        alphap = alpha_index(indices, p);
        l = PyDict_GetItem(P, p);
        lsize = (long)PyList_Size(l);

        for(long j=0; j<lsize; j++) {

            t = PyList_GetItem(l, j);
            r1 = PyTuple_GetItem(t, 0);
            d1 = PyTuple_GetItem(t, 1);
            ar1 = a_index(indices, r1);
            br1 = b_index(indices, r1);
            rating1 = get_rating(data, r1, p, d1);

            for(long k=0; k<lsize; k++) {

                t = PyList_GetItem(l, k);
                r2 = PyTuple_GetItem(t, 0);
                d2 = PyTuple_GetItem(t, 1);
                ar2 = a_index(indices, r2);
                br2 = b_index(indices, r2);
                rating2 = get_rating(data, r2, p, d2);

                deltad = PyFloat_AsDouble(d2) - PyFloat_AsDouble(d1);

                // derivatives wrt alphap
                A[alphap][ar1] += (2. / N) * deltad * rating1;
                A[alphap][ar2] -= (2. / N) * deltad * rating2;
                A[alphap][br1] += (2. / N) * deltad;
                A[alphap][br2] -= (2. / N) * deltad;
                A[alphap][alphap] += (2. / N) * deltad * deltad;

                // derivative wrt ar
                A[ar1][ar1] += (2. / N) * rating1 * rating1;
                A[ar2][ar1] -= (2. / N) * rating2 * rating1;
                A[ar1][ar2] -= (2. / N) * rating1 * rating2;
                A[ar2][ar2] += (2. / N) * rating2 * rating2;

                A[ar1][br1] += (2. / N) * rating1;
                A[ar2][br1] -= (2. / N) * rating2;
                A[ar1][br2] -= (2. / N) * rating1;
                A[ar2][br2] += (2. / N) * rating2;

                A[ar1][alphap] += (2. / N) * rating1 * deltad;
                A[ar2][alphap] -= (2. / N) * rating2 * deltad;


                // derivatives wrt br
                A[br1][ar1] += (2. / N) * rating1;
                A[br2][ar1] -= (2. / N) * rating1;
                A[br1][ar2] -= (2. / N) * rating2;
                A[br2][ar2] += (2. / N) * rating2;

                A[br1][br1] += (2. / N);
                A[br2][br1] -= (2. / N);
                A[br1][br2] -= (2. / N);
                A[br2][br2] += (2. / N);

                A[br1][alphap] += (2. / N) * deltad;
                A[br2][alphap] -= (2. / N) * deltad;

            }
        }
    }



    // get rid of singular behavior of A wrt b
    // remove equation for b0 and replace with the condition hat
    // the first b is zero
    for(long i=0; i<num_params; i++) {
        A[num_reviewers][i] = 0.;
    }
    A[num_reviewers][num_reviewers] = 1.;

    // get rid of singular behavior of A wrt to alpha
    // if a person only has one day of rating
    long row; PyObject *set;
    for(long i=0; i<num_P; i++) {
        p = PyList_GetItem(P_keys, i);
        l = PyDict_GetItem(P, p);
        set = PySet_New(PyList_New(0));
        for(long j=0; j<(long)PyList_Size(l); j++) {
            PySet_Add(set, PyTuple_GetItem(PyList_GetItem(l, j), 1));
        }
        if((long)PySet_Size(set) == 1) {
            row = alpha_index(indices, p);
            A[row][row] = 1.;
        }

    }


    // make python lists a and c from A, C
    PyObject* a = PyList_New(num_params);
    PyObject* c = PyList_New(num_params);
    for(long row=0; row<num_params; row++) {
        PyList_SetItem(c, row, PyFloat_FromDouble(C[row]));
        l = PyList_New(num_params);
        for(long col=0; col<num_params; col++) {
            PyList_SetItem(l, col, PyFloat_FromDouble(A[row][col]));
        }
        PyList_SetItem(a, row, l);
    }

    // free memory
    for(long i=0; i<num_params; i++) {
        free(A[i]);
    }
    free(A); free(C);

    return two_tuple(a, c);
}



// Create the module.

static PyMethodDef CGenerateMatrixMethods[] = {
    {
        "c_generate_matrix",
        c_generate_matrix,
        METH_VARARGS,
        c_generate_matrix_docstring
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef CGenerateMatrixModule = {
    PyModuleDef_HEAD_INIT,
    _c_generate_matrix_name,  // name of module
    _c_generate_matrix_docstring,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module,
         // or -1 if the module keeps state in global variables.
    CGenerateMatrixMethods
};


PyMODINIT_FUNC PyInit__c_generate_matrix(void) {
    // MUST BE PyInit_modulename.
    return PyModule_Create(&CGenerateMatrixModule);
}
