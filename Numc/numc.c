#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initialization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}


/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 * You dont have to use PyArg_UnpackTuple, but you need to make sure that args is indeed of matrix type
 * before you cast it.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    // Exception Handling
    if (NULL == args || NULL == self) {
       PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurrs in Matrix61c_add.");
       return NULL;
    } else if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Not both a and b are of type numc.Matrix.");
        return NULL;
    } else if (((Matrix61c *)args)->mat->rows != self->mat->rows
            || ((Matrix61c *)args)->mat->cols != self->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "mat1 and mat2 do not have the same dimensions.");
        return NULL;
    }

    //create a new numc.Matric object
    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(self->mat->rows),
                                    PyLong_FromLong(self->mat->cols));
    //adding
    add_matrix(result->mat, self->mat,
                ((Matrix61c *)args)->mat);
    return (PyObject *)result;
}


/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    // Exception Handling
    if (NULL == args || NULL == self) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurrs in Matrix61c_sub.");
        return NULL;
    } else if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Not both a and b are of type numc.Matrix.");
        return NULL;
    } else if (((Matrix61c *)args)->mat->rows != self->mat->rows
              || ((Matrix61c *)args)->mat->cols != self->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "mat1 and mat2 do not have the same dimensions.");
        return NULL;
    }

    //create a new numc.Matric object
    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(self->mat->rows),
                                    PyLong_FromLong(self->mat->cols));
    //Substracting
    sub_matrix(result->mat, self->mat,
                ((Matrix61c *)args)->mat);
    return (PyObject *)result;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    //Exception Handling
    if (NULL == args || NULL == self){
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurrs in Matrix61c_multiply.");
        return NULL;
    } else if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Not both a and b are of type numc.Matrix");
        return NULL;
    } else if (self->mat->cols != ((Matrix61c *) args)->mat->rows) {
        PyErr_SetString(PyExc_ValueError, "mat1's number of columns is not equal to mat2's number of rows.");
        return NULL;
    }

    //create a new numc.Matric
    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, ((Matrix61c *) args)->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(new_mat->rows),
                                    PyLong_FromLong(new_mat->cols));

    //multiplying
    mul_matrix(&(result->mat), self->mat,
                ((Matrix61c *) args)->mat);
    return (PyObject *) result;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    if (NULL == self){
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurrs in Matrix61c_neg.");
        return NULL;
    }
    //create a new numc.Matrix object
    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(new_mat->rows),
                                    PyLong_FromLong(new_mat->cols));
    //negating
    neg_matrix(result->mat, self->mat);
    return (PyObject *)result;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    if (NULL == self){
        PyErr_SetString(PyExc_RuntimeError, "Runtime error ocurrs in Matrix61c_abs.");
        return NULL;
    }
    //create a new numc.Matrix object
    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(self->mat->rows),
                                    PyLong_FromLong(self->mat->cols));
    //absoluting
    abs_matrix(result->mat, self->mat);
    return (PyObject *)result;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    // exception handling
    if (NULL == pow) {
       PyErr_SetString(PyExc_RuntimeError, "Runtime error occurs in Matrix61c_pow");
       return NULL;
    } else if (!PyLong_Check(pow)) {
        PyErr_SetString(PyExc_TypeError, "pow is not of the type int.");
        return NULL;
    } else if (self->mat->rows != self->mat->cols || PyLong_AsLong(pow) < 0) {
        PyErr_SetString(PyExc_ValueError, "The matrix is not a square matrix or pow is negative.");
        return NULL;
    }

    matrix *new_mat;
    allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    pow_matrix(&new_mat, self->mat, PyLong_AsLong(pow));
    result->mat = new_mat;
    result->shape = PyTuple_Pack(2, PyLong_FromLong(self->mat->rows),
                                    PyLong_FromLong(self->mat->cols));
    return (PyObject *) result;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    .nb_add = (binaryfunc) Matrix61c_add,
    .nb_subtract = (binaryfunc) Matrix61c_sub,
    .nb_multiply = (binaryfunc) Matrix61c_multiply,
    .nb_power = (ternaryfunc) Matrix61c_pow,
    .nb_negative = (unaryfunc) Matrix61c_neg,
    .nb_absolute = (unaryfunc) Matrix61c_abs
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    // Exception Handling
    PyObject *row = NULL;
    PyObject *col = NULL;
    PyObject *val = NULL;
    if (NULL == self) {
        PyErr_SetString(PyExc_ValueError, "self is a null pointer.");
        return NULL;
    } else if (!PyArg_UnpackTuple(args, "args", 3, 3, &row, &col, &val)) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occurs in Matrix61c_set_value.");
        return NULL;
    } else if (!PyLong_Check(row) || !PyLong_Check(col)
            || (!PyLong_Check(val) && !PyFloat_Check(val))) {
        PyErr_SetString(PyExc_TypeError, "Wrong type in Matrix61c_set_value.");
        return NULL;
    } else if (PyLong_AsLong(row) < 0 || PyLong_AsLong(row) >= self->mat->rows
            || PyLong_AsLong(col) < 0 || PyLong_AsLong(col) >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "i or j or both are out of range.");
        return NULL;
    }

    if (PyLong_Check(val)) {
        set(self->mat, PyLong_AsLong(row), PyLong_AsLong(col), PyLong_AsLong(val));
    } else {
        set(self->mat, PyLong_AsLong(row), PyLong_AsLong(col), PyFloat_AsDouble(val));
    }

    return Py_None;
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    // Exception Handling
    PyObject *row = NULL;
    PyObject *col = NULL;
    if (NULL == self) {
        PyErr_SetString(PyExc_ValueError, "self is a null pointer.");
        return NULL;
    } else if (!(PyArg_UnpackTuple(args, "args", 2, 2, &row, &col))) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occurs in Matrix61c_get_value.");
        return NULL;
    } else if (!PyLong_Check(row) || !PyLong_Check(col)) {
        PyErr_SetString(PyExc_TypeError, "Wrong type in Matrix61c_get_value.");
        return NULL;
    } else if (PyLong_AsLong(row) < 0 || PyLong_AsLong(row) >= self->mat->rows
            || PyLong_AsLong(col) < 0 || PyLong_AsLong(col) >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "i or j or both are out of range.");
        return NULL;
    }

    double val = get(self->mat, PyLong_AsLong(row), PyLong_AsLong(col));
    return (PyObject *) PyFloat_FromDouble(val);
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    {"get", (PyCFunction)Matrix61c_get_value, 1, "Getter function."},
    {"set", (PyCFunction)Matrix61c_set_value, 1, "Setter function."},
    {NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    /* TODO: YOUR CODE HERE */
    if (NULL == key || NULL == self) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occurs in Matrix61c_subscript.");
        return NULL;
    }

    if (self->mat->is_1d) {
        // matrix is 1D
        if (PyObject_TypeCheck(key, &PyLong_Type)) {
            // Key is a single number and we should return a single number

            PyObject *args = NULL;
            if (1 == self->mat->rows) {
                if (PyLong_AsLong(key) >= self->mat->cols) {
                    PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                    return NULL;
                }
                args = PyTuple_Pack(2, PyLong_FromLong(0), key);
            } else {
                if (PyLong_AsLong(key) >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                    return NULL;
                }
                args = PyTuple_Pack(2, key, PyLong_FromLong(0));
            }
            return Matrix61c_get_value(self, args);
        } else if (PyObject_TypeCheck(key, &PySlice_Type)) {
            // Key is a single slice and we should return a matrix with (stop - start) elements. e.x. a[0:2]
            // Two cases. 1. (stop - start) = 1    we should return a single number
            //            2. (stop - start) > 1     we should return a new matrix
            Py_ssize_t start, stop, step, slicelength;
            Py_ssize_t length = PyLong_AsSsize_t(PyLong_FromLong(self->mat->rows * self->mat->cols));
            if (-1 == PySlice_GetIndicesEx(key, length, &start, &stop, &step, &slicelength)){
                PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                return NULL;
            } else if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "For 1D, Slice info is not valid.");
                return NULL;
            }

            if (1 == slicelength) {
                // 1. (stop - start) = 1    we should return a single number
                PyObject *args = NULL;
                if (1 == self->mat->rows) {
                    args = PyTuple_Pack(2, PyLong_FromLong(0), PyLong_FromSsize_t(start));
                } else {
                    args = PyTuple_Pack(2, PyLong_FromSsize_t(start), PyLong_FromLong(0));
                }
                return Matrix61c_get_value(self, args);
            } else {
                // 2.(stop - start) > 1     we should return a new matrix
                matrix *new_mat;
                Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                if (1 == self->mat->rows) {
                    allocate_matrix_ref(&new_mat, self->mat, 0, start, 1, slicelength);
                    result->mat = new_mat;
                    result->shape = PyTuple_Pack(2, PyLong_FromLong(1),
                                                    PyLong_FromSsize_t(slicelength));
                } else {
                    allocate_matrix_ref(&new_mat, self->mat, start, 0, slicelength, 1);
                    result->mat = new_mat;
                    result->shape = PyTuple_Pack(2, PyLong_FromSsize_t(slicelength),
                                                    PyLong_FromLong(1));
                }
                return (PyObject *) result;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "For 1D, key must be an integer or a slice.");
            return NULL;
        }
    } else {
        // matrix is 2D
        if (PyObject_TypeCheck(key, &PyLong_Type)) {
            // Key is a single number and we should return a single row
            int keyth_row = PyLong_AsLong(key);
            if (keyth_row >= self->mat->rows) {
                PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                return NULL;
            }
            matrix *new_mat;
            allocate_matrix_ref(&new_mat, self->mat, keyth_row, 0, 1, self->mat->cols);
            Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
            result->mat = new_mat;
            result->shape = PyTuple_Pack(2, PyLong_FromLong(1),
                                            PyLong_FromLong(self->mat->cols));
            return (PyObject *) result;

        } else if (PyObject_TypeCheck(key, &PySlice_Type)) {
            // Key is a single slice and we should return a single row or multi rows
            // case 1: (stop - start) = 1, we should return a single row.
            // case 2: (stop- start) > 1, we should return multi rows.
            //PyObject *slice = PyTuple_GetItem(key, 0);
            Py_ssize_t start, stop, step, slicelength;
            Py_ssize_t length = self->mat->rows;
            if (-1 == PySlice_GetIndicesEx(key, length, &start, &stop, &step, &slicelength)) {
                PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                return NULL;
            } else if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "For 2D, Slice info is not valid.");
                return NULL;
            }

            if (1 == slicelength) {
                // case 1: (stop - start) = 1, we should return a single row.
                int keyth_row = start;
                matrix *new_mat;
                allocate_matrix_ref(&new_mat, self->mat, keyth_row, 0, 1, self->mat->cols);
                Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                result->mat = new_mat;
                result->shape = PyTuple_Pack(2, PyLong_FromLong(1),
                                                PyLong_FromLong(self->mat->cols));
                return (PyObject *) result;
            } else {
                // case 2: (stop - start) > 1, we should return multi rows.
                if (start >= self->mat->rows || stop > self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "For 2D, start or stop is out of range.");
                    return NULL;
                }
                matrix *new_mat;
                allocate_matrix_ref(&new_mat, self->mat, start, 0, slicelength, self->mat->cols);
                Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                result->mat = new_mat;
                result->shape = PyTuple_Pack(2, PyLong_FromSsize_t(slicelength),
                                                PyLong_FromLong(self->mat->cols));
                return (PyObject *) result;
            }
        } else if (PyObject_TypeCheck(key, &PyTuple_Type) && PyTuple_GET_SIZE(key) == 2) {
                // key is a tuple of two slices
                // case 1: key is a tuple of two slices
                // case 2: key is a tuple of (slice, int)
                // case 3: key is a tuple of (int, slice)
                // case 4: key is a tuple of (int, int)
                PyObject *item_one = PyTuple_GetItem(key, 0);
                PyObject *item_two = PyTuple_GetItem(key, 1);
                if (PyObject_TypeCheck(item_one, &PySlice_Type)
                        && PyObject_TypeCheck(item_two, &PySlice_Type)) {
                    // case 1: key is a tuple of two slices
                    // There are two cases
                    // case 1: return one single number.  e.x. a[2:3, 3:4]
                    // case 2: return multi rows e.x a[0:2, 0:2]
                    Py_ssize_t row_length = self->mat->rows;
                    Py_ssize_t col_length = self->mat->cols;
                    Py_ssize_t row_start, row_stop, row_step, row_slicelength;
                    Py_ssize_t col_start, col_stop, col_step, col_slicelength;

                    if (-1 == PySlice_GetIndicesEx(item_one, row_length, &row_start, &row_stop, &row_step, &row_slicelength)
                        || -1 == PySlice_GetIndicesEx(item_two, col_length, &col_start, &col_stop, &col_step, &col_slicelength)) {
                            PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                            return NULL;
                        } else if (row_slicelength < 1 || row_step != 1
                                    || col_slicelength < 1 || col_step != 1) {
                            PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                            return NULL;
                        }


                    // do something
                    if (1 == row_slicelength && 1 == col_slicelength) {
                        //case 1: return one single number.
                        PyObject *args = PyTuple_Pack(2, PyLong_FromSsize_t(row_start),
                                                         PyLong_FromSsize_t(col_start));
                        return Matrix61c_get_value(self, args);
                    } else {
                        // case 2: return multi rows.
                        matrix *new_mat;
                        allocate_matrix_ref(&new_mat, self->mat, row_start, col_start, row_slicelength, col_slicelength);
                        Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                        result->mat = new_mat;
                        result->shape = PyTuple_Pack(2, PyLong_FromSsize_t(row_slicelength),
                                                        PyLong_FromSsize_t(col_slicelength));
                        return (PyObject *) result;
                    }
                } else if (PyObject_TypeCheck(item_one, &PySlice_Type)
                            && PyObject_TypeCheck(item_two, &PyLong_Type)) {
                    // case 2: key is a tuple of (slice, int)
                    // There are two cases
                    // case 1: return one single number.  e.x. a[0:1，1]
                    // case 2: return multi rows e.x a[0:2, 0]
                    Py_ssize_t row_length = self->mat->rows;
                    Py_ssize_t row_start, row_stop, row_step, row_slicelength;
                    if (-1 == PySlice_GetIndicesEx(item_one, row_length, &row_start, &row_stop, &row_step, &row_slicelength)) {
                        PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                        return NULL;
                    } else if (row_slicelength < 1 || row_step != 1) {
                        PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                        return NULL;
                    } else if (PyLong_AsLong(item_two) >= self->mat->cols) {
                        PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                        return NULL;
                    }


                    if (1 == row_slicelength) {
                        // case 1: return one single number.
                        PyObject *args = PyTuple_Pack(2, PyLong_FromSsize_t(row_start),
                                                         PyLong_FromLong(PyLong_AsLong(item_two)));
                        //printf("row_start = %d, item_two = %d", row_start, PyLong_AsLong(item_two));
                        return Matrix61c_get_value(self, args);
                    } else {
                        // case 2: return multi rows.
                        matrix *new_mat;
                        allocate_matrix_ref(&new_mat, self->mat, row_start, PyLong_AsLong(item_two), row_slicelength, 1);
                        Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                        result->mat = new_mat;
                        result->shape = PyTuple_Pack(2, PyLong_FromSsize_t(row_slicelength),
                                                        PyLong_FromLong(1));
                        return (PyObject *) result;
                    }
                } else if (PyObject_TypeCheck(item_one, &PyLong_Type)
                            && PyObject_TypeCheck(item_two, &PySlice_Type)) {
                    // case 3: key is a tuple of (int, slice)
                    // There are two cases
                    // case 1: return one single number.  e.x. a[1，0:1]
                    // case 2: return multi rows e.x a[0, 0:2]
                    Py_ssize_t col_length = self->mat->cols;
                    Py_ssize_t col_start, col_stop, col_step, col_slicelength;
                    if (-1 == PySlice_GetIndicesEx(item_two, col_length, &col_start, &col_stop, &col_step, &col_slicelength)) {
                        PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                        return NULL;
                    } else if (col_slicelength < 1 || col_step != 1) {
                        PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                        return NULL;
                    } else if (PyLong_AsLong(item_one) >= self->mat->rows) {
                        PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                        return NULL;
                    }

                    if (1 == col_slicelength) {
                        // case 1: return one single number.
                        PyObject *args = PyTuple_Pack(2, PyLong_FromLong(PyLong_AsLong(item_one)),
                                                        PyLong_FromSsize_t(col_start));
                        return Matrix61c_get_value(self, args);
                    } else{
                        // case 2: return multi rows.
                        matrix *new_mat;
                        allocate_matrix_ref(&new_mat, self->mat, PyLong_AsLong(item_one), col_start, 1, col_slicelength);
                        Matrix61c *result = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                        result->mat = new_mat;
                        result->shape = PyTuple_Pack(2, PyLong_FromLong(1),
                                                        PyLong_FromSsize_t(col_slicelength));
                        return (PyObject *) result;
                    }
                } else if (PyObject_TypeCheck(item_one, &PyLong_Type)
                            && PyObject_TypeCheck(item_two, &PyLong_Type) ) {
                    // case 4: key is a tuple of (int, int)
                    // return a single number.
                    if (PyLong_AsLong(item_one) >= self->mat->rows
                            || PyLong_AsLong(item_two) >= self->mat->cols) {
                       PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                       return NULL;
                    }

                    PyObject *args = PyTuple_Pack(2, PyLong_FromLong(PyLong_AsLong(item_one)),
                                                    PyLong_FromLong(PyLong_AsLong(item_two)));
                    return Matrix61c_get_value(self, args);
                }
        } else {
            PyErr_SetString(PyExc_TypeError, "For 2D, key must be an integer, a slice, or a length-2 tuple of slices/ints.");
            return NULL;
        }
    }
}


/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    if (NULL == self || NULL == key || NULL == v) {
        PyErr_SetString(PyExc_RuntimeError, "Runtime error occurs in Matrix61c_set_subscript.");
        return -1;
    }

    if (self->mat->is_1d) {
        // self->mat is 1D
        if (PyObject_TypeCheck(key, &PyLong_Type)) {
            // Key is a single number, so we should reset a single number
            // Exception Handling
            if (-1 == list_checker(v, 1)) {
                PyErr_SetString(PyExc_TypeError, "v is not a float or int.");
                return -1;
            }

            // reset a single number
            PyObject *args = NULL;
            if (1 == self->mat->rows) {
                if (PyLong_AsLong(key) >= self->mat->cols) {
                    PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                    return -1;
                }
                args = PyTuple_Pack(3, PyLong_FromLong(0), key, v);
            } else {
                if (PyLong_AsLong(key) >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                    return -1;
                }
                args = PyTuple_Pack(3, key, PyLong_FromLong(0), v);
            }
            Matrix61c_set_value(self, args);
            return 0;
        } else if (PyObject_TypeCheck(key, &PySlice_Type)) {
            // key is a slice. Two cases.
            Py_ssize_t start, stop, step, slicelength;
            Py_ssize_t length = PyLong_AsSsize_t(PyLong_FromLong(self->mat->rows * self->mat->cols));
            if (-1 == PySlice_GetIndicesEx(key, length, &start, &stop, &step, &slicelength)){
                PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                return -1;
            } else if (slicelength < 1 || step != 1) {
                PyErr_SetString(PyExc_ValueError, "For 1D, Slice info is not valid.");
                return -1;
            }

            // case 1: reset a single number
            if (1 == slicelength) {
                // Exception Handling
                if (-1 == list_checker(v, 1)){
                    PyErr_SetString(PyExc_TypeError, "v is not a float or int.");
                    return -1;
                }

                // do something
                PyObject *args = NULL;
                if (1 == self->mat->rows) {
                    args = PyTuple_Pack(3, PyLong_FromLong(0), PyLong_FromSsize_t(start), v);
                } else {
                    args = PyTuple_Pack(3, PyLong_FromSsize_t(start), PyLong_FromLong(0), v);
                }
                Matrix61c_set_value(self, args);
                return 0;
            } else {
            // case 2: reset multi numbers
                // Exception Handling
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "v is not a 1D list.");
                    return -1;
                } else if (PyList_Size(v) != slicelength || -1 == list_checker(v, 0)) {
                    PyErr_SetString(PyExc_ValueError, "v has the wrong length or at least one element of v is not a float or int.");
                    return -1;
                }

                // reset every number
                PyObject *args = NULL;
                if (1 == self->mat->rows) {
                    for (int i = 0; i < slicelength; i++) {
                        args = PyTuple_Pack(3, PyLong_FromLong(0),
                                                PyLong_FromSsize_t(start+i), PyList_GetItem(v, i));
                        Matrix61c_set_value(self, args);
                    }
                }else {
                    for (int i = 0; i < slicelength; i++) {
                        args = PyTuple_Pack(3, PyLong_FromSsize_t(start+i),
                                                PyLong_FromLong(0), PyList_GetItem(v, i));
                        Matrix61c_set_value(self, args);
                    }
                }
                return 0;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "For 1D, key must be an integer or a slice.");
            return -1;
        }
    }else {
        // self->mat is 2D
        // matrix is 2D
        if (PyObject_TypeCheck(key, &PyLong_Type)) {
               // Key is a single number, so reset a single row
               // Exception Handling
               if (!PyList_Check(v)) {
                   PyErr_SetString(PyExc_TypeError, "v is not a 1D list.");
                   return -1;
               } else if (PyList_Size(v) != self->mat->cols || -1 == list_checker(v, 0)) {
                   PyErr_SetString(PyExc_ValueError, "v has the wrong length or at least one element of v is not a float or int.");
                   return -1;
               } else if (PyLong_AsLong(key) >= self->mat->rows) {
                   PyErr_SetString(PyExc_IndexError, "Index is out of range.");
                   return -1;
               }
               // reset one row
               PyObject *args = NULL;
               for (int i = 0; i < PyList_Size(v); i++) {
                   args = PyTuple_Pack(3, key, PyLong_FromLong(i), PyList_GetItem(v, i));
                   Matrix61c_set_value(self, args);
               }
               return 0;
        } else if (PyObject_TypeCheck(key, &PySlice_Type)) {
               // Key is a single slice
               Py_ssize_t start, stop, step, slicelength;
               Py_ssize_t length = self->mat->rows;
               if (-1 == PySlice_GetIndicesEx(key, length, &start, &stop, &step, &slicelength)) {
                   PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                   return -1;
               } else if (slicelength < 1 || step != 1) {
                   PyErr_SetString(PyExc_ValueError, "For 2D, Slice info is not valid.");
                   return -1;
               }

               if (1 == slicelength) {
                   // case 1: reset a single row

                   // Exception Handling
                   if (!PyList_Check(v)) {
                       PyErr_SetString(PyExc_TypeError, "v is not a 1D list.");
                       return -1;
                   } else if (PyList_Size(v) != self->mat->cols || -1 == list_checker(v, 0)) {
                       PyErr_SetString(PyExc_ValueError, "v has the wrong length or at least one element of v is not a float or int.");
                       return -1;
                   }
                    // reset one row
                    PyObject *args = NULL;
                    for (int i = 0; i < PyList_Size(v); i++) {
                        args = PyTuple_Pack(3, PyLong_FromSsize_t(start), PyLong_FromLong(i), PyList_GetItem(v, i));
                        Matrix61c_set_value(self, args);
                    }
                    return 0;
               }else {
                   // case 2: reset multi rows

                   // Exception Handling
                   if (!PyList_Check(v)) {
                       PyErr_SetString(PyExc_TypeError, "v is not a list.");
                       return -1;
                   }
                   for (int i = 0; i < PyList_Size(v); i++) {
                       PyObject *ls = PyList_GetItem(v, i);
                       if (!PyList_Check(ls)) {
                           PyErr_SetString(PyExc_ValueError, "v is not a 2D list.");
                           return -1;
                       } else if (PyList_Size(ls) != self->mat->cols || -1 == list_checker(ls, 0)) {
                           PyErr_SetString(PyExc_ValueError, "ls has the wrong length or at least one element of ls is not a float or int.");
                           return -1;
                       }
                   }

                   // reset multi rows
                   PyObject *args = NULL;
                   for (int i = start; i < stop; i++) {
                       PyObject *ls = PyList_GetItem(v, i);
                       for (int j = 0; j < self->mat->cols; j++) {
                           args = PyTuple_Pack(3, PyLong_FromSsize_t(start+i), PyLong_FromLong(j), PyList_GetItem(ls, j));
                           Matrix61c_set_value(self, args);
                       }
                   }
                   return 0;
               }
        } else if (PyObject_TypeCheck(key, &PyTuple_Type) && PyTuple_GET_SIZE(key) == 2) {
                // Key is a tuple of size two
                PyObject *item_one = PyTuple_GetItem(key, 0);
                PyObject *item_two = PyTuple_GetItem(key, 1);
                if (PyObject_TypeCheck(item_one, &PySlice_Type)
                        && PyObject_TypeCheck(item_two, &PySlice_Type)) {
                    // case 1: slice/slice
                        // case 1.1 reset one row
                        // case 1.2 reset multi rows
                    Py_ssize_t row_length = self->mat->rows;
                    Py_ssize_t col_length = self->mat->cols;
                    Py_ssize_t row_start, row_stop, row_step, row_slicelength;
                    Py_ssize_t col_start, col_stop, col_step, col_slicelength;
                    if (-1 == PySlice_GetIndicesEx(item_one, row_length, &row_start, &row_stop, &row_step, &row_slicelength)
                        || -1 == PySlice_GetIndicesEx(item_two, col_length, &col_start, &col_stop, &col_step, &col_slicelength)) {
                            PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                            return -1;
                    } else if (row_slicelength < 1 || row_step != 1
                            || col_slicelength < 1 || col_step != 1) {
                            PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                            return -1;
                    }

                    if (1 == row_slicelength && 1 == col_slicelength) {
                        //case 1: reset one single number.
                        if (-1 == list_checker(v, 1)) {
                            PyErr_SetString(PyExc_TypeError, "v is not a float or int.");
                            return -1;
                        }
                        PyObject *args = PyTuple_Pack(3, PyLong_FromSsize_t(row_start),
                                                            PyLong_FromSsize_t(col_start), v);
                        Matrix61c_set_value(self, args);
                        return 0;
                    } else {
                        // case 2: reset multi rows.
                        // Exception Handling
                        if (!PyList_Check(v)) {
                            PyErr_SetString(PyExc_TypeError, "v is not a list.");
                            return -1;
                        }
                        for (int i = 0; i < PyList_Size(v); i++) {
                            PyObject *ls = PyList_GetItem(v, i);
                            if (!PyList_Check(ls)) {
                                PyErr_SetString(PyExc_ValueError, "v is not a 2D list.");
                                return -1;
                            } else if (PyList_Size(ls) != col_slicelength || -1 == list_checker(ls, 0)) {
                                PyErr_SetString(PyExc_ValueError, "ls has the wrong length or at least one element of ls is not a float or int.");
                                return -1;
                            }
                        }

                        // reset multi rows
                        PyObject *args = NULL;
                        for (Py_ssize_t i = 0; i < row_slicelength; i++) {
                            PyObject *ls = PyList_GetItem(v, i);
                            for (Py_ssize_t j = 0; j < col_slicelength; j++) {
                                args = PyTuple_Pack(3, PyLong_FromSsize_t(row_start+i), PyLong_FromSsize_t(col_start+j), PyList_GetItem(ls, j));
                                Matrix61c_set_value(self, args);
                            }
                        }
                        return 0;
                    }
                } else if (PyObject_TypeCheck(item_one, &PySlice_Type)
                        && PyObject_TypeCheck(item_two, &PyLong_Type)) {
                    // case 2: slice/int
                        // case 2.1 reset one row
                        // case 2.2 reset multi rows
                        Py_ssize_t row_length = self->mat->rows;
                        Py_ssize_t row_start, row_stop, row_step, row_slicelength;
                        if (-1 == PySlice_GetIndicesEx(item_one, row_length, &row_start, &row_stop, &row_step, &row_slicelength)) {
                            PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                            return -1;
                        } else if (row_slicelength < 1 || row_step != 1) {
                            PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                            return -1;
                        } else if (PyLong_AsLong(item_two) >= self->mat->cols) {
                            PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                            return -1;
                        }

                        if (1 == row_slicelength) {
                            // case 1: reset one single number.
                            if (!list_checker(v,1)) {
                                PyErr_SetString(PyExc_TypeError, "v is not a float or int.");
                                return -1;
                            }
                            PyObject *args = PyTuple_Pack(3, PyLong_FromSsize_t(row_start), item_two, v);
                            Matrix61c_set_value(self, args);
                            return 0;
                        } else {
                            // case 2: reset multi rows.
                            if (!PyList_Check(v)) {
                                PyErr_SetString(PyExc_TypeError, "v is not a list.");
                                return -1;
                            } else if (PyList_Size(v) != row_slicelength || -1 == list_checker(v,0)) {
                                PyErr_SetString(PyExc_ValueError, "v has the wrong length or at least one element of ls is not a float or int.");
                                return -1;
                            }

                            for (int i = 0; i < row_slicelength; i++) {
                                PyObject *args = PyTuple_Pack(3, PyLong_FromSsize_t(row_start+i), item_two, PyList_GetItem(v,i));
                                Matrix61c_set_value(self, args);
                            }
                            return 0;
                        }
                } else if (PyObject_TypeCheck(item_one, &PyLong_Type)
                        && PyObject_TypeCheck(item_two, &PySlice_Type)) {
                    // case 3: int/slice
                        // case 3.1 reset one row
                        // case 3.2 reset multi rows
                        Py_ssize_t col_length = self->mat->cols;
                        Py_ssize_t col_start, col_stop, col_step, col_slicelength;
                        if (-1 == PySlice_GetIndicesEx(item_two, col_length, &col_start, &col_stop, &col_step, &col_slicelength)) {
                            PyErr_SetString(PyExc_RuntimeError, "PySlice_GetIndicesEx() fails.");
                            return -1;
                        } else if (col_slicelength < 1 || col_step != 1) {
                            PyErr_SetString(PyExc_ValueError, "slicelength or step info is wrong.");
                            return -1;
                        } else if (PyLong_AsLong(item_one) >= self->mat->rows) {
                            PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                            return -1;
                        }

                        if (1 == col_slicelength) {
                            // case 3.1 reset one row
                            if (!list_checker(v,1)) {
                                PyErr_SetString(PyExc_TypeError, "v is not a float or int.");
                                return -1;
                            }
                            PyObject *args = PyTuple_Pack(3, item_one, PyLong_FromSsize_t(col_start), v);
                            Matrix61c_set_value(self, args);
                            return 0;
                        } else {
                            // case 3.2 reset multi cols
                            if (!PyList_Check(v)) {
                                PyErr_SetString(PyExc_TypeError, "v is not a list.");
                                return -1;
                            } else if (PyList_Size(v) != col_slicelength || -1 == list_checker(v,0)) {
                                PyErr_SetString(PyExc_ValueError, "v has the wrong length or at least one element of ls is not a float or int.");
                                return -1;
                            }

                            for (int i = 0; i < col_slicelength; i++) {
                                PyObject *args = PyTuple_Pack(3, item_one, PyLong_FromSsize_t(col_start+i), PyList_GetItem(v,i));
                                Matrix61c_set_value(self, args);
                            }
                            return 0;
                        }
                } else if (PyObject_TypeCheck(item_one, &PyLong_Type)
                        && PyObject_TypeCheck(item_two, &PyLong_Type)) {
                    // case 4: int/int
                        // reset one number
                        if (PyLong_AsLong(item_one) >= self->mat->rows || PyLong_AsLong(item_two) >= self->mat->cols) {
                            PyErr_SetString(PyExc_IndexError, "For 2D, key is an integer but is out of range.");
                            return -1;
                        }

                        PyObject *args = PyTuple_Pack(3, item_one, item_two, v);
                        Matrix61c_set_value(self, args);
                        return 0;
                }
        } else {
            PyErr_SetString(PyExc_TypeError, "For 2D, key must be an integer, a slice, or a length-2 tuple of slices/ints.");
            return -1;
        }
    }
}

int number_checker(PyObject *v) {
    if (!PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type)) {
            return -1;
    }
    return 0;
}

int list_checker(PyObject *v, int is_single){
    if (1 == is_single) {
        return number_checker(v);
    } else {
        for (int i = 0; i < PyList_Size(v); i++) {
            if (-1 == number_checker(PyList_GetItem(v,i))) {
                return -1;
            }
        }
        return 0;
    }
}






PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}
