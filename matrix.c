#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 * throw
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        PyErr_SetString(PyExc_ValueError, "Either `rows` or `cols` or both have invalid values");
        return -1;
    }

    //double *real_data = (double *) calloc(rows*cols, sizeof(double));
    double *real_data = (double *) malloc(rows*cols * sizeof(double));
    memset(real_data, 0, rows*cols*sizeof(double));
    if (NULL == real_data) {
        PyErr_SetString(PyExc_RuntimeError, "allocate_matrix fails to allocate space for real_data.");
        return -1;
    }

    double **data_rows = (double **) malloc(rows * sizeof(double *));
    if (NULL == data_rows) {
        free(real_data);
        PyErr_SetString(PyExc_RuntimeError, "allocate_matrix fails to allocate space for data_rows.");
        return -1;
    }

    matrix *new_matrix = (matrix *) malloc(sizeof(struct matrix));
    if (NULL == new_matrix) {
        free(real_data);
        free(data_rows);
        PyErr_SetString(PyExc_RuntimeError, "allocate_matrix fails to allocate space for new_matrix.");
        return -1;
    }

    #pragma omp parallel for
    for (int i = 0; i < rows / 4 * 4; i += 4) {
        *(data_rows + i) = real_data + i * cols;
        *(data_rows + i + 1) = real_data + (i + 1)*cols;
        *(data_rows + i + 2) = real_data + (i + 2)*cols;
        *(data_rows + i + 3) = real_data + (i + 3)*cols;
    }

    // Tail case
    for (int i = rows / 4 * 4; i < rows; i += 1) {
        *(data_rows + i) = real_data + i * cols;
    }

    new_matrix->rows = rows;
    new_matrix->cols = cols;
    new_matrix->is_1d = (rows == 1 || cols == 1)? 1 : 0;
    new_matrix->ref_cnt = (int *) malloc(sizeof(int));
    *(new_matrix->ref_cnt) = 1;
    new_matrix->parent = NULL;
    new_matrix->data = data_rows;
    *(mat) = new_matrix;
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * The result includes the start index(row_offset), but excludes the end index(row_offset + rows).
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */


int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    if (rows < 1 || cols < 1) {
        PyErr_SetString(PyExc_ValueError, "Either `rows` or `cols` or both have invalid values");
        return -1;
    }

    matrix *matrix_ref = (matrix *) malloc(sizeof(struct matrix));
    if (NULL == matrix_ref) {
        PyErr_SetString(PyExc_RuntimeError, "allocate_matrix_ref fails to allocate space.");
        return -1;
    }
    double **data_rows = (double **) malloc(rows * sizeof(double *));
    if (NULL == data_rows) {
        free(matrix_ref);
        PyErr_SetString(PyExc_RuntimeError, "allocate_matrix_ref fails to allocate space.");
        return -1;
    }

    // Unrolling could be applied    WRONG NEED TO DEBUG OR DOUBLE CHECK
    #pragma omp parallel for
    for (int i = 0; i < rows / 4 * 4; i += 4) {
        *(data_rows + i) = *(from->data + row_offset + i) + col_offset;
        *(data_rows + i + 1) = *(from->data + row_offset + i + 1) + col_offset;
        *(data_rows + i + 2) = *(from->data + row_offset + i + 2) + col_offset;
        *(data_rows + i + 3) = *(from->data + row_offset + i + 3) + col_offset;
    }

    // edge case
    for (int i = rows / 4 * 4; i < rows; i++) {
        *(data_rows + i) = *(from->data + row_offset + i) + col_offset;
    }

    matrix_ref->rows = rows;
    matrix_ref->cols = cols;
    matrix_ref->is_1d = (rows == 1 || cols == 1)? 1 : 0;
    matrix_ref->ref_cnt = from->ref_cnt;
    *(matrix_ref->ref_cnt) += 1;
    matrix_ref->data = data_rows;
    matrix_ref->parent = from;
    *(mat) = matrix_ref;
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * deallocate_matrix will only be called once on any matrix.
 * See the spec for more information.  piazza post @3645_f37
 * you just have to make sure all of A's data is gone after all matrices referring to its data are gone.
 * For the deallocate function, since there can be multiple matrices that refer to the same data array in the memory,
 * you must not free the data until you call deallocate on the last existing matrix that refers to that data.
 * If you are having some difficulties implementing this,
 * here’s a hint: you can keep the matrix struct in the memory even if you have already called deallocate on that matrix.
 * You only need to make sure to that the struct is freed once the last matrix referring to its data is deallocated.
 */
void deallocate_matrix(matrix *mat) {
    if (NULL == mat) {
        PyErr_SetString(PyExc_ValueError, "mat is a null pointer.");
        return;
    }
    if (NULL == mat->parent) {
        // mat is not a slice of any other matrices.
        if (1 == *(mat->ref_cnt)) {
            // mat does not have any children.
            free(*(mat->data));
            free(mat->data);
            free(mat->ref_cnt);
            free(mat);
        }else {
            // mat has children.
            // Do not free mat! Keep it.
            *(mat->ref_cnt) -= 1;
        }
    }else{
        // mat is a slice of a matrix.
        if (1 == *(mat->ref_cnt)) {
            // mat is the last matrix referring to 'data'
            // free its parent and itself.
            matrix *parent = mat->parent;
            free(*(parent->data));
            free(parent->data);
            free(parent->ref_cnt);
            free(parent);
            free(mat->data);
            free(mat);
        }else {
            // mat is not the last matrix referring to 'data'
            // free mat but not real_data
            *(mat->ref_cnt) -= 1;
            free(mat->data);
            free(mat);
        }
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return *(*(mat->data + row) + col);
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    *(*(mat->data + row) + col) = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    if (NULL == mat) {
        PyErr_SetString(PyExc_ValueError, "mat is a null pointer.");
    }
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++){
            *(*(mat->data + i) + j) = val;
        }
    }
}


/*
 * Store the result of adding mat1 and mat2 to `result`.
 * 'mat1' and 'mat2' must have the same dimensions.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int size = mat1->rows * mat1->cols;
    int size_16 = size/16 * 16;
    int size_4 = size/4 * 4;
    omp_set_num_threads(16);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < size_16; i+=16) {
            __m256d vec_mat1, vec_mat2;
            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i);
            _mm256_storeu_pd(*(result->data) + i, _mm256_add_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 4);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 4);
            _mm256_storeu_pd(*(result->data) + i + 4, _mm256_add_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 8);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 8);
            _mm256_storeu_pd(*(result->data) + i + 8, _mm256_add_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 12);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 12);
            _mm256_storeu_pd(*(result->data) + i + 12, _mm256_add_pd(vec_mat1, vec_mat2));
        }
    }

    __m256d vec_mat3, vec_mat4;
    for (int i = size_16; i < size_4; i+=4) {
        vec_mat3 = _mm256_loadu_pd(*(mat1->data) + i);
        vec_mat4 = _mm256_loadu_pd(*(mat2->data) + i);
        _mm256_storeu_pd(*(result->data) + i, _mm256_add_pd(vec_mat3, vec_mat4));
    }

    for (int i = size_4; i < size; i++) {
        *(*(result->data) + i) = *(*(mat1->data) + i) + *(*(mat2->data) + i);
    }
    return 0;
}


/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * 'mat1' and 'mat2' must have the same dimensions.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int size = mat1->rows * mat1->cols;
    int size_16 = size /16 * 16;
    int size_4 = size /4 * 4;
    omp_set_num_threads(16);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < size_16; i+=16) {
            __m256d vec_mat1, vec_mat2;
            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i);
            _mm256_storeu_pd(*(result->data) + i, _mm256_sub_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 4);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 4);
            _mm256_storeu_pd(*(result->data) + i + 4, _mm256_sub_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 8);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 8);
            _mm256_storeu_pd(*(result->data) + i + 8, _mm256_sub_pd(vec_mat1, vec_mat2));

            vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i + 12);
            vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i + 12);
            _mm256_storeu_pd(*(result->data) + i + 12, _mm256_sub_pd(vec_mat1, vec_mat2));
        }
    }


    __m256d vec_mat1, vec_mat2;
    for (int i = size_16; i < size_4; i+=4) {
        vec_mat1 = _mm256_loadu_pd(*(mat1->data) + i);
        vec_mat2 = _mm256_loadu_pd(*(mat2->data) + i);
        _mm256_storeu_pd(*(result->data) + i, _mm256_sub_pd(vec_mat1, vec_mat2));
    }


    for (int i = size_4; i < size; i++) {
        *(*(result->data) + i) = *(*(mat1->data) + i) - *(*(mat2->data) + i);
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * We shoule check that mat1’s number of columns is equal to mat2’s number of rows.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix **result, matrix *mat1, matrix *mat2) {
    if (*(result) == mat1 || *(result) == mat2) {
        // *(result)  may point to the same matrix either as mat1 or mat2 or even both. For pow_matrix.
        // mat1 and mat2 may share the same matrix but it does not matter in cache optimized implementation.
        // auguments are passed by value that is why result is changed to a pointer to pointer to a matrix struct.
        allocate_matrix(result, mat1->rows, mat2->cols);
    }

    if (mat1->rows < 100  || mat2->cols < 100) {
        for (int j = 0; j < mat1->rows; j++) {
            for (int k = 0; k < mat2->rows; k++) {
                for (int i = 0; i < mat2->cols;i++){
                    *(*((*result)->data + j) + i) = *(*((*result)->data + j) + i) + *(*(mat1->data + j) + k) * *(*(mat2->data + k) + i);
                }
            }
        }
    } else {
        double *transpose_data = (double *) malloc(mat2->cols *mat2->rows * sizeof(double));
        if (NULL == transpose_data) {
            PyErr_SetString(PyExc_RuntimeError, "Fails to allocate space for transpose_data.");
            return -1;
        }

        int blocksize = 16; // 64
        int col = mat2->cols;
        int row = mat2->rows;
        double *real_data = *(mat2->data);
        for (int i = 0; i < row; i += blocksize) {
            for (int j = 0; j < col; j += blocksize) {
            // transpose the block from [i,j] to [i+blocksize, j+blocksize]
                int k_stop = i + blocksize < row ? (i + blocksize) : row;
                int j_stop = j + blocksize < col ? (j + blocksize) : col;
                for (int k = i; k < k_stop; k++) {
                    for (int l = j; l < j_stop; l++) {
                        transpose_data[l*row + k] = real_data[k*col +l];
                    }
                }
            }
        }

        // Cache Blocking + Data-level Parallelism + Thread-level Parallelism
        #pragma omp parallel for
        for (int i = 0; i < mat1->rows; i+=blocksize) {
            for (int j = 0; j < mat2->cols; j+=blocksize) {
                int stopRow = (i + blocksize) < mat1->rows? (i + blocksize) : mat1->rows;
                int stopCol = (j + blocksize) < mat2->cols? (j + blocksize) : mat2->cols;
                for (int blockRow = i; blockRow < stopRow; blockRow++) {
                    for (int blockCol = j; blockCol < stopCol; blockCol++) {
                        // Compute the dot product of row mat1 and row transpose_data
                        double sum_holder[4];
                        __m256d sum = _mm256_setzero_pd();
                        for(int k = 0; k < mat1->cols / 24 * 24; k += 24) {
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k), sum);
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k+4),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k+4), sum);
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k+8),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k+8), sum);
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k+12),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k+12), sum);
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k+16),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k+16), sum);
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k+20),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k+20), sum);
                        }
                        for (int k = mat1->cols / 24 * 24; k < mat1->cols / 4 * 4; k += 4){
                            sum = _mm256_fmadd_pd(_mm256_loadu_pd(*(mat1->data + blockRow) + k),  _mm256_loadu_pd(transpose_data + blockCol * mat2->rows + k), sum);
                        }
                        _mm256_storeu_pd(sum_holder, sum);
                        double val = sum_holder[0] + sum_holder[1] + sum_holder[2] + sum_holder[3];
                        for (int k = mat1->cols / 4 * 4; k < mat1->cols; k++){
                            val += *(*(mat1->data + blockRow) + k) * *(transpose_data + blockCol * mat2->rows + k);
                        }
                        *(*((*result)->data + blockRow) + blockCol) = val;
                    }
                }
            }
        }
        free(transpose_data);
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
 int pow_matrix(matrix **result, matrix *mat, int pow) {
    if (0 == pow) {
        for (int i = 0; i < mat->rows;  i++) {
            *(*((*result)->data + i) + i) = 1;
        }
    } else if (1 == pow) {
        deallocate_matrix(*result);
        allocate_matrix_ref(result, mat, 0, 0, mat->rows, mat->cols);
    } else {
        double *result_data = *((*result)->data);
        double *mat_data = *(mat->data);
        pow_helper(&result_data, mat_data, pow, mat->rows);

        #pragma omp parallel for
        for (int i = 0; i < (*result)->rows / 4 * 4; i += 4) {
            *((*result)->data + i) = result_data + i * (*result)->cols;
            *((*result)->data + i + 1) = result_data + (i + 1)* (*result)->cols;
            *((*result)->data + i + 2) = result_data + (i + 2)* (*result)->cols;
            *((*result)->data + i + 3) = result_data + (i + 3)* (*result)->cols;
        }

        for (int i = (*result)->rows / 4 * 4; i < (*result)->rows; i += 1) {
            *((*result)->data + i) = result_data + i * (*result)->cols;
        }

    }
    return 0;
}


int pow_helper(double **result_data, double *mat_data, int pow, int n) {
    if (1 == pow) {
        for (int i = 0; i < n * n; i++) {
            *(*result_data + i) = *(mat_data + i);
        }
    } else if (pow % 2 == 0) {
        pow_helper(result_data, mat_data, pow / 2, n);
        double *intermediate_data = (double *) malloc(n * n * sizeof(double));
        if (NULL == intermediate_data) {
            PyErr_SetString(PyExc_RuntimeError, "Fails to allocate space for intermediate_data.");
            return -1;
        }
        mul_helper(intermediate_data, *result_data, *result_data, n);
        double *temp = *result_data;
        *result_data = intermediate_data;
        free(temp);
    } else {
        pow_helper(result_data, mat_data, pow / 2, n);
        double *intermediate_data = (double *) malloc(n * n * sizeof(double));
        if (NULL == intermediate_data) {
            PyErr_SetString(PyExc_RuntimeError, "Fails to allocate space for intermediate_data.");
            return -1;
        }
        // mul_helper(intermediate_data, *result_data, *result_data, n);
        // mul_helper(*result_data, intermediate_data, mat_data, n);
        // // free(intermediate_data);
        // intermediate_data = NULL;
        mul_helper(intermediate_data, *result_data, *result_data, n);
        mul_helper(*result_data, intermediate_data, mat_data, n);
        free(intermediate_data);
    }
    return 0;
}


int mul_helper(double *result_data, double *mat1_data, double *mat2_data, int n) {


    double *transpose_mat2_data = (double *) malloc(n * n * sizeof(double));
    if (NULL == transpose_mat2_data) {
        PyErr_SetString(PyExc_RuntimeError, "Fails to allocate space for transpose_data.");
        return -1;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            *(transpose_mat2_data + j * n + i) = *(mat2_data + i * n + j);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum_holder[4];
            __m256d sum = _mm256_setzero_pd();
            for(int k = 0; k < n / 16 * 16; k += 16) {
                //* __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i * n + k),  _mm256_loadu_pd(transpose_mat2_data + j * n + k), sum);
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i * n + k+4),  _mm256_loadu_pd(transpose_mat2_data + j * n + k+4), sum);
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i * n + k+8),  _mm256_loadu_pd(transpose_mat2_data + j * n + k+8), sum);
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i * n + k+12),  _mm256_loadu_pd(transpose_mat2_data + j * n + k+12), sum);
            }
            for (int k = n / 16 * 16; k < n / 4 * 4; k += 4){
                sum = _mm256_fmadd_pd(_mm256_loadu_pd(mat1_data + i * n + k),  _mm256_loadu_pd(transpose_mat2_data + j * n + k), sum);
            }
            _mm256_storeu_pd(sum_holder, sum);
            double val = sum_holder[0] + sum_holder[1] + sum_holder[2] + sum_holder[3];
            for (int k = n / 4 * 4; k < n; k++) {
                val += *(mat1_data + i * n + k) * *(transpose_mat2_data + j * n + k);
            }
            *(result_data + i * n + j) = val;
        }
    }
    free(transpose_mat2_data);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    int size = mat->rows * mat->cols;
    int size_16 = size /16 * 16;
    int size_4 = size /4 * 4;
    __m256d xor_mask = _mm256_set1_pd(-0.);

    omp_set_num_threads(16);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < size_16; i+=16) {
            __m256d *vec_mat1;
            vec_mat1 = (__m256d *)(*(mat->data) + i);
            _mm256_storeu_pd(*(result->data) + i, _mm256_xor_pd(xor_mask , *vec_mat1));

            vec_mat1 = (__m256d *)(*(mat->data) + i + 4);
            _mm256_storeu_pd(*(result->data) + i + 4, _mm256_xor_pd(xor_mask, *vec_mat1));

            vec_mat1 = (__m256d *)(*(mat->data) + i + 8);
            _mm256_storeu_pd(*(result->data) + i + 8, _mm256_xor_pd(xor_mask, *vec_mat1));

            vec_mat1 = (__m256d *)(*(mat->data) + i + 12);
            _mm256_storeu_pd(*(result->data) + i + 12, _mm256_xor_pd(xor_mask, *vec_mat1));
        }
    }
    __m256d *vec_mat1;
    for (int i = size_16; i < size_4; i+=4) {
        vec_mat1 = (__m256d *)(*(mat->data) + i);
        _mm256_storeu_pd(*(result->data) + i, _mm256_xor_pd(xor_mask ,*vec_mat1));
    }
    for (int i = size_4; i < size; i++) {
        *(*(result->data) + i) = *(*(mat->data) + i) * (-1);
    }
    return 0;
}



/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int size = mat->rows * mat->cols;
    int size_16 = size /16 * 16;
    int size_4 = size /4 * 4;
    double *mat_data = mat->data[0];
    double *res_data = result->data[0];
    uint64_t num_int = 0x7FFFFFFFFFFFFFFF;
    double *db_ptr = (double*) &num_int;
    __m256d mask = _mm256_set1_pd(*db_ptr);


    #pragma omp parallel for
    for (int i = 0; i < size_16; i+=16) {

        __m256d vec_mat1 = _mm256_loadu_pd(&mat_data[i]);
        _mm256_storeu_pd(&res_data[i], _mm256_and_pd(mask ,vec_mat1));

        __m256d vec_mat2 = _mm256_loadu_pd(&mat_data[i + 4]);
        _mm256_storeu_pd(&res_data[i + 4], _mm256_and_pd(mask ,vec_mat2));

        __m256d vec_mat3 = _mm256_loadu_pd(&mat_data[i + 8]);
        _mm256_storeu_pd(&res_data[i + 8], _mm256_and_pd(mask ,vec_mat3));

        __m256d vec_mat4 = _mm256_loadu_pd(&mat_data[i + 12]);
        _mm256_storeu_pd(&res_data[i + 12], _mm256_and_pd(mask ,vec_mat4));
    }


    for (int i = size_16; i < size_4; i+=4) {
        __m256d vec_mat1 = _mm256_loadu_pd(&mat_data[i]);
        _mm256_storeu_pd(&res_data[i], _mm256_and_pd(mask ,vec_mat1));
    }

    for (int i = size_4; i < size; i++) {
        res_data[i] = mat_data[i] > 0? mat_data[i] : (-mat_data[i]);
    }
    return 0;
}
