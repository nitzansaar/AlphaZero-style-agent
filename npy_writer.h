#pragma once

/*
 * npy_writer.h — Minimal NumPy v1.0 .npy file writer (float32 only).
 *
 * Writes files loadable with np.load() or numpy.lib.format.read_array().
 * No external dependencies beyond libc.
 *
 * Usage:
 *   int dims[] = {N, 3, 9, 9};
 *   npy_write_float32("states.npy", data_ptr, N*3*81, 4, dims);
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Write a float32 array to path in NPY v1.0 format.
 *
 *   path   : output file path (created/overwritten)
 *   data   : row-major float32 data, n_elem elements
 *   n_elem : total number of elements (= product of dims[0..ndim-1])
 *   ndim   : number of dimensions (≥ 1)
 *   dims   : shape array, length ndim
 *
 * Returns 0 on success, -1 on error (check errno).
 */
int npy_write_float32(const char  *path,
                      const float *data,
                      int          n_elem,
                      int          ndim,
                      const int   *dims);

#ifdef __cplusplus
}
#endif
