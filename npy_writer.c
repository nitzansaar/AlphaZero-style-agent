/*
 * npy_writer.c — Minimal NumPy v1.0 .npy writer (float32 only).
 *
 * Format spec (v1.0):
 *   Bytes 0-5  : magic  \x93NUMPY
 *   Byte  6    : major version = 1
 *   Byte  7    : minor version = 0
 *   Bytes 8-9  : HEADER_LEN  (uint16, little-endian)
 *   Bytes 10 .. 10+HEADER_LEN-1 : header string (dict + spaces + '\n')
 *
 * Constraint: (10 + HEADER_LEN) must be a multiple of 64.
 */

#include "npy_writer.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* 8-byte prefix: magic (6) + major (1) + minor (1) */
static const uint8_t NPY_PREFIX[8] = {
    0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00
};

int npy_write_float32(const char  *path,
                      const float *data,
                      int          n_elem,
                      int          ndim,
                      const int   *dims)
{
    /* ── Build shape string, e.g. "(100,)" or "(100, 82)" ─────────────── */
    char shape[256];
    int  slen = 0;
    shape[slen++] = '(';
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape + slen, sizeof(shape) - (size_t)slen,
                         "%d", dims[i]);
        if (i < ndim - 1) {
            shape[slen++] = ',';
            shape[slen++] = ' ';
        } else if (ndim == 1) {
            shape[slen++] = ',';  /* trailing comma for 1-tuple: (N,) */
        }
    }
    shape[slen++] = ')';
    shape[slen]   = '\0';

    /* ── Build dict string ─────────────────────────────────────────────── */
    char dict[512];
    int  dict_len = snprintf(dict, sizeof(dict),
                             "{'descr': '<f4', 'fortran_order': False, "
                             "'shape': %s, }",
                             shape);

    /* ── Compute padding so (10 + HEADER_LEN) % 64 == 0 ──────────────── */
    /*
     * HEADER_LEN = dict_len + num_spaces + 1  (the +1 is for '\n')
     * We need: (10 + HEADER_LEN) % 64 == 0
     */
    int min_hlen   = dict_len + 1;                       /* dict + '\n'  */
    int pad        = (64 - (10 + min_hlen) % 64) % 64;  /* spaces to add */
    int header_len = min_hlen + pad;

    /* ── Allocate and build padded header ──────────────────────────────── */
    char *header = (char *)malloc((size_t)header_len);
    if (!header) return -1;

    memcpy(header, dict, (size_t)dict_len);
    memset(header + dict_len, ' ', (size_t)pad);
    header[dict_len + pad] = '\n';   /* terminates at index header_len-1 */

    /* ── Write file ────────────────────────────────────────────────────── */
    FILE *fp = fopen(path, "wb");
    if (!fp) { free(header); return -1; }

    fwrite(NPY_PREFIX, 1, 8, fp);

    uint16_t hl = (uint16_t)header_len;
    fwrite(&hl, 2, 1, fp);

    fwrite(header, 1, (size_t)header_len, fp);

    fwrite(data, sizeof(float), (size_t)n_elem, fp);

    fclose(fp);
    free(header);
    return 0;
}
