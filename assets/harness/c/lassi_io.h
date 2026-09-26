/* lassi_io.h: read and write one named array per binary file (LASSI Harness Contract).
 *
 * One self-contained header for C99 and C++17: static inline functions and
 * standard C I/O only. A program reads its inputs with lassi_io_read and
 * writes its outputs with lassi_io_write; the harness writes the inputs and
 * reads the outputs with lassi/harness/lassi_io.py, which reads and writes the
 * same bytes.
 *
 * File format, version 1, little-endian throughout, offsets from the start of
 * the file:
 *
 *   0      8 bytes     magic: the ASCII text LASSIIO and one zero byte
 *   8      u32         version, 1
 *   12     u32         dtype code, one of the LASSI_IO_<DTYPE> constants
 *   16     u32         rank
 *   20     u64 x rank  dims, in C order
 *   then   u32         name length in bytes, 1 to 255
 *   then   the name    each byte an ASCII letter, digit, '_', '.', or '-'
 *   then   zero bytes up to the next multiple of 8
 *   then   the data    product(dims) * itemsize bytes in C order, and nothing after it
 *
 * A rank-0 array is a scalar of one element; a dim of 0 gives an empty array
 * with no data. The functions move bytes and convert no value: an f16 element
 * is its IEEE binary16 bit pattern, and a bf16 element is the upper 16 bits
 * of the f32 of the same value.
 *
 * lassi_io_read and lassi_io_write return 0 on success. On a refusal they
 * return 1 and print one line, "lassi_io: <path>: <reason>", to stderr.
 * lassi_io_read refuses a file that is not exactly a valid version 1 file (bad
 * magic, version, dtype code, name length, name, or padding; a file that ends
 * inside its header; data longer or shorter than the dims say), checking each
 * header field against the file's size before using it. It zeroes *out first
 * (whenever out is not NULL), so *out stays zeroed after any refusal, a NULL
 * path included. lassi_io_write refuses a bad argument before it creates the file,
 * and removes the file when a write to it fails. Both refuse on a big-endian
 * host, since the data is copied as it is.
 */
#ifndef LASSI_IO_H
#define LASSI_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The 8 bytes every file starts with: this text and its terminating zero byte. */
#define LASSI_IO_MAGIC "LASSIIO"
#define LASSI_IO_VERSION 1u
#define LASSI_IO_MAX_NAME 255u

/* dtype codes, the same as lassi.harness.lassi_io.DTYPES. */
#define LASSI_IO_F32 1u
#define LASSI_IO_F16 2u
#define LASSI_IO_BF16 3u
#define LASSI_IO_I32 4u
#define LASSI_IO_U32 5u
#define LASSI_IO_I8 6u
#define LASSI_IO_U8 7u
#define LASSI_IO_I64 8u
#define LASSI_IO_F64 9u

/* One array read from a file. Zero it before the first use. lassi_io_read overwrites it, so release a
 * previous read with lassi_io_free first. */
typedef struct lassi_io_array lassi_io_array;
struct lassi_io_array {
    char name[LASSI_IO_MAX_NAME + 1u]; /* NUL-terminated */
    uint32_t dtype;                    /* a LASSI_IO_<DTYPE> code */
    uint32_t rank;
    uint64_t *dims;  /* rank values in C order; NULL when rank is 0 */
    void *data;      /* the element bytes in C order, 8-byte aligned; not NULL after a read */
    uint64_t count;  /* the number of elements: the product of dims, 1 for a scalar */
    size_t nbytes;   /* count * lassi_io_itemsize(dtype) */
    void *storage;   /* internal: the file's bytes, which data points into */
};

/* Return the size in bytes of one element of dtype, or 0 for an unknown code. */
static inline size_t lassi_io_itemsize(uint32_t dtype) {
    switch (dtype) {
    case LASSI_IO_I8:
    case LASSI_IO_U8:
        return 1u;
    case LASSI_IO_F16:
    case LASSI_IO_BF16:
        return 2u;
    case LASSI_IO_F32:
    case LASSI_IO_I32:
    case LASSI_IO_U32:
        return 4u;
    case LASSI_IO_I64:
    case LASSI_IO_F64:
        return 8u;
    default:
        return 0u;
    }
}

/* Release what a successful lassi_io_read allocated and zero the array; a zeroed array is left as it is. */
static inline void lassi_io_free(lassi_io_array *array) {
    if (array == NULL) {
        return;
    }
    free(array->storage);
    free(array->dims);
    memset(array, 0, sizeof *array);
}

/* Internal helpers; their names start with lassi_io_detail_. */

static inline int lassi_io_detail_fail(const char *path, const char *reason) {
    fprintf(stderr, "lassi_io: %s: %s\n", path != NULL ? path : "(no path)", reason);
    return 1;
}

static inline int lassi_io_detail_fail_number(const char *path, const char *reason, uint64_t value) {
    fprintf(stderr, "lassi_io: %s: %s %llu\n", path != NULL ? path : "(no path)", reason,
            (unsigned long long)value);
    return 1;
}

static inline int lassi_io_detail_fail_size(const char *path, uint64_t need, uint64_t have) {
    fprintf(stderr, "lassi_io: %s: data size: the dims need %llu bytes, but %llu bytes follow the header\n",
            path, (unsigned long long)need, (unsigned long long)have);
    return 1;
}

static inline int lassi_io_detail_truncated(const char *path) {
    return lassi_io_detail_fail(path, "truncated: the file ends inside its header");
}

static inline int lassi_io_detail_little_endian(void) {
    const uint16_t probe = 1u;
    unsigned char first = 0u;
    memcpy(&first, &probe, 1u);
    return first == 1;
}

static inline int lassi_io_detail_name_ok(const char *name, size_t length) {
    size_t i;
    if (length == 0u || length > LASSI_IO_MAX_NAME) {
        return 0;
    }
    for (i = 0u; i < length; ++i) {
        const char c = name[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
              c == '.' || c == '-')) {
            return 0;
        }
    }
    return 1;
}

/* Set *count to the product of rank dims; return 1 when it does not fit in 64 bits (a zero dim gives 0). */
static inline int lassi_io_detail_count(uint32_t rank, const uint64_t *dims, uint64_t *count) {
    uint64_t product = 1u;
    uint32_t i;
    for (i = 0u; i < rank; ++i) {
        if (dims[i] == 0u) {
            *count = 0u;
            return 0;
        }
    }
    for (i = 0u; i < rank; ++i) {
        if (product > ~(uint64_t)0u / dims[i]) {
            return 1;
        }
        product *= dims[i];
    }
    *count = product;
    return 0;
}

static inline uint32_t lassi_io_detail_get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t lassi_io_detail_get_u64(const unsigned char *p) {
    return (uint64_t)lassi_io_detail_get_u32(p) | ((uint64_t)lassi_io_detail_get_u32(p + 4) << 32);
}

static inline int lassi_io_detail_put(FILE *file, const void *bytes, size_t length) {
    return length == 0u || fwrite(bytes, 1u, length, file) == length;
}

static inline int lassi_io_detail_put_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4];
    size_t i;
    for (i = 0u; i < 4u; ++i) {
        bytes[i] = (unsigned char)((value >> (8u * i)) & 0xffu);
    }
    return lassi_io_detail_put(file, bytes, 4u);
}

static inline int lassi_io_detail_put_u64(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    size_t i;
    for (i = 0u; i < 8u; ++i) {
        bytes[i] = (unsigned char)((value >> (8u * i)) & 0xffu);
    }
    return lassi_io_detail_put(file, bytes, 8u);
}

/* Read the whole file into a new buffer; return 0 and set *bytes and *size, or print why and return 1. */
static inline int lassi_io_detail_slurp(const char *path, unsigned char **bytes, size_t *size) {
    FILE *file = fopen(path, "rb");
    long end = -1L;
    size_t length = 0u;
    unsigned char *buffer = NULL;
    if (file == NULL) {
        return lassi_io_detail_fail(path, "cannot open the file for reading");
    }
    if (fseek(file, 0L, SEEK_END) == 0) {
        end = ftell(file);
    }
    if (end < 0L || fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return lassi_io_detail_fail(path, "cannot find the file's size");
    }
    length = (size_t)end;
    buffer = (unsigned char *)malloc(length > 0u ? length : 1u);
    if (buffer == NULL) {
        fclose(file);
        return lassi_io_detail_fail(path, "out of memory for the file's bytes");
    }
    if (length > 0u && fread(buffer, 1u, length, file) != length) {
        free(buffer);
        fclose(file);
        return lassi_io_detail_fail(path, "cannot read the file");
    }
    fclose(file);
    *bytes = buffer;
    *size = length;
    return 0;
}

/* Check the magic, version, dtype code, and rank, in file order, into *array; return 0, or print why and return 1. */
static inline int lassi_io_detail_fixed(const char *path, const unsigned char *bytes, size_t size,
                                        lassi_io_array *array) {
    uint32_t version = 0u;
    if (size < 8u && (size == 0u || memcmp(bytes, LASSI_IO_MAGIC, size) == 0)) {
        return lassi_io_detail_truncated(path);
    }
    if (size < 8u || memcmp(bytes, LASSI_IO_MAGIC, 8u) != 0) {
        return lassi_io_detail_fail(path, "bad magic: not a lassi_io file");
    }
    if (size < 12u) {
        return lassi_io_detail_truncated(path);
    }
    version = lassi_io_detail_get_u32(bytes + 8);
    if (version != LASSI_IO_VERSION) {
        return lassi_io_detail_fail_number(path, "unsupported version", version);
    }
    if (size < 16u) {
        return lassi_io_detail_truncated(path);
    }
    array->dtype = lassi_io_detail_get_u32(bytes + 12);
    if (lassi_io_itemsize(array->dtype) == 0u) {
        return lassi_io_detail_fail_number(path, "unknown dtype code", array->dtype);
    }
    if (size < 20u) {
        return lassi_io_detail_truncated(path);
    }
    array->rank = lassi_io_detail_get_u32(bytes + 16);
    if ((uint64_t)array->rank > (uint64_t)(size - 20u) / 8u) {
        return lassi_io_detail_truncated(path);
    }
    return 0;
}

/* Parse the whole header into *array and set *start to the data's offset; return 0, or print why and return 1. */
static inline int lassi_io_detail_header(const char *path, const unsigned char *bytes, size_t size,
                                         lassi_io_array *array, size_t *start) {
    size_t offset = 0u, length = 0u, end = 0u, i = 0u;
    if (lassi_io_detail_fixed(path, bytes, size, array) != 0) {
        return 1;
    }
    if (array->rank > 0u) {
        array->dims = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)array->rank);
        if (array->dims == NULL) {
            return lassi_io_detail_fail(path, "out of memory for the dims");
        }
        for (i = 0u; i < (size_t)array->rank; ++i) {
            array->dims[i] = lassi_io_detail_get_u64(bytes + 20u + 8u * i);
        }
    }
    offset = 20u + 8u * (size_t)array->rank;
    if (size - offset < 4u) {
        return lassi_io_detail_truncated(path);
    }
    length = (size_t)lassi_io_detail_get_u32(bytes + offset);
    if (length == 0u || length > LASSI_IO_MAX_NAME) {
        return lassi_io_detail_fail_number(path, "bad name length, not 1 to 255:", length);
    }
    offset += 4u;
    if (size - offset < length) {
        return lassi_io_detail_truncated(path);
    }
    if (!lassi_io_detail_name_ok((const char *)(bytes + offset), length)) {
        return lassi_io_detail_fail(path, "bad name: not only ASCII letters, digits, '_', '.', and '-'");
    }
    memcpy(array->name, bytes + offset, length);
    array->name[length] = '\0';
    offset += length;
    end = (offset + 7u) & ~(size_t)7u;
    if (size < end) {
        return lassi_io_detail_truncated(path);
    }
    for (i = offset; i < end; ++i) {
        if (bytes[i] != 0) {
            return lassi_io_detail_fail(path, "bad padding: the bytes after the name must be zero");
        }
    }
    *start = end;
    return 0;
}

/* Read the array in the file at path into *out, which it zeroes first whenever out is not NULL; return 0,
 * or print why and return 1 with *out zeroed. */
static inline int lassi_io_read(const char *path, lassi_io_array *out) {
    lassi_io_array array;
    unsigned char *bytes = NULL;
    size_t size = 0u, start = 0u, itemsize = 0u;
    uint64_t count = 0u, available = 0u;
    int status = 1;
    if (out != NULL) {
        memset(out, 0, sizeof *out);
    }
    memset(&array, 0, sizeof array);
    if (out == NULL || path == NULL) {
        return lassi_io_detail_fail(path, "no path or no array to read into");
    }
    if (!lassi_io_detail_little_endian()) {
        return lassi_io_detail_fail(path, "this host is big-endian; lassi_io files are little-endian");
    }
    if (lassi_io_detail_slurp(path, &bytes, &size) != 0) {
        return 1;
    }
    if (lassi_io_detail_header(path, bytes, size, &array, &start) == 0) {
        itemsize = lassi_io_itemsize(array.dtype);
        available = (uint64_t)(size - start);
        if (lassi_io_detail_count(array.rank, array.dims, &count) != 0 || count > ~(uint64_t)0u / itemsize) {
            lassi_io_detail_fail(path, "data size: the dims need more than 2**64 bytes");
        } else if (count * itemsize != available) {
            lassi_io_detail_fail_size(path, count * itemsize, available);
        } else {
            status = 0;
        }
    }
    if (status != 0) {
        free(array.dims);
        free(bytes);
        return 1;
    }
    array.count = count;
    array.nbytes = (size_t)(count * itemsize);
    array.data = bytes + start;
    array.storage = bytes;
    *out = array;
    return 0;
}

/* Write one array to path; dims may be NULL for rank 0 and data NULL for an empty array. Return 0, or print
 * why and return 1: a bad argument is refused before the file is created, and a failed write removes it. */
static inline int lassi_io_write(const char *path, const char *name, uint32_t dtype, uint32_t rank,
                                 const uint64_t *dims, const void *data) {
    const unsigned char zeros[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    size_t length = 0u, itemsize = lassi_io_itemsize(dtype), nbytes = 0u, padding = 0u;
    uint64_t count = 0u;
    uint32_t i;
    int ok = 1;
    FILE *file = NULL;
    if (path == NULL) {
        return lassi_io_detail_fail(path, "no path to write");
    }
    if (!lassi_io_detail_little_endian()) {
        return lassi_io_detail_fail(path, "this host is big-endian; lassi_io files are little-endian");
    }
    while (name != NULL && length <= LASSI_IO_MAX_NAME && name[length] != '\0') {
        ++length;
    }
    if (name == NULL || !lassi_io_detail_name_ok(name, length)) {
        return lassi_io_detail_fail(path, "bad name: a name is 1 to 255 ASCII letters, digits, '_', '.', or '-'");
    }
    if (itemsize == 0u) {
        return lassi_io_detail_fail_number(path, "unknown dtype code", dtype);
    }
    if (rank > 0u && dims == NULL) {
        return lassi_io_detail_fail_number(path, "no dims for rank", rank);
    }
    if (lassi_io_detail_count(rank, dims, &count) != 0 || count > (uint64_t)(~(size_t)0u) / itemsize) {
        return lassi_io_detail_fail(path, "data size: the dims need more bytes than this host can address");
    }
    nbytes = (size_t)count * itemsize;
    if (nbytes > 0u && data == NULL) {
        return lassi_io_detail_fail(path, "no data for an array that is not empty");
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return lassi_io_detail_fail(path, "cannot open the file for writing");
    }
    /* The header before the padding is 20 + 8 * rank + 4 + length bytes; 8 * rank adds nothing modulo 8. */
    padding = (8u - (24u + length) % 8u) % 8u;
    ok = lassi_io_detail_put(file, LASSI_IO_MAGIC, 8u) && lassi_io_detail_put_u32(file, LASSI_IO_VERSION) &&
         lassi_io_detail_put_u32(file, dtype) && lassi_io_detail_put_u32(file, rank);
    for (i = 0u; ok && i < rank; ++i) {
        ok = lassi_io_detail_put_u64(file, dims[i]);
    }
    ok = ok && lassi_io_detail_put_u32(file, (uint32_t)length) && lassi_io_detail_put(file, name, length) &&
         lassi_io_detail_put(file, zeros, padding) && lassi_io_detail_put(file, data, nbytes);
    if (fclose(file) != 0 || !ok) {
        remove(path);
        return lassi_io_detail_fail(path, "cannot write the file");
    }
    return 0;
}

#endif /* LASSI_IO_H */
