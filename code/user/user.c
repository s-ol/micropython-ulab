
/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020-2021 Zoltán Vörös
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "user.h"

#if ULAB_HAS_USER_MODULE
static ndarray_obj_t *get_or_create_dense_out(mp_obj_t out, uint8_t ndim, size_t *shape, uint8_t dtype) {
    if (out == mp_const_none) {
        return ndarray_new_dense_ndarray(ndim, shape, dtype);
    }

    if(!mp_obj_is_type(out, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("output must be an ndarray"));
    }

    ndarray_obj_t *results = MP_OBJ_TO_PTR(out);

    if(!ndarray_is_dense(results)) {
        mp_raise_TypeError(MP_ERROR_TEXT("output must be a dense ndarray"));
    }

    if(results->ndim != ndim || memcmp(results->shape, shape, sizeof(results->shape)) != 0) {
        mp_raise_TypeError(MP_ERROR_TEXT("output does not match the expected shape"));
    }

    if(results->dtype != dtype) {
        mp_raise_TypeError(MP_ERROR_TEXT("output does not match the expected dtype"));
    }

    return results;
}

static ndarray_obj_t *get_or_create_dense_out_like(mp_obj_t out, ndarray_obj_t *ndarray) {
    return get_or_create_dense_out(out, ndarray->ndim, ndarray->shape, ndarray->dtype);
}

static mp_obj_t user_hsv2rgb(mp_obj_t dest, mp_obj_t src) {
    // raise a TypeError exception, if the input is not an ndarray
    if(!mp_obj_is_type(src, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be an ndarray"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(src);

    if(ndarray->dtype != NDARRAY_UINT8) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be have dtype uint8"));
    }

    if (ndarray->ndim != 2 || ndarray->shape[ULAB_MAX_DIMS - 1] != 3) {
       mp_raise_TypeError(MP_ERROR_TEXT("input must have shape (N, 3)"));
    }

    ndarray_obj_t *results = get_or_create_dense_out_like(dest, ndarray);
    uint8_t *array = (uint8_t *)results->array;
    uint8_t *inputs = (uint8_t *)ndarray->array;
    int32_t *strides = ndarray->strides;

    size_t k = 0;
    do {
        do {
            // adapted from https://www.vagrearg.org/content/hsvrgb
            uint8_t *r = array+0;
            uint8_t *g = array+1;
            uint8_t *b = array+2;

            uint8_t s = *(inputs + (strides)[ULAB_MAX_DIMS - 1]);
            uint8_t v = *(inputs + (strides)[ULAB_MAX_DIMS - 1]*2);

            if(s == 0) {
                // grayscale case
                *r = *g = *b = v;
                break;
            }

            uint16_t h = *((uint8_t *)inputs) * ((uint16_t) 6);
            uint8_t sextant = h >> 8;

            // swap pointers to account for correct sextant
            uint8_t *tmp;
            if(sextant & 2) {
                tmp = r; r = b; b = tmp;
            }
            if(sextant & 4) {
                tmp = g; g = b; b = tmp; }
            if(!(sextant & 6)) {
                if(!(sextant & 1)) {
                    tmp = r; r = g; g = tmp;
                }
            } else {
                if (sextant & 1) {
                    tmp = r; r = g; g = tmp;
                }
            }

            // do the math
            *g = v;

            uint16_t ww;
            ww = v * (255 - s);
            ww += 1;
            ww += ww >> 8;
            *b = ww >> 8;

            uint8_t h_fraction = h & 0xff;
            uint32_t d;

            if(!(sextant & 1)) {
              d = v * (uint32_t)((255 << 8) - (uint16_t)(s * (256 - h_fraction)));
              d += d >> 8;
              d += v;
              *r = d >> 16;
            } else {
              d = v * (uint32_t)((255 << 8) - (uint16_t)(s * h_fraction));
              d += d >> 8;
              d += v;
              *r = d >> 16;
            }

        } while (0);

        array += 3;

        (inputs) += (strides)[ULAB_MAX_DIMS - 2];
        k++;
    } while(k < (results)->shape[ULAB_MAX_DIMS - 2]);

    // at the end, return a micropython object
    return MP_OBJ_FROM_PTR(results);
}

static mp_obj_t user_hsv2rgb16(mp_obj_t dest, mp_obj_t src) {
    // raise a TypeError exception, if the input is not an ndarray
    if(!mp_obj_is_type(src, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be an ndarray"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(src);

    if(ndarray->dtype != NDARRAY_INT16) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be have dtype int16"));
    }

    if (ndarray->ndim != 2 || ndarray->shape[ULAB_MAX_DIMS - 1] != 3) {
       mp_raise_TypeError(MP_ERROR_TEXT("input must have shape (N, 3)"));
    }

    ndarray_obj_t *results = get_or_create_dense_out(dest, ndarray->ndim, ndarray->shape, NDARRAY_UINT8);
    uint8_t *array = (uint8_t *)results->array;
    uint8_t *inputs = (uint8_t *)ndarray->array;
    int32_t *strides = ndarray->strides;

    size_t k = 0;
    do {
        do {
            // adapted from https://www.vagrearg.org/content/hsvrgb
            uint8_t *r = array+0;
            uint8_t *g = array+1;
            uint8_t *b = array+2;

            uint8_t s = *((int16_t *)(inputs + (strides)[ULAB_MAX_DIMS - 1]));
            uint8_t v = *((int16_t *)(inputs + (strides)[ULAB_MAX_DIMS - 1]*2));

            if(s == 0) {
                // grayscale case
                *r = *g = *b = v;
                break;
            }

            uint16_t h = *((int16_t *)inputs) * 6;
            uint8_t sextant = h >> 8;

            // swap pointers to account for correct sextant
            uint8_t *tmp;
            if(sextant & 2) {
                tmp = r; r = b; b = tmp;
            }
            if(sextant & 4) {
                tmp = g; g = b; b = tmp; }
            if(!(sextant & 6)) {
                if(!(sextant & 1)) {
                    tmp = r; r = g; g = tmp;
                }
            } else {
                if (sextant & 1) {
                    tmp = r; r = g; g = tmp;
                }
            }

            // do the math
            *g = v;

            uint16_t ww;
            ww = v * (255 - s);
            ww += 1;
            ww += ww >> 8;
            *b = ww >> 8;

            uint8_t h_fraction = h & 0xff;
            uint32_t d;

            if(!(sextant & 1)) {
              d = v * (uint32_t)((255 << 8) - (uint16_t)(s * (256 - h_fraction)));
              d += d >> 8;
              d += v;
              *r = d >> 16;
            } else {
              d = v * (uint32_t)((255 << 8) - (uint16_t)(s * h_fraction));
              d += d >> 8;
              d += v;
              *r = d >> 16;
            }
        } while (0);

        array += 3;

        (inputs) += (strides)[ULAB_MAX_DIMS - 2];
        k++;
    } while(k < (results)->shape[ULAB_MAX_DIMS - 2]);

    // at the end, return a micropython object
    return MP_OBJ_FROM_PTR(results);
}

static mp_obj_t user_inplace_wrap8(mp_obj_t src) {
    // raise a TypeError exception, if the input is not an ndarray
    if(!mp_obj_is_type(src, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be an ndarray"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(src);

    if(ndarray->dtype != NDARRAY_INT16) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be have dtype int16"));
    }

    LOOP1_START(ndarray, val)
        *((int16_t *)val) = (*((int16_t *)val) + 256) % 256;
    LOOP1_END(ndarray->shape, ndarray, val)

    return mp_const_none;
}

static mp_obj_t user_inplace_clamp8(mp_obj_t src) {
    // raise a TypeError exception, if the input is not an ndarray
    if(!mp_obj_is_type(src, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be an ndarray"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(src);

    if(ndarray->dtype != NDARRAY_INT16) {
        mp_raise_TypeError(MP_ERROR_TEXT("input must be have dtype int16"));
    }

    LOOP1_START(ndarray, val)
        int16_t tmp = *((int16_t *)val);
        tmp = tmp < 0 ? 0 : tmp;
        tmp = tmp > 255 ? 255 : tmp;
        *((int16_t *)val) = tmp;
    LOOP1_END(ndarray->shape, ndarray, val)

    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_2(user_hsv2rgb_obj, user_hsv2rgb);
MP_DEFINE_CONST_FUN_OBJ_2(user_hsv2rgb16_obj, user_hsv2rgb16);
MP_DEFINE_CONST_FUN_OBJ_1(user_inplace_wrap8_obj, user_inplace_wrap8);
MP_DEFINE_CONST_FUN_OBJ_1(user_inplace_clamp8_obj, user_inplace_clamp8);

static const mp_rom_map_elem_t ulab_user_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_user) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_hsv2rgb), (mp_obj_t)&user_hsv2rgb_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_hsv2rgb16), (mp_obj_t)&user_hsv2rgb16_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_inplace_wrap8), (mp_obj_t)&user_inplace_wrap8_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_inplace_clamp8), (mp_obj_t)&user_inplace_clamp8_obj },
};

static MP_DEFINE_CONST_DICT(mp_module_ulab_user_globals, ulab_user_globals_table);

const mp_obj_module_t ulab_user_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_user_globals,
};
#if CIRCUITPY_ULAB
MP_REGISTER_MODULE(MP_QSTR_ulab_dot_user, ulab_user_module);
#endif
#endif

