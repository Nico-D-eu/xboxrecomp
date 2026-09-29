/**
 * The pushbuffer executor's GPU backend: NV2A draws on Direct3D 11.
 *
 * nv2a_pb_exec.c decodes the command stream and keeps the NV2A state; with
 * RECOMP_GPU=d3d11 it hands each draw, clear and flip to this instead of its
 * own rasteriser. Vertex programs and combiner setups become HLSL
 * (nv2a_hlsl.c), guest textures and render targets become D3D11 resources
 * cached by address, and a finished frame is read back into the framebuffer
 * window -- a frame or two late, so the GPU is never waited on.
 *
 * Everything here runs on the thread that executes the pushbuffer.
 */
#ifndef NV2A_D3D11_H
#define NV2A_D3D11_H

#include <stdint.h>
#include "nv2a_vp.h"
#include "nv2a_combiner.h"

typedef struct {
    uint32_t offset;        /* guest VA, or byte offset into `inline_base` */
    uint32_t type, size, stride;   /* NV097 format fields; size 0 = absent */
} Nv2aHwAttr;

typedef struct {
    uint32_t va;            /* resolved guest VA of level 0 */
    uint32_t format;        /* raw SET_TEXTURE_FORMAT */
    uint32_t color;         /* colour-format code */
    uint32_t width, height, pitch;
    uint32_t address;       /* raw SET_TEXTURE_ADDRESS */
    uint32_t filter;        /* raw SET_TEXTURE_FILTER */
    uint32_t border;        /* raw SET_TEXTURE_BORDER_COLOR */
    int      enabled;       /* valid and CONTROL0 enable */
} Nv2aHwTexture;

typedef struct {
    /* Targets: resolved guest VAs, 0 = none. */
    uint32_t color_va, zeta_va;
    uint32_t pitch, zpitch, format;
    uint32_t clip_x, clip_y, clip_w, clip_h;

    uint32_t depth_test, depth_func, depth_mask;
    uint32_t stencil_test, stencil_mask, stencil_func, stencil_ref,
             stencil_func_mask, stencil_fail, stencil_zfail, stencil_zpass;
    uint32_t alpha_test, alpha_func, alpha_ref;
    uint32_t blend_enable, blend_sfactor, blend_dfactor, blend_eq, blend_color;
    uint32_t color_mask;
    uint32_t cull_enable, cull_face, front_face;
    uint32_t fog_enable, fog_mode;
    float    fog_param[3];

    /* Geometry. */
    uint32_t prim;
    const uint32_t *idx;
    uint32_t idx_count;
    Nv2aHwAttr attr[16];
    const uint8_t *inline_base;    /* non-NULL: offsets are into this */
    uint32_t inline_bytes;
    int use_vp;                    /* else pre-transformed */
    uint32_t diffuse_attr, tex0_attr;   /* pre-transformed roles */
    const float (*cur_attr)[4];    /* current values, for absent attributes */
    uint32_t cur_attr_set;         /* which of them the title has set */

    const Nv2aVpState  *vp;
    const Nv2aCombiner *comb;
    Nv2aHwTexture tex[4];
} Nv2aHwDraw;

/* Is the backend in use? Initialises it on first call when RECOMP_GPU=d3d11;
 * 0 if it is not asked for or the device could not be made, and the
 * software path carries on. */
int  nv2a_d3d11_active(void);

void nv2a_d3d11_draw(const Nv2aHwDraw *d);

/* CLEAR_SURFACE: `flags` is its parameter (0x01 Z, 0x02 stencil, 0xF0 colour). */
void nv2a_d3d11_clear(const Nv2aHwDraw *target, uint32_t flags,
                      uint32_t color, uint32_t zclear);

/* The title flipped to the surface at `color_va`. */
void nv2a_d3d11_flip(uint32_t color_va);

/* Presented frames the render thread has still to show, or -1 when there is
 * no render thread (RECOMP_GPU_THREAD=0). The executor's flip waits on it. */
int  nv2a_d3d11_frames_in_flight(void);

void nv2a_d3d11_report(void);

#endif
