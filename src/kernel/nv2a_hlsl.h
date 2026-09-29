/**
 * HLSL for the NV2A's two programmable stages, for the D3D11 backend.
 *
 * The software executor interprets both on the CPU (nv2a_vp.c, nv2a_combiner.c);
 * this turns the same state into shader source instead, so a vertex program
 * or a combiner setup costs one compile rather than a CPU pass per vertex or
 * per pixel. Both generators follow their interpreter line for line -- the
 * interpreters are the reference that has already been checked against the
 * title's output, and the point is for the GPU to draw the same picture.
 *
 * The interface the generated shaders share with nv2a_d3d11.c:
 *
 *   VS input   ATTR0..ATTR15, uint4 each: the raw attribute bits, decoded in
 *              the shader by g_afmt[i] (see NV2A_HLSL_AFMT_*), so one input
 *              signature serves every vertex format.
 *   VS b0      float4 c[192]: the program's constant memory.
 *   VS b1      Nv2aHlslVsParams.
 *   VS -> PS   SV_Position, COLOR0, COLOR1, TEXCOORD0..3, FOG.
 *   PS b0      Nv2aHlslPsParams.
 *   PS t0..t3  the four texture stages, s0..s3 their samplers.
 */
#ifndef NV2A_HLSL_H
#define NV2A_HLSL_H

#include <stddef.h>
#include <stdint.h>
#include "nv2a_vp.h"
#include "nv2a_combiner.h"

/* How the vertex shader decodes attribute i (g_afmt[i].x). */
enum {
    NV2A_HLSL_AFMT_ABSENT = 0,   /* use g_adef[i] */
    NV2A_HLSL_AFMT_FLOAT  = 1,   /* R32 x size, bits as float          */
    NV2A_HLSL_AFMT_COLOR  = 2,   /* D3DCOLOR: bytes B,G,R,A            */
    NV2A_HLSL_AFMT_UBYTE  = 3,   /* normalised unsigned bytes          */
    NV2A_HLSL_AFMT_S1     = 4,   /* normalised signed shorts           */
    NV2A_HLSL_AFMT_S32K   = 5,   /* signed shorts, as integers         */
    NV2A_HLSL_AFMT_CMP    = 6    /* packed 11:11:10 normal in one dword */
};

typedef struct {
    float    adef[16][4];        /* value of an absent attribute        */
    uint32_t afmt[16][4];        /* x = NV2A_HLSL_AFMT_*, y = components */
    float    vp[4];              /* 2/width, 2/height, 1/zmax, unused   */
    uint32_t roles[4];           /* pass-through: diffuse, tex0 attribute */
} Nv2aHlslVsParams;

typedef struct {
    float    cf0[8][4], cf1[8][4];   /* per-stage combiner constants    */
    float    ff0[4], ff1[4];         /* final combiner constants        */
    float    fogc[4];                /* fog colour                       */
    float    tsc[4][4];              /* per stage: uv scale x,y; z alpha=1; w bound */
    uint32_t atest[4];               /* alpha test: enable, func, ref   */
} Nv2aHlslPsParams;

/* The combiner state a pixel shader is compiled from: everything that
 * changes the code, nothing that only changes a constant. */
typedef struct {
    uint32_t programmed;
    uint32_t count;                  /* stages in use, 0..8             */
    uint32_t control;                /* mux-msb and unique-constant bits */
    uint32_t stage_program;
    uint32_t color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8];
    uint32_t final_cw0, final_cw1;
} Nv2aHlslPsKey;

void nv2a_hlsl_ps_key(const Nv2aCombiner *c, Nv2aHlslPsKey *key);

/* Source for the program running at vp->program_start. Returns the length
 * written, or 0 if the program is empty or did not fit. *writes_consts is set
 * if the program stores into constant memory, which a shader cannot do. */
size_t nv2a_hlsl_vs(const Nv2aVpState *vp, char *buf, size_t size,
                    int *writes_consts);

/* The vertex shader for pre-transformed (screen-space) vertices. */
size_t nv2a_hlsl_vs_passthrough(char *buf, size_t size);

/* The pixel shader for a combiner setup. */
size_t nv2a_hlsl_ps(const Nv2aHlslPsKey *key, char *buf, size_t size);

#endif
