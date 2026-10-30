/**
 * NV2A register combiners, evaluated on the CPU for the software pushbuffer
 * executors.
 *
 * Every Xbox D3D pixel goes through these: the fixed-function texture stage
 * states a title sets are compiled by D3D into combiner programs, and titles
 * with their own pixel shaders write them directly. Up to eight general stages
 * combine the textures, the two vertex colours, fog and two constants per
 * stage into registers; a final stage produces the pixel's colour and alpha.
 * The executor used to stand in with "texel * diffuse" from stage 0, which is
 * right for a menu quad and wrong for anything that mixes two textures --
 * Burnout 3 composites its whole 3D scene into the frame that way.
 *
 * Semantics follow xemu's hw/xbox/nv2a/pgraph/psh.c (glsl/psh.c) and NVIDIA's
 * NV_register_combiners: input mappings, AB/CD products or dot products, mux,
 * output scale and bias, the final A*B + (1-A)*C + D. Colours are R, G, B, A
 * floats; texture and vertex inputs arrive in [0,1]. Portable C, no graphics
 * API.
 *
 * Two evaluators share the register state, each written for its executor:
 *
 *   nv2a_comb_*  nv2a_pb_exec.c's software path, and the reference that
 *                nv2a_hlsl.c's generated pixel shaders were transcribed from
 *                (the D3D11 backend);
 *   nv2a_rc_*    nv2a_pb_exec_upstream.c's threaded rasteriser
 *                (tests/nv2a_combiner checks it).
 *
 * They differ in a few edge cases (nv2a_combiner.c lists them); each keeps the
 * behaviour its executor was verified with.
 */
#ifndef NV2A_COMBINER_H
#define NV2A_COMBINER_H

#include <stdint.h>

typedef struct {
    uint32_t color_icw[8], color_ocw[8];    /* SET_COMBINER_COLOR_ICW/OCW */
    uint32_t alpha_icw[8], alpha_ocw[8];    /* SET_COMBINER_ALPHA_ICW/OCW */
    uint32_t factor0[8], factor1[8];        /* SET_COMBINER_FACTOR0/1 (ARGB) */
    /* SET_COMBINER_SPECULAR_FOG_CW0/1 and SET_SPECULAR_FOG_FACTOR. Each has
     * the name either evaluator was written with. */
    union { uint32_t final_cw0;     uint32_t final0;   };
    union { uint32_t final_cw1;     uint32_t final1;   };
    union { uint32_t final_factor0; uint32_t final_c0; };
    union { uint32_t final_factor1; uint32_t final_c1; };
    uint32_t control;             /* SET_COMBINER_CONTROL */
    uint32_t stage_program;       /* SET_SHADER_STAGE_PROGRAM */
    uint32_t other_stage_input;   /* SET_SHADER_OTHER_STAGE_INPUT */
    uint32_t dot_mapping;         /* SET_DOT_RGBMAPPING */
    uint32_t fog_color;           /* ARGB */
    int      programmed;          /* COMBINER_CONTROL has been written */
    uint32_t gen;                 /* bumped by every combiner method */
} Nv2aCombiner;

/* Shader stage modes (SET_SHADER_STAGE_PROGRAM, five bits per stage). */
enum {
    NV2A_TEXMODE_NONE = 0, NV2A_TEXMODE_2D_PROJECTIVE = 1,
    NV2A_TEXMODE_3D_PROJECTIVE = 2, NV2A_TEXMODE_CUBE = 3,
    NV2A_TEXMODE_PASS_THROUGH = 4, NV2A_TEXMODE_CLIP_PLANE = 5
};

/* ── nv2a_pb_exec.c's evaluator ─────────────────────────────────────────── */

/* Consume a combiner method. Returns 1 if the method belonged here. */
int nv2a_comb_method(Nv2aCombiner *c, uint32_t method, uint32_t param);

/* The mode of texture stage 0..3. */
uint32_t nv2a_comb_stage_mode(const Nv2aCombiner *c, int stage);

/* One pixel. v0/v1 are the diffuse and specular colours, fog_factor the
 * fog blend weight (1 = no fog), t[i] the value texture stage i produced
 * (zero for a stage in mode NONE). out receives R, G, B, A in [0,1]. */
void nv2a_comb_eval(const Nv2aCombiner *c, const float v0[4], const float v1[4],
                    float fog_factor, const float t[4][4], float out[4]);

/* ── nv2a_pb_exec_upstream.c's evaluator ────────────────────────────────── */

/* One pixel. Colours are r,g,b,a in [0,1]; `fog` is (fog colour, factor).
 * t[i] is what texture stage i produced. The result is clamped to [0,1].
 * Reads final0/final1/final_c0/final_c1 and the stage words; fog_color,
 * programmed and gen are not used (the caller passes fog in). */
void nv2a_rc_eval(const Nv2aCombiner *rc, const float v0[4],
                  const float v1[4], const float fog[4],
                  const float t[4][4], float out[4]);

/* D3DCOLOR (0xAARRGGBB) to r,g,b,a floats. */
void nv2a_rc_unpack(uint32_t argb, float out[4]);

#endif
