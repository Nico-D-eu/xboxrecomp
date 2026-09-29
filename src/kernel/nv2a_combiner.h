/**
 * NV2A register combiners, evaluated on the CPU for the software pushbuffer
 * executor.
 *
 * Every Xbox D3D pixel goes through these: the fixed-function texture stage
 * states a title sets are compiled by D3D into combiner programs, and titles
 * with their own pixel shaders write them directly. Up to eight general stages
 * combine the textures, the two vertex colours, fog and two constants per
 * stage into registers; a final stage produces the pixel's colour and alpha.
 *
 * Semantics follow xemu's hw/xbox/nv2a/pgraph/psh.c and NVIDIA's
 * NV_register_combiners: input mappings, AB/CD products or dot products, mux,
 * output scale and bias, the final A*B + (1-A)*C + D. Colours are R, G, B, A
 * floats; texture and vertex inputs arrive in [0,1].
 */
#ifndef NV2A_COMBINER_H
#define NV2A_COMBINER_H

#include <stdint.h>

typedef struct {
    uint32_t color_icw[8], color_ocw[8];
    uint32_t alpha_icw[8], alpha_ocw[8];
    uint32_t factor0[8], factor1[8];
    uint32_t final_cw0, final_cw1;
    uint32_t final_factor0, final_factor1;
    uint32_t control;             /* SET_COMBINER_CONTROL */
    uint32_t stage_program;       /* SET_SHADER_STAGE_PROGRAM */
    uint32_t other_stage_input;
    uint32_t fog_color;           /* ARGB */
    int      programmed;          /* COMBINER_CONTROL has been written */
} Nv2aCombiner;

/* Shader stage modes (SET_SHADER_STAGE_PROGRAM, five bits per stage). */
enum {
    NV2A_TEXMODE_NONE = 0, NV2A_TEXMODE_2D_PROJECTIVE = 1,
    NV2A_TEXMODE_3D_PROJECTIVE = 2, NV2A_TEXMODE_CUBE = 3,
    NV2A_TEXMODE_PASS_THROUGH = 4, NV2A_TEXMODE_CLIP_PLANE = 5
};

/* Consume a combiner method. Returns 1 if the method belonged here. */
int nv2a_comb_method(Nv2aCombiner *c, uint32_t method, uint32_t param);

/* The mode of texture stage 0..3. */
uint32_t nv2a_comb_stage_mode(const Nv2aCombiner *c, int stage);

/* One pixel. v0/v1 are the diffuse and specular colours, fog_factor the
 * fog blend weight (1 = no fog), t[i] the value texture stage i produced
 * (zero for a stage in mode NONE). out receives R, G, B, A in [0,1]. */
void nv2a_comb_eval(const Nv2aCombiner *c, const float v0[4], const float v1[4],
                    float fog_factor, const float t[4][4], float out[4]);

#endif
