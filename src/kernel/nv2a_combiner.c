/**
 * NV2A register combiners on the CPU. See nv2a_combiner.h.
 *
 * The register encodings are the ones Xbox D3D writes, which are its PS_*
 * macros (xgraphics/xbox pixel-shader headers) and what xemu's psh.c parses:
 *
 *   input byte   [3:0] register, [4] channel (RGB/BLUE = 0, ALPHA = 1),
 *                [7:5] mapping;  A B C D from the top byte down
 *   output word  [3:0] CD dst, [7:4] AB dst, [11:8] SUM dst, [31:12] flags:
 *                0x01 CD dot, 0x02 AB dot, 0x04 mux, 0x38 scale/bias,
 *                0x40 CD blue-to-alpha, 0x80 AB blue-to-alpha
 *   control      [7:0] stage count, [8] mux on MSB, [12] unique C0 per
 *                stage, [16] unique C1 per stage
 *   final        CW0 = A B C D, CW1 = E F G, then flags: 0x80 clamp the
 *                V1+R0 sum, 0x40 complement V1, 0x20 complement R0
 *
 * Registers: 0 zero, 1 C0, 2 C1, 3 fog, 4 V0 (diffuse), 5 V1 (specular),
 * 8-11 T0-T3, 12 R0, 13 R1, and for the final stage only 14 V1+R0, 15 E*F.
 * R0's alpha starts as T0's alpha, as the hardware does.
 */
#include "nv2a_combiner.h"

#include <string.h>

enum {
    R_ZERO = 0, R_C0 = 1, R_C1 = 2, R_FOG = 3, R_V0 = 4, R_V1 = 5,
    R_T0 = 8, R_R0 = 12, R_R1 = 13, R_SUM = 14, R_EF = 15
};

static int comb_method(Nv2aCombiner *c, uint32_t method, uint32_t param)
{
    if (method >= 0x0260 && method < 0x0280) {             /* ALPHA_ICW */
        c->alpha_icw[(method - 0x0260) / 4] = param;
        return 1;
    }
    if (method >= 0x0A60 && method < 0x0A80) {             /* FACTOR0 */
        c->factor0[(method - 0x0A60) / 4] = param;
        return 1;
    }
    if (method >= 0x0A80 && method < 0x0AA0) {             /* FACTOR1 */
        c->factor1[(method - 0x0A80) / 4] = param;
        return 1;
    }
    if (method >= 0x0AA0 && method < 0x0AC0) {             /* ALPHA_OCW */
        c->alpha_ocw[(method - 0x0AA0) / 4] = param;
        return 1;
    }
    if (method >= 0x0AC0 && method < 0x0AE0) {             /* COLOR_ICW */
        c->color_icw[(method - 0x0AC0) / 4] = param;
        return 1;
    }
    if (method >= 0x1E40 && method < 0x1E60) {             /* COLOR_OCW */
        c->color_ocw[(method - 0x1E40) / 4] = param;
        return 1;
    }
    switch (method) {
    case 0x0288: c->final_cw0 = param; return 1;           /* SPECULAR_FOG_CW0 */
    case 0x028C: c->final_cw1 = param; return 1;           /* SPECULAR_FOG_CW1 */
    case 0x02A8: c->fog_color = param; return 1;           /* FOG_COLOR */
    case 0x1E20: c->final_factor0 = param; return 1;       /* SPECULAR_FOG_FACTOR */
    case 0x1E24: c->final_factor1 = param; return 1;
    case 0x1E60: c->control = param; c->programmed = 1; return 1;
    case 0x1E70: c->stage_program = param; return 1;
    case 0x1E78: c->other_stage_input = param; return 1;
    default: return 0;
    }
}

int nv2a_comb_method(Nv2aCombiner *c, uint32_t method, uint32_t param)
{
    if (!comb_method(c, method, param))
        return 0;
    c->gen++;
    return 1;
}

uint32_t nv2a_comb_stage_mode(const Nv2aCombiner *c, int stage)
{
    return (c->stage_program >> (5 * stage)) & 0x1Fu;
}

static void unpack(uint32_t argb, float v[4])
{
    v[0] = (float)((argb >> 16) & 0xFF) / 255.0f;
    v[1] = (float)((argb >>  8) & 0xFF) / 255.0f;
    v[2] = (float)( argb        & 0xFF) / 255.0f;
    v[3] = (float)( argb >> 24        ) / 255.0f;
}

static float clamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float map_input(float x, uint32_t mapping)
{
    switch (mapping) {
    case 0: return clamp(x, 0.0f, 1.0f);                   /* UNSIGNED_IDENTITY */
    case 1: return 1.0f - clamp(x, 0.0f, 1.0f);            /* UNSIGNED_INVERT */
    case 2: return 2.0f * clamp(x, 0.0f, 1.0f) - 1.0f;     /* EXPAND_NORMAL */
    case 3: return -2.0f * clamp(x, 0.0f, 1.0f) + 1.0f;    /* EXPAND_NEGATE */
    case 4: return clamp(x, 0.0f, 1.0f) - 0.5f;            /* HALFBIAS_NORMAL */
    case 5: return -clamp(x, 0.0f, 1.0f) + 0.5f;           /* HALFBIAS_NEGATE */
    case 6: return x;                                      /* SIGNED_IDENTITY */
    default: return -x;                                    /* SIGNED_NEGATE */
    }
}

static float scale_bias(float x, uint32_t flags)
{
    switch (flags & 0x38u) {
    case 0x08: x = x - 0.5f; break;                        /* BIAS */
    case 0x10: x = x * 2.0f; break;                        /* SHIFTLEFT_1 */
    case 0x18: x = (x - 0.5f) * 2.0f; break;               /* SHIFTLEFT_1_BIAS */
    case 0x20: x = x * 4.0f; break;                        /* SHIFTLEFT_2 */
    case 0x30: x = x * 0.5f; break;                        /* SHIFTRIGHT_1 */
    default: break;
    }
    return clamp(x, -1.0f, 1.0f);
}

/* Read one input: four components, each already mapped. alpha_sel picks the
 * alpha channel (broadcast) instead of RGB for a colour input, or alpha
 * instead of blue for an alpha input. */
static void read_input(float regs[16][4], uint32_t byte, int for_alpha,
                       float out[4])
{
    uint32_t reg = byte & 0xF, alpha_sel = (byte >> 4) & 1, mapping = (byte >> 5) & 7;
    const float *r = regs[reg];
    int k;

    if (for_alpha) {
        float v = map_input(alpha_sel ? r[3] : r[2], mapping);
        out[0] = out[1] = out[2] = out[3] = v;
        return;
    }
    for (k = 0; k < 3; k++)
        out[k] = map_input(alpha_sel ? r[3] : r[k], mapping);
    out[3] = map_input(r[3], mapping);
}

static int mux_select(const float r0[4], uint32_t control)
{
    float a = clamp(r0[3], 0.0f, 1.0f);
    if (control & 0x100u)                                  /* MUX_MSB */
        return a >= 0.5f;
    return ((uint32_t)(a * 255.0f + 0.5f)) & 1u;           /* MUX_LSB */
}

void nv2a_comb_eval(const Nv2aCombiner *c, const float v0[4], const float v1[4],
                    float fog_factor, const float t[4][4], float out[4])
{
    float regs[16][4];
    uint32_t count = c->control & 0xFFu, i;
    int k;

    memset(regs, 0, sizeof regs);
    memcpy(regs[R_V0], v0, sizeof regs[0]);
    memcpy(regs[R_V1], v1, sizeof regs[0]);
    for (k = 0; k < 4; k++)
        memcpy(regs[R_T0 + k], t[k], sizeof regs[0]);
    unpack(c->fog_color, regs[R_FOG]);
    {   /* SET_FOG_COLOR keeps red in the low byte (A B G R). */
        float r = regs[R_FOG][2];
        regs[R_FOG][2] = regs[R_FOG][0];
        regs[R_FOG][0] = r;
    }
    regs[R_FOG][3] = fog_factor;
    regs[R_R0][3] = t[0][3];                               /* R0.a = T0.a */

    if (count > 8)
        count = 8;
    for (i = 0; i < count; i++) {
        float a[4], b[4], cc[4], d[4];
        float col_ab[4], col_cd[4], col_sum[4];
        float al_ab, al_cd, al_sum;
        uint32_t ci = c->color_icw[i], co = c->color_ocw[i];
        uint32_t ai = c->alpha_icw[i], ao = c->alpha_ocw[i];
        uint32_t cflags = co >> 12, aflags = ao >> 12;
        int sel = mux_select(regs[R_R0], c->control);

        unpack(c->factor0[(c->control & 0x1000u) ? i : 0], regs[R_C0]);
        unpack(c->factor1[(c->control & 0x10000u) ? i : 0], regs[R_C1]);

        /* Colour half. */
        read_input(regs, ci >> 24, 0, a);
        read_input(regs, ci >> 16, 0, b);
        read_input(regs, ci >>  8, 0, cc);
        read_input(regs, ci,       0, d);
        if (cflags & 0x02u) {
            float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            col_ab[0] = col_ab[1] = col_ab[2] = dot;
        } else {
            for (k = 0; k < 3; k++) col_ab[k] = a[k] * b[k];
        }
        if (cflags & 0x01u) {
            float dot = cc[0] * d[0] + cc[1] * d[1] + cc[2] * d[2];
            col_cd[0] = col_cd[1] = col_cd[2] = dot;
        } else {
            for (k = 0; k < 3; k++) col_cd[k] = cc[k] * d[k];
        }
        for (k = 0; k < 3; k++) {
            col_sum[k] = (cflags & 0x04u) ? (sel ? col_cd[k] : col_ab[k])
                                          : col_ab[k] + col_cd[k];
            col_ab[k]  = scale_bias(col_ab[k],  cflags);
            col_cd[k]  = scale_bias(col_cd[k],  cflags);
            col_sum[k] = scale_bias(col_sum[k], cflags);
        }

        /* Alpha half. */
        read_input(regs, ai >> 24, 1, a);
        read_input(regs, ai >> 16, 1, b);
        read_input(regs, ai >>  8, 1, cc);
        read_input(regs, ai,       1, d);
        al_ab = a[0] * b[0];
        al_cd = cc[0] * d[0];
        al_sum = (aflags & 0x04u) ? (sel ? al_cd : al_ab) : al_ab + al_cd;
        al_ab  = scale_bias(al_ab,  aflags);
        al_cd  = scale_bias(al_cd,  aflags);
        al_sum = scale_bias(al_sum, aflags);

        /* Writes, after both halves have read. Register 0 discards. */
        {
            uint32_t cd_dst = co & 0xF, ab_dst = (co >> 4) & 0xF, sum_dst = (co >> 8) & 0xF;
            uint32_t acd = ao & 0xF, aab = (ao >> 4) & 0xF, asum = (ao >> 8) & 0xF;
            if (ab_dst)  for (k = 0; k < 3; k++) regs[ab_dst][k]  = col_ab[k];
            if (cd_dst)  for (k = 0; k < 3; k++) regs[cd_dst][k]  = col_cd[k];
            if (sum_dst) for (k = 0; k < 3; k++) regs[sum_dst][k] = col_sum[k];
            if (aab)  regs[aab][3]  = al_ab;
            if (acd)  regs[acd][3]  = al_cd;
            if (asum) regs[asum][3] = al_sum;
            /* Blue-to-alpha: a dot product's result also lands in alpha. */
            if ((cflags & 0x80u) && ab_dst) regs[ab_dst][3] = col_ab[2];
            if ((cflags & 0x40u) && cd_dst) regs[cd_dst][3] = col_cd[2];
        }
    }

    /* Final combiner. */
    {
        uint32_t cw0 = c->final_cw0, cw1 = c->final_cw1, fl = cw1 & 0xFFu;
        float A[4], B[4], C[4], D[4], E[4], F[4], G[4];

        unpack(c->final_factor0, regs[R_C0]);
        unpack(c->final_factor1, regs[R_C1]);
        for (k = 0; k < 4; k++) {
            float r0 = (fl & 0x20u) ? 1.0f - regs[R_R0][k] : regs[R_R0][k];
            float s1 = (fl & 0x40u) ? 1.0f - regs[R_V1][k] : regs[R_V1][k];
            regs[R_SUM][k] = r0 + s1;
            if (fl & 0x80u)
                regs[R_SUM][k] = clamp(regs[R_SUM][k], 0.0f, 1.0f);
        }
        /* The final stage's mappings are unsigned identity or invert only. */
        read_input(regs, (cw1 >> 24) & 0x3F, 0, E);
        read_input(regs, (cw1 >> 16) & 0x3F, 0, F);
        for (k = 0; k < 3; k++)
            regs[R_EF][k] = E[k] * F[k];
        read_input(regs, (cw0 >> 24) & 0x3F, 0, A);
        read_input(regs, (cw0 >> 16) & 0x3F, 0, B);
        read_input(regs, (cw0 >>  8) & 0x3F, 0, C);
        read_input(regs,  cw0        & 0x3F, 0, D);
        read_input(regs, (cw1 >>  8) & 0x3F, 1, G);
        for (k = 0; k < 3; k++)
            out[k] = clamp(A[k] * B[k] + (1.0f - A[k]) * C[k] + D[k], 0.0f, 1.0f);
        out[3] = clamp(G[0], 0.0f, 1.0f);
    }
}
