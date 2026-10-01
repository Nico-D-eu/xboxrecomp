/**
 * NV2A vertex programs and register combiners as HLSL. See nv2a_hlsl.h.
 *
 * Both generators are transcriptions of their interpreters rather than fresh
 * readings of the hardware documentation: nv2a_vp.c's field table and
 * operation semantics, nv2a_combiner.c's input mapping, scale/bias, mux and
 * final-combiner arithmetic. Where the interpreter makes a choice (0 times
 * anything is 0, R0.a starts as T0.a, which unit's result a slot writes to
 * R1) the shader makes the same one, so the two paths can be compared pixel
 * for pixel on the same frame.
 */
#include "nv2a_hlsl.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char  *p;
    size_t size, len;
    int    overflow;
} Sb;

static void sb(Sb *s, const char *fmt, ...)
{
    va_list ap;
    int n;

    if (s->overflow)
        return;
    va_start(ap, fmt);
    n = vsnprintf(s->p + s->len, s->size - s->len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= s->size - s->len) {
        s->overflow = 1;
        return;
    }
    s->len += (size_t)n;
}

static size_t sb_done(Sb *s)
{
    return s->overflow ? 0 : s->len;
}

/* ── Vertex programs ─────────────────────────────────────────────────── */

enum { MUX_R = 1, MUX_V = 2, MUX_C = 3 };
enum {
    MAC_NOP, MAC_MOV, MAC_MUL, MAC_ADD, MAC_MAD, MAC_DP3, MAC_DPH, MAC_DP4,
    MAC_DST, MAC_MIN, MAC_MAX, MAC_SLT, MAC_SGE, MAC_ARL
};
enum { ILU_NOP, ILU_MOV, ILU_RCP, ILU_RCC, ILU_RSQ, ILU_EXP, ILU_LOG, ILU_LIT };

static uint32_t fld(const uint32_t *w, int dword, int start, int len)
{
    return (w[dword] >> start) & ((1u << len) - 1u);
}

/* The declarations every vertex shader starts with: the constant buffers,
 * the one input signature, the attribute decoder and the helpers that give
 * the vertex engine's arithmetic. */
static void vs_common(Sb *s)
{
    int i;

    sb(s, "cbuffer VSC : register(b0) { float4 c[192]; };\n"
          "cbuffer VSD : register(b1) {\n"
          "  float4 g_adef[16]; uint4 g_afmt[16]; float4 g_vp; uint4 g_roles;\n"
          "  float4 g_fog;   /* param 0, param 1, mode (0 off, 1 linear, 2 exp, 3 exp2), abs */\n"
          "};\n"
          "struct VSIn {\n");
    for (i = 0; i < 16; i++)
        sb(s, "  uint4 a%d : ATTR%d;\n", i, i);
    sb(s, "};\n"
          "struct VSOut {\n"
          "  float4 pos : SV_Position; float4 d0 : COLOR0; float4 d1 : COLOR1;\n"
          "  float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1;\n"
          "  float4 t2 : TEXCOORD2; float4 t3 : TEXCOORD3; float fog : FOG;\n"
          "};\n"
          /* The raw bits of one attribute, as the title laid it out. */
          "float4 decode(uint4 r, uint4 f, float4 def) {\n"
          "  float4 v;\n"
          "  [branch] switch (f.x) {\n"
          "  case 1: v = asfloat(r); break;\n"
          "  case 2: v = float4(r.z, r.y, r.x, r.w) / 255.0; break;\n"
          "  case 3: v = float4(r) / 255.0; break;\n"
          "  case 4: v = max(float4((int4)(r << 16) >> 16) / 32767.0, -1.0); break;\n"
          "  case 5: v = float4((int4)(r << 16) >> 16); break;\n"
          "  case 6: {\n"
          "    uint u = r.x;\n"
          "    v = float4(float((int)(u << 21) >> 21) / 1023.0,\n"
          "               float((int)(u << 10) >> 21) / 1023.0,\n"
          "               float((int)u >> 22) / 511.0, 1.0);\n"
          "    return v; }\n"
          "  default: return def;\n"
          "  }\n"
          "  if (f.y < 2) v.y = 0.0;\n"
          "  if (f.y < 3) v.z = 0.0;\n"
          "  if (f.y < 4) v.w = 1.0;\n"
          "  return v;\n"
          "}\n"
          /* 0 times anything -- inf and NaN included -- is 0. */
          "float4 mul0(float4 a, float4 b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }\n"
          "float4 rcc(float x) {\n"
          "  float r = 1.0 / x;\n"
          "  if (r > 0.0) r = clamp(r, 5.42101e-20, 1.884467e19);\n"
          "  else r = clamp(r, -1.884467e19, -5.42101e-20);\n"
          "  return r.xxxx;\n"
          "}\n"
          "float4 vexp(float x) { float f = floor(x); return float4(exp2(f), x - f, exp2(x), 1.0); }\n"
          "float4 vlog(float x) {\n"
          "  x = abs(x);\n"
          "  if (x == 0.0) return float4(-asfloat(0x7F800000u), 1.0, -asfloat(0x7F800000u), 1.0);\n"
          "  float e = floor(log2(x));\n"
          "  return float4(e, x / exp2(e), log2(x), 1.0);\n"
          "}\n"
          "float4 vlit(float4 c) {\n"
          "  float d = max(c.x, 0.0), s = max(c.y, 0.0), p = clamp(c.w, -128.0, 128.0);\n"
          "  return float4(1.0, d, c.x > 0.0 ? pow(s, p) : 0.0, 1.0);\n"
          "}\n"
          /* Screen pixels and zeta units back to clip space. Xbox D3D ends
           * every program with the viewport transform and the divide by w,
           * so oPos arrives as window x, y, z and clip w; multiplying back by
           * w is what lets the GPU clip and interpolate perspective-
           * correctly. */
          /* The fog factor from the program's oFog.x, as xemu computes it:
           * Xbox D3D biases SET_FOG_PARAMS so that these offsets land on
           * D3D's own formulas. 1 = no fog. */
          "float fog_factor(float d) {\n"
          "  float f = 1.0;\n"
          "  if (g_fog.z == 1.0) f = g_fog.x + d * g_fog.y - 1.0;\n"
          "  else if (g_fog.z == 2.0) f = g_fog.x + exp2(d * g_fog.y * 16.0) - 1.5;\n"
          "  else if (g_fog.z == 3.0) f = g_fog.x + exp2(-d * d * g_fog.y * g_fog.y * 32.0) - 1.5;\n"
          "  if (g_fog.w != 0.0) f = abs(f);\n"
          "  return f;\n"
          "}\n"
          "float4 to_clip(float4 p, float w) {\n"
          "  if (w == 0.0 || isinf(w) || isnan(w)) w = 1.0;\n"
          "  return float4((p.x * g_vp.x - 1.0) * w, (1.0 - p.y * g_vp.y) * w,\n"
          "                p.z * g_vp.z * w, w);\n"
          "}\n");
}

static void vs_inputs(Sb *s)
{
    int i;
    for (i = 0; i < 16; i++)
        sb(s, "  float4 v%d = decode(i.a%d, g_afmt[%d], g_adef[%d]);\n", i, i, i, i);
}

static const char *const SWZ = "xyzw";

static void vs_src(Sb *s, const uint32_t *w, uint32_t mux, uint32_t r_idx,
                   uint32_t neg, uint32_t swz)
{
    char base[48];
    int k;

    switch (mux) {
    case MUX_R:
        if (r_idx == 12)
            snprintf(base, sizeof base, "oPos");
        else if (r_idx < NV2A_VP_TEMPS)
            snprintf(base, sizeof base, "r%u", r_idx);
        else
            snprintf(base, sizeof base, "float4(0,0,0,0)");
        break;
    case MUX_V:
        snprintf(base, sizeof base, "v%u", fld(w, 1, 9, 4));
        break;
    case MUX_C: {
        uint32_t c = fld(w, 1, 13, 8);
        if (fld(w, 3, 1, 1))
            snprintf(base, sizeof base, "c[a0 + %u]", c);
        else if (c < NV2A_VP_CONSTANTS)
            snprintf(base, sizeof base, "c[%u]", c);
        else
            snprintf(base, sizeof base, "float4(0,0,0,0)");
        break;
    }
    default:
        snprintf(base, sizeof base, "float4(0,0,0,0)");
        break;
    }
    sb(s, neg ? "(-(%s)." : "((%s).", base);
    for (k = 0; k < 4; k++)
        sb(s, "%c", SWZ[(swz >> (6 - 2 * k)) & 3u]);
    sb(s, ")");
}

static void mask_str(uint32_t mask, char out[6])
{
    int n = 0;
    out[n++] = '.';
    if (mask & 8u) out[n++] = 'x';
    if (mask & 4u) out[n++] = 'y';
    if (mask & 2u) out[n++] = 'z';
    if (mask & 1u) out[n++] = 'w';
    out[n] = 0;
}

/* Output register names, indexed like NV2A_VP_O*. */
static const char *out_name(uint32_t addr)
{
    static const char *const names[NV2A_VP_OUTPUTS] = {
        "oPos", NULL, NULL, "oD0", "oD1", "oFog", "oPts", "oB0", "oB1",
        "oT0", "oT1", "oT2", "oT3"
    };
    return addr < NV2A_VP_OUTPUTS ? names[addr] : NULL;
}

static const char *temp_name(uint32_t r, char buf[8])
{
    if (r == 12)
        return "oPos";
    if (r < NV2A_VP_TEMPS) {
        snprintf(buf, 8, "r%u", r);
        return buf;
    }
    return NULL;
}

size_t nv2a_hlsl_vs(const Nv2aVpState *vp, char *buf, size_t size,
                    int *writes_consts)
{
    Sb s = { buf, size, 0, 0 };
    uint32_t pc;
    int finished = 0, i;

    *writes_consts = 0;
    vs_common(&s);
    sb(&s, "VSOut main(VSIn i) {\n");
    vs_inputs(&s);
    for (i = 0; i < NV2A_VP_TEMPS; i++)
        sb(&s, "  float4 r%d = 0;\n", i);
    sb(&s, "  float4 oPos = 0, oFog = 0, oPts = 0;\n"
           "  float4 oD0 = float4(0,0,0,1), oD1 = float4(0,0,0,1);\n"
           "  float4 oB0 = float4(0,0,0,1), oB1 = float4(0,0,0,1);\n"
           "  float4 oT0 = float4(0,0,0,1), oT1 = float4(0,0,0,1);\n"
           "  float4 oT2 = float4(0,0,0,1), oT3 = float4(0,0,0,1);\n"
           "  int a0 = 0;\n");

    for (pc = vp->program_start; pc < NV2A_VP_SLOTS; pc++) {
        const uint32_t *w = vp->program[pc];
        uint32_t mac = fld(w, 1, 21, 4), ilu = fld(w, 1, 25, 3);
        uint32_t c_r = (fld(w, 2, 0, 2) << 2) | fld(w, 3, 30, 2);
        uint32_t out_r = fld(w, 3, 20, 4);
        uint32_t mac_mask = fld(w, 3, 24, 4), ilu_mask = fld(w, 3, 16, 4);
        uint32_t o_mask = fld(w, 3, 12, 4);
        char m[6], nb[8];
        const char *dst;

        sb(&s, "  { // slot %u: %08X %08X %08X\n", pc, w[1], w[2], w[3]);
        sb(&s, "    float4 A = ");
        vs_src(&s, w, fld(w, 2, 26, 2), fld(w, 2, 28, 4), fld(w, 1, 8, 1), fld(w, 1, 0, 8));
        sb(&s, ";\n    float4 B = ");
        vs_src(&s, w, fld(w, 2, 11, 2), fld(w, 2, 13, 4), fld(w, 2, 25, 1), fld(w, 2, 17, 8));
        sb(&s, ";\n    float4 C = ");
        vs_src(&s, w, fld(w, 3, 28, 2), c_r, fld(w, 2, 10, 1), fld(w, 2, 2, 8));
        sb(&s, ";\n    float4 M = 0, I = 0;\n");

        switch (mac) {
        case MAC_MOV: sb(&s, "    M = A;\n"); break;
        case MAC_MUL: sb(&s, "    M = mul0(A, B);\n"); break;
        case MAC_ADD: sb(&s, "    M = A + C;\n"); break;
        case MAC_MAD: sb(&s, "    M = mul0(A, B) + C;\n"); break;
        case MAC_DP3: sb(&s, "    { float4 p = mul0(A, B); M = (p.x + p.y + p.z).xxxx; }\n"); break;
        case MAC_DPH: sb(&s, "    { float4 p = mul0(A, B); M = (p.x + p.y + p.z + B.w).xxxx; }\n"); break;
        case MAC_DP4: sb(&s, "    { float4 p = mul0(A, B); M = (p.x + p.y + p.z + p.w).xxxx; }\n"); break;
        case MAC_DST: sb(&s, "    M = float4(1.0, A.y * B.y, A.z, B.w);\n"); break;
        case MAC_MIN: sb(&s, "    M = min(A, B);\n"); break;
        case MAC_MAX: sb(&s, "    M = max(A, B);\n"); break;
        case MAC_SLT: sb(&s, "    M = (A < B) ? 1.0 : 0.0;\n"); break;
        case MAC_SGE: sb(&s, "    M = (A >= B) ? 1.0 : 0.0;\n"); break;
        default: break;
        }
        switch (ilu) {
        case ILU_MOV: sb(&s, "    I = C;\n"); break;
        case ILU_RCP: sb(&s, "    I = (1.0 / C.x).xxxx;\n"); break;
        case ILU_RCC: sb(&s, "    I = rcc(C.x);\n"); break;
        case ILU_RSQ: sb(&s, "    I = rsqrt(abs(C.x)).xxxx;\n"); break;
        case ILU_EXP: sb(&s, "    I = vexp(C.x);\n"); break;
        case ILU_LOG: sb(&s, "    I = vlog(C.x);\n"); break;
        case ILU_LIT: sb(&s, "    I = vlit(C);\n"); break;
        default: break;
        }

        /* Writes, in the interpreter's order: MAC, then ILU, then the
         * output register. */
        if (mac != MAC_NOP && mac != MAC_ARL && mac_mask
                && (dst = temp_name(out_r, nb)) != NULL) {
            mask_str(mac_mask, m);
            sb(&s, "    %s%s = M%s;\n", dst, m, m);
        }
        if (ilu != ILU_NOP && ilu_mask) {
            uint32_t ir = (mac != MAC_NOP) ? 1u : out_r;
            if ((dst = temp_name(ir, nb)) != NULL) {
                mask_str(ilu_mask, m);
                sb(&s, "    %s%s = I%s;\n", dst, m, m);
            }
        }
        if (o_mask) {
            const char *v = fld(w, 3, 2, 1) ? "I" : "M";
            uint32_t addr = fld(w, 3, 3, 8);
            mask_str(o_mask, m);
            if (fld(w, 3, 11, 1)) {
                if ((dst = out_name(addr & 0xFu)) != NULL)
                    sb(&s, "    %s%s = %s%s;\n", dst, m, v, m);
            } else {
                *writes_consts = 1;
                sb(&s, "    // constant write c[%u] dropped\n", addr);
            }
        }
        if (mac == MAC_ARL)
            sb(&s, "    a0 = (int)floor(A.x);\n");
        sb(&s, "  }\n");
        if (fld(w, 3, 0, 1)) {
            finished = 1;
            break;
        }
    }
    if (!finished)
        return 0;

    sb(&s, "  VSOut o;\n"
           "  o.pos = to_clip(oPos, oPos.w);\n"
           "  o.d0 = saturate(oD0); o.d1 = saturate(oD1);\n"
           "  o.t0 = oT0; o.t1 = oT1; o.t2 = oT2; o.t3 = oT3;\n"
           "  o.fog = fog_factor(oFog.x);\n"
           "  return o;\n"
           "}\n");
    return sb_done(&s);
}

size_t nv2a_hlsl_vs_passthrough(char *buf, size_t size)
{
    Sb s = { buf, size, 0, 0 };

    vs_common(&s);
    sb(&s, "VSOut main(VSIn i) {\n"
           "  float4 v[16];\n");
    /* The array form, because two of the attributes are chosen at run time. */
    {
        int k;
        for (k = 0; k < 16; k++)
            sb(&s, "  v[%d] = decode(i.a%d, g_afmt[%d], g_adef[%d]);\n", k, k, k, k);
    }
    /* x, y in pixels, z in zeta units, and 1/rhw as w -- what the software
     * path does with the same vertex. */
    sb(&s, "  float4 p = v[0];\n"
           "  float w = (p.w > 0.0 && !isinf(p.w)) ? 1.0 / p.w : 1.0;\n"
           "  VSOut o;\n"
           "  o.pos = to_clip(p, w);\n"
           "  o.d0 = saturate(v[g_roles.x]);\n"
           "  o.d1 = saturate(v[4]);\n"
           "  o.t0 = v[g_roles.y]; o.t1 = v[10]; o.t2 = v[11]; o.t3 = v[12];\n"
           "  o.fog = 1.0;\n"
           "  return o;\n"
           "}\n");
    return sb_done(&s);
}

/* ── Register combiners ──────────────────────────────────────────────── */

void nv2a_hlsl_ps_key(const Nv2aCombiner *c, Nv2aHlslPsKey *key)
{
    uint32_t i;

    memset(key, 0, sizeof *key);
    key->programmed = (uint32_t)c->programmed;
    key->stage_program = c->stage_program;
    key->other_stage_input = c->other_stage_input;
    key->dot_mapping = c->dot_mapping;
    if (!c->programmed)
        return;
    key->count = c->control & 0xFFu;
    if (key->count > 8)
        key->count = 8;
    key->control = c->control & 0x11100u;       /* mux msb, unique c0 / c1 */
    for (i = 0; i < key->count; i++) {
        key->color_icw[i] = c->color_icw[i];
        key->color_ocw[i] = c->color_ocw[i];
        key->alpha_icw[i] = c->alpha_icw[i];
        key->alpha_ocw[i] = c->alpha_ocw[i];
    }
    key->final_cw0 = c->final_cw0;
    key->final_cw1 = c->final_cw1;
}

/* Register names as nv2a_combiner.c numbers them. 6 and 7 are unused and
 * read as zero; 14 and 15 exist only in the final stage. */
static const char *comb_reg(uint32_t r)
{
    static const char *const names[16] = {
        "zero", "c0", "c1", "fog", "v0", "v1", "zero", "zero",
        "t0", "t1", "t2", "t3", "r0", "r1", "sum", "ef"
    };
    return names[r & 15u];
}

/* Only these can be written. */
static int comb_writable(uint32_t r)
{
    return r == 4 || r == 5 || (r >= 8 && r <= 13);
}

/* One input, mapped. `rgb` selects the colour form (a float3 of R,G,B or of
 * alpha broadcast); otherwise the alpha form (alpha, or blue). */
static void comb_input(Sb *s, uint32_t byte, int rgb)
{
    uint32_t reg = byte & 0xFu, alpha_sel = (byte >> 4) & 1u, mapping = (byte >> 5) & 7u;
    char x[32];

    if (rgb)
        snprintf(x, sizeof x, "%s.%s", comb_reg(reg), alpha_sel ? "aaa" : "rgb");
    else
        snprintf(x, sizeof x, "%s.%s", comb_reg(reg), alpha_sel ? "a" : "b");
    switch (mapping) {
    case 0: sb(s, "saturate(%s)", x); break;
    case 1: sb(s, "(1.0 - saturate(%s))", x); break;
    case 2: sb(s, "(2.0 * saturate(%s) - 1.0)", x); break;
    case 3: sb(s, "(-2.0 * saturate(%s) + 1.0)", x); break;
    case 4: sb(s, "(saturate(%s) - 0.5)", x); break;
    case 5: sb(s, "(-saturate(%s) + 0.5)", x); break;
    case 6: sb(s, "(%s)", x); break;
    default: sb(s, "(-(%s))", x); break;
    }
}

static const char *scale_bias(uint32_t flags)
{
    switch (flags & 0x38u) {
    case 0x08: return "clamp((%s) - 0.5, -1.0, 1.0)";
    case 0x10: return "clamp((%s) * 2.0, -1.0, 1.0)";
    case 0x18: return "clamp(((%s) - 0.5) * 2.0, -1.0, 1.0)";
    case 0x20: return "clamp((%s) * 4.0, -1.0, 1.0)";
    case 0x30: return "clamp((%s) * 0.5, -1.0, 1.0)";
    default:   return "clamp(%s, -1.0, 1.0)";
    }
}

static void comb_stage(Sb *s, const Nv2aHlslPsKey *k, uint32_t i)
{
    uint32_t ci = k->color_icw[i], co = k->color_ocw[i];
    uint32_t ai = k->alpha_icw[i], ao = k->alpha_ocw[i];
    uint32_t cflags = co >> 12, aflags = ao >> 12;
    const char *csb = scale_bias(cflags), *asb = scale_bias(aflags);
    char e[64];

    sb(s, "  { // stage %u\n", i);
    sb(s, "    c0 = cf0[%u]; c1 = cf1[%u];\n",
       (k->control & 0x1000u) ? i : 0u, (k->control & 0x10000u) ? i : 0u);
    if (k->control & 0x100u)
        sb(s, "    bool sel = saturate(r0.a) >= 0.5;\n");
    else
        sb(s, "    bool sel = (((uint)(saturate(r0.a) * 255.0 + 0.5)) & 1u) != 0u;\n");

    sb(s, "    float3 A = "); comb_input(s, ci >> 24, 1);
    sb(s, ";\n    float3 B = "); comb_input(s, ci >> 16, 1);
    sb(s, ";\n    float3 C = "); comb_input(s, ci >> 8, 1);
    sb(s, ";\n    float3 D = "); comb_input(s, ci, 1);
    sb(s, ";\n");
    if (cflags & 0x02u) sb(s, "    float3 ab = dot(A, B).xxx;\n");
    else                sb(s, "    float3 ab = A * B;\n");
    if (cflags & 0x01u) sb(s, "    float3 cd = dot(C, D).xxx;\n");
    else                sb(s, "    float3 cd = C * D;\n");
    if (cflags & 0x04u) sb(s, "    float3 su = sel ? cd : ab;\n");
    else                sb(s, "    float3 su = ab + cd;\n");
    snprintf(e, sizeof e, "ab"); sb(s, "    ab = "); sb(s, csb, e); sb(s, ";\n");
    snprintf(e, sizeof e, "cd"); sb(s, "    cd = "); sb(s, csb, e); sb(s, ";\n");
    snprintf(e, sizeof e, "su"); sb(s, "    su = "); sb(s, csb, e); sb(s, ";\n");

    sb(s, "    float a = "); comb_input(s, ai >> 24, 0);
    sb(s, ";\n    float b = "); comb_input(s, ai >> 16, 0);
    sb(s, ";\n    float c = "); comb_input(s, ai >> 8, 0);
    sb(s, ";\n    float d = "); comb_input(s, ai, 0);
    sb(s, ";\n    float aab = a * b, acd = c * d;\n");
    if (aflags & 0x04u) sb(s, "    float asu = sel ? acd : aab;\n");
    else                sb(s, "    float asu = aab + acd;\n");
    snprintf(e, sizeof e, "aab"); sb(s, "    aab = "); sb(s, asb, e); sb(s, ";\n");
    snprintf(e, sizeof e, "acd"); sb(s, "    acd = "); sb(s, asb, e); sb(s, ";\n");
    snprintf(e, sizeof e, "asu"); sb(s, "    asu = "); sb(s, asb, e); sb(s, ";\n");

    /* Writes, after both halves have read, in the interpreter's order. */
    {
        uint32_t cd_dst = co & 0xFu, ab_dst = (co >> 4) & 0xFu, sum_dst = (co >> 8) & 0xFu;
        uint32_t acd_d = ao & 0xFu, aab_d = (ao >> 4) & 0xFu, asu_d = (ao >> 8) & 0xFu;
        if (ab_dst && comb_writable(ab_dst))   sb(s, "    %s.rgb = ab;\n", comb_reg(ab_dst));
        if (cd_dst && comb_writable(cd_dst))   sb(s, "    %s.rgb = cd;\n", comb_reg(cd_dst));
        if (sum_dst && comb_writable(sum_dst)) sb(s, "    %s.rgb = su;\n", comb_reg(sum_dst));
        if (aab_d && comb_writable(aab_d))     sb(s, "    %s.a = aab;\n", comb_reg(aab_d));
        if (acd_d && comb_writable(acd_d))     sb(s, "    %s.a = acd;\n", comb_reg(acd_d));
        if (asu_d && comb_writable(asu_d))     sb(s, "    %s.a = asu;\n", comb_reg(asu_d));
        if ((cflags & 0x80u) && ab_dst && comb_writable(ab_dst))
            sb(s, "    %s.a = ab.b;\n", comb_reg(ab_dst));
        if ((cflags & 0x40u) && cd_dst && comb_writable(cd_dst))
            sb(s, "    %s.a = cd.b;\n", comb_reg(cd_dst));
    }
    sb(s, "  }\n");
}

size_t nv2a_hlsl_ps(const Nv2aHlslPsKey *k, char *buf, size_t size)
{
    Sb s = { buf, size, 0, 0 };
    int st;
    uint32_t i;

    sb(&s, "Texture2D tx0 : register(t0); Texture2D tx1 : register(t1);\n"
           "Texture2D tx2 : register(t2); Texture2D tx3 : register(t3);\n"
           "TextureCube cx0 : register(t4); TextureCube cx1 : register(t5);\n"
           "TextureCube cx2 : register(t6); TextureCube cx3 : register(t7);\n"
           /* The dot mappings (PS_DOTMAPPING): how a texel becomes a vector.
            * MINUS1_TO_1_D3D is biased, (x - 128) / 127, so a flat normal
            * map texel (128, 128, 255) is (0, 0, 1); MINUS1_TO_1_GL is two's
            * complement centred on the half steps; MINUS1_TO_1 is plain two's
            * complement. The D3D one was two's complement here, which turned
            * a flat normal into about (-1, -1, 0) and made Amped 2's
            * normal-mapped menu rider reflect the environment map all over:
            * chrome instead of cloth. */
           "float3 dm_unsigned(float3 c) { return c; }\n"
           "float3 dm_d3d(float3 c) { float3 t = floor(c * 255.0 + 0.5);\n"
           "  return clamp((t - 128.0) / 127.0, -1.0, 1.0); }\n"
           "float3 dm_gl(float3 c) { float3 t = floor(c * 255.0 + 0.5);\n"
           "  return t >= 128.0 ? (t - 255.5) / 127.5 : (t + 0.5) / 127.5; }\n"
           "float3 dm_signed(float3 c) { float3 t = floor(c * 255.0 + 0.5);\n"
           "  return clamp((t >= 128.0 ? t - 256.0 : t) / 127.0, -1.0, 1.0); }\n"
           "SamplerState s0 : register(s0); SamplerState s1 : register(s1);\n"
           "SamplerState s2 : register(s2); SamplerState s3 : register(s3);\n"
           "cbuffer PSC : register(b0) {\n"
           "  float4 cf0[8]; float4 cf1[8]; float4 ff0; float4 ff1; float4 fogc;\n"
           "  float4 tsc[4]; uint4 atest;\n"
           "};\n"
           "struct PSIn {\n"
           "  float4 pos : SV_Position; float4 d0 : COLOR0; float4 d1 : COLOR1;\n"
           "  float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1;\n"
           "  float4 t2 : TEXCOORD2; float4 t3 : TEXCOORD3; float fog : FOG;\n"
           "};\n"
           "bool cmp(uint f, float a, float b) {\n"
           "  switch (f) {\n"
           "  case 0x200: return false;  case 0x201: return a < b;\n"
           "  case 0x202: return a == b; case 0x203: return a <= b;\n"
           "  case 0x204: return a > b;  case 0x205: return a != b;\n"
           "  case 0x206: return a >= b; default: return true;\n"
           "  }\n"
           "}\n"
           "float4 main(PSIn i) : SV_Target {\n"
           "  float4 zero = 0, c0 = 0, c1 = 0, sum = 0, ef = 0;\n"
           "  float4 v0 = saturate(i.d0), v1 = saturate(i.d1);\n"
           "  float4 t0 = 0, t1 = 0, t2 = 0, t3 = 0;\n"
           "  bool kill = false;\n");

    /* The texture stages, before anything that could discard, so every
     * sample has its derivatives. */
    sb(&s, "  float dot1 = 0.0, dot2 = 0.0;\n");
    for (st = 0; st < 4; st++) {
        uint32_t mode = (k->stage_program >> (5 * st)) & 0x1Fu;
        /* The dot-product modes read another stage's texel as a vector
         * (PS_INPUTTEXTURE: the input for stage n in bits 16+4(n-1)) through
         * that stage's dot mapping (PS_DOTMAPPING: bits 4(n-1)). */
        uint32_t in_st = st ? (k->other_stage_input >> (16 + 4 * (st - 1))) & 0xFu : 0;
        uint32_t map = st ? (k->dot_mapping >> (4 * (st - 1))) & 0x7u : 0;
        const char *dm = map == 0 ? "dm_unsigned" : map == 1 ? "dm_d3d"
                       : map == 2 ? "dm_gl" : "dm_signed";
        if (in_st >= st)
            in_st = st ? st - 1 : 0;
        if (!k->programmed && st > 0)
            break;
        if (!k->programmed)
            mode = NV2A_TEXMODE_2D_PROJECTIVE;
        switch (mode) {
        case NV2A_TEXMODE_NONE:
            break;
        case NV2A_TEXMODE_CUBE:
            sb(&s, "  t%d = cx%d.Sample(s%d, i.t%d.xyz);\n", st, st, st, st);
            sb(&s, "  if (tsc[%d].w == 0.0) t%d = 0;\n", st, st);
            break;
        case 0x11:                                   /* DOT_PRODUCT */
            sb(&s, "  dot%d = dot(i.t%d.xyz, %s(t%u.rgb));\n", st == 1 ? 1 : 2, st, dm, in_st);
            break;
        case 0x09:                                   /* DOT_ST */
            sb(&s, "  t%d = tx%d.Sample(s%d, float2(dot1, dot(i.t%d.xyz, %s(t%u.rgb))) * tsc[%d].xy);\n",
               st, st, st, st, dm, in_st, st);
            break;
        case 0x0B:                                   /* DOT_RFLCT_DIFF */
        case 0x0C:                                   /* DOT_RFLCT_SPEC */
        case 0x0E:                                   /* DOT_STR_CUBE */
        case 0x12: {                                 /* DOT_RFLCT_SPEC_CONST */
            /* The normal from the three dot products; for stage 2 (diffuse)
             * the third comes from stage 3's coordinates, as on hardware. */
            if (mode == 0x0B)
                sb(&s, "  { float3 n = float3(dot1, dot(i.t%d.xyz, %s(t%u.rgb)), dot(i.t3.xyz, %s(t%u.rgb)));\n",
                   st, dm, in_st, dm, in_st);
            else
                sb(&s, "  { float3 n = float3(dot1, dot2, dot(i.t%d.xyz, %s(t%u.rgb)));\n",
                   st, dm, in_st);
            if (mode == 0x0C)
                sb(&s, "    float3 e = float3(i.t1.w, i.t2.w, i.t3.w);\n"
                       "    float3 r = 2.0 * n * dot(n, e) / max(dot(n, n), 1e-8) - e;\n");
            else if (mode == 0x12)
                sb(&s, "    float3 e = float3(0.0, 0.0, 1.0);\n"
                       "    float3 r = 2.0 * n * dot(n, e) / max(dot(n, n), 1e-8) - e;\n");
            else
                sb(&s, "    float3 r = n;\n");
            sb(&s, "    t%d = cx%d.Sample(s%d, r); }\n", st, st, st);
            sb(&s, "  if (tsc[%d].w == 0.0) t%d = 0;\n", st, st);
            break;
        }
        case NV2A_TEXMODE_2D_PROJECTIVE:
        case NV2A_TEXMODE_3D_PROJECTIVE: {
            int proj = 1;
            sb(&s, "  {\n    float4 tc = i.t%d;\n", st);
            if (proj)
                sb(&s, "    float q = (tc.w != 0.0) ? tc.w : 1.0;\n");
            else
                sb(&s, "    float q = 1.0;\n");
            sb(&s, "    t%d = tx%d.Sample(s%d, tc.xy / q * tsc[%d].xy);\n"
                   "    if (tsc[%d].z > 0.0) t%d.a = 1.0;\n"
                   "    if (tsc[%d].w == 0.0) t%d = 0;\n  }\n",
               st, st, st, st, st, st, st, st);
            break;
        }
        case NV2A_TEXMODE_PASS_THROUGH:
            sb(&s, "  t%d = i.t%d;\n", st, st);
            break;
        case NV2A_TEXMODE_CLIP_PLANE:
            sb(&s, "  if (any(i.t%d < 0.0)) kill = true;\n", st);
            break;
        default:
            sb(&s, "  // stage %d: mode 0x%X not implemented\n", st, mode);
            break;
        }
    }
    sb(&s, "  if (atest.w == 11u) t1 = 1.0;   /* RECOMP_GPU_PS_DEBUG=11: stage 1 white */\n"
           "  float4 fog = float4(fogc.rgb, saturate(i.fog));\n"
           "  if (atest.w == 4u) fog.a = 1.0;   /* RECOMP_GPU_PS_DEBUG=4: no fog */\n"
           "  if (atest.w == 5u) return float4(saturate(i.fog).xxx, 1.0); /* =5: fog factor */\n"
           "  if (atest.w == 6u) return float4(t1.rgb, 1.0);   /* =6..8: texture stage 1..3 */\n"
           "  if (atest.w == 7u) return float4(t2.rgb, 1.0);\n"
           "  if (atest.w == 8u) return float4(t3.rgb, 1.0);\n"
           "  if (atest.w == 9u) return float4(i.d1.rgb, 1.0); /* =9: specular colour */\n"
           "  if (atest.w == 10u) { float q1 = i.t1.w != 0.0 ? i.t1.w : 1.0;   /* =10: stage-1 coords */\n"
           "    return float4(frac(i.t1.xy / q1), i.t1.w == 0.0 ? 1.0 : 0.0, 1.0); }\n"
           "  float4 r0 = float4(0, 0, 0, t0.a), r1 = 0;\n"
           "  float4 outc;\n");

    if (!k->programmed) {
        /* Before the title has programmed the combiners: texture times
         * diffuse, the software path's default. */
        sb(&s, "  outc = (tsc[0].w != 0.0) ? t0 * v0 : v0;\n");
    } else {
        uint32_t cw0 = k->final_cw0, cw1 = k->final_cw1, fl = cw1 & 0xFFu;

        for (i = 0; i < k->count; i++)
            comb_stage(&s, k, i);
        sb(&s, "  c0 = ff0; c1 = ff1;\n");
        sb(&s, "  sum = %s + %s;\n", (fl & 0x20u) ? "(1.0 - r0)" : "r0",
           (fl & 0x40u) ? "(1.0 - v1)" : "v1");
        if (fl & 0x80u)
            sb(&s, "  sum = saturate(sum);\n");
        sb(&s, "  { float3 E = "); comb_input(&s, (cw1 >> 24) & 0x3Fu, 1);
        sb(&s, ";\n    float3 F = "); comb_input(&s, (cw1 >> 16) & 0x3Fu, 1);
        sb(&s, ";\n    ef.rgb = E * F; }\n");
        sb(&s, "  { float3 A = "); comb_input(&s, (cw0 >> 24) & 0x3Fu, 1);
        sb(&s, ";\n    float3 B = "); comb_input(&s, (cw0 >> 16) & 0x3Fu, 1);
        sb(&s, ";\n    float3 C = "); comb_input(&s, (cw0 >> 8) & 0x3Fu, 1);
        sb(&s, ";\n    float3 D = "); comb_input(&s, cw0 & 0x3Fu, 1);
        sb(&s, ";\n    float G = "); comb_input(&s, (cw1 >> 8) & 0x3Fu, 0);
        sb(&s, ";\n    outc = float4(saturate(A * B + (1.0 - A) * C + D), saturate(G)); }\n");
    }
    /* RECOMP_GPU_PS_DEBUG (atest.w): 1 diffuse, 2 texture 0, 3 magenta --
     * opaque and past the alpha test, to tell missing geometry from a pixel
     * that is computed transparent. */
    sb(&s, "  if (atest.w == 1u) return float4(v0.rgb, 1.0);\n"
           "  if (atest.w == 2u) return float4(t0.rgb, 1.0);\n"
           "  if (atest.w == 3u) return float4(1.0, 0.0, 1.0, 1.0);\n");
    sb(&s, "  if (kill) discard;\n"
           "  if (atest.x != 0u) {\n"
           "    float a = floor(saturate(outc.a) * 255.0 + 0.5);\n"
           "    if (!cmp(atest.y, a, (float)(atest.z & 0xFFu))) discard;\n"
           "  }\n"
           "  return outc;\n"
           "}\n");
    return sb_done(&s);
}
