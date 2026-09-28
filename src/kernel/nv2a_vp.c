/**
 * NV2A vertex program interpreter. See nv2a_vp.h.
 *
 * Each 128-bit slot carries one MAC (vector) and one ILU (scalar/special) op
 * issued together: both read the registers as they were before the slot, then
 * both write. Field positions and semantics follow xemu's hw/xbox/nv2a/vsh.c:
 *
 *   dword 1:  ILU [27:25]  MAC [24:21]  CONST [20:13]  V [12:9]
 *             A: neg [8]  swizzle x,y,z,w [7:0]
 *   dword 2:  A: R [31:28]  mux [27:26]
 *             B: neg [25]  swizzle [24:17]  R [16:13]  mux [12:11]
 *             C: neg [10]  swizzle [9:2]  R high [1:0]
 *   dword 3:  C: R low [31:30]  mux [29:28]
 *             out: MAC mask [27:24]  R [23:20]  ILU mask [19:16]
 *                  O mask [15:12]  ORB [11]  address [10:3]  mux [2]
 *             A0X [1]  FINAL [0]
 *
 * Operand mux: 1 temp, 2 input, 3 constant. Write masks: bit 3 is x, bit 0 w.
 * R12 is oPos. When a slot has both a MAC and an ILU op writing temps, the
 * ILU result goes to R1. The output-register write takes the MAC result when
 * OUT_MUX is 0 and the ILU result when it is 1, and lands in an output
 * register when ORB is 1 or in constant memory when it is 0.
 */
#include "nv2a_vp.h"

#include <math.h>
#include <string.h>

enum { MUX_R = 1, MUX_V = 2, MUX_C = 3 };

enum {
    MAC_NOP, MAC_MOV, MAC_MUL, MAC_ADD, MAC_MAD, MAC_DP3, MAC_DPH, MAC_DP4,
    MAC_DST, MAC_MIN, MAC_MAX, MAC_SLT, MAC_SGE, MAC_ARL
};
enum { ILU_NOP, ILU_MOV, ILU_RCP, ILU_RCC, ILU_RSQ, ILU_EXP, ILU_LOG, ILU_LIT };

#define NV097_SET_TRANSFORM_PROGRAM          0x0B00   /* 32 dwords */
#define NV097_SET_TRANSFORM_CONSTANT         0x0B80   /* 32 dwords */
#define NV097_SET_TRANSFORM_EXECUTION_MODE   0x1E94
#define NV097_SET_TRANSFORM_PROGRAM_CXT_WRITE_EN 0x1E98
#define NV097_SET_TRANSFORM_PROGRAM_LOAD     0x1E9C
#define NV097_SET_TRANSFORM_PROGRAM_START    0x1EA0
#define NV097_SET_TRANSFORM_CONSTANT_LOAD    0x1EA4
/* The viewport is not a separate register on NV2A: SET_VIEWPORT_SCALE and
 * _OFFSET land in constant memory at XFCTX_VPSCL / _VPOFF (nv2a_regs.h), and
 * Xbox D3D's epilogue to every vertex program reads them there as c[-38] and
 * c[-37]. Without these the epilogue multiplied oPos.xyz by zero. */
#define NV097_SET_VIEWPORT_OFFSET            0x0A20   /* 4 floats */
#define NV097_SET_VIEWPORT_SCALE             0x0AF0   /* 4 floats */
#define XFCTX_VPSCL                          0x3A
#define XFCTX_VPOFF                          0x3B

static uint32_t fld(const uint32_t *w, int dword, int start, int len)
{
    return (w[dword] >> start) & ((1u << len) - 1u);
}

static float u2f(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int nv2a_vp_method(Nv2aVpState *vp, uint32_t method, uint32_t param)
{
    if (method >= NV097_SET_TRANSFORM_PROGRAM
            && method < NV097_SET_TRANSFORM_PROGRAM + 32 * 4) {
        uint32_t word = ((method - NV097_SET_TRANSFORM_PROGRAM) / 4) % 4;
        if (vp->program_load < NV2A_VP_SLOTS)
            vp->program[vp->program_load][word] = param;
        if (word == 3)
            vp->program_load++;
        vp->program_writes++;
        return 1;
    }
    if (method >= NV097_SET_TRANSFORM_CONSTANT
            && method < NV097_SET_TRANSFORM_CONSTANT + 32 * 4) {
        uint32_t comp = ((method - NV097_SET_TRANSFORM_CONSTANT) / 4) % 4;
        if (vp->constant_load < NV2A_VP_CONSTANTS)
            vp->constants[vp->constant_load][comp] = u2f(param);
        if (comp == 3)
            vp->constant_load++;
        return 1;
    }
    if (method >= NV097_SET_VIEWPORT_OFFSET && method < NV097_SET_VIEWPORT_OFFSET + 16) {
        vp->constants[XFCTX_VPOFF][(method - NV097_SET_VIEWPORT_OFFSET) / 4] = u2f(param);
        return 0;   /* also leave it visible as a method to the executor's report */
    }
    if (method >= NV097_SET_VIEWPORT_SCALE && method < NV097_SET_VIEWPORT_SCALE + 16) {
        vp->constants[XFCTX_VPSCL][(method - NV097_SET_VIEWPORT_SCALE) / 4] = u2f(param);
        return 0;
    }
    switch (method) {
    case NV097_SET_TRANSFORM_EXECUTION_MODE:
        vp->execution_mode = param;
        return 1;
    case NV097_SET_TRANSFORM_PROGRAM_CXT_WRITE_EN:
        return 1;
    case NV097_SET_TRANSFORM_PROGRAM_LOAD:
        vp->program_load = param;
        return 1;
    case NV097_SET_TRANSFORM_PROGRAM_START:
        vp->program_start = param;
        return 1;
    case NV097_SET_TRANSFORM_CONSTANT_LOAD:
        vp->constant_load = param;
        return 1;
    default:
        return 0;
    }
}

int nv2a_vp_active(const Nv2aVpState *vp)
{
    return (vp->execution_mode & 3u) == 2u;
}

typedef struct {
    float r[NV2A_VP_TEMPS][4];
    float (*o)[4];                  /* outputs; o[OPOS] doubles as R12 */
    int   a0;
} Regs;

static void read_src(const Nv2aVpState *vp, const Regs *g,
                     const float in[NV2A_VP_INPUTS][4], const uint32_t *w,
                     uint32_t mux, uint32_t r_idx, uint32_t neg,
                     uint32_t swz, float out[4])
{
    const float *src;
    static const float zero[4] = {0, 0, 0, 0};
    int i;

    switch (mux) {
    case MUX_R:
        src = (r_idx == 12) ? g->o[NV2A_VP_OPOS]
            : (r_idx < NV2A_VP_TEMPS ? g->r[r_idx] : zero);
        break;
    case MUX_V:
        src = in[fld(w, 1, 9, 4)];
        break;
    case MUX_C: {
        int c = (int)fld(w, 1, 13, 8);
        if (fld(w, 3, 1, 1))
            c += g->a0;
        src = (c >= 0 && c < NV2A_VP_CONSTANTS) ? vp->constants[c] : zero;
        break;
    }
    default:
        src = zero;
        break;
    }
    /* swz holds x,y,z,w selectors from the top two bits down. */
    for (i = 0; i < 4; i++) {
        float v = src[(swz >> (6 - 2 * i)) & 3u];
        out[i] = neg ? -v : v;
    }
}

static void write_masked(float dst[4], const float v[4], uint32_t mask)
{
    if (mask & 8u) dst[0] = v[0];
    if (mask & 4u) dst[1] = v[1];
    if (mask & 2u) dst[2] = v[2];
    if (mask & 1u) dst[3] = v[3];
}

static void bcast(float d[4], float v)
{
    d[0] = d[1] = d[2] = d[3] = v;
}

int nv2a_vp_run(Nv2aVpState *vp, const float in[NV2A_VP_INPUTS][4],
                float out[NV2A_VP_OUTPUTS][4])
{
    Regs g;
    uint32_t pc;
    int n;

    memset(&g, 0, sizeof g);
    g.o = out;
    memset(out, 0, sizeof(float) * 4 * NV2A_VP_OUTPUTS);
    out[NV2A_VP_OD0][3] = out[NV2A_VP_OD1][3] = 1.0f;
    out[NV2A_VP_OB0][3] = out[NV2A_VP_OB1][3] = 1.0f;
    out[NV2A_VP_OT0][3] = out[NV2A_VP_OT1][3] = 1.0f;
    out[NV2A_VP_OT2][3] = out[NV2A_VP_OT3][3] = 1.0f;

    for (pc = vp->program_start, n = 0; pc < NV2A_VP_SLOTS; pc++) {
        const uint32_t *w = vp->program[pc];
        uint32_t mac = fld(w, 1, 21, 4), ilu = fld(w, 1, 25, 3);
        float a[4], b[4], c[4], mres[4] = {0, 0, 0, 0}, ires[4] = {0, 0, 0, 0};
        uint32_t c_r = (fld(w, 2, 0, 2) << 2) | fld(w, 3, 30, 2);
        uint32_t out_r = fld(w, 3, 20, 4);
        uint32_t mac_mask = fld(w, 3, 24, 4), ilu_mask = fld(w, 3, 16, 4);
        uint32_t o_mask = fld(w, 3, 12, 4);
        int new_a0 = g.a0, i;

        n++;
        read_src(vp, &g, in, w, fld(w, 2, 26, 2), fld(w, 2, 28, 4),
                 fld(w, 1, 8, 1), fld(w, 1, 0, 8), a);
        read_src(vp, &g, in, w, fld(w, 2, 11, 2), fld(w, 2, 13, 4),
                 fld(w, 2, 25, 1), fld(w, 2, 17, 8), b);
        read_src(vp, &g, in, w, fld(w, 3, 28, 2), c_r,
                 fld(w, 2, 10, 1), fld(w, 2, 2, 8), c);

        switch (mac) {
        case MAC_MOV: memcpy(mres, a, sizeof mres); break;
        case MAC_MUL: for (i = 0; i < 4; i++) mres[i] = a[i] * b[i]; break;
        case MAC_ADD: for (i = 0; i < 4; i++) mres[i] = a[i] + c[i]; break;
        case MAC_MAD: for (i = 0; i < 4; i++) mres[i] = a[i] * b[i] + c[i]; break;
        case MAC_DP3: bcast(mres, a[0]*b[0] + a[1]*b[1] + a[2]*b[2]); break;
        case MAC_DPH: bcast(mres, a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + b[3]); break;
        case MAC_DP4: bcast(mres, a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]); break;
        case MAC_DST:
            mres[0] = 1.0f; mres[1] = a[1] * b[1]; mres[2] = a[2]; mres[3] = b[3];
            break;
        case MAC_MIN: for (i = 0; i < 4; i++) mres[i] = a[i] < b[i] ? a[i] : b[i]; break;
        case MAC_MAX: for (i = 0; i < 4; i++) mres[i] = a[i] > b[i] ? a[i] : b[i]; break;
        case MAC_SLT: for (i = 0; i < 4; i++) mres[i] = a[i] <  b[i] ? 1.0f : 0.0f; break;
        case MAC_SGE: for (i = 0; i < 4; i++) mres[i] = a[i] >= b[i] ? 1.0f : 0.0f; break;
        case MAC_ARL: new_a0 = (int)floorf(a[0]); break;
        default: break;
        }

        switch (ilu) {
        case ILU_MOV: memcpy(ires, c, sizeof ires); break;
        case ILU_RCP: bcast(ires, 1.0f / c[0]); break;
        case ILU_RCC: {
            float r = 1.0f / c[0];
            if (r > 0.0f)
                r = r < 5.42101e-20f ? 5.42101e-20f : (r > 1.884467e19f ? 1.884467e19f : r);
            else
                r = r > -5.42101e-20f ? -5.42101e-20f : (r < -1.884467e19f ? -1.884467e19f : r);
            bcast(ires, r);
            break;
        }
        case ILU_RSQ: bcast(ires, 1.0f / sqrtf(fabsf(c[0]))); break;
        case ILU_EXP: {
            float fl = floorf(c[0]);
            ires[0] = exp2f(fl); ires[1] = c[0] - fl; ires[2] = exp2f(c[0]); ires[3] = 1.0f;
            break;
        }
        case ILU_LOG: {
            float x = fabsf(c[0]);
            if (x == 0.0f) {
                ires[0] = ires[2] = -INFINITY; ires[1] = 1.0f;
            } else {
                float e = floorf(log2f(x));
                ires[0] = e; ires[1] = x / exp2f(e); ires[2] = log2f(x);
            }
            ires[3] = 1.0f;
            break;
        }
        case ILU_LIT: {
            float d = c[0] > 0.0f ? c[0] : 0.0f;
            float s = c[1] > 0.0f ? c[1] : 0.0f;
            float p = c[3] < -128.0f ? -128.0f : (c[3] > 128.0f ? 128.0f : c[3]);
            ires[0] = 1.0f; ires[1] = d;
            ires[2] = c[0] > 0.0f ? powf(s, p) : 0.0f; ires[3] = 1.0f;
            break;
        }
        default: break;
        }

        /* Writes, after both units have read. */
        if (mac != MAC_NOP && mac != MAC_ARL && mac_mask) {
            if (out_r == 12)
                write_masked(out[NV2A_VP_OPOS], mres, mac_mask);
            else if (out_r < NV2A_VP_TEMPS)
                write_masked(g.r[out_r], mres, mac_mask);
        }
        if (ilu != ILU_NOP && ilu_mask) {
            uint32_t ir = (mac != MAC_NOP) ? 1u : out_r;
            if (ir == 12)
                write_masked(out[NV2A_VP_OPOS], ires, ilu_mask);
            else if (ir < NV2A_VP_TEMPS)
                write_masked(g.r[ir], ires, ilu_mask);
        }
        if (o_mask) {
            const float *v = fld(w, 3, 2, 1) ? ires : mres;
            uint32_t addr = fld(w, 3, 3, 8);
            if (fld(w, 3, 11, 1)) {                /* output register */
                addr &= 0xFu;
                if (addr < NV2A_VP_OUTPUTS)
                    write_masked(out[addr], v, o_mask);
            } else if (addr < NV2A_VP_CONSTANTS) {  /* constant memory */
                write_masked(vp->constants[addr], v, o_mask);
            }
        }
        g.a0 = new_a0;

        if (fld(w, 3, 0, 1))                       /* FINAL */
            return n;
    }
    return 0;
}
