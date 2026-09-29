/**
 * NV2A draws on Direct3D 11. See nv2a_d3d11.h.
 *
 * The software executor was correct and slow: every pixel of every triangle
 * went through a C rasteriser and a C combiner evaluator, and in game that is
 * about one and a half frames a second. This keeps the executor's decoding and
 * its state exactly as they are and replaces only what happens at a draw:
 *
 *   vertices   the title's own arrays, copied per draw into a ring buffer and
 *              described to the input assembler as raw bits; the vertex shader
 *              decodes them (nv2a_hlsl.c), so any layout needs no conversion.
 *   programs   vertex programs and combiner setups are compiled to HLSL once
 *              and cached by content.
 *   textures   decoded from guest memory on first use and cached by address
 *              and format, re-checked by hash at most once a frame.
 *   targets    colour and depth surfaces are D3D11 textures keyed by guest
 *              address. A texture bound at a surface's address samples the
 *              surface, which is how render-to-texture keeps working.
 *   output     each flipped frame is copied to a staging texture and read two
 *              flips later into the framebuffer window, so the dumps and the
 *              window show what the GPU drew without ever waiting for it.
 *
 * Windows only; elsewhere nv2a_d3d11_active() is 0 and the software path runs.
 */
#include "nv2a_d3d11.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)

#define COBJMACROS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "nv2a_hlsl.h"
#include "../d3d/d3d8_swizzle.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern void xbox_FramebufferWindowPresentArgb(const void *px, uint32_t w,
                                              uint32_t h, uint32_t row_pitch);
/* From nv2a_pb_exec.c: one texture level decoded to A8R8G8B8, and the size
 * of a texel. The same decoder the software sampler uses, so the two paths
 * cannot disagree about a format. */
extern int nv2a_pb_decode_level(uint32_t color, const uint8_t *src, uint32_t w,
                                uint32_t h, uint32_t pitch, uint32_t *out);
extern uint32_t nv2a_pb_texel_bytes(uint32_t color);

static int s_state = -1;                 /* -1 not tried, 0 off, 1 on */
static ID3D11Device         *s_dev;
static ID3D11DeviceContext  *s_ctx;
static ID3D11DeviceContext1 *s_ctx1;

static uint32_t s_frame = 1;             /* flips seen */

/* Ring buffers for vertex and index data, written NO_OVERWRITE and
 * discarded when they wrap. */
#define VB_RING_BYTES (64u << 20)
#define IB_RING_BYTES (16u << 20)
static ID3D11Buffer *s_vb, *s_ib, *s_zero_vb;
/* Start full, so the first map of each ring is a DISCARD. */
static uint32_t s_vb_pos = VB_RING_BYTES, s_ib_pos = IB_RING_BYTES;

static ID3D11Buffer *s_cb_vsc, *s_cb_vsd, *s_cb_psc;
static uint32_t s_vsc_gen = 0xFFFFFFFFu;
static const Nv2aVpState *s_vsc_owner;

static ID3D11VertexShader *s_vs_pass;
static ID3DBlob *s_sig_blob;             /* bytecode with the shared input signature */

static struct {
    uint32_t draws, tris, skipped, vs_compiled, ps_compiled, compile_failed;
    uint32_t tex_uploads, tex_bad, rt_created, ds_created, readbacks;
    uint32_t vs_const_writes, fallback_attrs, points_lines;
    uint64_t vb_bytes, ib_bytes, tex_hash_bytes;
    uint64_t draw_ticks;              /* QPC ticks inside nv2a_d3d11_draw */
} s_st;
static LARGE_INTEGER s_qpf;

/* ── Small helpers ───────────────────────────────────────────────────── */

static uint64_t hash64(const void *data, size_t n, uint64_t h)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h ^= v;
        h *= 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    for (; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

/* A cheap "has this changed" for a large texture: 64 spans of 64 bytes spread
 * across it. Caught: anything that rewrites the image -- a video frame, a
 * streamed-in level texture, a surface the title copies into. Missed until
 * the next full check: an edit smaller than the gaps. */
static uint64_t hash_sparse(const uint8_t *p, size_t n)
{
    uint64_t h = 0x51ED270B27u;
    size_t k, step;

    if (n <= 4096)
        return hash64(p, n, h);
    step = (n - 64) / 63;
    for (k = 0; k < 64; k++)
        h = hash64(p + k * step, 64, h);
    return h;
}

static void unpack_argb(uint32_t argb, float out[4])
{
    out[0] = (float)((argb >> 16) & 0xFF) / 255.0f;
    out[1] = (float)((argb >>  8) & 0xFF) / 255.0f;
    out[2] = (float)( argb        & 0xFF) / 255.0f;
    out[3] = (float)( argb >> 24        ) / 255.0f;
}

static const uint8_t *guest(void)
{
    return (const uint8_t *)xbox_GetMemoryOffset();
}

/* ── Shaders ─────────────────────────────────────────────────────────── */

static ID3DBlob *compile(const char *src, size_t len, const char *target,
                         UINT flags)
{
    ID3DBlob *code = NULL, *err = NULL;
    HRESULT hr = D3DCompile(src, len, NULL, NULL, NULL, "main", target,
                            flags, 0, &code, &err);

    if (FAILED(hr)) {
        static int dumped;
        s_st.compile_failed++;
        fprintf(stderr, "  [D3D11] %s compile failed: %s\n", target,
                err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        if (dumped < 4) {
            char path[64];
            FILE *f;
            snprintf(path, sizeof path, "gpu_shader_fail_%d.hlsl", dumped++);
            f = fopen(path, "wb");
            if (f) {
                fwrite(src, 1, len, f);
                fclose(f);
                fprintf(stderr, "  [D3D11]   source written to %s\n", path);
            }
        }
        if (err)
            ID3D10Blob_Release(err);
        if (code)
            ID3D10Blob_Release(code);
        return NULL;
    }
    if (err)
        ID3D10Blob_Release(err);
    return code;
}

#define SRC_BYTES (256u << 10)
static char s_src[SRC_BYTES];

typedef struct { uint64_t key; void *obj; } CacheSlot;

#define VS_CACHE 2048
#define PS_CACHE 2048
static CacheSlot s_vs_cache[VS_CACHE], s_ps_cache[PS_CACHE];

static void **cache_find(CacheSlot *tab, uint32_t n, uint64_t key, int *found)
{
    uint32_t i, at = (uint32_t)(key ^ (key >> 32)) & (n - 1);

    for (i = 0; i < n; i++) {
        CacheSlot *s = &tab[(at + i) & (n - 1)];
        if (s->obj && s->key == key) {
            *found = 1;
            return &s->obj;
        }
        if (!s->obj) {
            s->key = key;
            *found = 0;
            return &s->obj;
        }
    }
    *found = 0;
    return NULL;
}

/* The program at program_start, as a hash of its slots up to FINAL. */
static uint64_t vp_hash(const Nv2aVpState *vp)
{
    uint64_t h = 0xCBF29CE484222325ull ^ vp->program_start;
    uint32_t pc;

    for (pc = vp->program_start; pc < NV2A_VP_SLOTS; pc++) {
        h = hash64(&vp->program[pc][1], 12, h);
        if (vp->program[pc][3] & 1u)
            break;
    }
    return h;
}

static ID3D11VertexShader *vs_for(const Nv2aVpState *vp)
{
    static const Nv2aVpState *last_vp;
    static uint32_t last_gen = 0xFFFFFFFFu, last_start = 0xFFFFFFFFu;
    static ID3D11VertexShader *last_vs;
    uint64_t key;
    void **slot;
    int found, writes_consts = 0;
    size_t len;
    ID3DBlob *code;
    ID3D11VertexShader *vs = NULL;

    if (vp == last_vp && vp->program_gen == last_gen
            && vp->program_start == last_start)
        return last_vs;
    key = vp_hash(vp);
    slot = cache_find(s_vs_cache, VS_CACHE, key, &found);
    if (slot && found) {
        vs = (ID3D11VertexShader *)*slot;
    } else {
        len = nv2a_hlsl_vs(vp, s_src, sizeof s_src, &writes_consts);
        if (writes_consts)
            s_st.vs_const_writes++;
        code = len ? compile(s_src, len, "vs_5_0", D3DCOMPILE_IEEE_STRICTNESS) : NULL;
        if (code) {
            if (SUCCEEDED(ID3D11Device_CreateVertexShader(s_dev,
                    ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code),
                    NULL, &vs))) {
                s_st.vs_compiled++;
                if (slot)
                    *slot = vs;
            }
            ID3D10Blob_Release(code);
        }
        /* A program that will not compile is not retried every draw. */
        if (!vs && slot)
            *slot = (void *)(uintptr_t)1;
    }
    if ((uintptr_t)vs == 1)
        vs = NULL;
    last_vp = vp;
    last_gen = vp->program_gen;
    last_start = vp->program_start;
    last_vs = vs;
    return vs;
}

static ID3D11PixelShader *ps_for(const Nv2aCombiner *c)
{
    static const Nv2aCombiner *last_c;
    static uint32_t last_gen = 0xFFFFFFFFu;
    static ID3D11PixelShader *last_ps;
    Nv2aHlslPsKey key;
    uint64_t h;
    void **slot;
    int found;
    ID3D11PixelShader *ps = NULL;

    if (c == last_c && c->gen == last_gen)
        return last_ps;
    nv2a_hlsl_ps_key(c, &key);
    h = hash64(&key, sizeof key, 0x84222325CBF29CE4ull);
    slot = cache_find(s_ps_cache, PS_CACHE, h, &found);
    if (slot && found) {
        ps = (ID3D11PixelShader *)*slot;
    } else {
        size_t len = nv2a_hlsl_ps(&key, s_src, sizeof s_src);
        ID3DBlob *code = len ? compile(s_src, len, "ps_5_0", 0) : NULL;
        if (code) {
            if (SUCCEEDED(ID3D11Device_CreatePixelShader(s_dev,
                    ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code),
                    NULL, &ps))) {
                s_st.ps_compiled++;
                if (slot)
                    *slot = ps;
            }
            ID3D10Blob_Release(code);
        }
        if (!ps && slot)
            *slot = (void *)(uintptr_t)1;
    }
    if ((uintptr_t)ps == 1)
        ps = NULL;
    last_c = c;
    last_gen = c->gen;
    last_ps = ps;
    return ps;
}

/* ── Fixed-function state objects, cached by descriptor ──────────────── */

#define STATE_CACHE 256
static CacheSlot s_blend_cache[STATE_CACHE], s_ds_cache[STATE_CACHE],
                 s_rs_cache[STATE_CACHE], s_smp_cache[STATE_CACHE],
                 s_il_cache[STATE_CACHE];

static D3D11_BLEND blend_factor(uint32_t f, int alpha)
{
    switch (f) {
    case 0x0000: return D3D11_BLEND_ZERO;
    case 0x0001: return D3D11_BLEND_ONE;
    case 0x0300: return alpha ? D3D11_BLEND_SRC_ALPHA : D3D11_BLEND_SRC_COLOR;
    case 0x0301: return alpha ? D3D11_BLEND_INV_SRC_ALPHA : D3D11_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D11_BLEND_SRC_ALPHA;
    case 0x0303: return D3D11_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D11_BLEND_DEST_ALPHA;
    case 0x0305: return D3D11_BLEND_INV_DEST_ALPHA;
    case 0x0306: return alpha ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_DEST_COLOR;
    case 0x0307: return alpha ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_DEST_COLOR;
    case 0x0308: return alpha ? D3D11_BLEND_ONE : D3D11_BLEND_SRC_ALPHA_SAT;
    case 0x8001: case 0x8003: return D3D11_BLEND_BLEND_FACTOR;
    case 0x8002: case 0x8004: return D3D11_BLEND_INV_BLEND_FACTOR;
    default:     return D3D11_BLEND_ONE;
    }
}

static D3D11_BLEND_OP blend_op(uint32_t eq)
{
    switch (eq) {
    case 0x800A: return D3D11_BLEND_OP_SUBTRACT;
    case 0x800B: return D3D11_BLEND_OP_REV_SUBTRACT;
    case 0x8007: return D3D11_BLEND_OP_MIN;
    case 0x8008: return D3D11_BLEND_OP_MAX;
    default:     return D3D11_BLEND_OP_ADD;
    }
}

static ID3D11BlendState *blend_state(const Nv2aHwDraw *d)
{
    D3D11_BLEND_DESC bd;
    D3D11_RENDER_TARGET_BLEND_DESC *rt = &bd.RenderTarget[0];
    uint64_t key;
    void **slot;
    int found;
    ID3D11BlendState *bs = NULL;
    UINT8 mask = 0;

    if (d->color_mask & 0x00010000u) mask |= D3D11_COLOR_WRITE_ENABLE_RED;
    if (d->color_mask & 0x00000100u) mask |= D3D11_COLOR_WRITE_ENABLE_GREEN;
    if (d->color_mask & 0x00000001u) mask |= D3D11_COLOR_WRITE_ENABLE_BLUE;
    if (d->color_mask & 0x01000000u) mask |= D3D11_COLOR_WRITE_ENABLE_ALPHA;

    memset(&bd, 0, sizeof bd);
    rt->BlendEnable = d->blend_enable != 0;
    rt->SrcBlend = blend_factor(d->blend_sfactor, 0);
    rt->DestBlend = blend_factor(d->blend_dfactor, 0);
    rt->BlendOp = blend_op(d->blend_eq);
    rt->SrcBlendAlpha = blend_factor(d->blend_sfactor, 1);
    rt->DestBlendAlpha = blend_factor(d->blend_dfactor, 1);
    rt->BlendOpAlpha = rt->BlendOp;
    rt->RenderTargetWriteMask = mask;

    key = hash64(&bd, sizeof bd, 1);
    slot = cache_find(s_blend_cache, STATE_CACHE, key, &found);
    if (slot && found)
        return (ID3D11BlendState *)*slot;
    if (SUCCEEDED(ID3D11Device_CreateBlendState(s_dev, &bd, &bs)) && slot)
        *slot = bs;
    return bs;
}

static D3D11_COMPARISON_FUNC cmp_func(uint32_t f)
{
    if (f >= 0x0200 && f <= 0x0207)
        return (D3D11_COMPARISON_FUNC)(D3D11_COMPARISON_NEVER + (f - 0x0200));
    return D3D11_COMPARISON_ALWAYS;
}

static D3D11_STENCIL_OP stencil_op(uint32_t op)
{
    switch (op) {
    case 0x0000: return D3D11_STENCIL_OP_ZERO;
    case 0x1E01: return D3D11_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D11_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D11_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D11_STENCIL_OP_INVERT;
    case 0x8507: return D3D11_STENCIL_OP_INCR;
    case 0x8508: return D3D11_STENCIL_OP_DECR;
    default:     return D3D11_STENCIL_OP_KEEP;
    }
}

static ID3D11DepthStencilState *depth_state(const Nv2aHwDraw *d)
{
    D3D11_DEPTH_STENCIL_DESC dd;
    uint64_t key;
    void **slot;
    int found;
    ID3D11DepthStencilState *ds = NULL;

    memset(&dd, 0, sizeof dd);
    /* GL semantics, which NV2A has: with the test off nothing is written.
     * RECOMP_GPU_NODEPTH=1 turns the test off everywhere, to find geometry
     * that is drawn but hidden. */
    static int nodepth = -1;
    if (nodepth < 0)
        nodepth = getenv("RECOMP_GPU_NODEPTH") != NULL;
    dd.DepthEnable = d->depth_test != 0 && !nodepth;
    dd.DepthWriteMask = d->depth_mask ? D3D11_DEPTH_WRITE_MASK_ALL
                                      : D3D11_DEPTH_WRITE_MASK_ZERO;
    dd.DepthFunc = cmp_func(d->depth_func);
    dd.StencilEnable = d->stencil_test != 0;
    dd.StencilReadMask = (UINT8)d->stencil_func_mask;
    dd.StencilWriteMask = (UINT8)d->stencil_mask;
    dd.FrontFace.StencilFunc = cmp_func(d->stencil_func);
    dd.FrontFace.StencilFailOp = stencil_op(d->stencil_fail);
    dd.FrontFace.StencilDepthFailOp = stencil_op(d->stencil_zfail);
    dd.FrontFace.StencilPassOp = stencil_op(d->stencil_zpass);
    dd.BackFace = dd.FrontFace;

    key = hash64(&dd, sizeof dd, 2);
    slot = cache_find(s_ds_cache, STATE_CACHE, key, &found);
    if (slot && found)
        return (ID3D11DepthStencilState *)*slot;
    if (SUCCEEDED(ID3D11Device_CreateDepthStencilState(s_dev, &dd, &ds)) && slot)
        *slot = ds;
    return ds;
}

static ID3D11RasterizerState *raster_state(const Nv2aHwDraw *d)
{
    static int cull = -1;
    D3D11_RASTERIZER_DESC rd;
    uint64_t key;
    void **slot;
    int found;
    ID3D11RasterizerState *rs = NULL;

    if (cull < 0)
        cull = getenv("RECOMP_GPU_CULL") != NULL;
    memset(&rd, 0, sizeof rd);
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    /* The software rasteriser never culled and drew right, so culling is
     * opt-in until its winding convention has been checked on screen. The
     * y flip into clip space reverses the winding, hence the inversion. */
    if (cull && d->cull_enable) {
        if (d->cull_face == 0x404)
            rd.CullMode = D3D11_CULL_FRONT;
        else if (d->cull_face == 0x405)
            rd.CullMode = D3D11_CULL_BACK;
        rd.FrontCounterClockwise = d->front_face == 0x900;
    }
    /* No z clip: the software path never clipped depth either. */
    rd.DepthClipEnable = FALSE;
    rd.ScissorEnable = TRUE;

    key = hash64(&rd, sizeof rd, 3);
    slot = cache_find(s_rs_cache, STATE_CACHE, key, &found);
    if (slot && found)
        return (ID3D11RasterizerState *)*slot;
    if (SUCCEEDED(ID3D11Device_CreateRasterizerState(s_dev, &rd, &rs)) && slot)
        *slot = rs;
    return rs;
}

static D3D11_TEXTURE_ADDRESS_MODE addr_mode(uint32_t m)
{
    switch (m) {
    case 2: return D3D11_TEXTURE_ADDRESS_MIRROR;
    case 3: case 5: return D3D11_TEXTURE_ADDRESS_CLAMP;
    case 4: return D3D11_TEXTURE_ADDRESS_BORDER;
    default: return D3D11_TEXTURE_ADDRESS_WRAP;
    }
}

static ID3D11SamplerState *sampler(const Nv2aHwTexture *t, uint32_t levels)
{
    D3D11_SAMPLER_DESC sd;
    uint32_t min = (t->filter >> 16) & 0x3Fu, mag = (t->filter >> 24) & 0xFu;
    int lmin, lmag, lmip, mips;
    uint64_t key;
    void **slot;
    int found;
    ID3D11SamplerState *ss = NULL;

    /* NV2A minification: 1 box/2 tent at level 0 only, 3/4 box/tent with
     * the nearest level, 5/6 box/tent blending two levels. */
    lmin = (min == 2 || min == 4 || min == 6 || min == 7);
    lmip = (min == 5 || min == 6);
    mips = (min >= 3 && min <= 6) && levels > 1;
    lmag = mag != 1;
    memset(&sd, 0, sizeof sd);
    sd.Filter = (D3D11_FILTER)((lmin << 4) | (lmag << 2) | lmip);
    sd.AddressU = addr_mode(t->address & 0xFu);
    sd.AddressV = addr_mode((t->address >> 8) & 0xFu);
    sd.AddressW = addr_mode((t->address >> 16) & 0xFu);
    sd.MaxAnisotropy = 1;
    sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
    unpack_argb(t->border, sd.BorderColor);
    sd.MinLOD = 0.0f;
    sd.MaxLOD = mips ? D3D11_FLOAT32_MAX : 0.0f;

    key = hash64(&sd, sizeof sd, 4);
    slot = cache_find(s_smp_cache, STATE_CACHE, key, &found);
    if (slot && found)
        return (ID3D11SamplerState *)*slot;
    if (SUCCEEDED(ID3D11Device_CreateSamplerState(s_dev, &sd, &ss)) && slot)
        *slot = ss;
    return ss;
}

/* ── Render targets and depth buffers ────────────────────────────────── */

typedef struct {
    uint32_t va, w, h;
    ID3D11Texture2D *tex;
    ID3D11RenderTargetView *rtv;
    ID3D11ShaderResourceView *srv;
    uint32_t drawn_frame;
} RtEntry;

typedef struct {
    uint32_t va, w, h;
    ID3D11Texture2D *tex;
    ID3D11DepthStencilView *dsv;
    ID3D11ShaderResourceView *srv;
    uint32_t used_frame;
} DsEntry;

#define RT_SLOTS 48
#define DS_SLOTS 48
static RtEntry s_rt[RT_SLOTS];
static DsEntry s_ds[DS_SLOTS];

static void rt_release(RtEntry *e)
{
    if (e->srv) ID3D11ShaderResourceView_Release(e->srv);
    if (e->rtv) ID3D11RenderTargetView_Release(e->rtv);
    if (e->tex) ID3D11Texture2D_Release(e->tex);
    memset(e, 0, sizeof *e);
}

static RtEntry *rt_find(uint32_t va)
{
    int i;
    for (i = 0; i < RT_SLOTS; i++)
        if (s_rt[i].tex && s_rt[i].va == va)
            return &s_rt[i];
    return NULL;
}

/* The colour surface at `va`, at least w x h, creating or growing it. A
 * grown surface keeps what was drawn into the old one. */
static RtEntry *rt_get(uint32_t va, uint32_t w, uint32_t h)
{
    RtEntry *e = rt_find(va), *victim = NULL;
    D3D11_TEXTURE2D_DESC td;
    RtEntry n;
    int i;

    if (e && e->w >= w && e->h >= h)
        return e;
    memset(&n, 0, sizeof n);
    n.va = va;
    n.w = e && e->w > w ? e->w : w;
    n.h = e && e->h > h ? e->h : h;
    memset(&td, 0, sizeof td);
    td.Width = n.w;
    td.Height = n.h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &n.tex)))
        return e;
    ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)n.tex, NULL, &n.rtv);
    ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)n.tex, NULL, &n.srv);
    {
        const float black[4] = { 0, 0, 0, 1 };
        ID3D11DeviceContext_ClearRenderTargetView(s_ctx, n.rtv, black);
    }
    s_st.rt_created++;
    if (e) {
        D3D11_BOX box = { 0, 0, 0, e->w, e->h, 1 };
        ID3D11DeviceContext_CopySubresourceRegion(s_ctx, (ID3D11Resource *)n.tex, 0,
                0, 0, 0, (ID3D11Resource *)e->tex, 0, &box);
        n.drawn_frame = e->drawn_frame;
        rt_release(e);
        *e = n;
        return e;
    }
    for (i = 0; i < RT_SLOTS; i++) {
        if (!s_rt[i].tex) {
            victim = &s_rt[i];
            break;
        }
        if (!victim || s_rt[i].drawn_frame < victim->drawn_frame)
            victim = &s_rt[i];
    }
    rt_release(victim);
    *victim = n;
    return victim;
}

static DsEntry *ds_get(uint32_t va, uint32_t w, uint32_t h)
{
    D3D11_TEXTURE2D_DESC td;
    D3D11_DEPTH_STENCIL_VIEW_DESC dv;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv;
    DsEntry *victim = NULL, n;
    int i;

    for (i = 0; i < DS_SLOTS; i++)
        if (s_ds[i].tex && s_ds[i].va == va && s_ds[i].w == w && s_ds[i].h == h) {
            s_ds[i].used_frame = s_frame;
            return &s_ds[i];
        }
    memset(&n, 0, sizeof n);
    n.va = va; n.w = w; n.h = h; n.used_frame = s_frame;
    memset(&td, 0, sizeof td);
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R24G8_TYPELESS;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &n.tex)))
        return NULL;
    memset(&dv, 0, sizeof dv);
    dv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    ID3D11Device_CreateDepthStencilView(s_dev, (ID3D11Resource *)n.tex, &dv, &n.dsv);
    memset(&sv, 0, sizeof sv);
    sv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels = 1;
    ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)n.tex, &sv, &n.srv);
    ID3D11DeviceContext_ClearDepthStencilView(s_ctx, n.dsv,
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
    s_st.ds_created++;
    for (i = 0; i < DS_SLOTS; i++) {
        if (!s_ds[i].tex) {
            victim = &s_ds[i];
            break;
        }
        if (!victim || s_ds[i].used_frame < victim->used_frame)
            victim = &s_ds[i];
    }
    if (victim->srv) ID3D11ShaderResourceView_Release(victim->srv);
    if (victim->dsv) ID3D11DepthStencilView_Release(victim->dsv);
    if (victim->tex) ID3D11Texture2D_Release(victim->tex);
    *victim = n;
    return victim;
}

/* How big a surface is: a swizzled one says so in its format, a pitched one
 * is as wide as its pitch and as tall as anything has drawn into it. */
static void surface_dims(const Nv2aHwDraw *d, uint32_t pitch, uint32_t bpp,
                         uint32_t *w, uint32_t *h)
{
    uint32_t right = d->clip_x + d->clip_w, bottom = d->clip_y + d->clip_h;

    if (((d->format >> 8) & 0xFu) == 2u) {
        *w = 1u << ((d->format >> 16) & 0xFFu);
        *h = 1u << ((d->format >> 24) & 0xFFu);
    } else {
        *w = bpp && pitch ? pitch / bpp : right;
        *h = bottom;
    }
    if (*w < right) *w = right;
    if (*h < bottom) *h = bottom;
    if (*w > 4096) *w = 4096;
    if (*h > 4096) *h = 4096;
}

static uint32_t color_bpp(uint32_t format)
{
    switch (format & 0xFu) {
    case 1: case 2: case 3: case 0xA: return 2;
    case 9: return 1;
    default: return 4;
    }
}

/* ── Textures ─────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t va, format, pitch, rect;       /* key */
    uint32_t w, h, levels, bytes;
    uint64_t hash, shash;             /* full, and sparse (hash_sparse) */
    uint32_t hashed_frame, used_frame, full_frame;
    ID3D11Texture2D *tex;
    ID3D11ShaderResourceView *srv;
    int bad;
} TexEntry;

#define TEX_SLOTS 4096
static TexEntry s_tex[TEX_SLOTS];

static uint32_t dxt_block(uint32_t color)
{
    return d3d8_format_dxt_block_bytes(color);
}

/* Bytes of one level, and the layout facts upload needs. */
static uint32_t level_bytes(uint32_t color, uint32_t w, uint32_t h, uint32_t pitch,
                            int linear)
{
    uint32_t bb = dxt_block(color);
    if (bb)
        return ((w + 3) / 4) * ((h + 3) / 4) * bb;
    if (linear)
        return pitch * h;
    return w * h * nv2a_pb_texel_bytes(color);
}

static void tex_release(TexEntry *e)
{
    if (e->srv) ID3D11ShaderResourceView_Release(e->srv);
    if (e->tex) ID3D11Texture2D_Release(e->tex);
    memset(e, 0, sizeof *e);
}

/* (Re)fill a texture from guest memory. */
static int tex_upload(TexEntry *e, const Nv2aHwTexture *t, int create)
{
    const uint8_t *src = guest() + t->va;
    uint32_t bb = dxt_block(t->color);
    int linear = !d3d8_format_is_swizzled(t->color) && !bb;
    int bc = bb && (e->w % 4 == 0) && (e->h % 4 == 0);
    uint32_t lv, w = e->w, h = e->h, off = 0;
    static uint32_t *scratch;
    static size_t scratch_n;

    if (create) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof td);
        td.Width = e->w;
        td.Height = e->h;
        td.MipLevels = e->levels;
        td.ArraySize = 1;
        td.Format = bc ? (bb == 8 ? DXGI_FORMAT_BC1_UNORM
                                  : (t->color == 0x0E ? DXGI_FORMAT_BC2_UNORM
                                                      : DXGI_FORMAT_BC3_UNORM))
                       : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &e->tex)))
            return 0;
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)e->tex,
                                              NULL, &e->srv);
    }
    for (lv = 0; lv < e->levels; lv++) {
        uint32_t lb = level_bytes(t->color, w, h, t->pitch, linear);
        if (bc) {
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)e->tex,
                    lv, NULL, src + off, ((w + 3) / 4) * bb, 0);
        } else {
            size_t need = (size_t)w * h;
            if (need > scratch_n) {
                free(scratch);
                scratch = (uint32_t *)malloc(need * 4);
                scratch_n = scratch ? need : 0;
                if (!scratch)
                    return 0;
            }
            if (!nv2a_pb_decode_level(t->color, src + off, w, h,
                                      linear ? t->pitch : 0, scratch))
                return 0;
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)e->tex,
                    lv, NULL, scratch, w * 4, 0);
        }
        off += lb;
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
    }
    s_st.tex_uploads++;
    return 1;
}

static ID3D11ShaderResourceView *tex_srv(const Nv2aHwTexture *t, uint32_t *out_w,
                                         uint32_t *out_h, uint32_t *out_levels)
{
    uint32_t bb = dxt_block(t->color);
    int swz = d3d8_format_is_swizzled(t->color) || bb;
    uint32_t w = t->width, h = t->height, levels = 1, bytes = 0, lv, lw, lh;
    uint32_t rect = (t->width << 16) | (t->height & 0xFFFFu);
    uint32_t at, i;
    TexEntry *e = NULL, *victim = NULL;

    if (swz) {
        levels = (t->format >> 16) & 0xFu;
        if (!levels)
            levels = 1;
        /* Stop where the chain would go below one texel -- or, for a block
         * format, where D3D11 would want a level smaller than the data. */
        for (lv = 1, lw = w, lh = h; lv < levels; lv++) {
            if (lw == 1 && lh == 1)
                break;
            lw = lw > 1 ? lw >> 1 : 1;
            lh = lh > 1 ? lh >> 1 : 1;
        }
        levels = lv;
    }
    if (!w || !h || w > 4096 || h > 4096)
        return NULL;
    if (!bb && !nv2a_pb_texel_bytes(t->color))
        return NULL;
    for (lv = 0, lw = w, lh = h; lv < levels; lv++) {
        bytes += level_bytes(t->color, lw, lh, t->pitch, !swz);
        lw = lw > 1 ? lw >> 1 : 1;
        lh = lh > 1 ? lh >> 1 : 1;
    }

    at = (t->va * 2654435761u ^ t->format * 40503u) & (TEX_SLOTS - 1);
    for (i = 0; i < 16; i++) {
        TexEntry *c = &s_tex[(at + i) & (TEX_SLOTS - 1)];
        if (c->tex || c->bad) {
            if (c->va == t->va && c->format == t->format && c->pitch == t->pitch
                    && c->rect == rect) {
                e = c;
                break;
            }
            if (!victim || c->used_frame < victim->used_frame)
                victim = c;
        } else if (!victim || victim->tex || victim->bad) {
            victim = c;
        }
    }
    if (!e) {
        tex_release(victim);
        e = victim;
        e->va = t->va; e->format = t->format; e->pitch = t->pitch; e->rect = rect;
        e->w = w; e->h = h; e->levels = levels; e->bytes = bytes;
        e->hash = hash64(guest() + t->va, bytes, 7);
        e->shash = hash_sparse(guest() + t->va, bytes);
        e->hashed_frame = e->full_frame = s_frame;
        if (!tex_upload(e, t, 1)) {
            static uint32_t said[16];
            static int nsaid;
            int k, fresh = 1;
            for (k = 0; k < nsaid; k++)
                if (said[k] == t->color) fresh = 0;
            if (fresh && nsaid < 16) {
                said[nsaid++] = t->color;
                fprintf(stderr, "  [D3D11] texture format 0x%02X not uploaded"
                        " (%ux%u at 0x%08X)\n", t->color, w, h, t->va);
            }
            if (e->srv) ID3D11ShaderResourceView_Release(e->srv);
            if (e->tex) ID3D11Texture2D_Release(e->tex);
            e->srv = NULL; e->tex = NULL;
            e->bad = 1;
            s_st.tex_bad++;
        }
    } else if (e->hashed_frame != s_frame && !e->bad) {
        /* Once a frame: has the title rewritten it? Small textures are hashed
         * whole; large ones by sample, with a whole hash every 16 frames,
         * staggered by slot so they do not all fall on one frame. Hashing
         * every texture whole was 12 MB of reads a frame in game. */
        const uint8_t *src = guest() + t->va;
        int full = e->bytes <= 65536
                || ((s_frame + (uint32_t)(e - s_tex)) & 15u) == 0;
        int changed;
        e->hashed_frame = s_frame;
        if (full) {
            uint64_t hh = hash64(src, e->bytes, 7);
            s_st.tex_hash_bytes += e->bytes;
            changed = hh != e->hash;
            e->hash = hh;
            e->full_frame = s_frame;
            if (changed)
                e->shash = hash_sparse(src, e->bytes);
        } else {
            uint64_t sh = hash_sparse(src, e->bytes);
            s_st.tex_hash_bytes += 4096;
            changed = sh != e->shash;
            e->shash = sh;
            if (changed)
                e->hash = hash64(src, e->bytes, 7);
        }
        if (changed)
            tex_upload(e, t, 0);
    }
    e->used_frame = s_frame;
    *out_w = e->w;
    *out_h = e->h;
    *out_levels = e->levels;
    return e->bad ? NULL : e->srv;
}

/* ── Vertex data ─────────────────────────────────────────────────────── */

/* The input-assembler format for one attribute, and how many bytes it reads;
 * 0 if this type is not passed through raw. */
static DXGI_FORMAT attr_format(const Nv2aHwAttr *a, uint32_t *bytes, uint32_t *mode)
{
    switch (a->type) {
    case 2:
        *mode = NV2A_HLSL_AFMT_FLOAT;
        *bytes = 4 * a->size;
        switch (a->size) {
        case 1: return DXGI_FORMAT_R32_UINT;
        case 2: return DXGI_FORMAT_R32G32_UINT;
        case 3: return DXGI_FORMAT_R32G32B32_UINT;
        default: *bytes = 16; return DXGI_FORMAT_R32G32B32A32_UINT;
        }
    case 0:
        *mode = NV2A_HLSL_AFMT_COLOR;
        *bytes = 4;
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case 4:
        *mode = NV2A_HLSL_AFMT_UBYTE;
        if (a->size == 1) { *bytes = 1; return DXGI_FORMAT_R8_UINT; }
        if (a->size == 2) { *bytes = 2; return DXGI_FORMAT_R8G8_UINT; }
        *bytes = 4;
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case 1:
    case 5:
        *mode = a->type == 1 ? NV2A_HLSL_AFMT_S1 : NV2A_HLSL_AFMT_S32K;
        if (a->size == 1) { *bytes = 2; return DXGI_FORMAT_R16_UINT; }
        if (a->size == 2) { *bytes = 4; return DXGI_FORMAT_R16G16_UINT; }
        *bytes = 8;
        return DXGI_FORMAT_R16G16B16A16_UINT;
    case 6:
        *mode = NV2A_HLSL_AFMT_CMP;
        *bytes = 4;
        return DXGI_FORMAT_R32_UINT;
    default:
        *mode = NV2A_HLSL_AFMT_ABSENT;
        *bytes = 0;
        return DXGI_FORMAT_UNKNOWN;
    }
}

typedef struct {
    uint32_t base, stride, end;       /* byte range of one element, from base */
} Stream;

#define ZERO_SLOT 16

static void *ring_map(ID3D11Buffer *buf, uint32_t *pos, uint32_t cap,
                      uint32_t bytes, uint32_t align)
{
    D3D11_MAPPED_SUBRESOURCE m;
    D3D11_MAP how = D3D11_MAP_WRITE_NO_OVERWRITE;
    uint32_t at = (*pos + align - 1) & ~(align - 1);

    if (bytes > cap)
        return NULL;
    if (at + bytes > cap) {
        how = D3D11_MAP_WRITE_DISCARD;
        at = 0;
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)buf, 0, how, 0, &m)))
        return NULL;
    *pos = at + bytes;
    return (uint8_t *)m.pData + at;
}

/* How many list indices a primitive of n vertices becomes. */
static uint32_t tri_indices(uint32_t prim, uint32_t n)
{
    switch (prim) {
    case 5:  return (n / 3) * 3;                       /* TRIANGLES */
    case 6:                                            /* TRIANGLE_STRIP */
    case 7:                                            /* TRIANGLE_FAN */
    case 10: return n >= 3 ? (n - 2) * 3 : 0;          /* POLYGON */
    case 8:  return (n / 4) * 6;                       /* QUADS */
    case 9:  return n >= 4 ? ((n - 2) / 2) * 6 : 0;    /* QUAD_STRIP */
    default: return 0;
    }
}

static void emit_tris(uint32_t prim, const uint32_t *idx, uint32_t n,
                      uint32_t base, uint32_t *out)
{
    uint32_t i, k = 0;

#define V(j) (idx[(j)] - base)
    switch (prim) {
    case 5:
        for (i = 0; i + 2 < n; i += 3) {
            out[k++] = V(i); out[k++] = V(i + 1); out[k++] = V(i + 2);
        }
        break;
    case 6:
        /* Alternate the winding so every triangle keeps the strip's. */
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) { out[k++] = V(i + 1); out[k++] = V(i); }
            else       { out[k++] = V(i);     out[k++] = V(i + 1); }
            out[k++] = V(i + 2);
        }
        break;
    case 7:
    case 10:
        for (i = 1; i + 1 < n; i++) {
            out[k++] = V(0); out[k++] = V(i); out[k++] = V(i + 1);
        }
        break;
    case 8:
        for (i = 0; i + 3 < n; i += 4) {
            out[k++] = V(i); out[k++] = V(i + 1); out[k++] = V(i + 2);
            out[k++] = V(i); out[k++] = V(i + 2); out[k++] = V(i + 3);
        }
        break;
    case 9:
        for (i = 0; i + 3 < n; i += 2) {
            out[k++] = V(i); out[k++] = V(i + 1); out[k++] = V(i + 3);
            out[k++] = V(i); out[k++] = V(i + 3); out[k++] = V(i + 2);
        }
        break;
    default:
        break;
    }
#undef V
}

/* ── Frame readback ──────────────────────────────────────────────────── */

#define STAGES 3
static ID3D11Texture2D *s_stage[STAGES];
static uint32_t s_stage_w[STAGES], s_stage_h[STAGES];
static int s_stage_full[STAGES];
static uint32_t s_stage_next;

static void readback_one(int i)
{
    D3D11_MAPPED_SUBRESOURCE m;

    if (!s_stage_full[i])
        return;
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_stage[i], 0,
                                       D3D11_MAP_READ, 0, &m)))
        return;
    xbox_FramebufferWindowPresentArgb(m.pData, s_stage_w[i], s_stage_h[i],
                                      m.RowPitch);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_stage[i], 0);
    s_stage_full[i] = 0;
    s_st.readbacks++;
}

/* ── Init ────────────────────────────────────────────────────────────── */

static ID3D11Buffer *make_buffer(UINT bytes, UINT bind, D3D11_USAGE usage,
                                 const void *init)
{
    D3D11_BUFFER_DESC bd;
    D3D11_SUBRESOURCE_DATA sd;
    ID3D11Buffer *b = NULL;

    memset(&bd, 0, sizeof bd);
    bd.ByteWidth = bytes;
    bd.BindFlags = bind;
    bd.Usage = usage;
    bd.CPUAccessFlags = usage == D3D11_USAGE_DYNAMIC ? D3D11_CPU_ACCESS_WRITE : 0;
    sd.pSysMem = init;
    sd.SysMemPitch = sd.SysMemSlicePitch = 0;
    ID3D11Device_CreateBuffer(s_dev, &bd, init ? &sd : NULL, &b);
    return b;
}

static int init(void)
{
    static const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0
    };
    UINT flags = 0;
    HRESULT hr;
    size_t len;
    static const uint8_t zeros[64] = { 0 };

    IDXGIAdapter1 *adapter = NULL;

    if (getenv("RECOMP_GPU_DEBUG"))
        flags |= D3D11_CREATE_DEVICE_DEBUG;
    /* The fast GPU, not the default one: on a laptop the default adapter is
     * the integrated GPU. RECOMP_GPU_ADAPTER=default keeps the default. */
    if (!getenv("RECOMP_GPU_ADAPTER")
            || _stricmp(getenv("RECOMP_GPU_ADAPTER"), "default") != 0) {
        IDXGIFactory6 *f6 = NULL;
        if (SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory6, (void **)&f6))) {
            IDXGIFactory6_EnumAdapterByGpuPreference(f6, 0,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, &IID_IDXGIAdapter1,
                    (void **)&adapter);
            IDXGIFactory6_Release(f6);
        }
    }
    hr = D3D11CreateDevice((IDXGIAdapter *)adapter,
                           adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
                           NULL, flags, levels, 2, D3D11_SDK_VERSION, &s_dev, NULL, &s_ctx);
    if (adapter)
        IDXGIAdapter1_Release(adapter);
    if (FAILED(hr)) {
        fprintf(stderr, "  [D3D11] device creation failed (0x%08lX); "
                "staying on the software rasteriser\n", (unsigned long)hr);
        return 0;
    }
    ID3D11DeviceContext_QueryInterface(s_ctx, &IID_ID3D11DeviceContext1,
                                       (void **)&s_ctx1);
    s_vb = make_buffer(VB_RING_BYTES, D3D11_BIND_VERTEX_BUFFER, D3D11_USAGE_DYNAMIC, NULL);
    s_ib = make_buffer(IB_RING_BYTES, D3D11_BIND_INDEX_BUFFER, D3D11_USAGE_DYNAMIC, NULL);
    s_zero_vb = make_buffer(sizeof zeros, D3D11_BIND_VERTEX_BUFFER,
                            D3D11_USAGE_IMMUTABLE, zeros);
    s_cb_vsc = make_buffer(NV2A_VP_CONSTANTS * 16, D3D11_BIND_CONSTANT_BUFFER,
                           D3D11_USAGE_DYNAMIC, NULL);
    s_cb_vsd = make_buffer(sizeof(Nv2aHlslVsParams), D3D11_BIND_CONSTANT_BUFFER,
                           D3D11_USAGE_DYNAMIC, NULL);
    s_cb_psc = make_buffer(sizeof(Nv2aHlslPsParams), D3D11_BIND_CONSTANT_BUFFER,
                           D3D11_USAGE_DYNAMIC, NULL);
    if (!s_vb || !s_ib || !s_zero_vb || !s_cb_vsc || !s_cb_vsd || !s_cb_psc) {
        fprintf(stderr, "  [D3D11] buffer creation failed\n");
        return 0;
    }
    len = nv2a_hlsl_vs_passthrough(s_src, sizeof s_src);
    s_sig_blob = compile(s_src, len, "vs_5_0", 0);
    if (!s_sig_blob || FAILED(ID3D11Device_CreateVertexShader(s_dev,
            ID3D10Blob_GetBufferPointer(s_sig_blob),
            ID3D10Blob_GetBufferSize(s_sig_blob), NULL, &s_vs_pass))) {
        fprintf(stderr, "  [D3D11] pass-through vertex shader failed\n");
        return 0;
    }
    {
        IDXGIDevice *dxgi = NULL;
        IDXGIAdapter *ad = NULL;
        DXGI_ADAPTER_DESC desc;
        char name[128] = "?";
        if (SUCCEEDED(ID3D11Device_QueryInterface(s_dev, &IID_IDXGIDevice, (void **)&dxgi))) {
            if (SUCCEEDED(IDXGIDevice_GetAdapter(dxgi, &ad))) {
                if (SUCCEEDED(IDXGIAdapter_GetDesc(ad, &desc)))
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name,
                                        sizeof name, NULL, NULL);
                IDXGIAdapter_Release(ad);
            }
            IDXGIDevice_Release(dxgi);
        }
        fprintf(stderr, "  [D3D11] GPU backend up on %s (feature level %X)\n",
                name, (unsigned)ID3D11Device_GetFeatureLevel(s_dev));
    }
    return 1;
}

int nv2a_d3d11_active(void)
{
    if (s_state < 0) {
        const char *e = getenv("RECOMP_GPU");
        s_state = 0;
        if (e && _stricmp(e, "d3d11") == 0)
            s_state = init();
        fflush(stderr);
    }
    return s_state > 0;
}

/* ── Draw ────────────────────────────────────────────────────────────── */

static ID3D11InputLayout *input_layout(const D3D11_INPUT_ELEMENT_DESC *el)
{
    uint64_t key = 5;
    void **slot;
    int found, i;
    ID3D11InputLayout *il = NULL;

    for (i = 0; i < 16; i++) {
        uint32_t k[3] = { (uint32_t)el[i].Format, el[i].InputSlot, el[i].AlignedByteOffset };
        key = hash64(k, sizeof k, key);
    }
    slot = cache_find(s_il_cache, STATE_CACHE, key, &found);
    if (slot && found)
        return (ID3D11InputLayout *)*slot;
    if (SUCCEEDED(ID3D11Device_CreateInputLayout(s_dev, el, 16,
            ID3D10Blob_GetBufferPointer(s_sig_blob),
            ID3D10Blob_GetBufferSize(s_sig_blob), &il)) && slot)
        *slot = il;
    return il;
}

/* Bind colour and depth targets for `d`. Returns the surface size in *w, *h
 * and the colour entry (NULL for a depth-only pass); 0 if there is nothing
 * to draw into. */
static int bind_targets(const Nv2aHwDraw *d, uint32_t *w, uint32_t *h, RtEntry **out_rt)
{
    RtEntry *rt = NULL;
    DsEntry *ds = NULL;
    ID3D11RenderTargetView *rtv = NULL;
    int want_color = d->color_va && (d->color_mask & 0x01010101u);

    if (want_color) {
        surface_dims(d, d->pitch, color_bpp(d->format), w, h);
        rt = rt_get(d->color_va, *w, *h);
        if (!rt)
            return 0;
        *w = rt->w;
        *h = rt->h;
        rtv = rt->rtv;
    } else {
        surface_dims(d, d->zpitch, 4, w, h);
    }
    if (d->zeta_va && !*w)
        return 0;
    if (d->zeta_va)
        ds = ds_get(d->zeta_va, *w, *h);
    if (!rtv && !ds)
        return 0;
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, rtv ? 1 : 0, rtv ? &rtv : NULL,
                                           ds ? ds->dsv : NULL);
    *out_rt = rt;
    return 1;
}

void nv2a_d3d11_draw(const Nv2aHwDraw *d)
{
    const uint8_t *mem = d->inline_base ? d->inline_base : guest();
    D3D11_INPUT_ELEMENT_DESC el[16];
    Stream st[16];
    ID3D11Buffer *vbs[17];
    UINT strides[17], offs[17];
    Nv2aHlslVsParams vsd;
    Nv2aHlslPsParams psd;
    uint32_t nstreams = 0, i, a, lo = 0xFFFFFFFFu, hi = 0, nidx, w = 0, h = 0;
    uint32_t count;
    RtEntry *rt = NULL;
    ID3D11VertexShader *vs;
    ID3D11PixelShader *ps;
    D3D11_MAPPED_SUBRESOURCE m;
    uint32_t *ib;
    uint32_t ib_at;

    LARGE_INTEGER t_in, t_out;

    if (d->idx_count < 3)
        return;
    QueryPerformanceCounter(&t_in);
    nidx = tri_indices(d->prim, d->idx_count);
    if (!nidx) {
        if (d->prim >= 1 && d->prim <= 4)
            s_st.points_lines++;
        return;
    }
    for (i = 0; i < d->idx_count; i++) {
        if (d->idx[i] < lo) lo = d->idx[i];
        if (d->idx[i] > hi) hi = d->idx[i];
    }
    count = hi - lo + 1;
    if (count > (1u << 20)) {
        s_st.skipped++;
        return;
    }

    vs = d->use_vp ? vs_for(d->vp) : s_vs_pass;
    ps = ps_for(d->comb);
    if (!vs || !ps) {
        s_st.skipped++;
        return;
    }
    if (!bind_targets(d, &w, &h, &rt)) {
        s_st.skipped++;
        return;
    }

    /* Vertices: one stream per interleaved buffer, copied whole across the
     * index range the batch uses, then described as raw bits. */
    memset(&vsd, 0, sizeof vsd);
    for (a = 0; a < 16; a++) {
        const Nv2aHwAttr *at = &d->attr[a];
        uint32_t bytes = 0, mode = 0, s;
        DXGI_FORMAT f;

        if (d->cur_attr)
            memcpy(vsd.adef[a], d->cur_attr[a], sizeof vsd.adef[a]);
        else
            vsd.adef[a][3] = 1.0f;
        el[a].SemanticName = "ATTR";
        el[a].SemanticIndex = a;
        el[a].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
        el[a].InstanceDataStepRate = 0;
        el[a].Format = DXGI_FORMAT_R32G32B32A32_UINT;
        el[a].InputSlot = ZERO_SLOT;
        el[a].AlignedByteOffset = 0;

        if (!at->size || !at->stride || (!d->inline_base && !at->offset))
            continue;
        f = attr_format(at, &bytes, &mode);
        if (f == DXGI_FORMAT_UNKNOWN)
            continue;
        if ((at->stride % 4) || (at->offset % 4)) {
            s_st.fallback_attrs++;
            continue;
        }
        for (s = 0; s < nstreams; s++)
            if (st[s].stride == at->stride && at->offset >= st[s].base
                    && at->offset + bytes <= st[s].base + at->stride)
                break;
        if (s == nstreams) {
            if (nstreams == 16)
                continue;
            st[s].base = at->offset;
            st[s].stride = at->stride;
            st[s].end = 0;
            nstreams++;
        }
        if (at->offset - st[s].base + bytes > st[s].end)
            st[s].end = at->offset - st[s].base + bytes;
        el[a].Format = f;
        el[a].InputSlot = s;
        el[a].AlignedByteOffset = at->offset - st[s].base;
        vsd.afmt[a][0] = mode;
        vsd.afmt[a][1] = at->size;
    }
    for (i = 0; i < nstreams; i++) {
        uint32_t span = (count - 1) * st[i].stride + st[i].end;
        uint32_t src = st[i].base + lo * st[i].stride;
        uint8_t *dst;

        if (d->inline_base && src + span > d->inline_bytes) {
            if (src >= d->inline_bytes)
                span = 0;
            else
                span = d->inline_bytes - src;
        }
        dst = (uint8_t *)ring_map(s_vb, &s_vb_pos, VB_RING_BYTES, span + 16, 16);
        if (!dst) {
            s_st.skipped++;
            return;
        }
        memcpy(dst, mem + src, span);
        memset(dst + span, 0, 16);
        s_st.vb_bytes += span;
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_vb, 0);
        vbs[i] = s_vb;
        strides[i] = st[i].stride;
        offs[i] = (UINT)(s_vb_pos - span - 16);
    }
    for (i = nstreams; i < ZERO_SLOT; i++) {
        vbs[i] = NULL;
        strides[i] = offs[i] = 0;
    }
    vbs[ZERO_SLOT] = s_zero_vb;
    strides[ZERO_SLOT] = 0;
    offs[ZERO_SLOT] = 0;

    ib = (uint32_t *)ring_map(s_ib, &s_ib_pos, IB_RING_BYTES, nidx * 4, 4);
    if (!ib) {
        s_st.skipped++;
        return;
    }
    ib_at = s_ib_pos - nidx * 4;
    emit_tris(d->prim, d->idx, d->idx_count, lo, ib);
    s_st.ib_bytes += nidx * 4;
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_ib, 0);

    /* Vertex-shader parameters. */
    vsd.vp[0] = 2.0f / (float)w;
    vsd.vp[1] = 2.0f / (float)h;
    vsd.vp[2] = 1.0f / (((d->format >> 4) & 0xFu) == 1u ? 65535.0f : 16777215.0f);
    if (d->fog_enable) {
        vsd.fog[0] = d->fog_param[0];
        vsd.fog[1] = d->fog_param[1];
        switch (d->fog_mode) {
        case 0x2601: vsd.fog[2] = 1.0f; break;                    /* LINEAR */
        case 0x0804: vsd.fog[2] = 1.0f; vsd.fog[3] = 1.0f; break; /* LINEAR_ABS */
        case 0x0800: vsd.fog[2] = 2.0f; break;                    /* EXP */
        case 0x0802: vsd.fog[2] = 2.0f; vsd.fog[3] = 1.0f; break; /* EXP_ABS */
        case 0x0801: vsd.fog[2] = 3.0f; break;                    /* EXP2 */
        case 0x0803: vsd.fog[2] = 3.0f; vsd.fog[3] = 1.0f; break; /* EXP2_ABS */
        default: break;
        }
    }
    vsd.roles[0] = d->diffuse_attr;
    vsd.roles[1] = d->tex0_attr;
    if (!d->use_vp && !vsd.afmt[d->diffuse_attr & 15][0]
            && !(d->cur_attr_set & (1u << (d->diffuse_attr & 15)))) {
        /* Pre-transformed with no colour stream: white, as the software
         * path draws it. */
        vsd.adef[d->diffuse_attr & 15][0] = vsd.adef[d->diffuse_attr & 15][1] =
            vsd.adef[d->diffuse_attr & 15][2] = 1.0f;
    }
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_vsd, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        memcpy(m.pData, &vsd, sizeof vsd);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_vsd, 0);
    }
    if (d->use_vp && (d->vp != s_vsc_owner || d->vp->constants_gen != s_vsc_gen)) {
        if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_vsc, 0,
                                              D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, d->vp->constants, sizeof d->vp->constants);
            ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_vsc, 0);
            s_vsc_owner = d->vp;
            s_vsc_gen = d->vp->constants_gen;
        }
    }

    /* Pixel-shader parameters and the four texture stages. */
    memset(&psd, 0, sizeof psd);
    for (i = 0; i < 8; i++) {
        unpack_argb(d->comb->factor0[i], psd.cf0[i]);
        unpack_argb(d->comb->factor1[i], psd.cf1[i]);
    }
    unpack_argb(d->comb->final_factor0, psd.ff0);
    unpack_argb(d->comb->final_factor1, psd.ff1);
    /* SET_FOG_COLOR is A B G R from the top byte down -- red is the low
     * byte, unlike every other colour register here. */
    unpack_argb(d->comb->fog_color, psd.fogc);
    {
        float r = psd.fogc[2];
        psd.fogc[2] = psd.fogc[0];
        psd.fogc[0] = r;
    }
    psd.atest[0] = d->alpha_test;
    psd.atest[1] = d->alpha_func;
    psd.atest[2] = d->alpha_ref;
    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("RECOMP_GPU_PS_DEBUG");
            dbg = e ? atoi(e) : 0;
        }
        psd.atest[3] = (uint32_t)dbg;
    }
    {
        ID3D11ShaderResourceView *srv[4] = { NULL, NULL, NULL, NULL };
        ID3D11SamplerState *smp[4] = { NULL, NULL, NULL, NULL };
        for (i = 0; i < 4; i++) {
            const Nv2aHwTexture *t = &d->tex[i];
            uint32_t tw = 0, th = 0, lv = 1;
            int norm = d3d8_format_is_swizzled(t->color) || dxt_block(t->color);
            RtEntry *src_rt;

            if (!t->enabled)
                continue;
            /* A surface drawn this frame or the last, at this address: the
             * title is sampling what it rendered. */
            src_rt = rt_find(t->va);
            if (src_rt && src_rt != rt && src_rt->drawn_frame + 2 >= s_frame) {
                srv[i] = src_rt->srv;
                tw = src_rt->w;
                th = src_rt->h;
                /* Normalised coordinates cover the texture's own size, which
                 * may be less than the surface allocated for it. */
                psd.tsc[i][0] = norm ? (float)t->width / (float)tw : 1.0f / (float)tw;
                psd.tsc[i][1] = norm ? (float)t->height / (float)th : 1.0f / (float)th;
            } else {
                srv[i] = tex_srv(t, &tw, &th, &lv);
                if (!srv[i])
                    continue;
                psd.tsc[i][0] = norm ? 1.0f : 1.0f / (float)tw;
                psd.tsc[i][1] = norm ? 1.0f : 1.0f / (float)th;
            }
            /* X8 formats carry no alpha: read it as opaque. */
            psd.tsc[i][2] = (t->color == 0x07 || t->color == 0x1E || t->color == 0x03
                             || t->color == 0x1C || t->color == 0x05 || t->color == 0x11)
                            ? 1.0f : 0.0f;
            psd.tsc[i][3] = 1.0f;
            smp[i] = sampler(t, lv);
        }
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 4, srv);
        ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 4, smp);
    }
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_psc, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        memcpy(m.pData, &psd, sizeof psd);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_psc, 0);
    }

    /* Fixed-function state. */
    {
        D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
        D3D11_RECT sc;
        float bf[4];
        sc.left = (LONG)d->clip_x;
        sc.top = (LONG)d->clip_y;
        sc.right = (LONG)(d->clip_x + d->clip_w);
        sc.bottom = (LONG)(d->clip_y + d->clip_h);
        ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
        ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
        ID3D11DeviceContext_RSSetState(s_ctx, raster_state(d));
        unpack_argb(d->blend_color, bf);
        if (d->blend_sfactor == 0x8003 || d->blend_sfactor == 0x8004
                || d->blend_dfactor == 0x8003 || d->blend_dfactor == 0x8004)
            bf[0] = bf[1] = bf[2] = bf[3];
        ID3D11DeviceContext_OMSetBlendState(s_ctx, blend_state(d), bf, 0xFFFFFFFFu);
        ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, depth_state(d),
                                                   d->stencil_ref & 0xFFu);
    }

    ID3D11DeviceContext_IASetInputLayout(s_ctx, input_layout(el));
    ID3D11DeviceContext_IASetVertexBuffers(s_ctx, 0, 17, vbs, strides, offs);
    ID3D11DeviceContext_IASetIndexBuffer(s_ctx, s_ib, DXGI_FORMAT_R32_UINT, 0);
    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(s_ctx, vs, NULL, 0);
    {
        ID3D11Buffer *vcb[2] = { s_cb_vsc, s_cb_vsd };
        ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 0, 2, vcb);
    }
    ID3D11DeviceContext_PSSetShader(s_ctx, ps, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb_psc);

    ID3D11DeviceContext_DrawIndexed(s_ctx, nidx, ib_at / 4, 0);
    s_st.draws++;
    s_st.tris += nidx / 3;
    if (rt)
        rt->drawn_frame = s_frame;
    QueryPerformanceCounter(&t_out);
    s_st.draw_ticks += (uint64_t)(t_out.QuadPart - t_in.QuadPart);
}

void nv2a_d3d11_clear(const Nv2aHwDraw *d, uint32_t flags, uint32_t color,
                      uint32_t zclear)
{
    uint32_t w = 0, h = 0;

    /* A clear writes whatever the flags name, whatever the colour mask. */
    if (flags & 0xF0u) {
        RtEntry *rt;
        if (!d->color_va)
            return;
        surface_dims(d, d->pitch, color_bpp(d->format), &w, &h);
        rt = rt_get(d->color_va, w, h);
        if (rt) {
            float c[4];
            D3D11_RECT r;
            unpack_argb(color, c);
            r.left = (LONG)d->clip_x;
            r.top = (LONG)d->clip_y;
            r.right = (LONG)(d->clip_x + d->clip_w);
            r.bottom = (LONG)(d->clip_y + d->clip_h);
            if (s_ctx1)
                ID3D11DeviceContext1_ClearView(s_ctx1, (ID3D11View *)rt->rtv, c, &r, 1);
            else
                ID3D11DeviceContext_ClearRenderTargetView(s_ctx, rt->rtv, c);
            rt->drawn_frame = s_frame;
            w = rt->w;
            h = rt->h;
        }
    }
    if ((flags & 0x03u) && d->zeta_va) {
        DsEntry *ds;
        UINT which = 0;
        if (!w)
            surface_dims(d, d->color_va ? d->pitch : d->zpitch,
                         d->color_va ? color_bpp(d->format) : 4, &w, &h);
        if (d->color_va) {
            RtEntry *rt = rt_find(d->color_va);
            if (rt) { w = rt->w; h = rt->h; }
        }
        ds = ds_get(d->zeta_va, w, h);
        if (flags & 0x01u) which |= D3D11_CLEAR_DEPTH;
        if (flags & 0x02u) which |= D3D11_CLEAR_STENCIL;
        if (ds) {
            float z = ((d->format >> 4) & 0xFu) == 1u
                    ? (float)(zclear & 0xFFFFu) / 65535.0f
                    : (float)(zclear >> 8) / 16777215.0f;
            ID3D11DeviceContext_ClearDepthStencilView(s_ctx, ds->dsv, which, z,
                                                      (UINT8)(zclear & 0xFFu));
        }
    }
}

void nv2a_d3d11_flip(uint32_t color_va)
{
    RtEntry *rt = rt_find(color_va);
    uint32_t i = s_stage_next % STAGES;

    s_frame++;
    if (!rt)
        return;
    /* The oldest copy first, then reuse its staging texture for this frame.
     * Three in flight means the one being read was copied two flips ago. */
    readback_one((int)i);
    if (!s_stage[i] || s_stage_w[i] != rt->w || s_stage_h[i] != rt->h) {
        D3D11_TEXTURE2D_DESC td;
        if (s_stage[i])
            ID3D11Texture2D_Release(s_stage[i]);
        s_stage[i] = NULL;
        memset(&td, 0, sizeof td);
        td.Width = rt->w;
        td.Height = rt->h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &s_stage[i])))
            return;
        s_stage_w[i] = rt->w;
        s_stage_h[i] = rt->h;
    }
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)s_stage[i],
                                     (ID3D11Resource *)rt->tex);
    s_stage_full[i] = 1;
    s_stage_next++;
    ID3D11DeviceContext_Flush(s_ctx);
}

void nv2a_d3d11_report(void)
{
    static uint32_t last_frame, last_draws;
    static uint64_t last_vb, last_ticks, last_hash;
    static long long last_scan;
    extern volatile long long g_nv2a_scan_ticks;
    uint32_t frames;

    if (s_state <= 0)
        return;
    if (!s_qpf.QuadPart)
        QueryPerformanceFrequency(&s_qpf);
    frames = s_frame - last_frame;
    if (frames) {
        fprintf(stderr, "  [D3D11] per frame over the last %u: %u draws, %.0f KB vertices,"
                " %.0f KB texture hashing, %.2f ms in draw\n", frames,
                (s_st.draws - last_draws) / frames,
                (double)(s_st.vb_bytes - last_vb) / frames / 1024.0,
                (double)(s_st.tex_hash_bytes - last_hash) / frames / 1024.0,
                (double)(s_st.draw_ticks - last_ticks) * 1000.0 / (double)s_qpf.QuadPart / frames);
        fprintf(stderr, "  [D3D11]   executor busy %.2f ms a frame (walk + execute, flip waits included)\n",
                (double)(g_nv2a_scan_ticks - last_scan) * 1000.0 / (double)s_qpf.QuadPart / frames);
    }
    last_frame = s_frame; last_draws = s_st.draws; last_vb = s_st.vb_bytes;
    last_ticks = s_st.draw_ticks; last_hash = s_st.tex_hash_bytes;
    last_scan = g_nv2a_scan_ticks;
    fprintf(stderr, "  [D3D11] %u draws, %u triangles, %u skipped, %u points/lines;"
            " shaders %u VS + %u PS (%u failed); %u texture uploads (%u bad);"
            " %u colour + %u depth targets; %u frames read back;"
            " %u programs writing constants, %u unaligned attributes\n",
            s_st.draws, s_st.tris, s_st.skipped, s_st.points_lines,
            s_st.vs_compiled, s_st.ps_compiled, s_st.compile_failed,
            s_st.tex_uploads, s_st.tex_bad, s_st.rt_created, s_st.ds_created,
            s_st.readbacks, s_st.vs_const_writes, s_st.fallback_attrs);
    fflush(stderr);
}

#else  /* !_WIN32 */

int  nv2a_d3d11_active(void) { return 0; }
void nv2a_d3d11_draw(const Nv2aHwDraw *d) { (void)d; }
void nv2a_d3d11_clear(const Nv2aHwDraw *d, uint32_t f, uint32_t c, uint32_t z)
{ (void)d; (void)f; (void)c; (void)z; }
void nv2a_d3d11_flip(uint32_t va) { (void)va; }
void nv2a_d3d11_report(void) {}

#endif
