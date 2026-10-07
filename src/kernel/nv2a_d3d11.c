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
#include "xbox_memory_layout.h"   /* xbox_PinHostThread */
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
    uint32_t tex_uploads, tex_bad, rt_created, ds_created, readbacks, cpu_frames;
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

/* Is [va, va+bytes) guest memory that exists: low RAM, the contiguous window
 * or its tiled alias? Addresses come from the title's pushbuffer, and one
 * misread command makes a vertex array or texture point anywhere -- reading
 * 0xEE0B0C10 took the whole process down. */
static int guest_ok(uint32_t va, uint32_t bytes)
{
    uint64_t end = (uint64_t)va + bytes;
    if (end <= 0x04000000u)
        return 1;
    if (va >= 0x80000000u && end <= 0x84000000ull)          /* contiguous window */
        return 1;
    return va >= 0xF0000000u && end <= 0xF4000000ull;
}

/* ── Shaders ─────────────────────────────────────────────────────────── */

/* Compiled shaders persist in a directory (RECOMP_SHADER_CACHE, default
 * "shader_cache"; "0" turns it off), one file per HLSL source, target and
 * flags. D3DCompile takes 100-400 ms for the larger generated programs, on
 * the render thread, so each new one used to be a visible hitch -- in Amped
 * 2's ML1 a few every 10 s for minutes, since programs keep turning up as
 * the run goes on. With the cache only the first run pays. */
static const char *shader_cache_dir(void)
{
    static int init;
    static char dir[260];
    if (!init) {
        const char *e = getenv("RECOMP_SHADER_CACHE");
        init = 1;
        if (e && strcmp(e, "0") == 0)
            dir[0] = 0;
        else {
            snprintf(dir, sizeof dir, "%s", e && *e ? e : "shader_cache");
            CreateDirectoryA(dir, NULL);
        }
    }
    return dir[0] ? dir : NULL;
}

static ID3DBlob *compile_uncached(const char *src, size_t len, const char *target,
                                  UINT flags);

/* The cached bytecode for this source, or NULL. */
static ID3DBlob *compile_cached_only(const char *src, size_t len, const char *target,
                                     UINT flags);

static ID3DBlob *compile(const char *src, size_t len, const char *target,
                         UINT flags)
{
    const char *dir = shader_cache_dir();
    char path[320];
    ID3DBlob *code = NULL;
    FILE *f;

    if (!dir)
        return compile_uncached(src, len, target, flags);
    if ((code = compile_cached_only(src, len, target, flags)) != NULL)
        return code;
    {
        uint64_t h = hash64(src, len, 0xCBF29CE484222325ull);
        h = hash64(target, strlen(target), h);
        h = hash64(&flags, sizeof flags, h);
        snprintf(path, sizeof path, "%s\\%s_%016llx.cso", dir, target,
                 (unsigned long long)h);
    }
    code = compile_uncached(src, len, target, flags);
    if (getenv("RECOMP_SHADER_CACHE_SRC")) {        /* the HLSL beside it */
        char hp[330];
        snprintf(hp, sizeof hp, "%s.hlsl", path);
        if ((f = fopen(hp, "wb")) != NULL) {
            fwrite(src, 1, len, f);
            fclose(f);
        }
    }
    if (code && (f = fopen(path, "wb")) != NULL) {
        fwrite(ID3D10Blob_GetBufferPointer(code), 1, ID3D10Blob_GetBufferSize(code), f);
        fclose(f);
    }
    return code;
}

static ID3DBlob *compile_cached_only(const char *src, size_t len, const char *target,
                                     UINT flags)
{
    const char *dir = shader_cache_dir();
    char path[320];
    ID3DBlob *code = NULL;
    FILE *f;
    uint64_t h;
    long n;

    if (!dir)
        return NULL;
    h = hash64(src, len, 0xCBF29CE484222325ull);
    h = hash64(target, strlen(target), h);
    h = hash64(&flags, sizeof flags, h);
    snprintf(path, sizeof path, "%s\\%s_%016llx.cso", dir, target,
             (unsigned long long)h);
    if (!(f = fopen(path, "rb")))
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 0 && SUCCEEDED(D3DCreateBlob((SIZE_T)n, &code))
            && fread(ID3D10Blob_GetBufferPointer(code), 1, (size_t)n, f) == (size_t)n) {
        fclose(f);
        return code;
    }
    if (code)
        ID3D10Blob_Release(code);
    fclose(f);
    return NULL;
}

static ID3DBlob *compile_uncached(const char *src, size_t len, const char *target,
                                  UINT flags)
{
    ID3DBlob *code = NULL, *err = NULL;
    LARGE_INTEGER ta, tb, tf;
    HRESULT hr;
    QueryPerformanceCounter(&ta);
    hr = D3DCompile(src, len, NULL, NULL, NULL, "main", target,
                    flags, 0, &code, &err);
    QueryPerformanceCounter(&tb);
    QueryPerformanceFrequency(&tf);
    if (getenv("RECOMP_PACE_TRACE") && (tb.QuadPart - ta.QuadPart) * 50 > tf.QuadPart)
        fprintf(stderr, "  [D3D11] D3DCompile %s took %.0f ms\n", target,
                (double)(tb.QuadPart - ta.QuadPart) * 1000.0 / tf.QuadPart);

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

/* Asynchronous shader creation (RECOMP_GPU_ASYNC_SHADERS=0 turns it off).
 *
 * A program never seen before -- not even in the disk cache -- compiles on a
 * worker thread, and the draws that need it are skipped until it is ready,
 * usually a frame or two. D3DCompile takes 100-300 ms for the larger
 * generated vertex programs, and Amped 2 meets new ones for minutes into a
 * level, so compiling on the render thread was a visible hitch each time.
 * The device is free-threaded for creation. The slot the result lands in is
 * never moved or reused (the caches only ever add), so the worker publishes
 * into it with one interlocked store; SHADER_PENDING marks it in flight. */
#define SHADER_FAILED  ((void *)(uintptr_t)1)
#define SHADER_PENDING ((void *)(uintptr_t)2)

typedef struct ShaderJob {
    struct ShaderJob *next;
    void **slot;
    int pixel;
    size_t len;
    char src[1];
} ShaderJob;

static CRITICAL_SECTION s_job_lock;
static HANDLE s_job_event;
static ShaderJob *s_job_head, *s_job_tail;
static int s_async = -1;

static DWORD WINAPI shader_worker(LPVOID unused)
{
    (void)unused;
    for (;;) {
        ShaderJob *j;

        WaitForSingleObject(s_job_event, INFINITE);
        for (;;) {
            ID3DBlob *code;
            void *obj = NULL;

            EnterCriticalSection(&s_job_lock);
            j = s_job_head;
            if (j && !(s_job_head = j->next))
                s_job_tail = NULL;
            LeaveCriticalSection(&s_job_lock);
            if (!j)
                break;
            code = compile(j->src, j->len, j->pixel ? "ps_5_0" : "vs_5_0",
                           j->pixel ? 0 : D3DCOMPILE_IEEE_STRICTNESS);
            if (code) {
                HRESULT hr = j->pixel
                    ? ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(code),
                          ID3D10Blob_GetBufferSize(code), NULL, (ID3D11PixelShader **)&obj)
                    : ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(code),
                          ID3D10Blob_GetBufferSize(code), NULL, (ID3D11VertexShader **)&obj);
                if (FAILED(hr))
                    obj = NULL;
                else if (j->pixel)
                    s_st.ps_compiled++;
                else
                    s_st.vs_compiled++;
                ID3D10Blob_Release(code);
            }
            InterlockedExchangePointer(j->slot, obj ? obj : SHADER_FAILED);
            free(j);
        }
    }
    return 0;
}

/* Queue src for the worker. 0 when asynchronous creation is off or the worker
 * could not start: the caller then compiles in place. */
static int shader_queue(void **slot, int pixel, const char *src, size_t len)
{
    ShaderJob *j;

    if (s_async < 0) {
        const char *e = getenv("RECOMP_GPU_ASYNC_SHADERS");
        s_async = !(e && *e == '0');
        if (s_async) {
            HANDLE th;
            InitializeCriticalSection(&s_job_lock);
            s_job_event = CreateEventA(NULL, FALSE, FALSE, NULL);
            th = s_job_event ? CreateThread(NULL, 0, shader_worker, NULL, 0, NULL) : NULL;
            if (th) {
                SetThreadPriority(th, THREAD_PRIORITY_BELOW_NORMAL);
                CloseHandle(th);
            } else {
                s_async = 0;
            }
        }
    }
    if (!s_async || !slot || !(j = (ShaderJob *)malloc(sizeof *j + len)))
        return 0;
    j->next = NULL;
    j->slot = slot;
    j->pixel = pixel;
    j->len = len;
    memcpy(j->src, src, len);
    *slot = SHADER_PENDING;
    EnterCriticalSection(&s_job_lock);
    if (s_job_tail)
        s_job_tail->next = j;
    else
        s_job_head = j;
    s_job_tail = j;
    LeaveCriticalSection(&s_job_lock);
    SetEvent(s_job_event);
    return 1;
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
        code = len ? compile_cached_only(s_src, len, "vs_5_0", D3DCOMPILE_IEEE_STRICTNESS) : NULL;
        if (!code && len && shader_queue(slot, 0, s_src, len)) {
            last_vp = NULL;                 /* ask again next draw */
            return NULL;
        }
        if (!code && len)
            code = compile(s_src, len, "vs_5_0", D3DCOMPILE_IEEE_STRICTNESS);
        if (code) {
            LARGE_INTEGER a, b, f;
            HRESULT hr;
            QueryPerformanceCounter(&a);
            hr = ID3D11Device_CreateVertexShader(s_dev,
                    ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code),
                    NULL, &vs);
            QueryPerformanceCounter(&b);
            QueryPerformanceFrequency(&f);
            if (getenv("RECOMP_PACE_TRACE") && (b.QuadPart - a.QuadPart) * 50 > f.QuadPart)
                fprintf(stderr, "  [D3D11] CreateVertexShader took %.0f ms\n",
                        (double)(b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart);
            if (SUCCEEDED(hr)) {
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
    if ((void *)vs == SHADER_PENDING) {
        last_vp = NULL;
        return NULL;
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
        ID3DBlob *code = len ? compile_cached_only(s_src, len, "ps_5_0", 0) : NULL;
        if (!code && len && shader_queue(slot, 1, s_src, len)) {
            last_c = NULL;
            return NULL;
        }
        if (!code && len)
            code = compile(s_src, len, "ps_5_0", 0);
        if (code) {
            LARGE_INTEGER a, b, f;
            HRESULT hr;
            QueryPerformanceCounter(&a);
            hr = ID3D11Device_CreatePixelShader(s_dev,
                    ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code),
                    NULL, &ps);
            QueryPerformanceCounter(&b);
            QueryPerformanceFrequency(&f);
            if (getenv("RECOMP_PACE_TRACE") && (b.QuadPart - a.QuadPart) * 50 > f.QuadPart)
                fprintf(stderr, "  [D3D11] CreatePixelShader took %.0f ms\n",
                        (double)(b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart);
            if (SUCCEEDED(hr)) {
                s_st.ps_compiled++;
                if (slot)
                    *slot = ps;
            }
            ID3D10Blob_Release(code);
        }
        if (!ps && slot)
            *slot = (void *)(uintptr_t)1;
    }
    if ((void *)ps == SHADER_PENDING) {
        last_c = NULL;
        return NULL;
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
    /* RECOMP_GPU_ANISO=<2-16>: anisotropic filtering, for textures the title
     * already filters linearly across mip levels -- ground and walls seen
     * at a grazing angle stay sharp. Point-sampled and unmipped textures
     * (UI, fonts, lookup tables) keep their own filter. */
    {
        static int aniso = -1;
        if (aniso < 0) {
            const char *e = getenv("RECOMP_GPU_ANISO");
            aniso = e ? atoi(e) : 0;
            if (aniso > 16) aniso = 16;
        }
        if (aniso > 1 && mips && lmin) {
            sd.Filter = D3D11_FILTER_ANISOTROPIC;
            sd.MaxAnisotropy = (UINT)aniso;
        }
    }
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
    /* A sampled signature of the guest memory under the surface when it was
     * last drawn. Rendering never writes guest memory here, so while that
     * memory is unchanged nothing else has claimed the address, and the
     * surface is still what a texture there means -- however many frames
     * ago it was drawn. */
    uint64_t mem_sig;
    uint32_t sig_frame;
} RtEntry;

static uint64_t rt_mem_sig(const RtEntry *e)
{
    const uint8_t *g = (const uint8_t *)xbox_GetMemoryOffset();
    uint64_t h = 0xCBF29CE484222325ull;
    uint32_t bytes = e->w * e->h * 4u, step, off;
    if (!guest_ok(e->va, bytes))
        return 0;
    step = bytes / 256u;
    if (step < 4)
        step = 4;
    step &= ~3u;
    for (off = 0; off < bytes; off += step) {
        h ^= *(const uint32_t *)(g + e->va + off);
        h *= 0x100000001B3ull;
    }
    return h;
}

/* Is the surface at src still what the title samples there? Drawn in the
 * last two frames, or the memory under it unchanged since it was. */
static int rt_still_current(RtEntry *src, uint32_t frame)
{
    if (src->drawn_frame + 2 >= frame)
        return 1;
    return src->sig_frame && rt_mem_sig(src) == src->mem_sig;
}

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

/* RECOMP_GPU_SCALE=<n>: every colour and depth surface is n times the
 * guest's size in each direction (2 = 1280x960 for a 640x480 title, 3 =
 * 1920x1440, 4 = 2560x1920), with viewports, scissors and clear rectangles
 * scaled to match. Surfaces keep their guest size in RtEntry/DsEntry --
 * texture coordinates and the presenter's aspect work in it -- and only
 * the D3D11 textures are larger. The guest never reads the pixels back
 * (it cannot: they stay on the GPU), so it cannot tell. 1 (default) to 8. */
static int gpu_scale(void)
{
    static int sc = 0;
    if (!sc) {
        const char *e = getenv("RECOMP_GPU_SCALE");
        sc = e ? atoi(e) : 1;
        if (sc < 1) sc = 1;
        if (sc > 8) sc = 8;
    }
    return sc;
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
    td.Width = n.w * (UINT)gpu_scale();
    td.Height = n.h * (UINT)gpu_scale();
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
        D3D11_BOX box = { 0, 0, 0, e->w * (UINT)gpu_scale(), e->h * (UINT)gpu_scale(), 1 };
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
    td.Width = w * (UINT)gpu_scale();
    td.Height = h * (UINT)gpu_scale();
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
    int cube;               /* six faces, each its own mip chain */
    ID3D11ShaderResourceView *rep;  /* texture pack replacement, not owned */
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
    uint32_t lv, w = e->w, h = e->h, off = 0, face, faces = e->cube ? 6 : 1;
    uint32_t face_stride = 0;
    static uint32_t *scratch;
    static size_t scratch_n;

    /* A cube map is six faces in a row, each a full mip chain, each face
     * starting on a 128-byte boundary. */
    for (lv = 0; lv < e->levels; lv++) {
        face_stride += level_bytes(t->color, w, h, t->pitch, linear);
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
    }
    face_stride = (face_stride + 127u) & ~127u;
    w = e->w;
    h = e->h;

    if (create) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof td);
        td.Width = e->w;
        td.Height = e->h;
        td.MipLevels = e->levels;
        td.ArraySize = faces;
        if (e->cube)
            td.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
        td.Format = bc ? (bb == 8 ? DXGI_FORMAT_BC1_UNORM
                                  : (t->color == 0x0E ? DXGI_FORMAT_BC2_UNORM
                                                      : DXGI_FORMAT_BC3_UNORM))
                       : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &e->tex)))
            return 0;
        if (e->cube) {
            D3D11_SHADER_RESOURCE_VIEW_DESC vd;
            memset(&vd, 0, sizeof vd);
            vd.Format = td.Format;
            vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
            vd.TextureCube.MipLevels = e->levels;
            ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)e->tex,
                                                  &vd, &e->srv);
        } else {
            ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)e->tex,
                                                  NULL, &e->srv);
        }
    }
    for (face = 0; face < faces; face++) {
    w = e->w;
    h = e->h;
    off = face * face_stride;
    for (lv = 0; lv < e->levels; lv++) {
        uint32_t lb = level_bytes(t->color, w, h, t->pitch, linear);
        uint32_t sub = lv + face * e->levels;
        if (bc) {
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)e->tex,
                    sub, NULL, src + off, ((w + 3) / 4) * bb, 0);
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
                    sub, NULL, scratch, w * 4, 0);
        }
        off += lb;
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
    }
    }
    s_st.tex_uploads++;
    return 1;
}

/* ── Volume textures ─────────────────────────────────────────────────────
 *
 * SET_TEXTURE_FORMAT bits 4-7 are the dimensionality and 28-31 log2 of the
 * depth. A 3D texture was uploaded as its first 2D slice. Amped 2's night
 * level (BR1) lights its trees and some slopes through a 32x64x32 X8R8G8B8
 * volume (stage 1, coordinates from the vertex's packed attribute), which
 * then read only the slice at w = 0. (The level's overall darkness was the
 * shadow map instead -- see tex_depth_max.) Level 0 only, 32-bit swizzled formats
 * only -- what the title uses; anything else keeps the old 2D path. A small
 * cache of its own, hashed sparsely once a frame like the 2D one. */
typedef struct {
    uint32_t va, format;
    uint64_t shash;
    uint32_t hashed_frame, used_frame;
    ID3D11Texture3D *tex;
    ID3D11ShaderResourceView *srv;
} VolEntry;
#define VOL_SLOTS 16
static VolEntry s_vol[VOL_SLOTS];

static int tex_is_volume(const Nv2aHwTexture *t)
{
    return ((t->format >> 4) & 0xFu) == 3u && !(t->format & 4u)
        && d3d8_format_is_swizzled(t->color) && !dxt_block(t->color)
        && nv2a_pb_texel_bytes(t->color) == 4;
}

static ID3D11ShaderResourceView *vol_srv(const Nv2aHwTexture *t)
{
    uint32_t w = t->width, h = t->height, dp = 1u << ((t->format >> 28) & 0xFu);
    uint32_t bytes = w * h * dp * 4u, i;
    VolEntry *e = NULL, *victim = NULL;
    const uint8_t *src;
    uint64_t sh;

    if (!w || !h || w > 512 || h > 512 || dp > 512 || !guest_ok(t->va, bytes)) {
        static int said;
        if (said++ < 16)
            fprintf(stderr, "[GPU] volume texture %08X refused: %ux%ux%u format %08X\n",
                    t->va, w, h, dp, t->format);
        return NULL;
    }
    src = guest() + t->va;
    for (i = 0; i < VOL_SLOTS; i++) {
        if (s_vol[i].tex && s_vol[i].va == t->va && s_vol[i].format == t->format) {
            e = &s_vol[i];
            break;
        }
        if (!victim || !s_vol[i].tex || (victim->tex && s_vol[i].used_frame < victim->used_frame))
            victim = &s_vol[i];
    }
    if (e && e->hashed_frame == s_frame) {
        e->used_frame = s_frame;
        return e->srv;
    }
    sh = hash_sparse(src, bytes);
    if (e && sh == e->shash) {
        e->hashed_frame = e->used_frame = s_frame;
        return e->srv;
    }
    if (!e) {
        D3D11_TEXTURE3D_DESC td;
        e = victim;
        if (e->srv) ID3D11ShaderResourceView_Release(e->srv);
        if (e->tex) ID3D11Texture3D_Release(e->tex);
        memset(e, 0, sizeof *e);
        memset(&td, 0, sizeof td);
        td.Width = w; td.Height = h; td.Depth = dp; td.MipLevels = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture3D(s_dev, &td, NULL, &e->tex)))
            return NULL;
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)e->tex, NULL, &e->srv);
        e->va = t->va;
        e->format = t->format;
        fprintf(stderr, "[GPU] volume texture %08X: %ux%ux%u fmt %02X\n", t->va, w, h, dp, t->color);
    }
    {
        static uint8_t *lin;
        static size_t lin_n;
        if (bytes > lin_n) {
            free(lin);
            lin = (uint8_t *)malloc(bytes);
            lin_n = lin ? bytes : 0;
            if (!lin)
                return NULL;
        }
        xbox_unswizzle_box(lin, src, w, h, dp, 4);
        if (getenv("RECOMP_VOL_DUMP")) {   /* diag/vol_<va>.bin, linear BGRA */
            char path[64];
            FILE *f;
            snprintf(path, sizeof path, "diag/vol_%08X.bin", t->va);
            if ((f = fopen(path, "wb")) != NULL) {
                fwrite(lin, 1, bytes, f);
                fclose(f);
            }
        }
        ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)e->tex, 0, NULL,
                                              lin, w * 4, w * h * 4);
        s_st.tex_uploads++;
    }
    e->shash = sh;
    e->hashed_frame = e->used_frame = s_frame;
    return e->srv;
}

/* ── Texture packs ───────────────────────────────────────────────────────
 *
 * RECOMP_TEXTURE_PACK=<dir>: replacement textures, one DDS per original,
 * named <crc32>_<W>x<H>_<fmt>.dds -- the CRC-32 (zlib's) of the top mip as
 * it sits in guest memory, its size, and the NV2A colour format, in hex.
 * That is the key a project's asset tools compute from the game files
 * (Amped 2: tools/assets, `tex-pack`), so a pack built offline finds its
 * textures here without any table. The DDS can be any size and format
 * D3D11 samples (BC7 from texconv, BC1-3, or 32-bit), with its own mips.
 *
 * The original is still uploaded as before; a replacement only changes the
 * view the draw samples. Looked up when a texture is created or its guest
 * memory changes, one file probe per key (hits and misses both cached).
 * Cube maps keep their originals. Several packs: separate the directories
 * with ';', highest priority first. RECOMP_TEXTURE_PACK_MISSES=<file>
 * appends the key of every texture the pack does not cover, once each,
 * to find what a pack still lacks (textures built at run time included).
 */
static uint32_t s_crc_table[256];

static uint32_t crc32_bytes(const uint8_t *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    size_t i;
    if (!s_crc_table[1]) {
        uint32_t k, j, v;
        for (k = 0; k < 256; k++) {
            for (v = k, j = 0; j < 8; j++)
                v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            s_crc_table[k] = v;
        }
    }
    for (i = 0; i < n; i++)
        c = s_crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* Load a DDS as a shader resource view, or NULL. */
static ID3D11ShaderResourceView *dds_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    uint8_t *buf = NULL;
    long n;
    uint32_t w, h, mips, flags, fourcc, bits, off = 128, block = 0, i;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    D3D11_TEXTURE2D_DESC td;
    D3D11_SUBRESOURCE_DATA sd[16];
    ID3D11Texture2D *tex = NULL;
    ID3D11ShaderResourceView *srv = NULL;
    const uint8_t *p;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 128 && (buf = (uint8_t *)malloc((size_t)n)) != NULL
            && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (!buf)
        return NULL;
    if (memcmp(buf, "DDS ", 4) != 0)
        goto done;
    memcpy(&h, buf + 12, 4);
    memcpy(&w, buf + 16, 4);
    memcpy(&mips, buf + 28, 4);
    memcpy(&flags, buf + 80, 4);
    memcpy(&fourcc, buf + 84, 4);
    memcpy(&bits, buf + 88, 4);
    if (!mips) mips = 1;
    if (mips > 16) mips = 16;
    if (fourcc == 0x30315844u) {                    /* "DX10" */
        uint32_t dxgi;
        if (n < 148) goto done;
        memcpy(&dxgi, buf + 128, 4);
        fmt = (DXGI_FORMAT)dxgi;
        off = 148;
    } else if (fourcc == 0x31545844u) {             /* "DXT1" */
        fmt = DXGI_FORMAT_BC1_UNORM;
    } else if (fourcc == 0x33545844u) {             /* "DXT3" */
        fmt = DXGI_FORMAT_BC2_UNORM;
    } else if (fourcc == 0x35545844u) {             /* "DXT5" */
        fmt = DXGI_FORMAT_BC3_UNORM;
    } else if ((flags & 0x40u) && bits == 32) {     /* RGB, 32 bits */
        uint32_t rmask;
        memcpy(&rmask, buf + 92, 4);
        fmt = rmask == 0x00FF0000u ? DXGI_FORMAT_B8G8R8A8_UNORM
                                   : DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    switch (fmt) {
    case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_UNORM:
        block = 8; break;
    case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        block = 16; break;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM:
        break;
    default:
        goto done;
    }
    p = buf + off;
    for (i = 0; i < mips; i++) {
        uint32_t lw = w >> i ? w >> i : 1, lh = h >> i ? h >> i : 1;
        uint32_t pitch = block ? ((lw + 3) / 4) * block : lw * 4;
        uint32_t size = block ? pitch * ((lh + 3) / 4) : pitch * lh;
        if (p + size > buf + n)
            break;
        sd[i].pSysMem = p;
        sd[i].SysMemPitch = pitch;
        sd[i].SysMemSlicePitch = size;
        p += size;
    }
    if (!i)
        goto done;
    memset(&td, 0, sizeof td);
    td.Width = w;
    td.Height = h;
    td.MipLevels = i;
    td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (SUCCEEDED(ID3D11Device_CreateTexture2D(s_dev, &td, sd, &tex))) {
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)tex, NULL, &srv);
        ID3D11Texture2D_Release(tex);        /* the view keeps it */
    }
done:
    free(buf);
    return srv;
}

typedef struct {
    uint32_t crc, wh, color, used;          /* used = 0: empty */
    ID3D11ShaderResourceView *srv;          /* NULL: not in the pack */
} PackEntry;
#define PACK_SLOTS 16384
static PackEntry *s_pack;
static const char *s_pack_dir;
static int s_pack_on = -1;
static uint32_t s_pack_misses, s_pack_loaded;

static ID3D11ShaderResourceView *tex_pack_lookup(const Nv2aHwTexture *t,
                                                 const TexEntry *e)
{
    uint32_t bb, crc, at, i, lb;
    int linear;
    PackEntry *pe = NULL;

    if (s_pack_on < 0) {
        s_pack_dir = getenv("RECOMP_TEXTURE_PACK");
        s_pack_on = s_pack_dir && *s_pack_dir
                 && (s_pack = (PackEntry *)calloc(PACK_SLOTS, sizeof *s_pack)) != NULL;
        if (s_pack_on)
            fprintf(stderr, "  [D3D11] texture pack: %s\n", s_pack_dir);
    }
    if (!s_pack_on || e->cube)
        return NULL;
    bb = dxt_block(t->color);
    linear = !d3d8_format_is_swizzled(t->color) && !bb;
    lb = level_bytes(t->color, e->w, e->h, t->pitch, linear);
    crc = crc32_bytes(guest() + t->va, lb);
    at = (crc ^ (e->w * 73856093u) ^ (e->h * 19349663u) ^ t->color) & (PACK_SLOTS - 1);
    for (i = 0; i < 64; i++) {
        PackEntry *c = &s_pack[(at + i) & (PACK_SLOTS - 1)];
        if (!c->used) { pe = c; break; }
        if (c->crc == crc && c->wh == ((e->w << 16) | e->h) && c->color == t->color)
            return c->srv;
    }
    if (!pe)
        return NULL;                          /* table full here: no probe */
    pe->used = 1;
    pe->crc = crc;
    pe->wh = (e->w << 16) | e->h;
    pe->color = t->color;
    {
        /* Several packs, ';'-separated: the first that has the key wins. */
        char path[1024];
        const char *d = s_pack_dir;
        pe->srv = NULL;
        while (d && *d && !pe->srv) {
            const char *end = strchr(d, ';');
            int len = end ? (int)(end - d) : (int)strlen(d);
            snprintf(path, sizeof path, "%.*s\\%08x_%ux%u_%02x.dds", len, d, crc,
                     e->w, e->h, t->color);
            pe->srv = dds_load(path);
            d = end ? end + 1 : NULL;
        }
        if (!pe->srv)
            s_pack_misses++;
        if (pe->srv) {
            if (s_pack_loaded++ < 8)
                fprintf(stderr, "  [D3D11] texture pack: %08x_%ux%u_%02x\n",
                        crc, e->w, e->h, t->color);
        } else {
            const char *mf = getenv("RECOMP_TEXTURE_PACK_MISSES");
            FILE *f = mf && *mf ? fopen(mf, "a") : NULL;
            if (f) {
                fprintf(f, "%08x_%ux%u_%02x\n", crc, e->w, e->h, t->color);
                fclose(f);
            }
        }
    }
    return pe->srv;
}

static ID3D11ShaderResourceView *tex_srv(const Nv2aHwTexture *t, uint32_t *out_w,
                                         uint32_t *out_h, uint32_t *out_levels)
{
    uint32_t bb = dxt_block(t->color);
    int swz = d3d8_format_is_swizzled(t->color) || bb;
    uint32_t w = t->width, h = t->height, levels = 1, bytes = 0, lv, lw, lh;
    uint32_t rect = (t->width << 16) | (t->height & 0xFFFFu);
    uint32_t at, i;
    int cube;
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
    /* SET_TEXTURE_FORMAT bit 2: a cube map, six square faces. */
    cube = (t->format & 4u) && swz && w == h;
    if (cube)
        bytes = ((bytes + 127u) & ~127u) * 6u;

    if (!guest_ok(t->va, bytes))
        return NULL;
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
        e->w = w; e->h = h; e->levels = levels; e->bytes = bytes; e->cube = cube;
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
        } else {
            e->rep = tex_pack_lookup(t, e);
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
        if (changed) {
            tex_upload(e, t, 0);
            e->rep = tex_pack_lookup(t, e);
        }
    }
    e->used_frame = s_frame;
    *out_w = e->w;
    *out_h = e->h;
    *out_levels = e->levels;
    if (e->bad)
        return NULL;
    return e->rep ? e->rep : e->srv;
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

static void queue_start(void);

int nv2a_d3d11_active(void)
{
    if (s_state < 0) {
        const char *e = getenv("RECOMP_GPU");
        s_state = 0;
        if (e && _stricmp(e, "d3d11") == 0)
            s_state = init();
        if (s_state > 0)
            queue_start();
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

/* The depth surface last drawn at `va`, whatever its size, or NULL. */
static DsEntry *ds_find(uint32_t va)
{
    DsEntry *best = NULL;
    int i;
    for (i = 0; i < DS_SLOTS; i++)
        if (s_ds[i].tex && s_ds[i].va == va
                && (!best || s_ds[i].used_frame > best->used_frame))
            best = &s_ds[i];
    return best;
}

/* A depth texture format's largest value (0 for a colour format): X8_Y24
 * 0x2A/0x2B swizzled and 0x2E/0x2F linear, Y16 0x2C/0x2D and 0x30/0x31.
 *
 * The NV2A samples a depth texture as a shadow map: the texel against the
 * coordinate's r/q (in the format's depth units), by SET_SHADOW_DEPTH_FUNC,
 * 1 or 0 in every channel. Amped 2's night level (BR1) blends each terrain
 * pixel between its lit colour and a dark-blue "in shadow" one (the
 * specular colour) by that result, from a 512x512 Y16 map at 0x80D20000
 * that it draws the rider and props into from above. It was read as a
 * colour texture from guest memory, which our depth pass never writes (the
 * depth stays on the GPU), so stage 1 read 0 -- the whole level in shadow,
 * the flat dark blue the player saw. So the map is our own depth surface
 * at that address, sampled through its SRV; a depth texture with no
 * surface drawn behind it reads as lit. */
static float tex_depth_max(uint32_t color)
{
    switch (color) {
    case 0x2A: case 0x2B: case 0x2E: case 0x2F: return 16777215.0f;
    case 0x2C: case 0x2D: case 0x30: case 0x31: return 65535.0f;
    default: return 0.0f;
    }
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

static int s_vis_in_test;        /* a visibility test is counting (do_zpass) */
static int s_vis_trace = -1;

static void vis_trace_draw(const Nv2aHwDraw *d, const char *what)
{
    static int said;
    if (s_vis_trace < 0)
        s_vis_trace = getenv("RECOMP_VIS_TRACE") != NULL;
    if (!s_vis_trace || !s_vis_in_test || said >= 16)
        return;
    said++;
    fprintf(stderr, "  [D3D11] in a visibility test, draw %s: prim %u, %u indices, colour %08X"
            " mask %08X, zeta %08X, depth %u func %04X, vp %d\n", what, d->prim, d->idx_count,
            d->color_va, d->color_mask, d->zeta_va, d->depth_test, d->depth_func, d->use_vp);
}

static void do_draw(const Nv2aHwDraw *d)
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
    {   /* RECOMP_SKIP_PS=<hash>[,<hash>]: leave out draws with these pixel
         * shaders (the DRAWDUMP key), to see what a pass contributes. */
        static int init;
        static uint64_t skip[4];
        static int nskip;
        if (!init) {
            const char *e = getenv("RECOMP_SKIP_PS");
            init = 1;
            while (e && *e && nskip < 4) {
                char *end;
                skip[nskip++] = _strtoui64(e, &end, 16);
                e = *end ? end + 1 : end;
            }
        }
        static int skip_blend = -1;      /* RECOMP_SKIP_BLEND: every blended draw */
        if (skip_blend < 0)
            skip_blend = getenv("RECOMP_SKIP_BLEND") != NULL;
        if (skip_blend && d->blend_enable)
            return;
        if (nskip && d->comb) {
            Nv2aHlslPsKey pk;
            uint64_t hk;
            int k;
            nv2a_hlsl_ps_key(d->comb, &pk);
            hk = hash64(&pk, sizeof pk, 3);
            for (k = 0; k < nskip; k++)
                if (skip[k] == hk)
                    return;
        }
    }
    if (!vs || !ps) {
        s_st.skipped++;
        return;
    }
    if (!bind_targets(d, &w, &h, &rt)) {
        s_st.skipped++;
        return;
    }
    /* Pixels the CPU wrote into this surface's memory. Rendering here never
     * writes guest memory, so if the memory under a surface changed since
     * its last frame of drawing, the title put an image there itself --
     * Amped 2's loading screen copies its controller picture straight into
     * the back buffer and draws only the text with the GPU. Bring those
     * pixels onto the surface before drawing over them. 32-bit only. */
    if (rt && rt->sig_frame != s_frame && color_bpp(d->format) == 4 && d->pitch
            && gpu_scale() == 1 && guest_ok(rt->va, d->pitch * rt->h)) {
        uint64_t sig = rt_mem_sig(rt);
        if (rt->sig_frame && sig != rt->mem_sig) {
            D3D11_BOX box = { 0, 0, 0, rt->w, rt->h, 1 };
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)rt->tex, 0,
                                                  &box, guest() + rt->va, d->pitch, 0);
            s_st.cpu_frames++;
        }
        rt->sig_frame = s_frame;
        rt->mem_sig = sig;
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
        if (!d->inline_base && !guest_ok(src, span)) {
            s_st.skipped++;
            return;
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
        ID3D11ShaderResourceView *srv[16] = { NULL };
        ID3D11SamplerState *smp[4] = { NULL, NULL, NULL, NULL };
        for (i = 0; i < 4; i++) {
            const Nv2aHwTexture *t = &d->tex[i];
            uint32_t tw = 0, th = 0, lv = 1;
            int norm = d3d8_format_is_swizzled(t->color) || dxt_block(t->color);
            RtEntry *src_rt;

            if (!t->enabled)
                continue;
            /* A depth texture -- a shadow map: the depth surface drawn at
             * that address, at t12+i, compared in the shader (tsc.w 3); with
             * no such surface, everything passes (tsc.w 4). */
            if (tex_depth_max(t->color) > 0.0f) {
                DsEntry *z = ds_find(t->va);
                int swz = t->color == 0x2A || t->color == 0x2B
                       || t->color == 0x2C || t->color == 0x2D;
                psd.tdz[i][0] = 1.0f / tex_depth_max(t->color);
                psd.tdz[i][1] = (float)d->shadow_func;
                psd.tsc[i][2] = 0.0f;
                psd.tsc[i][3] = 4.0f;
                if (z && z->srv) {
                    srv[12 + i] = z->srv;
                    psd.tsc[i][0] = swz ? 1.0f : 1.0f / (float)z->w;
                    psd.tsc[i][1] = swz ? 1.0f : 1.0f / (float)z->h;
                    psd.tsc[i][3] = 3.0f;
                    smp[i] = sampler(t, 1);
                }
                continue;
            }
            /* A volume texture: Texture3D tv<i> at t8+i, flagged by tsc.w 2. */
            if (tex_is_volume(t)) {
                srv[8 + i] = vol_srv(t);
                if (srv[8 + i]) {
                    psd.tsc[i][0] = psd.tsc[i][1] = 1.0f;
                    psd.tsc[i][2] = (t->color == 0x07 || t->color == 0x1E) ? 1.0f : 0.0f;
                    psd.tsc[i][3] = 2.0f;
                    smp[i] = sampler(t, 1);
                }
                continue;
            }
            /* A surface drawn this frame or the last, at this address: the
             * title is sampling what it rendered. */
            src_rt = rt_find(t->va);
            if (src_rt && src_rt != rt && !rt_still_current(src_rt, s_frame)) {
                static int said;
                if (said < 12 && getenv("RECOMP_RT_TRACE")) {
                    said++;
                    fprintf(stderr, "  [D3D11] texture %08X %ux%u fmt %02X is surface %ux%u drawn %u"
                            " frames ago, memory %s -- sampling guest memory instead\n",
                            t->va, t->width, t->height, t->color, src_rt->w, src_rt->h,
                            s_frame - src_rt->drawn_frame,
                            src_rt->sig_frame ? "changed" : "never signed");
                }
            }
            if (src_rt && src_rt != rt && rt_still_current(src_rt, s_frame)) {
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
                if ((t->format & 4u) && t->width == t->height
                        && (d3d8_format_is_swizzled(t->color) || dxt_block(t->color))) {
                    srv[4 + i] = srv[i];          /* TextureCube cx<i> */
                    srv[i] = NULL;
                }
                psd.tsc[i][0] = norm ? 1.0f : 1.0f / (float)tw;
                psd.tsc[i][1] = norm ? 1.0f : 1.0f / (float)th;
            }
            /* RECOMP_TEX_LOG=<n>: the first n distinct textures drawn with,
             * where they come from and how much of their memory is set. */
            {
                static int left = -1, nseen;
                static uint32_t seen[512], seen_nz[512];
                int k;
                if (left < 0) {
                    const char *e = getenv("RECOMP_TEX_LOG");
                    left = e ? atoi(e) : 0;
                }
                if (left > 0) {
                    uint32_t key = t->va ^ (t->color << 24) ^ (t->width << 12) ^ t->height;
                    for (k = 0; k < nseen && seen[k] != key; k++)
                        ;
                    if (((k == nseen && nseen < 512) || (k < nseen && seen_nz[k] == 0))
                            && guest_ok(t->va, t->width * t->height * 4)) {
                        const uint8_t *g = guest() + t->va;
                        uint32_t n = t->width * t->height * 4, nz = 0, b;
                        if (n > (4u << 20)) n = 4u << 20;
                        for (b = 0; b < n; b += 64)
                            nz += g[b] != 0;
                        if (k == nseen) {
                            seen[nseen++] = key;
                            seen_nz[k] = nz;
                        } else if (nz == 0) {
                            goto tex_log_done;     /* still empty */
                        } else {
                            seen_nz[k] = nz;
                        }
                        left--;
                        fprintf(stderr, "  [D3D11] tex%u %08X %ux%u fmt %02X pitch %u %s, "
                                "%u/%u sampled bytes set (frame %u)\n", i, t->va, t->width,
                                t->height, t->color, t->pitch,
                                srv[i] == (src_rt ? src_rt->srv : NULL) && src_rt ? "from surface" : "from memory",
                                nz, n / 64, s_frame);
                    }
                }
            tex_log_done:;
            }
            /* X8 formats carry no alpha: read it as opaque. */
            psd.tsc[i][2] = (t->color == 0x07 || t->color == 0x1E || t->color == 0x03
                             || t->color == 0x1C || t->color == 0x05 || t->color == 0x11)
                            ? 1.0f : 0.0f;
            psd.tsc[i][3] = 1.0f;
            smp[i] = sampler(t, lv);
        }
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 16, srv);
        ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 4, smp);
        /* RECOMP_TEXVS=<va>: the first draw sampling that texture on stage 1
         * prints its vertex shader and the program's constants -- to see how
         * a stage's coordinates are made. */
        {
            static int done = -1;
            static uint32_t want;
            if (done < 0) {
                const char *e = getenv("RECOMP_TEXVS");
                want = e ? (uint32_t)strtoul(e, NULL, 0) : 0;
                done = want ? 0 : 1;
            }
            static uint64_t seen_vs[6];
            static int nseen_vs;
            if (!done && d->tex[1].enabled && d->tex[1].va == want && d->use_vp && d->vp) {
                static char vsrc[256 * 1024];
                int wc = 0, c, k2;
                size_t n = nv2a_hlsl_vs(d->vp, vsrc, sizeof vsrc, &wc);
                uint64_t hv = hash64(vsrc, (uint32_t)n, 7);
                for (k2 = 0; k2 < nseen_vs && seen_vs[k2] != hv; k2++)
                    ;
                if (k2 < nseen_vs)
                    goto texvs_done;           /* this program already shown */
                seen_vs[nseen_vs++] = hv;
                if (nseen_vs >= 6)
                    done = 1;
                fprintf(stderr, "  [TEXVS] draw with texture %08X on stage 1 (%u indices,"
                        " program %d): vertex shader (%u bytes) follows\n", want,
                        d->idx_count, nseen_vs, (unsigned)n);
                fwrite(vsrc, 1, n, stderr);
                fprintf(stderr, "\n  [TEXVS] attributes:");
                for (c = 0; c < 16; c++)
                    if (d->attr[c].size)
                        fprintf(stderr, " %d:type %u size %u stride %u off %08X", c,
                                d->attr[c].type, d->attr[c].size, d->attr[c].stride,
                                d->attr[c].offset);
                fprintf(stderr, "\n  [TEXVS] attr1 offset %08X, inline %d, %u indices, idx0 %u",
                        d->attr[1].offset, d->inline_base != NULL, d->idx_count,
                        d->idx_count ? d->idx[0] : 0);
                {
                    uint32_t tb = d->tex[1].width * d->tex[1].height * 4u;
                    FILE *tf = guest_ok(want, tb) ? fopen("diag/texvs_stage1.bin", "wb") : NULL;
                    if (tf) {
                        fwrite(guest() + want, 1, tb, tf);
                        fclose(tf);
                        fprintf(stderr, "\n  [TEXVS] stage-1 texture %ux%u fmt %02X (format word %08X)"
                                " written to diag/texvs_stage1.bin", d->tex[1].width,
                                d->tex[1].height, d->tex[1].color, d->tex[1].format);
                    }
                }
                if (d->inline_base && d->attr[1].size && d->idx_count) {
                    int vi;
                    for (vi = 0; vi < 4 && vi < (int)d->idx_count; vi++) {
                        uint32_t off = d->attr[1].offset + d->idx[vi] * d->attr[1].stride;
                        if (off + 4 <= d->inline_bytes) {
                            uint32_t u = *(const uint32_t *)(d->inline_base + off);
                            fprintf(stderr, "\n  [TEXVS] inline vertex %u attr1 = %08X -> CMP"
                                    " %.3f %.3f %.3f; as bytes %u %u %u %u", d->idx[vi], u,
                                    (float)((int32_t)(u << 21) >> 21) / 1023.0f,
                                    (float)((int32_t)(u << 10) >> 21) / 1023.0f,
                                    (float)((int32_t)u >> 22) / 511.0f,
                                    u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, u >> 24);
                        }
                    }
                }
                if (!d->inline_base && d->attr[1].size && d->idx_count) {
                    int vi;
                    for (vi = 0; vi < 4 && vi < (int)d->idx_count; vi++) {
                        uint32_t va = d->attr[1].offset + d->idx[vi] * d->attr[1].stride;
                        if (guest_ok(va, 4)) {
                            uint32_t u = *(const uint32_t *)(guest() + va);
                            fprintf(stderr, "\n  [TEXVS] vertex %u attr1 @%08X = %08X -> CMP"
                                    " %.3f %.3f %.3f", d->idx[vi], va, u,
                                    (float)((int32_t)(u << 21) >> 21) / 1023.0f,
                                    (float)((int32_t)(u << 10) >> 21) / 1023.0f,
                                    (float)((int32_t)u >> 22) / 511.0f);
                        }
                    }
                }
                fprintf(stderr, "\n  [TEXVS] constants (non-zero):\n");
                for (c = 0; c < NV2A_VP_CONSTANTS; c++) {
                    const float *k = d->vp->constants[c];
                    if (k[0] || k[1] || k[2] || k[3])
                        fprintf(stderr, "  [TEXVS] c[%d] = %g %g %g %g\n", c,
                                k[0], k[1], k[2], k[3]);
                }
                if (d->comb) {                 /* and the pixel shader */
                    Nv2aHlslPsKey pk;
                    nv2a_hlsl_ps_key(d->comb, &pk);
                    n = nv2a_hlsl_ps(&pk, vsrc, sizeof vsrc);
                    fprintf(stderr, "  [TEXVS] pixel shader (%u bytes) follows\n", (unsigned)n);
                    fwrite(vsrc, 1, n, stderr);
                    fprintf(stderr, "\n");
                }
                fflush(stderr);
            }
        texvs_done:;
        }
        /* RECOMP_DRAW_DUMP=<min indices>: each distinct pixel shader used by
         * a draw that large (after frame 600) prints its four stages' state
         * and its source -- once each, up to 24. */
        {
            static int min_idx = -1, nkeys;
            static uint64_t keys[24];
            if (min_idx < 0) {
                const char *e = getenv("RECOMP_DRAW_DUMP");
                min_idx = e ? atoi(e) : 0;
            }
            if (min_idx > 0 && nkeys < 24 && d->comb && s_frame > 600
                    && d->idx_count >= (uint32_t)min_idx) {
                static char psrc[64 * 1024];
                Nv2aHlslPsKey pk;
                uint64_t hk;
                int k2;
                size_t n;
                nv2a_hlsl_ps_key(d->comb, &pk);
                hk = hash64(&pk, sizeof pk, 3);
                for (k2 = 0; k2 < nkeys && keys[k2] != hk; k2++)
                    ;
                if (k2 == nkeys) {
                    keys[nkeys++] = hk;
                    fprintf(stderr, "  [DRAWDUMP] frame %u, %u indices, shader %016llx tag %06X:\n",
                            s_frame, d->idx_count, (unsigned long long)hk,
                            nv2a_hlsl_ps_tag(&pk));
                    fprintf(stderr, "  [DRAWDUMP]   blend %u src %04X dst %04X eq %04X color %08X"
                            " mask %08X depth %u func %04X zmask %u fog %u\n", d->blend_enable,
                            d->blend_sfactor, d->blend_dfactor, d->blend_eq, d->blend_color,
                            d->color_mask, d->depth_test, d->depth_func, d->depth_mask,
                            d->fog_enable);
                    fprintf(stderr, "  [DRAWDUMP]   factor0 %08X %08X factor1 %08X %08X final %08X %08X"
                            " fog %08X\n", d->comb->factor0[0], d->comb->factor0[1],
                            d->comb->factor1[0], d->comb->factor1[1], d->comb->final_factor0,
                            d->comb->final_factor1, d->comb->fog_color);
                    for (k2 = 0; k2 < 4; k2++) {
                        const Nv2aHwTexture *t = &d->tex[k2];
                        uint32_t tb = dxt_block(t->color)
                            ? t->width * t->height / (t->color == 0x0C ? 2u : 1u)
                            : t->width * t->height * nv2a_pb_texel_bytes(t->color);
                        fprintf(stderr, "  [DRAWDUMP]   stage %d: en %d va %08X format %08X"
                                " color %02X %ux%u addr %08X filter %08X\n", k2, t->enabled,
                                t->va, t->format, t->color, t->width, t->height,
                                t->address, t->filter);
                        tb *= 2;                /* with room for the mip chain */
                        if (t->va && tb && tb <= (1u << 21) && guest_ok(t->va, tb)) {
                            char path[64];      /* raw, diag/tex_<va>.bin */
                            FILE *f;
                            snprintf(path, sizeof path, "diag/tex_%08X.bin", t->va);
                            if ((f = fopen(path, "wb")) != NULL) {
                                fwrite(guest() + t->va, 1, tb, f);
                                fclose(f);
                            }
                        }
                    }
                    n = nv2a_hlsl_ps(&pk, psrc, sizeof psrc);
                    fwrite(psrc, 1, n, stderr);
                    fprintf(stderr, "\n");
                    fflush(stderr);
                }
            }
        }
    }
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_psc, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        memcpy(m.pData, &psd, sizeof psd);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_psc, 0);
    }

    /* Fixed-function state. */
    {
        const int S = gpu_scale();
        D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)(w * S), (float)(h * S), 0.0f, 1.0f };
        D3D11_RECT sc;
        float bf[4];
        sc.left = (LONG)(d->clip_x * S);
        sc.top = (LONG)(d->clip_y * S);
        sc.right = (LONG)((d->clip_x + d->clip_w) * S);
        sc.bottom = (LONG)((d->clip_y + d->clip_h) * S);
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

    vis_trace_draw(d, "submitted");
    ID3D11DeviceContext_DrawIndexed(s_ctx, nidx, ib_at / 4, 0);
    s_st.draws++;
    s_st.tris += nidx / 3;
    if (rt) {
        rt->drawn_frame = s_frame;
        if (rt->sig_frame != s_frame) {       /* once a frame per surface */
            rt->sig_frame = s_frame;
            rt->mem_sig = rt_mem_sig(rt);
        }
    }
    QueryPerformanceCounter(&t_out);
    s_st.draw_ticks += (uint64_t)(t_out.QuadPart - t_in.QuadPart);
}

static void do_clear(const Nv2aHwDraw *d, uint32_t flags, uint32_t color,
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
            r.left = (LONG)(d->clip_x * gpu_scale());
            r.top = (LONG)(d->clip_y * gpu_scale());
            r.right = (LONG)((d->clip_x + d->clip_w) * gpu_scale());
            r.bottom = (LONG)((d->clip_y + d->clip_h) * gpu_scale());
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

/* ── Presenting to the window ────────────────────────────────────────────
 *
 * A flip-model swap chain on the framebuffer window, fed by a fullscreen
 * triangle that samples the flipped surface: the window can be resized or
 * maximised and the image scales to fit, letterboxed to the title's 4:3,
 * with no CPU copy. GDI stops painting the moment this claims the window.
 * The readback path stays for dumps (RECOMP_FB_DUMP) and runs only then.
 * RECOMP_GPU_PRESENT=gdi keeps the old path. */
extern HWND xbox_FramebufferWindowHwnd(void);
extern void xbox_FramebufferWindowClaim(void);

static IDXGISwapChain1 *s_swap;
static ID3D11RenderTargetView *s_bb_rtv;
static UINT s_bb_w, s_bb_h;
static ID3D11VertexShader *s_blit_vs;
static ID3D11PixelShader *s_blit_ps;
static ID3D11PixelShader *s_blit_fxaa_ps;    /* RECOMP_GPU_FXAA */
static ID3D11SamplerState *s_blit_smp;
static ID3D11Buffer *s_blit_cb;
static int s_present_mode = -1;          /* 0 gdi, 1 swap chain */

static const char s_blit_hlsl[] =
    "cbuffer B : register(b0) { float4 uvscale; };\n"
    "Texture2D tx : register(t0); SamplerState sm : register(s0);\n"
    "struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; };\n"
    "V vs(uint id : SV_VertexID) {\n"
    "  V o; float2 t = float2((id << 1) & 2, id & 2);\n"
    "  o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "  o.uv = t * uvscale.xy; return o;\n"
    "}\n"
    "float4 ps(V i) : SV_Target { return float4(tx.Sample(sm, i.uv).rgb, 1); }\n"
    /* RECOMP_GPU_FXAA: FXAA on the finished frame (the classic PC version:
     * luma edges from the four diagonal neighbours, then two or four taps
     * along the edge). uvscale.zw is one source texel. */
    "float lum(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }\n"
    "float3 at(float2 uv) { return tx.SampleLevel(sm, uv, 0).rgb; }\n"
    "float4 ps_fxaa(V i) : SV_Target {\n"
    "  float2 px = uvscale.zw, uv = i.uv;\n"
    "  float3 m = at(uv);\n"
    "  float nw = lum(at(uv + float2(-1, -1) * px)), ne = lum(at(uv + float2(1, -1) * px));\n"
    "  float sw = lum(at(uv + float2(-1, 1) * px)), se = lum(at(uv + float2(1, 1) * px));\n"
    "  float lm = lum(m);\n"
    "  float lmin = min(lm, min(min(nw, ne), min(sw, se)));\n"
    "  float lmax = max(lm, max(max(nw, ne), max(sw, se)));\n"
    "  float2 d = float2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));\n"
    "  float red = max((nw + ne + sw + se) * (0.25 / 8.0), 1.0 / 128.0);\n"
    "  float r = 1.0 / (min(abs(d.x), abs(d.y)) + red);\n"
    "  d = clamp(d * r, -8.0, 8.0) * px;\n"
    "  float3 a = 0.5 * (at(uv + d * (1.0 / 3.0 - 0.5)) + at(uv + d * (2.0 / 3.0 - 0.5)));\n"
    "  float3 b = a * 0.5 + 0.25 * (at(uv - d * 0.5) + at(uv + d * 0.5));\n"
    "  float lb = lum(b);\n"
    "  return float4((lb < lmin || lb > lmax) ? a : b, 1);\n"
    "}\n";

static int present_init(HWND hwnd)
{
    IDXGIDevice *dxgi = NULL;
    IDXGIAdapter *ad = NULL;
    IDXGIFactory2 *f2 = NULL;
    DXGI_SWAP_CHAIN_DESC1 sd;
    ID3DBlob *vb = NULL, *pb = NULL, *err = NULL;
    D3D11_SAMPLER_DESC smp;
    HRESULT hr;

    if (FAILED(ID3D11Device_QueryInterface(s_dev, &IID_IDXGIDevice, (void **)&dxgi)))
        return 0;
    IDXGIDevice_GetAdapter(dxgi, &ad);
    IDXGIDevice_Release(dxgi);
    if (!ad)
        return 0;
    IDXGIAdapter_GetParent(ad, &IID_IDXGIFactory2, (void **)&f2);
    IDXGIAdapter_Release(ad);
    if (!f2)
        return 0;
    memset(&sd, 0, sizeof sd);
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    hr = IDXGIFactory2_CreateSwapChainForHwnd(f2, (IUnknown *)s_dev, hwnd, &sd,
                                               NULL, NULL, &s_swap);
    if (SUCCEEDED(hr))
        IDXGIFactory2_MakeWindowAssociation(f2, hwnd, DXGI_MWA_NO_ALT_ENTER);
    IDXGIFactory2_Release(f2);
    if (FAILED(hr)) {
        fprintf(stderr, "  [D3D11] swap chain failed (0x%08lX); presenting through GDI\n",
                (unsigned long)hr);
        return 0;
    }
    if (FAILED(D3DCompile(s_blit_hlsl, sizeof s_blit_hlsl - 1, NULL, NULL, NULL,
                          "vs", "vs_5_0", 0, 0, &vb, &err))
            || FAILED(D3DCompile(s_blit_hlsl, sizeof s_blit_hlsl - 1, NULL, NULL, NULL,
                                 "ps", "ps_5_0", 0, 0, &pb, &err))) {
        fprintf(stderr, "  [D3D11] blit shader: %s\n",
                err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        return 0;
    }
    ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(vb),
                                    ID3D10Blob_GetBufferSize(vb), NULL, &s_blit_vs);
    ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(pb),
                                   ID3D10Blob_GetBufferSize(pb), NULL, &s_blit_ps);
    ID3D10Blob_Release(vb);
    ID3D10Blob_Release(pb);
    {
        const char *fx = getenv("RECOMP_GPU_FXAA");
        if (fx && *fx && *fx != '0') {
            ID3DBlob *fb = NULL;
            if (SUCCEEDED(D3DCompile(s_blit_hlsl, sizeof s_blit_hlsl - 1, NULL, NULL,
                                     NULL, "ps_fxaa", "ps_5_0", 0, 0, &fb, &err))) {
                ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(fb),
                                               ID3D10Blob_GetBufferSize(fb), NULL,
                                               &s_blit_fxaa_ps);
                ID3D10Blob_Release(fb);
                fprintf(stderr, "  [D3D11] FXAA on\n");
            } else {
                fprintf(stderr, "  [D3D11] FXAA shader: %s\n",
                        err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
            }
        }
    }
    memset(&smp, 0, sizeof smp);
    smp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    smp.AddressU = smp.AddressV = smp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    smp.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11Device_CreateSamplerState(s_dev, &smp, &s_blit_smp);
    s_blit_cb = make_buffer(16, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, NULL);
    xbox_FramebufferWindowClaim();
    fprintf(stderr, "  [D3D11] presenting through a swap chain on the window\n");
    return 1;
}

/* Returns 1 if the frame went to the window. */
/* ── The video overlay (PVIDEO) ──────────────────────────────────────────
 *
 * XMV movies are not drawn: D3D's UpdateOverlay (Amped 2 0x00329240)
 * programs the NV2A's video scaler, and the display hardware puts a YUY2
 * surface over the picture. The registers it writes (base 0xFD008000):
 * BUFFER 0x700 (1 = show buffer 0), STOP 0x704, BASE 0x900, OFFSET 0x920
 * (the surface's physical address plus the source origin, 64-byte aligned;
 * the rest is POINT_IN 0x930 in 1/16 pixel), SIZE_IN 0x928 (h << 16 | w,
 * w counting from OFFSET), POINT_OUT 0x948 and SIZE_OUT 0x950 (the
 * destination, y << 16 | x), FORMAT 0x958 (pitch in bits 0-12, colour 1 =
 * YUY2 in bit 16, colour key on in bit 20) and the key at 0xB00. The
 * aperture is plain memory here, so they are simply read. A movie player
 * updates the overlay and waits for vblank without presenting, so while it
 * is on the render thread presents on its own (present_overlay_tick). */
typedef struct {
    uint32_t va, w, h, pitch, x, y, ow, oh, key;
    int keyed;
    float s0;
} Overlay;

static ID3D11Texture2D *s_ov_tex;
static ID3D11ShaderResourceView *s_ov_srv;
static uint32_t s_ov_tw, s_ov_th;
static ID3D11VertexShader *s_ov_vs;
static ID3D11PixelShader *s_ov_ps;
static ID3D11Buffer *s_ov_cb;
static int s_ov_failed;
static ULONGLONG s_last_present_ms;
static uint32_t s_last_flip_va, s_last_flip_w = 640, s_last_flip_h = 480;

static uint32_t pvideo(uint32_t off)
{
    return *(const volatile uint32_t *)(guest() + 0xFD008000u + off);
}

static int overlay_read(Overlay *o)
{
    uint32_t phys, sin, fmt;

    if (!(pvideo(0x700) & 1u) || (pvideo(0x704) & 1u))
        return 0;
    fmt = pvideo(0x958);
    if (((fmt >> 16) & 0xFu) != 1u)            /* YUY2 is all D3D asks for */
        return 0;
    phys = (pvideo(0x900) + pvideo(0x920)) & 0x0FFFFFFFu;
    o->va = (phys >= 0x04000000u && phys < 0x08000000u) ? phys - 0x04000000u
                                                        : 0x80000000u + phys;
    sin = pvideo(0x928);
    o->w = sin & 0xFFFFu;
    o->h = sin >> 16;
    o->s0 = (float)(pvideo(0x930) & 0xFFFFu) / 16.0f;
    o->pitch = fmt & 0x1FFFu;
    o->x = pvideo(0x948) & 0xFFFFu;
    o->y = pvideo(0x948) >> 16;
    o->ow = pvideo(0x950) & 0xFFFFu;
    o->oh = pvideo(0x950) >> 16;
    o->keyed = (fmt & 0x00100000u) != 0;
    o->key = pvideo(0xB00);
    if (!o->w || !o->h || o->w > 2048 || o->h > 2048 || !o->ow || !o->oh
            || o->pitch < o->w * 2u || !guest_ok(o->va, o->pitch * o->h))
        return 0;
    return 1;
}

static const char s_ov_hlsl[] =
    "cbuffer O : register(b0) { float4 src; float4 dst; float4 frm; uint4 key; };\n"
    /* src: w, h, s0, unused (source pixels); dst: x, y, w, h (guest pixels);
     * frm: frame w, h (guest pixels) and its uv scale; key: on, r, g, b */
    "Texture2D<float4> yuy : register(t0); Texture2D frame : register(t1);\n"
    "SamplerState sm : register(s0);\n"
    "struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; };\n"
    "V vs(uint id : SV_VertexID) {\n"
    "  V o; float2 t = float2((id << 1) & 2, id & 2);\n"
    "  o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = t; return o;\n"
    "}\n"
    "float4 ps(V i) : SV_Target {\n"
    "  float px = src.z + i.uv.x * (src.x - src.z);\n"
    "  int2 tc = int2(min(px, src.x - 1) * 0.5, min(i.uv.y * src.y, src.y - 1));\n"
    "  float4 q = yuy.Load(int3(tc, 0));      /* Y0 U Y1 V */\n"
    "  float y = (frac(px * 0.5) < 0.5 ? q.r : q.b) - 16.0 / 255.0;\n"
    "  float u = q.g - 0.5, v = q.a - 0.5;\n"
    "  float3 rgb = saturate(float3(1.164 * y + 1.596 * v,\n"
    "                               1.164 * y - 0.391 * u - 0.813 * v,\n"
    "                               1.164 * y + 2.018 * u));\n"
    "  if (key.x != 0) {\n"
    "    float2 g = dst.xy + i.uv * dst.zw;\n"
    "    float3 f = frame.Sample(sm, g / frm.xy * frm.zw).rgb;\n"
    "    uint3 fb = uint3(f * 255.0 + 0.5);\n"
    "    if (any(fb != key.yzw)) return float4(f, 1);\n"
    "  }\n"
    "  return float4(rgb, 1);\n"
    "}\n";

static int overlay_init(void)
{
    ID3DBlob *vb = NULL, *pb = NULL, *err = NULL;
    if (s_ov_ps || s_ov_failed)
        return s_ov_ps != NULL;
    if (FAILED(D3DCompile(s_ov_hlsl, sizeof s_ov_hlsl - 1, NULL, NULL, NULL, "vs", "vs_5_0",
                          0, 0, &vb, &err))
            || FAILED(D3DCompile(s_ov_hlsl, sizeof s_ov_hlsl - 1, NULL, NULL, NULL, "ps",
                                 "ps_5_0", 0, 0, &pb, &err))) {
        fprintf(stderr, "  [D3D11] overlay shader: %s\n",
                err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "compile failed");
        if (err) ID3D10Blob_Release(err);
        if (vb) ID3D10Blob_Release(vb);
        s_ov_failed = 1;
        return 0;
    }
    ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(vb),
                                    ID3D10Blob_GetBufferSize(vb), NULL, &s_ov_vs);
    ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(pb),
                                   ID3D10Blob_GetBufferSize(pb), NULL, &s_ov_ps);
    ID3D10Blob_Release(vb);
    ID3D10Blob_Release(pb);
    s_ov_cb = make_buffer(64, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, NULL);
    return s_ov_ps && s_ov_vs && s_ov_cb;
}

/* Draw the overlay into the bound back buffer, over the frame already blitted
 * into `pic` (the letterboxed picture's viewport, in back-buffer pixels). */
static void overlay_compose(RtEntry *rt, uint32_t fw, uint32_t fh, const D3D11_VIEWPORT *pic)
{
    Overlay o;
    D3D11_VIEWPORT vp;
    D3D11_MAPPED_SUBRESOURCE m;
    uint32_t tw, y;
    ID3D11ShaderResourceView *srvs[2];

    if (!overlay_read(&o) || !overlay_init())
        return;
    {   /* RECOMP_PVIDEO_TRACE: each new source surface (the first 40). */
        static uint32_t last_va;
        static int said, trace = -1;
        if (trace < 0)
            trace = getenv("RECOMP_PVIDEO_TRACE") != NULL;
        if (trace && o.va != last_va && said++ < 40)
            fprintf(stderr, "  [PVIDEO] overlay source %08X (frame %u)\n", o.va, s_frame);
        last_va = o.va;
    }
    tw = (o.w + 1) / 2;
    if (!s_ov_tex || s_ov_tw != tw || s_ov_th != o.h) {
        D3D11_TEXTURE2D_DESC td;
        if (s_ov_srv) ID3D11ShaderResourceView_Release(s_ov_srv);
        if (s_ov_tex) ID3D11Texture2D_Release(s_ov_tex);
        s_ov_srv = NULL;
        s_ov_tex = NULL;
        memset(&td, 0, sizeof td);
        td.Width = tw; td.Height = o.h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &s_ov_tex)))
            return;
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)s_ov_tex, NULL, &s_ov_srv);
        s_ov_tw = tw;
        s_ov_th = o.h;
        fprintf(stderr, "  [D3D11] video overlay: %ux%u YUY2 at %08X pitch %u -> %ux%u at"
                " %u,%u%s\n", o.w, o.h, o.va, o.pitch, o.ow, o.oh, o.x, o.y,
                o.keyed ? " (colour keyed)" : "");
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_ov_tex, 0,
                                       D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    for (y = 0; y < o.h; y++)
        memcpy((uint8_t *)m.pData + (size_t)y * m.RowPitch,
               guest() + o.va + (size_t)y * o.pitch, (size_t)tw * 4);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_ov_tex, 0);

    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_ov_cb, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        float *f = (float *)m.pData;
        uint32_t *u = (uint32_t *)m.pData;
        f[0] = (float)o.w; f[1] = (float)o.h; f[2] = o.s0; f[3] = 0;
        f[4] = (float)o.x; f[5] = (float)o.y; f[6] = (float)o.ow; f[7] = (float)o.oh;
        f[8] = (float)fw; f[9] = (float)fh;
        f[10] = rt ? (float)fw / (float)rt->w : 1.0f;
        f[11] = rt ? (float)fh / (float)rt->h : 1.0f;
        u[12] = (o.keyed && rt) ? 1u : 0u;
        u[13] = (o.key >> 16) & 0xFFu; u[14] = (o.key >> 8) & 0xFFu; u[15] = o.key & 0xFFu;
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_ov_cb, 0);
    }
    /* The destination, in guest pixels, mapped into the picture's viewport. */
    vp.TopLeftX = pic->TopLeftX + (float)o.x * pic->Width / (float)fw;
    vp.TopLeftY = pic->TopLeftY + (float)o.y * pic->Height / (float)fh;
    vp.Width = (float)o.ow * pic->Width / (float)fw;
    vp.Height = (float)o.oh * pic->Height / (float)fh;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_ov_vs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_ov_ps, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_ov_cb);
    srvs[0] = s_ov_srv;
    srvs[1] = rt ? rt->srv : NULL;
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 2, srvs);
    ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 1, &s_blit_smp);
    ID3D11DeviceContext_Draw(s_ctx, 3, 0);
    srvs[0] = srvs[1] = NULL;
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 2, srvs);
}

static int present_frame(RtEntry *rt, uint32_t w, uint32_t h);
static void present_shot(void);

/* From the render thread's idle loop: while the overlay is on and nothing has
 * flipped for a frame's time, show the last frame with the overlay over it. */
static void present_overlay_tick(void)
{
    Overlay o;
    ULONGLONG now = GetTickCount64();
    if (now - s_last_present_ms < 16 || !overlay_read(&o))
        return;
    present_frame(s_last_flip_va ? rt_find(s_last_flip_va) : NULL,
                  s_last_flip_w, s_last_flip_h);
}

static int present_frame(RtEntry *rt, uint32_t w, uint32_t h)
{
    HWND hwnd;
    RECT cr;
    UINT cw, ch;
    D3D11_VIEWPORT vp;
    D3D11_MAPPED_SUBRESOURCE m;
    float scale;

    if (s_present_mode < 0) {
        const char *e = getenv("RECOMP_GPU_PRESENT");
        s_present_mode = !(e && _stricmp(e, "gdi") == 0);
    }
    if (!s_present_mode)
        return 0;
    hwnd = xbox_FramebufferWindowHwnd();
    if (!hwnd)
        return 0;
    if (!s_swap && !present_init(hwnd)) {
        s_present_mode = 0;
        return 0;
    }
    GetClientRect(hwnd, &cr);
    cw = (UINT)(cr.right - cr.left);
    ch = (UINT)(cr.bottom - cr.top);
    if (!cw || !ch)
        return 1;                              /* minimised */
    if (!s_bb_rtv || cw != s_bb_w || ch != s_bb_h) {
        ID3D11Texture2D *bb = NULL;
        if (s_bb_rtv) {
            ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 0, NULL, NULL);
            ID3D11RenderTargetView_Release(s_bb_rtv);
            s_bb_rtv = NULL;
        }
        if (FAILED(IDXGISwapChain1_ResizeBuffers(s_swap, 0, cw, ch, DXGI_FORMAT_UNKNOWN, 0)))
            return 0;
        IDXGISwapChain1_GetBuffer(s_swap, 0, &IID_ID3D11Texture2D, (void **)&bb);
        if (!bb)
            return 0;
        ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)bb, NULL, &s_bb_rtv);
        ID3D11Texture2D_Release(bb);
        s_bb_w = cw;
        s_bb_h = ch;
    }
    {
        const float black[4] = { 0, 0, 0, 1 };
        ID3D11DeviceContext_ClearRenderTargetView(s_ctx, s_bb_rtv, black);
    }
    /* Letterbox the title's picture into the window. */
    scale = (float)cw / (float)w < (float)ch / (float)h ? (float)cw / (float)w
                                                       : (float)ch / (float)h;
    vp.Width = (float)w * scale;
    vp.Height = (float)h * scale;
    vp.TopLeftX = ((float)cw - vp.Width) * 0.5f;
    vp.TopLeftY = ((float)ch - vp.Height) * 0.5f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_blit_cb, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        float uv[4] = { rt ? (float)w / (float)rt->w : 1.0f, rt ? (float)h / (float)rt->h : 1.0f,
                        rt ? 1.0f / (float)rt->w : 0.0f, rt ? 1.0f / (float)rt->h : 0.0f };
        memcpy(m.pData, uv, sizeof uv);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_blit_cb, 0);
    }
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &s_bb_rtv, NULL);
    ID3D11DeviceContext_OMSetBlendState(s_ctx, NULL, NULL, 0xFFFFFFFFu);
    ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, NULL, 0);
    ID3D11DeviceContext_RSSetState(s_ctx, NULL);
    ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
    ID3D11DeviceContext_IASetInputLayout(s_ctx, NULL);
    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_blit_vs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_blit_fxaa_ps ? s_blit_fxaa_ps : s_blit_ps,
                                    NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 0, 1, &s_blit_cb);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_blit_cb);
    if (rt) {
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &rt->srv);
        ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 1, &s_blit_smp);
        ID3D11DeviceContext_Draw(s_ctx, 3, 0);
        {
            ID3D11ShaderResourceView *none = NULL;
            ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
        }
    }
    overlay_compose(rt, w, h, &vp);
    present_shot();
    IDXGISwapChain1_Present(s_swap, 0, 0);
    s_last_present_ms = GetTickCount64();
    return 1;
}

/* A B8G8R8A8 texture's top-left w x h as a 24-bit BMP. A blocking readback. */
static void shot_write(ID3D11Texture2D *tex, uint32_t w, uint32_t h, const char *path)
{
    D3D11_TEXTURE2D_DESC td;
    D3D11_MAPPED_SUBRESOURCE m;
    ID3D11Texture2D *st = NULL;
    FILE *f;
    uint32_t x, y, row = w * 3, pad = (4 - row % 4) % 4;

    ID3D11Texture2D_GetDesc(tex, &td);
    if (w > td.Width) w = td.Width;
    if (h > td.Height) h = td.Height;
    row = w * 3; pad = (4 - row % 4) % 4;
    td.MipLevels = 1; td.ArraySize = 1; td.SampleDesc.Count = 1; td.SampleDesc.Quality = 0;
    td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    td.BindFlags = 0; td.MiscFlags = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &st)))
        return;
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)st, (ID3D11Resource *)tex);
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)st, 0, D3D11_MAP_READ, 0, &m))
            && (f = fopen(path, "wb")) != NULL) {
        uint8_t hdr[54] = { 'B', 'M' };
        uint32_t size = 54 + (row + pad) * h;
        memcpy(hdr + 2, &size, 4); hdr[10] = 54; hdr[14] = 40;
        memcpy(hdr + 18, &w, 4); memcpy(hdr + 22, &h, 4); hdr[26] = 1; hdr[28] = 24;
        fwrite(hdr, 1, 54, f);
        for (y = h; y-- > 0; ) {
            const uint8_t *src = (const uint8_t *)m.pData + (size_t)y * m.RowPitch;
            uint8_t zero[3] = { 0 };
            for (x = 0; x < w; x++)
                fwrite(src + x * 4, 1, 3, f);
            fwrite(zero, 1, pad, f);
        }
        fclose(f);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)st, 0);
    }
    ID3D11Texture2D_Release(st);
}

/* RECOMP_GPU_SHOT=<prefix>: the presented frame as <prefix>NNNN.bmp every
 * RECOMP_GPU_SHOT_MS (default 1000) -- the title's own picture, whatever
 * window is in front of it. A blocking readback, so a hitch per shot. */
static void gpu_shot(RtEntry *rt)
{
    static const char *prefix = (const char *)1;
    static DWORD last, every;
    static unsigned seq;
    char path[300];
    FILE *f;

    if (prefix == (const char *)1) {
        const char *e = getenv("RECOMP_GPU_SHOT_MS");
        prefix = getenv("RECOMP_GPU_SHOT");
        every = e ? (DWORD)atoi(e) : 1000;
    }
    if (!prefix || GetTickCount() - last < every)
        return;
    last = GetTickCount();
    snprintf(path, sizeof path, "%s%04u.bmp", prefix, seq++);
    shot_write(rt->tex, rt->w * (uint32_t)gpu_scale(), rt->h * (uint32_t)gpu_scale(), path);
    /* RECOMP_GPU_SHOT_MEM: the guest memory under the surface as well, to
     * see what the CPU put there. */
    if (getenv("RECOMP_GPU_SHOT_MEM") && gpu_scale() == 1
            && (snprintf(path, sizeof path, "%s%04u_mem.bmp", prefix, seq - 1),
                f = fopen(path, "wb")) != NULL) {
        const uint8_t *g = guest() + rt->va;
        uint32_t x, y, w = rt->w, h = rt->h, row = w * 3, pad = (4 - row % 4) % 4;
        uint8_t hdr[54] = { 'B', 'M' };
        uint32_t size = 54 + (row + pad) * h;
        memcpy(hdr + 2, &size, 4); hdr[10] = 54; hdr[14] = 40;
        memcpy(hdr + 18, &w, 4); memcpy(hdr + 22, &h, 4); hdr[26] = 1; hdr[28] = 24;
        fwrite(hdr, 1, 54, f);
        for (y = h; y-- > 0; ) {
            uint8_t zero[3] = { 0 };
            for (x = 0; x < w; x++)
                fwrite(g + (size_t)y * w * 4 + x * 4, 1, 3, f);
            fwrite(zero, 1, pad, f);
        }
        fclose(f);
    }
}

/* RECOMP_GPU_SHOT_PRESENT=<prefix>: the window's back buffer as presented --
 * letterboxing and the video overlay included, which RECOMP_GPU_SHOT (the
 * title's render target) does not show. Same RECOMP_GPU_SHOT_MS interval. */
static void present_shot(void)
{
    static const char *prefix = (const char *)1;
    static DWORD last, every;
    static unsigned seq;
    ID3D11Texture2D *bb = NULL;
    char path[300];

    if (prefix == (const char *)1) {
        const char *e = getenv("RECOMP_GPU_SHOT_MS");
        prefix = getenv("RECOMP_GPU_SHOT_PRESENT");
        every = e ? (DWORD)atoi(e) : 1000;
    }
    if (!prefix || GetTickCount() - last < every)
        return;
    last = GetTickCount();
    IDXGISwapChain1_GetBuffer(s_swap, 0, &IID_ID3D11Texture2D, (void **)&bb);
    if (!bb)
        return;
    snprintf(path, sizeof path, "%s%04u.bmp", prefix, seq++);
    shot_write(bb, s_bb_w, s_bb_h, path);
    ID3D11Texture2D_Release(bb);
}

static void do_flip(uint32_t color_va)
{
    RtEntry *rt = rt_find(color_va);
    uint32_t i = s_stage_next % STAGES;

    s_frame++;
    {   /* RECOMP_PVIDEO_TRACE: the overlay registers when they change. */
        static int trace = -1;
        static uint32_t last[12];
        if (trace < 0)
            trace = getenv("RECOMP_PVIDEO_TRACE") != NULL;
        if (trace) {      /* the NV2A aperture (0xFD000000) is mapped */
            static const uint16_t off[12] = { 0x700, 0x704, 0x900, 0x908, 0x920, 0x928,
                                              0x930, 0x938, 0x940, 0x948, 0x950, 0x958 };
            const uint8_t *pv = guest() + 0xFD008000u;
            uint32_t cur[12];
            int k, changed = 0;
            for (k = 0; k < 12; k++) {
                cur[k] = *(const volatile uint32_t *)(pv + off[k]);
                changed |= cur[k] != last[k];
            }
            if (changed) {
                static int said;
                if (said++ < 40)
                    fprintf(stderr, "  [PVIDEO] frame %u rt %08X: buffer %X stop %X base %08X"
                            " limit %08X offset %08X size_in %08X point_in %08X ds_dx %08X"
                            " dt_dy %08X point_out %08X size_out %08X format %08X key %08X\n",
                            s_frame, color_va, cur[0], cur[1], cur[2], cur[3], cur[4], cur[5],
                            cur[6], cur[7], cur[8], cur[9], cur[10], cur[11],
                            *(const volatile uint32_t *)(pv + 0xB00));
                memcpy(last, cur, sizeof last);
            }
        }
    }
    if (!rt)
        return;
    s_last_flip_va = color_va;
    s_last_flip_w = rt->w;
    s_last_flip_h = rt->h;
    gpu_shot(rt);
    {
        static int dumps = -1;
        if (dumps < 0)
            dumps = getenv("RECOMP_FB_DUMP") != NULL;
        /* On the window already, and nobody wants the pixels on the CPU. */
        if (present_frame(rt, rt->w, rt->h) && !dumps)
            return;
        /* A scaled surface is not the guest's size, so there is nothing
         * to read back into guest memory. */
        if (gpu_scale() > 1)
            return;
    }
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

/* ── The render thread ───────────────────────────────────────────────────
 *
 * D3D11 submission was on the executor thread, and D3D's flush waits make
 * the title wait for the executor: every microsecond of driver overhead was
 * the title's too. So the executor now only snapshots a draw -- the draw
 * description, the indices rebased to zero, and the vertex bytes it reads
 * out of guest memory -- into a packet on a ring, and this thread does the
 * D3D11 work from the packets. Vertex-program and combiner state travel as
 * packets of their own, sent only when their change counters move.
 *
 * Every draw arrives here as an inline draw over its packet's copy of the
 * vertices, so do_draw is unchanged. Textures are still read from guest
 * memory when the draw is submitted, up to a ring's worth after it was
 * recorded; D3D's own fencing keeps a title from rewriting a texture the GPU
 * may still read, and that holds here for the same reason.
 *
 * RECOMP_GPU_THREAD=0 submits on the executor thread, as before. */

enum { PKT_WRAP = 1, PKT_DRAW, PKT_CLEAR, PKT_FLIP, PKT_VP_PROG, PKT_VP_CONST, PKT_COMB,
       PKT_ZPASS, PKT_REPORT };

typedef struct {
    uint32_t type, bytes;               /* bytes includes this header */
} PktHdr;

typedef struct {
    PktHdr   h;
    Nv2aHwDraw d;
    float    cur[16][4];
    uint32_t nidx, blob_bytes;
    /* uint32_t idx[nidx]; then the vertex blob, 16-aligned */
} PktDraw;

typedef struct {
    PktHdr   h;
    Nv2aHwDraw d;
    uint32_t flags, color, zclear;
} PktClear;

typedef struct {
    PktHdr   h;
    uint32_t color_va;
} PktFlip;

typedef struct {
    PktHdr   h;
    uint32_t a, b;      /* PKT_ZPASS: a = enable; PKT_REPORT: a, b = slot VAs */
} PktVis;

typedef struct {
    PktHdr   h;
    uint32_t program_start, program_gen, execution_mode;
    uint32_t program[NV2A_VP_SLOTS][4];
} PktVpProg;

typedef struct {
    PktHdr   h;
    uint32_t constants_gen;
    float    constants[NV2A_VP_CONSTANTS][4];
} PktVpConst;

typedef struct {
    PktHdr   h;
    Nv2aCombiner comb;
} PktComb;

#define RING_BYTES (64u << 20)
static uint8_t *s_ring;
static volatile LONG64 s_ring_head, s_ring_tail;   /* bytes ever written / read */
static volatile LONG s_consumer_idle;
static HANDLE s_ring_wake;
static int s_threaded;
/* Frames queued by the executor and presented by the render thread. */
static volatile LONG s_flips_queued, s_flips_done;

/* At most one presented frame per sixtieth of a second, on this thread's
 * clock: the display's pace, which the executor then follows through
 * nv2a_d3d11_frames_in_flight instead of sleeping itself. */
static void pace_display(void)
{
    static LARGE_INTEGER qf;
    static LONGLONG due;
    static int off = -1;
    LARGE_INTEGER now;

    if (off < 0) {
        off = getenv("RECOMP_FLIP_NOWAIT") != NULL;
        QueryPerformanceFrequency(&qf);
    }
    if (off)
        return;
    QueryPerformanceCounter(&now);
    {
        /* RECOMP_PACE_TRACE: every 600 flips, how many arrived late (the
         * frame was not ready when it was due), how many resynced, the
         * time spent waiting here, and the render thread's idle time --
         * whether the display or the producer sets the rate. */
        static int trace = -1;
        static unsigned n, late, resync;
        static LONGLONG t0, waited, entry;
        extern LONGLONG g_d3d11_idle_ticks;
        if (trace < 0)
            trace = getenv("RECOMP_PACE_TRACE") != NULL;
        if (trace) {
            if (due && now.QuadPart > due) late++;
            if (due && now.QuadPart - due >= qf.QuadPart / 30) {
                static LONGLONG start;
                if (!start) start = now.QuadPart;
                resync++;
                fprintf(stderr, "  [PACE] frame %.0f ms late at %.2f s (render idle so far %.0f ms;"
                        " %u VS + %u PS compiled, %u texture uploads)\n",
                        (double)(now.QuadPart - due) * 1000.0 / qf.QuadPart,
                        (double)(now.QuadPart - start) / qf.QuadPart,
                        (double)g_d3d11_idle_ticks * 1000.0 / qf.QuadPart,
                        s_st.vs_compiled, s_st.ps_compiled, s_st.tex_uploads);
            }
            entry = now.QuadPart;
        }
        while (due && now.QuadPart < due) {
            if (due - now.QuadPart > qf.QuadPart / 500)
                Sleep(1);
            else
                YieldProcessor();
            QueryPerformanceCounter(&now);
        }
        if (trace) {
            waited += now.QuadPart - entry;
            if (!t0) t0 = now.QuadPart;
            if (++n == 600) {
                double secs = (double)(now.QuadPart - t0) / (double)qf.QuadPart;
                fprintf(stderr, "  [PACE] 600 flips in %.2f s (%.1f/s): %u late, %u resynced, "
                        "%.1f ms/flip pacing wait, %.1f ms/flip render idle\n",
                        secs, 600.0 / secs, late, resync,
                        (double)waited * 1000.0 / qf.QuadPart / 600.0,
                        (double)g_d3d11_idle_ticks * 1000.0 / qf.QuadPart / 600.0);
                n = late = resync = 0;
                waited = 0;
                g_d3d11_idle_ticks = 0;
                t0 = now.QuadPart;
            }
        }
    }
    /* Keep the 60 Hz schedule through a late frame; resync after a stall. */
    due = (due && now.QuadPart - due < qf.QuadPart / 30) ? due + qf.QuadPart / 60
                                                       : now.QuadPart + qf.QuadPart / 60;
}

LONGLONG g_d3d11_idle_ticks;

int nv2a_d3d11_frames_in_flight(void)
{
    if (!s_threaded)
        return -1;
    return (int)(s_flips_queued - s_flips_done);
}

/* Render-side copies of the state the packets carry. */
static Nv2aVpState  s_rvp;
static Nv2aCombiner s_rcomb;

static uint8_t *ring_reserve(uint32_t bytes)
{
    for (;;) {
        LONG64 head = s_ring_head, tail = s_ring_tail;
        uint32_t pos = (uint32_t)(head % RING_BYTES);
        uint32_t need = bytes;

        if (pos + bytes > RING_BYTES)
            need += RING_BYTES - pos;        /* the wrap, then the packet */
        if ((LONG64)RING_BYTES - (head - tail) >= (LONG64)need) {
            if (pos + bytes > RING_BYTES) {
                if (RING_BYTES - pos >= sizeof(PktHdr)) {
                    PktHdr *w = (PktHdr *)(s_ring + pos);
                    w->type = PKT_WRAP;
                    w->bytes = RING_BYTES - pos;
                }
                MemoryBarrier();
                InterlockedExchange64(&s_ring_head, head + (RING_BYTES - pos));
                continue;
            }
            return s_ring + pos;
        }
        SetEvent(s_ring_wake);                /* full: make sure it drains */
        SwitchToThread();
    }
}

static void ring_commit(uint32_t bytes)
{
    MemoryBarrier();
    InterlockedExchange64(&s_ring_head, s_ring_head + bytes);
    if (s_consumer_idle)
        SetEvent(s_ring_wake);
}

static uint32_t align16(uint32_t n)
{
    return (n + 15u) & ~15u;
}

static void send_state(const Nv2aHwDraw *d)
{
    static const Nv2aVpState *vp_owner;
    static uint32_t prog_gen = 0xFFFFFFFFu, prog_start = 0xFFFFFFFFu, const_gen = 0xFFFFFFFFu;
    static const Nv2aCombiner *comb_owner;
    static uint32_t comb_gen = 0xFFFFFFFFu;

    if (d->vp && (d->vp != vp_owner || d->vp->program_gen != prog_gen
                  || d->vp->program_start != prog_start)) {
        PktVpProg *p = (PktVpProg *)ring_reserve(align16(sizeof *p));
        p->h.type = PKT_VP_PROG;
        p->h.bytes = align16(sizeof *p);
        p->program_start = d->vp->program_start;
        p->program_gen = d->vp->program_gen;
        p->execution_mode = d->vp->execution_mode;
        memcpy(p->program, d->vp->program, sizeof p->program);
        ring_commit(p->h.bytes);
        prog_gen = d->vp->program_gen;
        prog_start = d->vp->program_start;
    }
    if (d->vp && (d->vp != vp_owner || d->vp->constants_gen != const_gen)) {
        PktVpConst *p = (PktVpConst *)ring_reserve(align16(sizeof *p));
        p->h.type = PKT_VP_CONST;
        p->h.bytes = align16(sizeof *p);
        p->constants_gen = d->vp->constants_gen;
        memcpy(p->constants, d->vp->constants, sizeof p->constants);
        ring_commit(p->h.bytes);
        const_gen = d->vp->constants_gen;
    }
    vp_owner = d->vp;
    if (d->comb && (d->comb != comb_owner || d->comb->gen != comb_gen)) {
        PktComb *p = (PktComb *)ring_reserve(align16(sizeof *p));
        p->h.type = PKT_COMB;
        p->h.bytes = align16(sizeof *p);
        p->comb = *d->comb;
        ring_commit(p->h.bytes);
        comb_owner = d->comb;
        comb_gen = d->comb->gen;
    }
}

/* Snapshot a draw: which bytes of which streams it reads, copied, with the
 * indices rebased so the copy starts at index 0. */
static void queue_draw(const Nv2aHwDraw *d)
{
    const uint8_t *mem = d->inline_base ? d->inline_base : guest();
    Stream st[16];
    uint32_t stream_of[16], nstreams = 0, lo = 0xFFFFFFFFu, hi = 0, i, a, count;
    uint32_t blob = 0, blob_off[16], span[16], src[16], total;
    PktDraw *p;
    uint32_t *idx;
    uint8_t *vb;

    if (d->idx_count < 3 || !tri_indices(d->prim, d->idx_count)) {
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
    for (a = 0; a < 16; a++) {
        const Nv2aHwAttr *at = &d->attr[a];
        uint32_t bytes = 0, mode = 0, s;

        stream_of[a] = 0xFFFFFFFFu;
        if (!at->size || !at->stride || (!d->inline_base && !at->offset))
            continue;
        if (attr_format(at, &bytes, &mode) == DXGI_FORMAT_UNKNOWN)
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
        stream_of[a] = s;
    }
    for (i = 0; i < nstreams; i++) {
        span[i] = (count - 1) * st[i].stride + st[i].end;
        src[i] = st[i].base + lo * st[i].stride;
        if (d->inline_base && src[i] + span[i] > d->inline_bytes)
            span[i] = src[i] >= d->inline_bytes ? 0 : d->inline_bytes - src[i];
        if (!d->inline_base && !guest_ok(src[i], span[i])) {
            s_st.skipped++;
            return;
        }
        blob_off[i] = blob;
        blob += align16(span[i] + 16);       /* the draw path reads a little past */
    }
    total = align16(sizeof(PktDraw) + d->idx_count * 4) + blob;
    if (total > RING_BYTES / 4) {
        s_st.skipped++;
        return;
    }
    send_state(d);

    p = (PktDraw *)ring_reserve(total);
    p->h.type = PKT_DRAW;
    p->h.bytes = total;
    p->d = *d;
    p->nidx = d->idx_count;
    p->blob_bytes = blob;
    if (d->cur_attr)
        memcpy(p->cur, d->cur_attr, sizeof p->cur);
    idx = (uint32_t *)(p + 1);
    for (i = 0; i < d->idx_count; i++)
        idx[i] = d->idx[i] - lo;
    vb = (uint8_t *)p + align16(sizeof(PktDraw) + d->idx_count * 4);
    for (i = 0; i < nstreams; i++) {
        memcpy(vb + blob_off[i], mem + src[i], span[i]);
        memset(vb + blob_off[i] + span[i], 0, 16);
    }
    /* Every attribute now lives in the blob; the rest are absent. */
    for (a = 0; a < 16; a++) {
        if (stream_of[a] == 0xFFFFFFFFu) {
            p->d.attr[a].size = 0;
            continue;
        }
        p->d.attr[a].offset = blob_off[stream_of[a]]
                            + (d->attr[a].offset - st[stream_of[a]].base);
    }
    p->d.inline_bytes = blob;
    ring_commit(total);
}

/* ── Visibility tests ─────────────────────────────────────────────────────
 *
 * D3D's BeginVisibilityTest pushes CLEAR_REPORT_VALUE and
 * SET_ZPASS_PIXEL_COUNT_ENABLE 1; EndVisibilityTest pushes the enable 0 and
 * GET_REPORT (0x17D0: type 1 in bits 24-27, the report's offset below),
 * having set the 16-byte slot's last word to 0xFFFFFFFF. The NV2A then
 * writes { 64-bit timestamp, pixels that passed the depth test, 0 }, and
 * GetVisibilityTestResult reads it. Amped 2 draws each floodlight's glow and
 * lens flare only when its test says the lamp is in view; with the method
 * ignored the slot never completed, and the flares were drawn through hills,
 * trees and the rider. Here each test is a D3D11 occlusion query, read back
 * without a stall once the GPU has it and written to the slot then. */
typedef struct {
    ID3D11Query *q;
    uint32_t va, va_alt;   /* the slot, or the other reading of the offset */
    int state;             /* 0 free, 1 counting, 2 ended, 3 reported */
} VisQuery;
#define VIS_SLOTS 256
static VisQuery s_vis[VIS_SLOTS];
static VisQuery *s_vis_open, *s_vis_last;
static int s_vis_pending;

/* The last count for each slot, by address: what a report says at once,
 * before its own query has come back (see nv2a_d3d11_get_report). */
#define VIS_CACHE 1024
static volatile uint32_t s_vis_cache_va[VIS_CACHE], s_vis_cache_n[VIS_CACHE];

static uint32_t vis_cache_slot(uint32_t va)
{
    return ((va >> 4) * 2654435761u) & (VIS_CACHE - 1);
}

static void vis_write(VisQuery *v, UINT64 samples)
{
    uint8_t *g = (uint8_t *)guest();
    uint32_t va = v->va, sc = (uint32_t)(gpu_scale() * gpu_scale());
    if (!guest_ok(va, 16))
        return;
    {
        LARGE_INTEGER now;
        uint32_t *s = (uint32_t *)(g + va), k = vis_cache_slot(va);
        QueryPerformanceCounter(&now);
        s[0] = (uint32_t)now.QuadPart;
        s[1] = (uint32_t)(now.QuadPart >> 32);
        s[2] = (uint32_t)(samples / (sc ? sc : 1));
        MemoryBarrier();
        s[3] = 0;                                  /* done */
        s_vis_cache_va[k] = va;
        s_vis_cache_n[k] = s[2];
        {   /* RECOMP_VIS_TRACE: the first few, then a tally every 2000 */
            static int said, trace = -1;
            static unsigned n, nz, maxv;
            if (trace < 0)
                trace = getenv("RECOMP_VIS_TRACE") != NULL;
            if (trace) {
                if (said < 6) {
                    said++;
                    fprintf(stderr, "  [D3D11] visibility report at %08X: %u pixels\n",
                            va, s[2]);
                }
                n++;
                if (s[2]) nz++;
                if (s[2] > maxv) maxv = s[2];
                if (n == 2000) {
                    fprintf(stderr, "  [D3D11] visibility: %u of 2000 reports saw pixels (max %u)\n",
                            nz, maxv);
                    n = nz = maxv = 0;
                }
            }
        }
    }
}

static void vis_poll(void)
{
    int i;
    if (!s_vis_pending)
        return;
    for (i = 0; i < VIS_SLOTS; i++) {
        VisQuery *v = &s_vis[i];
        UINT64 n;
        if (v->state < 2)
            continue;
        if (ID3D11DeviceContext_GetData(s_ctx, (ID3D11Asynchronous *)v->q, &n, sizeof n,
                                        D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
            continue;
        if (v->state == 3)
            vis_write(v, n);
        v->state = 0;
        s_vis_pending--;
    }
}

static void do_zpass(int on)
{
    int i;
    if (s_vis_open) {                              /* close a test left open */
        ID3D11DeviceContext_End(s_ctx, (ID3D11Asynchronous *)s_vis_open->q);
        s_vis_open->state = 2;
        s_vis_pending++;
        s_vis_last = s_vis_open;
        s_vis_open = NULL;
        s_vis_in_test = 0;
    }
    if (!on)
        return;
    vis_poll();
    for (i = 0; i < VIS_SLOTS && s_vis[i].state; i++)
        ;
    if (i == VIS_SLOTS) {
        static int said;
        if (!said++)
            fprintf(stderr, "  [D3D11] more than %d visibility tests in flight\n", VIS_SLOTS);
        return;
    }
    if (!s_vis[i].q) {
        D3D11_QUERY_DESC qd = { D3D11_QUERY_OCCLUSION, 0 };
        if (FAILED(ID3D11Device_CreateQuery(s_dev, &qd, &s_vis[i].q)))
            return;
    }
    ID3D11DeviceContext_Begin(s_ctx, (ID3D11Asynchronous *)s_vis[i].q);
    s_vis[i].state = 1;
    s_vis_in_test = 1;
    s_vis_open = &s_vis[i];
    s_vis_last = NULL;
}

static void do_get_report(uint32_t va, uint32_t va_alt)
{
    VisQuery *v = s_vis_last;
    s_vis_last = NULL;
    if (!v || v->state != 2) {
        /* A report with no test behind it: say nothing passed. */
        VisQuery tmp = { NULL, va, va_alt, 3 };
        vis_write(&tmp, 0);
        return;
    }
    v->va = va;
    v->va_alt = va_alt;
    v->state = 3;
}

static DWORD WINAPI render_thread(LPVOID unused)
{
    (void)unused;
    xbox_PinHostThread();
    for (;;) {
        LONG64 tail = s_ring_tail;
        PktHdr *h;

        if (tail == s_ring_head) {
            LARGE_INTEGER a, b;
            if (s_vis_pending) {         /* a title may be waiting on one */
                ID3D11DeviceContext_Flush(s_ctx);
                vis_poll();
            }
            present_overlay_tick();      /* a movie playing over no flips */
            QueryPerformanceCounter(&a);
            InterlockedExchange(&s_consumer_idle, 1);
            if (tail == s_ring_head)
                WaitForSingleObject(s_ring_wake, 2);
            InterlockedExchange(&s_consumer_idle, 0);
            QueryPerformanceCounter(&b);
            g_d3d11_idle_ticks += b.QuadPart - a.QuadPart;
            continue;
        }
        MemoryBarrier();
        h = (PktHdr *)(s_ring + (uint32_t)(tail % RING_BYTES));
        switch (h->type) {
        case PKT_WRAP:
            break;
        case PKT_VP_PROG: {
            const PktVpProg *p = (const PktVpProg *)h;
            s_rvp.program_start = p->program_start;
            s_rvp.program_gen = p->program_gen;
            s_rvp.execution_mode = p->execution_mode;
            memcpy(s_rvp.program, p->program, sizeof s_rvp.program);
            break;
        }
        case PKT_VP_CONST: {
            const PktVpConst *p = (const PktVpConst *)h;
            s_rvp.constants_gen = p->constants_gen;
            memcpy(s_rvp.constants, p->constants, sizeof s_rvp.constants);
            break;
        }
        case PKT_COMB:
            s_rcomb = ((const PktComb *)h)->comb;
            break;
        case PKT_DRAW: {
            PktDraw *p = (PktDraw *)h;
            p->d.idx = (const uint32_t *)(p + 1);
            p->d.idx_count = p->nidx;
            p->d.inline_base = (const uint8_t *)p + align16(sizeof(PktDraw) + p->nidx * 4);
            p->d.cur_attr = (const float (*)[4])p->cur;
            p->d.vp = &s_rvp;
            p->d.comb = &s_rcomb;
            vis_trace_draw(&p->d, "received");
            do_draw(&p->d);
            break;
        }
        case PKT_CLEAR: {
            const PktClear *p = (const PktClear *)h;
            do_clear(&p->d, p->flags, p->color, p->zclear);
            break;
        }
        case PKT_FLIP:
            pace_display();
            do_flip(((const PktFlip *)h)->color_va);
            InterlockedIncrement(&s_flips_done);
            vis_poll();
            break;
        case PKT_ZPASS:
            do_zpass((int)((const PktVis *)h)->a);
            break;
        case PKT_REPORT:
            do_get_report(((const PktVis *)h)->a, ((const PktVis *)h)->b);
            break;
        default:
            break;
        }
        MemoryBarrier();
        InterlockedExchange64(&s_ring_tail, tail + h->bytes);
    }
    return 0;
}

static void queue_start(void)
{
    const char *e = getenv("RECOMP_GPU_THREAD");
    HANDLE th;

    if (e && *e == '0')
        return;
    s_ring = (uint8_t *)VirtualAlloc(NULL, RING_BYTES, MEM_COMMIT | MEM_RESERVE,
                                     PAGE_READWRITE);
    s_ring_wake = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (!s_ring || !s_ring_wake)
        return;
    th = CreateThread(NULL, 0, render_thread, NULL, 0, NULL);
    if (!th)
        return;
    CloseHandle(th);
    s_threaded = 1;
    fprintf(stderr, "  [D3D11] submitting on a render thread (%u MB command ring)\n",
            RING_BYTES >> 20);
}

void nv2a_d3d11_draw(const Nv2aHwDraw *d)
{
    if (!s_threaded) {
        do_draw(d);
        return;
    }
    queue_draw(d);
}

void nv2a_d3d11_clear(const Nv2aHwDraw *d, uint32_t flags, uint32_t color,
                      uint32_t zclear)
{
    PktClear *p;

    if (!s_threaded) {
        do_clear(d, flags, color, zclear);
        return;
    }
    p = (PktClear *)ring_reserve(align16(sizeof *p));
    p->h.type = PKT_CLEAR;
    p->h.bytes = align16(sizeof *p);
    p->d = *d;
    p->flags = flags;
    p->color = color;
    p->zclear = zclear;
    ring_commit(p->h.bytes);
}

void nv2a_d3d11_flip(uint32_t color_va)
{
    PktFlip *p;

    if (!s_threaded) {
        do_flip(color_va);
        vis_poll();
        return;
    }
    InterlockedIncrement(&s_flips_queued);
    p = (PktFlip *)ring_reserve(align16(sizeof *p));
    p->h.type = PKT_FLIP;
    p->h.bytes = align16(sizeof *p);
    p->color_va = color_va;
    ring_commit(p->h.bytes);
}

static void queue_vis(uint32_t type, uint32_t a, uint32_t b)
{
    PktVis *p = (PktVis *)ring_reserve(align16(sizeof *p));
    p->h.type = type;
    p->h.bytes = align16(sizeof *p);
    p->a = a;
    p->b = b;
    ring_commit(p->h.bytes);
}

void nv2a_d3d11_zpass(int enable)
{
    if (!s_threaded) {
        do_zpass(enable);
        return;
    }
    queue_vis(PKT_ZPASS, (uint32_t)enable, 0);
}

void nv2a_d3d11_get_report(uint32_t va, uint32_t va_alt)
{
    uint8_t *g = (uint8_t *)guest();
    uint32_t slot = 0;

    /* The slot is the reading D3D marked pending (last word 0xFFFFFFFF):
     * the offset's bit 24 is shared with the report type. */
    if (guest_ok(va, 16) && *(volatile uint32_t *)(g + va + 12) == 0xFFFFFFFFu)
        slot = va;
    else if (guest_ok(va_alt, 16) && *(volatile uint32_t *)(g + va_alt + 12) == 0xFFFFFFFFu)
        slot = va_alt;
    /* Nothing marked pending: not a report D3D asked for -- a misread
     * stream, say. Write nowhere rather than 16 bytes at a guess. */
    if (!slot)
        return;
    /* Complete it now with the slot's last count. The query behind it comes
     * back frames later -- the render thread and the GPU run behind the
     * walk -- and the title reads its results a frame on: a slot still
     * pending then counts as not visible, so every floodlight's glow went
     * out. The real count replaces this when it arrives. */
    {
        uint32_t *s = (uint32_t *)(g + slot), k = vis_cache_slot(slot);
        s[0] = s[1] = 0;
        s[2] = s_vis_cache_va[k] == slot ? s_vis_cache_n[k] : 0;
        MemoryBarrier();
        s[3] = 0;
    }
    if (!s_threaded) {
        do_get_report(slot, slot);
        vis_poll();
        return;
    }
    queue_vis(PKT_REPORT, slot, slot);
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
            " %u colour + %u depth targets; %u frames read back, %u CPU-written frames shown;"
            " %u programs writing constants, %u unaligned attributes\n",
            s_st.draws, s_st.tris, s_st.skipped, s_st.points_lines,
            s_st.vs_compiled, s_st.ps_compiled, s_st.compile_failed,
            s_st.tex_uploads, s_st.tex_bad, s_st.rt_created, s_st.ds_created,
            s_st.readbacks, s_st.cpu_frames, s_st.vs_const_writes, s_st.fallback_attrs);
    if (s_pack_on > 0)
        fprintf(stderr, "  [D3D11] texture pack: %u replaced, %u not in the pack\n",
                s_pack_loaded, s_pack_misses);
    fflush(stderr);
}

#else  /* !_WIN32 */

int  nv2a_d3d11_active(void) { return 0; }
void nv2a_d3d11_draw(const Nv2aHwDraw *d) { (void)d; }
void nv2a_d3d11_clear(const Nv2aHwDraw *d, uint32_t f, uint32_t c, uint32_t z)
{ (void)d; (void)f; (void)c; (void)z; }
void nv2a_d3d11_flip(uint32_t va) { (void)va; }
int  nv2a_d3d11_frames_in_flight(void) { return -1; }
void nv2a_d3d11_report(void) {}
void nv2a_d3d11_zpass(int enable) { (void)enable; }
void nv2a_d3d11_get_report(uint32_t va, uint32_t va_alt) { (void)va; (void)va_alt; }

#endif
