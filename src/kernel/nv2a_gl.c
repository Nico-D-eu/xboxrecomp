/**
 * nv2a_gl.c - the pushbuffer executor's OpenGL 3.3 back end, for the hosts
 * where nv2a_d3d11.c has no Direct3D: Linux now, macOS later. nv2a_d3d11.c
 * forwards its entry points here there, so the executor is unchanged.
 *
 * Stage 1, what this file holds so far:
 *   - a context of its own, made on the executor's thread (the window's is
 *     on the main thread; the two share nothing);
 *   - colour surfaces as textures and depth/stencil surfaces as
 *     renderbuffers, keyed by guest address like nv2a_d3d11.c's;
 *   - clears;
 *   - the flip, which reads the frame back and hands it to the window
 *     (xbox_FramebufferWindowPresentArgb).
 * Draws are counted and dropped: they come next, with nv2a_glsl_* for the
 * shaders. Off unless RECOMP_GPU=gl.
 *
 * Rows: the GLSL shaders flip y, so row 0 of every target is the top, as in
 * guest memory. Scissor rectangles take the guest's y unchanged and
 * glReadPixels returns the top row first.
 *
 * Under WSLg, GALLIUM_DRIVER=d3d12 puts GL on the GPU; Mesa's default there
 * is llvmpipe, on the CPU.
 */
#if !defined(_WIN32)

#include "nv2a_d3d11.h"
#include "nv2a_hlsl.h"
#include "../d3d/d3d8_swizzle.h"

#include <SDL.h>
#include <epoxy/gl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void xbox_FramebufferWindowPresentArgb(const void *px, uint32_t w,
                                              uint32_t h, uint32_t row_pitch);
extern ptrdiff_t xbox_GetMemoryOffset(void);
extern uint32_t g_xbox_contig_size, g_xbox_ram_top;   /* kernel.h: 64 or 128 MB */
/* nv2a_pb_exec.c: a texture level as A8R8G8B8, and the size of a texel --
 * the software sampler's decoder, so the paths agree on every format. */
extern int nv2a_pb_decode_level(uint32_t color, const uint8_t *src, uint32_t w,
                                uint32_t h, uint32_t pitch, uint32_t *out);
extern uint32_t nv2a_pb_texel_bytes(uint32_t color);

#define GL_SURFACES 16

typedef struct {
    uint32_t va, w, h;
    GLuint   obj;            /* texture (colour) or renderbuffer (depth) */
    uint32_t used;           /* LRU tick */
    uint32_t drawn;          /* last frame drawn or cleared into */
} GlSurface;

static int        s_state;   /* 0 not tried, 1 up, -1 off or failed */
static pthread_t  s_thread;
static SDL_Window *s_win;
static SDL_GLContext s_ctx;
static GLuint     s_fbo;
static GlSurface  s_color[GL_SURFACES], s_depth[GL_SURFACES];
/* What s_fbo has attached (bind_targets). Invalid after anything that may
 * have changed it: a surface made or deleted (GL reuses names), the flip. */
static int        s_att_valid, s_att_complete;
static GLuint     s_att_color, s_att_depth;
static uint32_t   s_vp_w, s_vp_h;
static uint32_t   s_tick;
static uint64_t   s_draws, s_clears, s_flips, s_other_thread;
static uint32_t   s_frame;   /* flips so far */
/* Where the time goes, for the [GL] report: nanoseconds in draws (all of
 * it), in texture checks and uploads (part of the draws), in flips. */
static uint64_t   s_ns_draw, s_ns_tex, s_ns_flip, s_rep_flips;
static uint64_t   s_rep_draw, s_rep_tex, s_rep_flip;
/* RECOMP_GL_PROFILE=1: the draw split into sections (prepare and program,
 * targets, vertices, indices, uniforms, textures, state, the draw call). */
static uint64_t   s_ns_part[8], s_rep_part[8];
static int        s_profile = -1;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static uint8_t   *s_readback;
static size_t     s_readback_cap;

/* A surface's size, as nv2a_d3d11.c works it out (there inside its _WIN32
 * block): swizzled surfaces carry their size in the format, linear ones
 * have the pitch and the clip rectangle. Shared code once the renderer is
 * split (PLAN Phase 12). */
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

static int draw_init(void);

static int init(void)
{
    const char *e = getenv("RECOMP_GPU");

    if (!e || strcmp(e, "gl") != 0)
        return 0;
    if (!SDL_WasInit(SDL_INIT_VIDEO) && SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "  [GL] no SDL video (%s); software rasteriser\n", SDL_GetError());
        return 0;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    /* Hidden: it only owns the context. Everything is drawn into
     * framebuffer objects and reaches the screen through the window. */
    s_win = SDL_CreateWindow("nv2a_gl", 0, 0, 16, 16, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    s_ctx = s_win ? SDL_GL_CreateContext(s_win) : NULL;
    if (!s_ctx || SDL_GL_MakeCurrent(s_win, s_ctx) != 0) {
        fprintf(stderr, "  [GL] no OpenGL 3.3 core context (%s); software rasteriser\n",
                SDL_GetError());
        if (s_win)
            SDL_DestroyWindow(s_win);
        s_win = NULL;
        return 0;
    }
    glGenFramebuffers(1, &s_fbo);
    s_thread = pthread_self();
    draw_init();
    fprintf(stderr, "  [GL] back end up: %s, OpenGL %s\n",
            (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));
    return 1;
}

int nv2a_gl_active(void)
{
    if (!s_state)
        s_state = init() ? 1 : -1;
    return s_state == 1;
}

/* The context belongs to the thread that made it. The executor makes every
 * call from one thread; anything else is counted and skipped. */
static int on_thread(void)
{
    if (s_state == 1 && pthread_equal(pthread_self(), s_thread))
        return 1;
    s_other_thread++;
    return 0;
}

/* The surface at va, (re)made at w x h. Least recently used goes first. */
static GlSurface *surface(GlSurface *tab, uint32_t va, uint32_t w, uint32_t h, int color)
{
    GlSurface *e = NULL, *lru = &tab[0];
    int i;

    for (i = 0; i < GL_SURFACES; i++) {
        if (tab[i].obj && tab[i].va == va) {
            e = &tab[i];
            break;
        }
        if (tab[i].used < lru->used)
            lru = &tab[i];
    }
    if (e && e->w == w && e->h == h) {
        e->used = ++s_tick;
        return e;
    }
    if (!e) {
        for (i = 0; i < GL_SURFACES && !e; i++)
            if (!tab[i].obj)
                e = &tab[i];
        if (!e)
            e = lru;
    }
    s_att_valid = 0;
    if (e->obj) {
        if (color)
            glDeleteTextures(1, &e->obj);
        else
            glDeleteRenderbuffers(1, &e->obj);
        e->obj = 0;
    }
    if (color) {
        glGenTextures(1, &e->obj);
        glBindTexture(GL_TEXTURE_2D, e->obj);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)w, (GLsizei)h, 0,
                     GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        /* One level: complete under any sampler, mipmapped ones too. */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    } else {
        glGenRenderbuffers(1, &e->obj);
        glBindRenderbuffer(GL_RENDERBUFFER, e->obj);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)w, (GLsizei)h);
    }
    e->va = va;
    e->w = w;
    e->h = h;
    e->used = ++s_tick;
    return e;
}

static GlSurface *surface_find(GlSurface *tab, uint32_t va)
{
    int i;
    for (i = 0; i < GL_SURFACES; i++)
        if (tab[i].obj && tab[i].va == va)
            return &tab[i];
    return NULL;
}

/* Attach what a draw or clear writes; the size it has, in *w, *h. */
static int bind_targets(const Nv2aHwDraw *d, int want_color, int want_depth,
                        uint32_t *w, uint32_t *h)
{
    GlSurface *rt = NULL, *ds = NULL;

    *w = *h = 0;
    want_color = want_color && d->color_va;
    want_depth = want_depth && d->zeta_va;
    if (want_color) {
        surface_dims(d, d->pitch, color_bpp(d->format), w, h);
        rt = surface(s_color, d->color_va, *w, *h, 1);
    } else if (want_depth) {
        GlSurface *c = d->color_va ? surface_find(s_color, d->color_va) : NULL;
        if (c) {
            *w = c->w;
            *h = c->h;
        } else {
            surface_dims(d, d->zpitch, 4, w, h);
        }
    }
    if (want_depth && *w)
        ds = surface(s_depth, d->zeta_va, *w, *h, 0);
    if (!rt && !ds)
        return 0;
    /* Attach only when the targets change: re-attaching for every draw
     * revalidates the framebuffer each time. */
    if (!s_att_valid || s_att_color != (rt ? rt->obj : 0) || s_att_depth != (ds ? ds->obj : 0)) {
        glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               rt ? rt->obj : 0, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, ds ? ds->obj : 0);
        glDrawBuffer(rt ? GL_COLOR_ATTACHMENT0 : GL_NONE);
        s_att_color = rt ? rt->obj : 0;
        s_att_depth = ds ? ds->obj : 0;
        s_att_complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        s_att_valid = 1;
    }
    if (!s_att_complete)
        return 0;
    if (s_vp_w != *w || s_vp_h != *h) {
        glViewport(0, 0, (GLsizei)*w, (GLsizei)*h);
        s_vp_w = *w;
        s_vp_h = *h;
    }
    return 1;
}

void nv2a_gl_clear(const Nv2aHwDraw *d, uint32_t flags, uint32_t color, uint32_t zclear)
{
    GLbitfield bits = 0;
    uint32_t w, h;

    if (!on_thread())
        return;
    s_clears++;
    if (!bind_targets(d, (flags & 0xF0u) != 0, (flags & 0x03u) != 0, &w, &h))
        return;
    glEnable(GL_SCISSOR_TEST);
    glScissor((GLint)d->clip_x, (GLint)d->clip_y, (GLsizei)d->clip_w, (GLsizei)d->clip_h);
    /* A clear writes whatever the flags name, whatever the masks. */
    if ((flags & 0xF0u) && d->color_va) {
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glClearColor((float)((color >> 16) & 0xFFu) / 255.0f, (float)((color >> 8) & 0xFFu) / 255.0f,
                     (float)(color & 0xFFu) / 255.0f, (float)(color >> 24) / 255.0f);
        bits |= GL_COLOR_BUFFER_BIT;
    }
    if ((flags & 0x01u) && d->zeta_va) {
        glDepthMask(GL_TRUE);
        glClearDepth(((d->format >> 4) & 0xFu) == 1u ? (double)(zclear & 0xFFFFu) / 65535.0
                                                     : (double)(zclear >> 8) / 16777215.0);
        bits |= GL_DEPTH_BUFFER_BIT;
    }
    if ((flags & 0x02u) && d->zeta_va) {
        glStencilMask(0xFFu);
        glClearStencil((GLint)(zclear & 0xFFu));
        bits |= GL_STENCIL_BUFFER_BIT;
    }
    if (bits)
        glClear(bits);
    if ((flags & 0xF0u) && d->color_va) {
        GlSurface *rt = surface_find(s_color, d->color_va);
        if (rt)
            rt->drawn = s_frame;
    }
    glDisable(GL_SCISSOR_TEST);
}

/* ── Draws ───────────────────────────────────────────────────────────────
 *
 * The same steps as nv2a_d3d11.c's do_draw, in GL: shaders from
 * nv2a_glsl_*, vertex streams copied whole over the index range and read
 * as raw integers (the vertex shader decodes the formats), primitives
 * expanded to a triangle list, textures decoded with the software path's
 * decoder. NV2A's state values are GL's own enums (compare functions,
 * stencil ops, blend factors and equations), so they go to GL unchanged.
 *
 * Not yet: volume textures, shadow maps (depth textures read as passing),
 * CPU-written pixels under a surface, texture packs, visibility counts,
 * anisotropy, render scale. */

static const uint8_t *guest(void)
{
    return (const uint8_t *)xbox_GetMemoryOffset();
}

/* nv2a_d3d11.c's guard: low RAM, the contiguous window or its tiled alias.
 * A misread command can point a vertex array anywhere. */
static int guest_ok(uint32_t va, uint32_t bytes)
{
    uint64_t end = (uint64_t)va + bytes;
    if (end <= g_xbox_ram_top)
        return 1;
    if (va >= 0x80000000u && end <= 0x80000000ull + g_xbox_contig_size)
        return 1;
    return va >= 0xF0000000u && end <= 0xF0000000ull + g_xbox_contig_size;
}

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

/* 64 spans of 64 bytes: has a large texture been rewritten? */
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

/* ── Shaders: one GL program per (vertex program, combiner setup) ── */

#define PROG_SLOTS 4096
#define SHADER_SLOTS 2048

typedef struct { uint64_t key; GLuint obj; } GlShader;
typedef struct { uint64_t vs, ps; GLuint prog; } GlProgram;

static GlShader  s_vs_cache[SHADER_SLOTS], s_ps_cache[SHADER_SLOTS];
static GlProgram s_prog[PROG_SLOTS];
static GLuint    s_vao;
static const Nv2aVpState *s_vsc_owner;

/* Streaming rings for vertices, indices and uniform blocks: each draw's data
 * goes at the next free offset of one large buffer.
 *
 * With GL 4.4 (buffer storage) the buffer is mapped once, persistently and
 * coherently, and written directly. The ring is in RING_PARTS parts; leaving
 * a part fences it, and entering one waits for its fence, so nothing the GPU
 * may still read is overwritten. Elsewhere (macOS's 4.1) each write maps its
 * range unsynchronised and a full ring is orphaned.
 *
 * Measured in Amped 2's ML1 under Mesa's d3d12 driver: a whole buffer
 * re-specified per draw, or a map and unmap per draw, both cost 60-120 us a
 * draw -- 150 ms a frame at about 830 draws. */
#define RING_PARTS 4

typedef struct {
    GLuint   buf;
    GLenum   target;
    uint32_t size, pos;
    uint32_t epoch;          /* wraps so far: earlier ranges are gone */
    uint8_t *map;            /* persistent mapping, or NULL */
    GLsync   fence[RING_PARTS];
} GlRing;

static GlRing   s_vring, s_iring, s_uring;
static GLint    s_ubo_align = 256;
static uint32_t s_vsc_epoch;
static int      s_persistent;

static void ring_init(GlRing *r, GLenum target, uint32_t size)
{
    glGenBuffers(1, &r->buf);
    r->target = target;
    r->size = size;
    r->pos = 0;
    glBindBuffer(target, r->buf);
    if (s_persistent) {
        GLbitfield f = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        glBufferStorage(target, size, NULL, f);
        r->map = (uint8_t *)glMapBufferRange(target, 0, size, f);
        if (r->map)
            return;
        /* No mapping after all: plain storage can't be re-specified, so
         * start over with an ordinary buffer. */
        glDeleteBuffers(1, &r->buf);
        glGenBuffers(1, &r->buf);
        glBindBuffer(target, r->buf);
    }
    glBufferData(target, size, NULL, GL_STREAM_DRAW);
}

/* Persistent rings: the GPU is done with part k before it is written. */
static void ring_enter(GlRing *r, uint32_t k)
{
    if (r->fence[k]) {
        glClientWaitSync(r->fence[k], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
        glDeleteSync(r->fence[k]);
        r->fence[k] = 0;
    }
}

static void ring_leave(GlRing *r, uint32_t k)
{
    if (r->fence[k])
        glDeleteSync(r->fence[k]);
    r->fence[k] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

/* Move the write position to p, fencing and waiting at part boundaries. */
static void ring_move(GlRing *r, uint32_t from, uint32_t to)
{
    uint32_t part = r->size / RING_PARTS, a = from / part, b;

    if (!r->map)
        return;
    if (to < from) {                       /* wrapped */
        for (b = a; b < RING_PARTS; b++)
            ring_leave(r, b);
        a = 0;
        ring_enter(r, 0);
    }
    for (b = a + 1; b <= to / part && b < RING_PARTS; b++) {
        ring_leave(r, b - 1);
        ring_enter(r, b);
    }
}

/* Room for n more bytes from here, wrapping (or orphaning) if not. */
static void ring_reserve(GlRing *r, uint32_t n)
{
    if (r->pos + n <= r->size)
        return;
    if (r->map) {
        ring_move(r, r->pos, 0);
    } else {
        glBindBuffer(r->target, r->buf);
        glBufferData(r->target, r->size, NULL, GL_STREAM_DRAW);
    }
    r->pos = 0;
    r->epoch++;
}

/* n bytes from src at the next offset aligned to align (*at). */
static int ring_put(GlRing *r, const void *src, uint32_t n, uint32_t align, uint32_t *at)
{
    uint32_t p;

    if (n + align > r->size)
        return 0;
    ring_reserve(r, n + align);
    p = (r->pos + align - 1) / align * align;
    if (r->map) {
        ring_move(r, r->pos, p + n);
        memcpy(r->map + p, src, n);
    } else {
        void *m;
        glBindBuffer(r->target, r->buf);
        m = glMapBufferRange(r->target, p, n, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT
                                              | GL_MAP_INVALIDATE_RANGE_BIT);
        if (!m)
            return 0;
        memcpy(m, src, n);
        glUnmapBuffer(r->target);
    }
    r->pos = p + n;
    *at = p;
    return 1;
}
static uint32_t  s_vsc_gen;
static uint64_t  s_skipped, s_compile_fail, s_tex_uploads;

static GLuint compile(GLenum type, const char *src, size_t len)
{
    GLuint sh = glCreateShader(type);
    GLint ok = 0, n = (GLint)len;

    glShaderSource(sh, 1, &src, &n);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        if (s_compile_fail++ < 4)
            fprintf(stderr, "  [GL] %s shader failed to compile: %s\n",
                    type == GL_VERTEX_SHADER ? "vertex" : "pixel", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

/* A shader from the cache, made on a miss. 0: none (failed, or full). */
static GLuint shader_for(GlShader *tab, uint64_t key, GLenum type, const void *arg)
{
    static char src[256 * 1024];
    uint32_t at = (uint32_t)(key % SHADER_SLOTS), i;
    size_t len;
    int wc = 0;

    for (i = 0; i < SHADER_SLOTS; i++) {
        GlShader *e = &tab[(at + i) % SHADER_SLOTS];
        if (e->key == key && e->obj)
            return e->obj;
        if (!e->obj) {
            if (type == GL_FRAGMENT_SHADER)
                len = nv2a_glsl_ps((const Nv2aHlslPsKey *)arg, src, sizeof src);
            else if (arg)
                len = nv2a_glsl_vs((const Nv2aVpState *)arg, src, sizeof src, &wc);
            else
                len = nv2a_glsl_vs_passthrough(src, sizeof src);
            if (!len || !(e->obj = compile(type, src, len)))
                return 0;
            e->key = key;
            return e->obj;
        }
    }
    return 0;
}

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

static GLuint program_for(const Nv2aHwDraw *d)
{
    static const char *const k_samplers[16] = {
        "tx0", "tx1", "tx2", "tx3", "cx0", "cx1", "cx2", "cx3",
        "tv0", "tv1", "tv2", "tv3", "tz0", "tz1", "tz2", "tz3"
    };
    Nv2aHlslPsKey pk;
    uint64_t vk, pkh;
    GLuint vs, ps;
    uint32_t at, i;

    vk = d->use_vp ? vp_hash(d->vp) : 1u;
    nv2a_hlsl_ps_key(d->comb, &pk);
    pkh = hash64(&pk, sizeof pk, 3);
    at = (uint32_t)((vk ^ (pkh * 31u)) % PROG_SLOTS);
    for (i = 0; i < PROG_SLOTS; i++) {
        GlProgram *e = &s_prog[(at + i) % PROG_SLOTS];
        GLint ok = 0;
        int k;

        if (e->prog && e->vs == vk && e->ps == pkh)
            return e->prog;
        if (e->prog)
            continue;
        vs = shader_for(s_vs_cache, vk, GL_VERTEX_SHADER, d->use_vp ? (const void *)d->vp : NULL);
        ps = shader_for(s_ps_cache, pkh, GL_FRAGMENT_SHADER, &pk);
        if (!vs || !ps)
            return 0;
        e->prog = glCreateProgram();
        glAttachShader(e->prog, vs);
        glAttachShader(e->prog, ps);
        glLinkProgram(e->prog);
        glGetProgramiv(e->prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(e->prog, sizeof log, NULL, log);
            if (s_compile_fail++ < 4)
                fprintf(stderr, "  [GL] program failed to link: %s\n", log);
            glDeleteProgram(e->prog);
            e->prog = 0;
            return 0;
        }
        glUniformBlockBinding(e->prog, glGetUniformBlockIndex(e->prog, "VSC"), 0);
        glUniformBlockBinding(e->prog, glGetUniformBlockIndex(e->prog, "VSD"), 1);
        glUniformBlockBinding(e->prog, glGetUniformBlockIndex(e->prog, "PSC"), 2);
        glUseProgram(e->prog);
        for (k = 0; k < 16; k++) {
            GLint loc = glGetUniformLocation(e->prog, k_samplers[k]);
            if (loc >= 0)
                glUniform1i(loc, k);
        }
        e->vs = vk;
        e->ps = pkh;
        return e->prog;
    }
    return 0;
}

/* ── Textures ── */

#define TEX_SLOTS 4096

typedef struct {
    uint32_t va, format, pitch, rect;     /* key */
    uint32_t w, h, levels, bytes;
    uint64_t shash;
    uint32_t checked_frame;
    GLuint   obj;
    int      cube;
} GlTexture;

static GlTexture s_tex[TEX_SLOTS];

static uint32_t level_bytes(uint32_t color, uint32_t w, uint32_t h, uint32_t pitch, int linear)
{
    uint32_t bb = d3d8_format_dxt_block_bytes(color);
    if (bb)
        return ((w + 3) / 4) * ((h + 3) / 4) * bb;
    if (linear)
        return pitch * h;
    return w * h * nv2a_pb_texel_bytes(color);
}

/* Every level (and face) decoded to A8R8G8B8 and uploaded. */
static int tex_upload(GlTexture *e, const Nv2aHwTexture *t)
{
    static uint32_t *scratch;
    static size_t scratch_n;
    const uint8_t *src = guest() + t->va;
    int linear = !d3d8_format_is_swizzled(t->color) && !d3d8_format_dxt_block_bytes(t->color);
    uint32_t lv, w, h, off, face, faces = e->cube ? 6 : 1, face_stride = 0;
    GLenum target = e->cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;

    for (lv = 0, w = e->w, h = e->h; lv < e->levels; lv++) {
        face_stride += level_bytes(t->color, w, h, t->pitch, linear);
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
    }
    face_stride = (face_stride + 127u) & ~127u;
    glBindTexture(target, e->obj);
    glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, (GLint)e->levels - 1);
    for (face = 0; face < faces; face++) {
        GLenum ft = e->cube ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + face : GL_TEXTURE_2D;
        off = face * face_stride;
        for (lv = 0, w = e->w, h = e->h; lv < e->levels; lv++) {
            size_t need = (size_t)w * h;
            if (need > scratch_n) {
                free(scratch);
                scratch = (uint32_t *)malloc(need * 4);
                scratch_n = scratch ? need : 0;
                if (!scratch)
                    return 0;
            }
            if (!nv2a_pb_decode_level(t->color, src + off, w, h, linear ? t->pitch : 0, scratch))
                return 0;
            glTexImage2D(ft, (GLint)lv, GL_RGBA8, (GLsizei)w, (GLsizei)h, 0,
                         GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, scratch);
            off += level_bytes(t->color, w, h, t->pitch, linear);
            if (w > 1) w >>= 1;
            if (h > 1) h >>= 1;
        }
    }
    s_tex_uploads++;
    return 1;
}

/* The texture a stage samples, current with guest memory (checked once a
 * frame, sparsely). 0: none. */
static GLuint tex_for_impl(const Nv2aHwTexture *t, int *cube, uint32_t *tw, uint32_t *th)
{
    uint32_t bb = d3d8_format_dxt_block_bytes(t->color);
    int swz = d3d8_format_is_swizzled(t->color) || bb;
    int linear = !swz;
    uint32_t w = t->width, h = t->height, levels = 1, lv, lw, lh, bytes = 0, faces;
    uint32_t rect = (t->width << 16) | (t->height & 0xFFFFu);
    uint32_t at, i;
    GlTexture *e = NULL;

    if (!w || !h || w > 4096 || h > 4096 || !t->va)
        return 0;
    if (!bb && !nv2a_pb_texel_bytes(t->color))
        return 0;
    if (swz) {
        levels = (t->format >> 16) & 0xFu;
        if (!levels)
            levels = 1;
        for (lv = 1, lw = w, lh = h; lv < levels; lv++) {
            if (lw == 1 && lh == 1)
                break;
            lw = lw > 1 ? lw >> 1 : 1;
            lh = lh > 1 ? lh >> 1 : 1;
        }
        levels = lv;
    }
    *cube = (t->format & 4u) && w == h && swz;
    faces = *cube ? 6 : 1;
    for (lv = 0, lw = w, lh = h; lv < levels; lv++) {
        bytes += level_bytes(t->color, lw, lh, t->pitch, linear);
        if (lw > 1) lw >>= 1;
        if (lh > 1) lh >>= 1;
    }
    if (faces > 1)
        bytes = ((bytes + 127u) & ~127u) * faces;
    if (!guest_ok(t->va, bytes))
        return 0;

    at = (t->va >> 6) * 2654435761u % TEX_SLOTS;
    for (i = 0; i < 64; i++) {
        GlTexture *c = &s_tex[(at + i) % TEX_SLOTS];
        if (c->obj && c->va == t->va && c->format == t->format && c->pitch == t->pitch
                && c->rect == rect) {
            e = c;
            break;
        }
        if (!c->obj && !e)
            e = c;
    }
    if (!e)
        e = &s_tex[at];                          /* full there: reuse the home slot */
    if (!e->obj || e->va != t->va || e->format != t->format || e->pitch != t->pitch
            || e->rect != rect) {
        if (e->obj)
            glDeleteTextures(1, &e->obj);
        memset(e, 0, sizeof *e);
        glGenTextures(1, &e->obj);
        e->va = t->va;
        e->format = t->format;
        e->pitch = t->pitch;
        e->rect = rect;
        e->w = w;
        e->h = h;
        e->levels = levels;
        e->bytes = bytes;
        e->cube = *cube;
        e->shash = hash_sparse(guest() + t->va, bytes);
        e->checked_frame = s_frame;
        if (!tex_upload(e, t))
            return 0;
    } else if (e->checked_frame != s_frame) {
        uint64_t sh = hash_sparse(guest() + t->va, bytes);
        e->checked_frame = s_frame;
        if (sh != e->shash) {
            e->shash = sh;
            tex_upload(e, t);
        }
    }
    *tw = w;
    *th = h;
    return e->obj;
}

static GLuint tex_for(const Nv2aHwTexture *t, int *cube, uint32_t *tw, uint32_t *th)
{
    uint64_t t0 = now_ns();
    GLuint r = tex_for_impl(t, cube, tw, th);
    s_ns_tex += now_ns() - t0;
    return r;
}

/* Sampler objects by filter, address and border. */
#define SAMPLER_SLOTS 256
static struct { uint64_t key; GLuint obj; } s_samplers[SAMPLER_SLOTS];

static GLint wrap_mode(uint32_t m)
{
    switch (m) {
    case 2: return GL_MIRRORED_REPEAT;
    case 3: case 5: return GL_CLAMP_TO_EDGE;
    case 4: return GL_CLAMP_TO_BORDER;
    default: return GL_REPEAT;
    }
}

static GLuint sampler_for(const Nv2aHwTexture *t, uint32_t levels)
{
    uint32_t min = (t->filter >> 16) & 0x3Fu, mag = (t->filter >> 24) & 0xFu;
    int lmin = (min == 2 || min == 4 || min == 6 || min == 7);
    int lmip = (min == 5 || min == 6);
    int mips = (min >= 3 && min <= 6) && levels > 1;
    uint64_t key = ((uint64_t)(t->address & 0xFFFFFu) << 40) ^ ((uint64_t)t->border << 8)
                 ^ (uint64_t)((lmin << 3) | (lmip << 2) | (mips << 1) | (mag != 1));
    uint32_t at = (uint32_t)(hash64(&key, sizeof key, 5) % SAMPLER_SLOTS), i;

    for (i = 0; i < SAMPLER_SLOTS; i++) {
        uint32_t k = (at + i) % SAMPLER_SLOTS;
        if (s_samplers[k].obj && s_samplers[k].key == key)
            return s_samplers[k].obj;
        if (!s_samplers[k].obj) {
            GLuint s;
            float border[4];
            glGenSamplers(1, &s);
            glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, mag != 1 ? GL_LINEAR : GL_NEAREST);
            glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER,
                                mips ? (lmin ? (lmip ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR_MIPMAP_NEAREST)
                                             : (lmip ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST))
                                     : (lmin ? GL_LINEAR : GL_NEAREST));
            glSamplerParameteri(s, GL_TEXTURE_WRAP_S, wrap_mode(t->address & 0xFu));
            glSamplerParameteri(s, GL_TEXTURE_WRAP_T, wrap_mode((t->address >> 8) & 0xFu));
            glSamplerParameteri(s, GL_TEXTURE_WRAP_R, wrap_mode((t->address >> 16) & 0xFu));
            unpack_argb(t->border, border);
            glSamplerParameterfv(s, GL_TEXTURE_BORDER_COLOR, border);
            s_samplers[k].key = key;
            s_samplers[k].obj = s;
            return s;
        }
    }
    return 0;
}

/* ── Geometry ── */

static void attr_gl(const Nv2aHwAttr *a, GLint *comps, GLenum *type, uint32_t *bytes,
                    uint32_t *mode)
{
    *comps = 0;
    switch (a->type) {
    case 2:
        *mode = NV2A_HLSL_AFMT_FLOAT;
        *comps = a->size >= 1 && a->size <= 3 ? (GLint)a->size : 4;
        *type = GL_UNSIGNED_INT;
        *bytes = 4u * (uint32_t)*comps;
        return;
    case 0:
        *mode = NV2A_HLSL_AFMT_COLOR;
        *comps = 4; *type = GL_UNSIGNED_BYTE; *bytes = 4;
        return;
    case 4:
        *mode = NV2A_HLSL_AFMT_UBYTE;
        *comps = a->size == 1 || a->size == 2 ? (GLint)a->size : 4;
        *type = GL_UNSIGNED_BYTE; *bytes = (uint32_t)*comps;
        return;
    case 1: case 5:
        *mode = a->type == 1 ? NV2A_HLSL_AFMT_S1 : NV2A_HLSL_AFMT_S32K;
        *comps = a->size == 1 || a->size == 2 ? (GLint)a->size : 4;
        *type = GL_UNSIGNED_SHORT; *bytes = 2u * (uint32_t)*comps;
        return;
    case 6:
        *mode = NV2A_HLSL_AFMT_CMP;
        *comps = 1; *type = GL_UNSIGNED_INT; *bytes = 4;
        return;
    default:
        *mode = NV2A_HLSL_AFMT_ABSENT;
        *bytes = 0;
        return;
    }
}

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

static void emit_tris(uint32_t prim, const uint32_t *idx, uint32_t n, uint32_t base, uint32_t *out)
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

static int draw_init(void)
{
    glGenVertexArrays(1, &s_vao);
    glBindVertexArray(s_vao);           /* first: the index ring binds into it */
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &s_ubo_align);
    s_persistent = epoxy_gl_version() >= 44 || epoxy_has_gl_extension("GL_ARB_buffer_storage");
    if (s_ubo_align < 16)
        s_ubo_align = 16;
    ring_init(&s_vring, GL_ARRAY_BUFFER, 32u << 20);
    ring_init(&s_iring, GL_ELEMENT_ARRAY_BUFFER, 8u << 20);
    ring_init(&s_uring, GL_UNIFORM_BUFFER, 16u << 20);
    /* No depth clip, as the D3D11 path (and the software one) draw. */
    glEnable(GL_DEPTH_CLAMP);
    return 1;
}

static void draw_impl(const Nv2aHwDraw *d)
{
    static uint8_t *vbuf;
    static size_t vbuf_cap;
    static uint32_t *ibuf;
    static size_t ibuf_cap;
    const uint8_t *mem;
    struct { uint32_t base, stride, end, at; } st[16];
    Nv2aHlslVsParams vsd;
    Nv2aHlslPsParams psd;
    uint32_t nstreams = 0, i, a, lo = 0xFFFFFFFFu, hi = 0, nidx, count, w, h, total;
    uint32_t vb_at = 0, ib_at = 0, at = 0;
    uint64_t mark = 0, tnow;
#define MARK(k) do { if (s_profile) { tnow = now_ns(); s_ns_part[k] += tnow - mark; mark = tnow; } } while (0)

    if (s_profile < 0)
        s_profile = getenv("RECOMP_GL_PROFILE") != NULL;
    if (s_profile)
        mark = now_ns();
    GLuint prog;

    if (!on_thread())
        return;
    s_draws++;
    if (d->idx_count < 3 || !d->comb || (d->use_vp && !d->vp))
        return;
    nidx = tri_indices(d->prim, d->idx_count);
    if (!nidx)
        return;
    for (i = 0; i < d->idx_count; i++) {
        if (d->idx[i] < lo) lo = d->idx[i];
        if (d->idx[i] > hi) hi = d->idx[i];
    }
    count = hi - lo + 1;
    if (count > (1u << 20) || !(prog = program_for(d))) {
        s_skipped++;
        return;
    }
    MARK(0);
    if (!bind_targets(d, d->color_va && (d->color_mask & 0x01010101u), d->zeta_va != 0, &w, &h)) {
        s_skipped++;
        return;
    }
    mem = d->inline_base ? d->inline_base : guest();

    MARK(1);
    /* Streams: one per interleaved buffer, copied over the index range. */
    memset(&vsd, 0, sizeof vsd);
    for (a = 0; a < 16; a++) {
        const Nv2aHwAttr *at = &d->attr[a];
        uint32_t bytes = 0, mode = 0, s;
        GLint comps;
        GLenum type = GL_UNSIGNED_INT;

        if (d->cur_attr)
            memcpy(vsd.adef[a], d->cur_attr[a], sizeof vsd.adef[a]);
        else
            vsd.adef[a][3] = 1.0f;
        glDisableVertexAttribArray(a);
        if (!at->size || !at->stride || (!d->inline_base && !at->offset))
            continue;
        attr_gl(at, &comps, &type, &bytes, &mode);
        if (!comps || (at->stride % 4) || (at->offset % 4))
            continue;
        for (s = 0; s < nstreams; s++)
            if (st[s].stride == at->stride && at->offset >= st[s].base
                    && at->offset + bytes <= st[s].base + at->stride)
                break;
        if (s == nstreams) {
            st[s].base = at->offset;
            st[s].stride = at->stride;
            st[s].end = 0;
            nstreams++;
        }
        if (at->offset - st[s].base + bytes > st[s].end)
            st[s].end = at->offset - st[s].base + bytes;
        vsd.afmt[a][0] = mode;
        vsd.afmt[a][1] = at->size;
    }
    for (i = 0, total = 0; i < nstreams; i++) {
        st[i].at = total;
        total += ((count - 1) * st[i].stride + st[i].end + 16 + 15) & ~15u;
    }
    if (vbuf_cap < total) {
        uint8_t *n = (uint8_t *)realloc(vbuf, total);
        if (!n)
            return;
        vbuf = n;
        vbuf_cap = total;
    }
    for (i = 0; i < nstreams; i++) {
        uint32_t span = (count - 1) * st[i].stride + st[i].end;
        uint32_t src = st[i].base + lo * st[i].stride;
        uint32_t avail = span;
        if (d->inline_base)
            avail = src >= d->inline_bytes ? 0 : (src + span > d->inline_bytes ? d->inline_bytes - src : span);
        else if (!guest_ok(src, span)) {
            s_skipped++;
            return;
        }
        memcpy(vbuf + st[i].at, mem + src, avail);
        memset(vbuf + st[i].at + avail, 0, span - avail + 16);
    }
    if (!ring_put(&s_vring, vbuf, total ? total : 16, 16, &vb_at)) {
        s_skipped++;
        return;
    }
    for (a = 0; a < 16; a++) {
        const Nv2aHwAttr *at = &d->attr[a];
        uint32_t bytes, mode, s;
        GLint comps;
        GLenum type = GL_UNSIGNED_INT;
        if (!vsd.afmt[a][0])
            continue;
        attr_gl(at, &comps, &type, &bytes, &mode);
        for (s = 0; s < nstreams; s++)
            if (st[s].stride == at->stride && at->offset >= st[s].base
                    && at->offset + bytes <= st[s].base + at->stride)
                break;
        glEnableVertexAttribArray(a);
        glVertexAttribIPointer(a, comps, type, (GLsizei)at->stride,
                               (const void *)(uintptr_t)(vb_at + st[s].at + at->offset - st[s].base));
    }

    MARK(2);
    if (ibuf_cap < nidx) {
        uint32_t *n = (uint32_t *)realloc(ibuf, (size_t)nidx * 4);
        if (!n)
            return;
        ibuf = n;
        ibuf_cap = nidx;
    }
    emit_tris(d->prim, d->idx, d->idx_count, lo, ibuf);
    if (!ring_put(&s_iring, ibuf, nidx * 4, 4, &ib_at)) {
        s_skipped++;
        return;
    }

    MARK(3);
    /* Vertex parameters, as nv2a_d3d11.c sets them. */
    vsd.vp[0] = 2.0f / (float)w;
    vsd.vp[1] = 2.0f / (float)h;
    vsd.vp[2] = 1.0f / (((d->format >> 4) & 0xFu) == 1u ? 65535.0f : 16777215.0f);
    if (d->fog_enable) {
        vsd.fog[0] = d->fog_param[0];
        vsd.fog[1] = d->fog_param[1];
        switch (d->fog_mode) {
        case 0x2601: vsd.fog[2] = 1.0f; break;
        case 0x0804: vsd.fog[2] = 1.0f; vsd.fog[3] = 1.0f; break;
        case 0x0800: vsd.fog[2] = 2.0f; break;
        case 0x0802: vsd.fog[2] = 2.0f; vsd.fog[3] = 1.0f; break;
        case 0x0801: vsd.fog[2] = 3.0f; break;
        case 0x0803: vsd.fog[2] = 3.0f; vsd.fog[3] = 1.0f; break;
        default: break;
        }
    }
    vsd.roles[0] = d->diffuse_attr;
    vsd.roles[1] = d->tex0_attr;
    if (!d->use_vp && !vsd.afmt[d->diffuse_attr & 15][0]
            && !(d->cur_attr_set & (1u << (d->diffuse_attr & 15))))
        vsd.adef[d->diffuse_attr & 15][0] = vsd.adef[d->diffuse_attr & 15][1] =
            vsd.adef[d->diffuse_attr & 15][2] = 1.0f;
    /* Room for all three blocks first: a wrap between them would leave the
     * ranges already bound pointing into the storage it replaced. */
    ring_reserve(&s_uring, sizeof vsd + sizeof psd + sizeof d->vp->constants
                           + 3u * (uint32_t)s_ubo_align);
    if (!ring_put(&s_uring, &vsd, sizeof vsd, (uint32_t)s_ubo_align, &at)) {
        s_skipped++;
        return;
    }
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, s_uring.buf, at, sizeof vsd);
    /* The constants only when they changed -- or the ring was replaced
     * under the range still bound. */
    if (d->use_vp && (d->vp != s_vsc_owner || d->vp->constants_gen != s_vsc_gen
                      || s_vsc_epoch != s_uring.epoch)) {
        if (!ring_put(&s_uring, d->vp->constants, sizeof d->vp->constants,
                      (uint32_t)s_ubo_align, &at)) {
            s_skipped++;
            return;
        }
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, s_uring.buf, at, sizeof d->vp->constants);
        s_vsc_owner = d->vp;
        s_vsc_gen = d->vp->constants_gen;
        s_vsc_epoch = s_uring.epoch;
    }

    MARK(4);
    /* Pixel parameters and textures. */
    memset(&psd, 0, sizeof psd);
    for (i = 0; i < 8; i++) {
        unpack_argb(d->comb->factor0[i], psd.cf0[i]);
        unpack_argb(d->comb->factor1[i], psd.cf1[i]);
    }
    unpack_argb(d->comb->final_factor0, psd.ff0);
    unpack_argb(d->comb->final_factor1, psd.ff1);
    unpack_argb(d->comb->fog_color, psd.fogc);
    {   /* SET_FOG_COLOR has red in the low byte. */
        float r = psd.fogc[2];
        psd.fogc[2] = psd.fogc[0];
        psd.fogc[0] = r;
    }
    psd.atest[0] = d->alpha_test;
    psd.atest[1] = d->alpha_func;
    psd.atest[2] = d->alpha_ref;
    for (i = 0; i < 4; i++) {
        const Nv2aHwTexture *t = &d->tex[i];
        GlSurface *src_rt;
        GLuint tex = 0, smp;
        uint32_t tw = 0, th = 0;
        int cube = 0;
        int norm = d3d8_format_is_swizzled(t->color) || d3d8_format_dxt_block_bytes(t->color);

        if (!t->enabled)
            continue;
        if (t->color >= 0x2A && t->color <= 0x31) {
            psd.tsc[i][3] = 4.0f;            /* a shadow map: read as passing, for now */
            continue;
        }
        src_rt = surface_find(s_color, t->va);
        if (src_rt && src_rt->va != d->color_va && src_rt->used && s_frame - src_rt->drawn <= 1) {
            /* What the title rendered this frame or the last. */
            tex = src_rt->obj;
            tw = src_rt->w;
            th = src_rt->h;
            psd.tsc[i][0] = norm ? (float)t->width / (float)tw : 1.0f / (float)tw;
            psd.tsc[i][1] = norm ? (float)t->height / (float)th : 1.0f / (float)th;
        } else {
            tex = tex_for(t, &cube, &tw, &th);
            if (!tex)
                continue;
            psd.tsc[i][0] = norm ? 1.0f : 1.0f / (float)tw;
            psd.tsc[i][1] = norm ? 1.0f : 1.0f / (float)th;
        }
        psd.tsc[i][2] = (t->color == 0x07 || t->color == 0x1E || t->color == 0x03
                         || t->color == 0x1C || t->color == 0x05 || t->color == 0x11)
                        ? 1.0f : 0.0f;
        psd.tsc[i][3] = 1.0f;
        glActiveTexture(GL_TEXTURE0 + (cube ? 4 + i : i));
        glBindTexture(cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D, tex);
        smp = sampler_for(t, 16);
        glBindSampler(cube ? 4 + i : i, smp);
    }
    if (!ring_put(&s_uring, &psd, sizeof psd, (uint32_t)s_ubo_align, &at)) {
        s_skipped++;
        return;
    }
    glBindBufferRange(GL_UNIFORM_BUFFER, 2, s_uring.buf, at, sizeof psd);

    MARK(5);
    /* Fixed function: NV2A's values are GL's enums. */
    glUseProgram(prog);
    glEnable(GL_SCISSOR_TEST);
    glScissor((GLint)d->clip_x, (GLint)d->clip_y, (GLsizei)d->clip_w, (GLsizei)d->clip_h);
    glColorMask((d->color_mask & 0x00010000u) != 0, (d->color_mask & 0x00000100u) != 0,
                (d->color_mask & 0x00000001u) != 0, (d->color_mask & 0x01000000u) != 0);
    if (d->blend_enable) {
        float bc[4];
        unpack_argb(d->blend_color, bc);
        glEnable(GL_BLEND);
        glBlendFunc(d->blend_sfactor, d->blend_dfactor);
        glBlendEquation(d->blend_eq ? d->blend_eq : GL_FUNC_ADD);
        glBlendColor(bc[0], bc[1], bc[2], bc[3]);
    } else {
        glDisable(GL_BLEND);
    }
    if (d->depth_test && d->zeta_va) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(d->depth_func >= 0x200 && d->depth_func <= 0x207 ? d->depth_func : GL_ALWAYS);
        glDepthMask(d->depth_mask ? GL_TRUE : GL_FALSE);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    if (d->stencil_test && d->zeta_va) {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(d->stencil_func >= 0x200 && d->stencil_func <= 0x207 ? d->stencil_func : GL_ALWAYS,
                      (GLint)(d->stencil_ref & 0xFFu), d->stencil_func_mask & 0xFFu);
        glStencilOp(d->stencil_fail ? d->stencil_fail : GL_ZERO,
                    d->stencil_zfail ? d->stencil_zfail : GL_ZERO,
                    d->stencil_zpass ? d->stencil_zpass : GL_ZERO);
        glStencilMask(d->stencil_mask & 0xFFu);
    } else {
        glDisable(GL_STENCIL_TEST);
    }
    glDisable(GL_CULL_FACE);        /* as nv2a_d3d11.c: culling not yet checked */

    MARK(6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_iring.buf);
    glDrawElements(GL_TRIANGLES, (GLsizei)nidx, GL_UNSIGNED_INT, (const void *)(uintptr_t)ib_at);
    MARK(7);
#undef MARK
    {
        GlSurface *rt = d->color_va ? surface_find(s_color, d->color_va) : NULL;
        if (rt)
            rt->drawn = s_frame;
    }
}

/* The finished frame, read back and handed to the window. Synchronous: a
 * pixel-buffer ring comes with the draws, when there is something to wait
 * for. */
static void flip_impl(uint32_t color_va)
{
    GlSurface *e;
    size_t need;

    if (!on_thread())
        return;
    s_flips++;
    s_frame++;
    if (!(e = surface_find(s_color, color_va)))
        return;
    need = (size_t)e->w * e->h * 4;
    if (s_readback_cap < need) {
        uint8_t *n = (uint8_t *)realloc(s_readback, need);
        if (!n)
            return;
        s_readback = n;
        s_readback_cap = need;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, e->obj, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);
    s_att_valid = 0;
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, (GLsizei)e->w, (GLsizei)e->h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                 s_readback);
    xbox_FramebufferWindowPresentArgb(s_readback, e->w, e->h, e->w * 4);
    {   /* RECOMP_GL_SHOT=<n>: every n-th frame as gl_shot_<frame>.ppm in the
         * working directory. */
        static int every = -1;
        if (every < 0) {
            const char *v = getenv("RECOMP_GL_SHOT");
            every = v ? atoi(v) : 0;
        }
        if (every > 0 && s_frame % (uint32_t)every == 0) {
            char path[64];
            FILE *f;
            snprintf(path, sizeof path, "gl_shot_%05u.ppm", s_frame);
            if ((f = fopen(path, "wb")) != NULL) {
                size_t k, n = (size_t)e->w * e->h;
                fprintf(f, "P6 %u %u 255\n", e->w, e->h);
                for (k = 0; k < n; k++) {
                    const uint8_t *p = s_readback + k * 4;   /* B, G, R, A */
                    fputc(p[2], f);
                    fputc(p[1], f);
                    fputc(p[0], f);
                }
                fclose(f);
            }
        }
    }
}

void nv2a_gl_draw(const Nv2aHwDraw *d)
{
    uint64_t t0 = now_ns();
    draw_impl(d);
    s_ns_draw += now_ns() - t0;
}

void nv2a_gl_flip(uint32_t color_va)
{
    uint64_t t0 = now_ns();
    flip_impl(color_va);
    s_ns_flip += now_ns() - t0;
}

/* Visibility tests are not counted yet: reports stay as the guest left
 * them, which reads as "hidden" (no lens flares). */
void nv2a_gl_zpass(int enable) { (void)enable; }
void nv2a_gl_get_report(uint32_t va, uint32_t va_alt) { (void)va; (void)va_alt; }

/* No queue yet: every call completes before it returns. */
int nv2a_gl_frames_in_flight(void) { return -1; }

void nv2a_gl_report(void)
{
    if (s_state != 1)
        return;
    fprintf(stderr, "  [GL] %llu flips, %llu clears, %llu draws (%llu skipped), %llu texture"
                    " uploads, %llu shader failures, %llu calls from another thread\n",
            (unsigned long long)s_flips, (unsigned long long)s_clears,
            (unsigned long long)s_draws, (unsigned long long)s_skipped,
            (unsigned long long)s_tex_uploads, (unsigned long long)s_compile_fail,
            (unsigned long long)s_other_thread);
    if (s_flips > s_rep_flips) {
        double n = (double)(s_flips - s_rep_flips);
        fprintf(stderr, "  [GL] per frame: %.2f ms in draws (%.2f of it textures),"
                        " %.2f ms in the flip\n",
                (double)(s_ns_draw - s_rep_draw) / n / 1e6,
                (double)(s_ns_tex - s_rep_tex) / n / 1e6,
                (double)(s_ns_flip - s_rep_flip) / n / 1e6);
        s_rep_flips = s_flips;
        s_rep_draw = s_ns_draw;
        s_rep_tex = s_ns_tex;
        s_rep_flip = s_ns_flip;
        if (s_profile > 0) {
            int k;
            fprintf(stderr, "  [GL] draw sections per frame (ms):");
            for (k = 0; k < 8; k++) {
                fprintf(stderr, " %.2f", (double)(s_ns_part[k] - s_rep_part[k]) / n / 1e6);
                s_rep_part[k] = s_ns_part[k];
            }
            fprintf(stderr, "\n");
        }
    }
}

#endif /* !_WIN32 */
