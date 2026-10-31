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

#include <SDL.h>
#include <epoxy/gl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void xbox_FramebufferWindowPresentArgb(const void *px, uint32_t w,
                                              uint32_t h, uint32_t row_pitch);

#define GL_SURFACES 16

typedef struct {
    uint32_t va, w, h;
    GLuint   obj;            /* texture (colour) or renderbuffer (depth) */
    uint32_t used;           /* LRU tick */
} GlSurface;

static int        s_state;   /* 0 not tried, 1 up, -1 off or failed */
static pthread_t  s_thread;
static SDL_Window *s_win;
static SDL_GLContext s_ctx;
static GLuint     s_fbo;
static GlSurface  s_color[GL_SURFACES], s_depth[GL_SURFACES];
static uint32_t   s_tick;
static uint64_t   s_draws, s_clears, s_flips, s_other_thread;
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
    fprintf(stderr, "  [GL] back end up: %s, OpenGL %s (draws not implemented yet)\n",
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
    glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           rt ? rt->obj : 0, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, ds ? ds->obj : 0);
    glDrawBuffer(rt ? GL_COLOR_ATTACHMENT0 : GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        return 0;
    glViewport(0, 0, (GLsizei)*w, (GLsizei)*h);
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
    glDisable(GL_SCISSOR_TEST);
}

void nv2a_gl_draw(const Nv2aHwDraw *d)
{
    (void)d;
    if (on_thread())
        s_draws++;
}

/* The finished frame, read back and handed to the window. Synchronous: a
 * pixel-buffer ring comes with the draws, when there is something to wait
 * for. */
void nv2a_gl_flip(uint32_t color_va)
{
    GlSurface *e;
    size_t need;

    if (!on_thread())
        return;
    s_flips++;
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
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, (GLsizei)e->w, (GLsizei)e->h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                 s_readback);
    xbox_FramebufferWindowPresentArgb(s_readback, e->w, e->h, e->w * 4);
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
    fprintf(stderr, "  [GL] %llu flips, %llu clears, %llu draws (dropped: not implemented),"
                    " %llu calls from another thread\n",
            (unsigned long long)s_flips, (unsigned long long)s_clears,
            (unsigned long long)s_draws, (unsigned long long)s_other_thread);
}

#endif /* !_WIN32 */
