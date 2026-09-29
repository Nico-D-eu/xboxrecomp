/**
 * Execute the parts of the title's pushbuffer that produce visible pixels.
 *
 * The title builds NV2A commands in guest RAM and advances DMA_PUT; without
 * something consuming them the framebuffer stays whatever it was, which is how
 * a fully booted title renders a black screen. This walks the same command
 * stream nv2a_pb_scan.c surveys and carries out the subset that decides what is
 * on screen: which surface is being drawn into, and clearing it.
 *
 * It also rasterises geometry, but only the part that can be drawn honestly:
 * batches whose attribute 0 is already in screen space, flat-shaded, straight
 * into the same guest framebuffer the clear writes. Titles draw their UI, HUD
 * and 2D overlays that way, so it is the first geometry to appear. Batches that
 * need a vertex program executed are counted and skipped rather than drawn
 * somewhere wrong -- see raster_batch(). Texturing, depth and vertex programs
 * are still a renderer, not a command decoder; the upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c.
 *
 * Everything this does not handle is counted and ranked by
 * nv2a_pb_exec_report(), so what remains is a list rather than a guess.
 *
 * Enabled with RECOMP_PB_EXEC. RECOMP_RASTER_TEST draws one known triangle
 * after every clear, which separates "the pixel path is broken" from "the title
 * has not given us any vertices". RECOMP_FB_DUMP=<prefix> writes the surface to
 * <prefix>NNN.bmp, so the result can be looked at without a display.
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"   /* XBOX_CONTIG_BASE / XBOX_CONTIG_SIZE */
#include "xbox_memory_layout.h"   /* xbox_Nv2aFrameCounterFlip */
/* The swizzle decoder the D3D8 layer already uses -- one implementation of
 * Morton order, not a second one that can disagree with it. */
#include "../d3d/d3d8_swizzle.h"
/* Vertex programs, run on the CPU: a title that transforms its vertices
 * hands over positions that only become screen space after the program. */
#include "nv2a_vp.h"
/* The register combiners, which decide every pixel's colour. */
#include "nv2a_combiner.h"
/* The GPU backend, when RECOMP_GPU=d3d11 asks for it. */
#include "nv2a_d3d11.h"

/* getenv once per name. Several switches are tested per method or per
 * batch, and getenv walks the whole environment with strchr each time -- it
 * was a fifth of the executor thread's samples in game. */
static int env_cached(int *slot, const char *name)
{
    if (*slot < 0)
        *slot = getenv(name) != NULL;
    return *slot;
}
static int s_env_verbose = -1, s_env_vp_off = -1, s_env_batch_trace = -1;

static Nv2aVpState s_vp;
static Nv2aCombiner s_comb;
/* Batches seen, for probes that name one. */
static unsigned s_batch_no;
/* Frames (full-screen target switches) and when the first one was seen. */
static unsigned s_frames_seen;
static unsigned long long s_frames_t0;
/* Set per batch: vertices go through s_vp instead of being read as screen
 * space. */
static int s_batch_vp;
static uint32_t s_vp_batches, s_vp_vertices, s_vp_failed, s_vp_behind_eye;

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowStart(void);
extern uint32_t g_xbox_image_lo, g_xbox_image_hi;

/* Would writing this surface land on the title's own image?
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset within the colour DMA object,
 * not a guest virtual address, and this executor has always used it as one.
 * That is harmless while the two happen to agree and catastrophic when they do
 * not: the Xbox Dashboard names surface 0x00088000 at 1280x960x4, so clearing
 * it wrote 4.9 MB of opaque black from 0x00088000 to 0x00538000 -- straight
 * over its own code, its D3D context at 0x000BBFC0 and the register-block
 * pointer at 0x000BE2C4. The symptom was a title that submitted one perfect
 * frame and then spun forever in a pushbuffer-full loop, three layers away,
 * with every D3D global reading 0xFF000000: the clear colour.
 *
 * So refuse, and say so. Getting the address right needs the DMA object base
 * this ignores (NV097_SET_CONTEXT_DMA_COLOR); until that exists, writing
 * nothing is strictly better than writing over the guest, and a title that
 * cannot draw is easier to debug than one that has been overwritten.
 */
static int surface_hits_image(uint32_t base, uint32_t bytes)
{
    if (!g_xbox_image_hi || !bytes)
        return 0;
    return base < g_xbox_image_hi && base + bytes > g_xbox_image_lo;
}

/* Where a DMA-object offset actually lives.
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset inside the colour DMA object,
 * and for a framebuffer that object covers physical memory -- so the offset
 * is a physical address, not a guest VA. Those are the same number in this
 * runtime, which is why treating it as a VA works until it does not: on
 * hardware the image is mapped at VA 0x00010000 from arbitrary physical
 * pages, so a framebuffer at physical 0x84000 does not overlap it. Here it
 * would.
 *
 * The title tells us which it is by where it allocated. Half-Life 2's
 * framebuffer comes from MmAllocateContiguousMemory, which this runtime
 * serves from the window at XBOX_CONTIG_BASE, so physical P is visible at
 * XBOX_CONTIG_BASE + P -- clear of the image, and the same bytes the title's
 * own writes and the framebuffer window reach.
 *
 * So: use the offset as a VA when that is credible, and fall back to the
 * physical mirror exactly when it is not. Titles whose surfaces already sit
 * in ordinary RAM (Wreckless renders to the tiled alias of physical
 * 0x01954000) keep the first path and are unaffected.
 */
static uint32_t dma_resolve(uint32_t offset)
{
    extern uint32_t xbox_ContiguousAllocatedBytes(void);

    /* Did this runtime hand the offset out as contiguous memory? Then the
     * bytes live in the window, and that is not a guess: the arena is a bump
     * allocator from XBOX_CONTIG_BASE, so everything below its high-water
     * mark is memory some MmAllocateContiguousMemory call returned. The
     * title's own writes go through the window, so the executor's must too.
     *
     * Checking this BEFORE the image test is the whole point. The image test
     * only catches an offset that would land on the title's code, and whether
     * it does is an accident of where the image happens to end: Half-Life 2's
     * colour surface is physical 0x00A6C000, which clears the image by 700 KB.
     * So it looked like an ordinary VA, and the executor cleared 1.2 MB of
     * black straight through the guest heap -- which faulted the title three
     * frames later on a pointer that had been overwritten, while the real
     * framebuffer at 0x80A6C000 stayed untouched and the screen stayed black. */
    if (offset < xbox_ContiguousAllocatedBytes())
        return XBOX_CONTIG_BASE + offset;
    if (!surface_hits_image(offset, 1))
        return offset;
    if ((uint64_t)offset < XBOX_CONTIG_SIZE)
        return XBOX_CONTIG_BASE + offset;
    return offset;                         /* nothing better to offer */
}

static int surface_write_refused(uint32_t base, uint32_t bytes, const char *what)
{
    static int said;

    if (!surface_hits_image(base, bytes))
        return 0;
    if (!said) {
        said = 1;
        fprintf(stderr,
                "  [GPU] REFUSING to %s surface 0x%08X..0x%08X: that overlaps "
                "the loaded image (0x%08X..0x%08X).\n"
                "  [GPU]   SET_SURFACE_COLOR_OFFSET is a DMA-object offset, not "
                "a guest VA, and this executor treats it as one. Writing here "
                "would destroy the title's own code and globals.\n",
                what, base, base + bytes, g_xbox_image_lo, g_xbox_image_hi);
        fflush(stderr);
    }
    return 1;
}

/* NV097 methods this executor acts on. */
/* Blending. The pair this title programs, read from its own pushbuffer
 * rather than guessed: BLEND_ENABLE written 1168 times and left on,
 * SFACTOR 0x0302 (SRC_ALPHA) and DFACTOR 0x0303 (ONE_MINUS_SRC_ALPHA).
 * ALPHA_TEST_ENABLE is written 390 times and left at zero, so this is
 * blending and not an alpha test. */
#define NV097_SET_BLEND_ENABLE            0x0304
#define NV097_SET_BLEND_FUNC_SFACTOR      0x0344
#define NV097_SET_BLEND_FUNC_DFACTOR      0x0348
#define NV_BLEND_SRC_ALPHA                0x0302
#define NV_BLEND_ONE_MINUS_SRC_ALPHA      0x0303

#define NV097_SET_SURFACE_CLIP_HORIZONTAL 0x0200
#define NV097_SET_SURFACE_CLIP_VERTICAL   0x0204
#define NV097_SET_SURFACE_FORMAT          0x0208
#define NV097_SET_SURFACE_PITCH           0x020C
#define NV097_SET_SURFACE_COLOR_OFFSET    0x0210
#define NV097_SET_COLOR_CLEAR_VALUE       0x1D90
#define NV097_CLEAR_SURFACE               0x1D94
#define NV097_SET_VERTEX_DATA_ARRAY_OFFSET 0x1720   /* +i*4, 16 attributes */
#define NV097_SET_VERTEX_DATA_ARRAY_FORMAT 0x1760   /* +i*4 */
#define NV097_SET_BEGIN_END               0x17FC
#define NV097_SET_TEXTURE_OFFSET          0x1B00   /* +i*0x40 */
#define NV097_SET_TEXTURE_FORMAT          0x1B04
#define NV097_SET_TEXTURE_ADDRESS         0x1B08
#define NV097_SET_TEXTURE_CONTROL1        0x1B10
#define NV097_SET_TEXTURE_IMAGE_RECT      0x1B1C
/* The buffer flip. A title double-buffers by telling the GPU which buffer
 * the CRTC reads and which it draws into, advancing the write index and
 * then stalling until the flip has happened. Ignoring these means the
 * stall never clears: Half-Life 2's loader submits its initialisation,
 * asks for a flip, and waits for it in a loop that makes no kernel calls
 * and burns no dispatch, which reads as a hang with no cause.
 *
 * ponytail: the flip completes the moment it is asked for, because there is
 * no scanout to be in the middle of. That makes every frame land instantly
 * and a title that paces itself on the flip runs as fast as it can draw.
 * Pacing wants the vblank clock in the kernel, not a sleep in here. */
#define NV097_SET_FLIP_READ               0x0120
#define NV097_SET_FLIP_WRITE              0x0124
#define NV097_SET_FLIP_MODULO             0x0128
#define NV097_FLIP_INCREMENT_WRITE        0x012C
#define NV097_FLIP_STALL                  0x0130
#define NV097_ARRAY_ELEMENT16             0x1800
/* Draw a run of vertices straight out of the arrays, with no index list:
 * bits 0..23 are the first vertex, bits 24..31 the count minus one. It may
 * appear several times inside one BEGIN_END to draw a longer run. */
#define NV097_DRAW_ARRAYS                 0x1810
#define NV097_INLINE_ARRAY                0x1818
/* Immediate-mode vertices. SET_VERTEX3F/4F carry the position, and writing
 * its last component completes a vertex using whatever the SET_VERTEX_DATA*
 * registers currently hold for the other attributes. This is how Half-Life
 * 2's Xbox loader and the game's own 2D drawing submit every quad -- neither
 * uses INLINE_ARRAY -- so without these the executor saw SET_BEGIN_END pairs
 * with nothing attached and reported `draws 0` while a million and a half
 * textured quads a minute went past it. */
#define NV097_SET_VERTEX3F                0x1500   /* +0..0x08, 3 floats */
#define NV097_SET_VERTEX4F                0x1518   /* +0..0x0C, 4 floats */
#define NV097_SET_VERTEX_DATA2F_M         0x1880   /* + attr*8,  2 floats */
#define NV097_SET_VERTEX_DATA4F_M         0x1A00   /* + attr*16, 4 floats */
#define NV097_SET_VERTEX_DATA4UB          0x1940   /* + attr*4,  D3DCOLOR */

/* One immediate vertex, as this file packs it for the shared draw path:
 * position float4, diffuse D3DCOLOR, texcoord0 float2. */
#define IMM_VERTEX_DWORDS 7

#define NV097_CLEAR_COLOR_MASK            0xF0   /* R,G,B,A bits */

/* One vertex attribute stream, as the title describes it. Attribute 0 is
 * position; the rest are colours, texture coordinates and so on. */
typedef struct {
    uint32_t offset;      /* guest address of element 0 */
    uint32_t type;        /* NV097 data type nibble */
    uint32_t size;        /* components per element */
    uint32_t stride;      /* bytes between elements */
} VertexAttr;

#define NV_VERTEX_ATTRS 16
/* Per batch. 4096 of each silently truncated the title's larger batches --
 * terrain and the rider are drawn as single arrays of more than that. */
#define NV_MAX_INDICES  65536
#define NV_MAX_INLINE   65536           /* dwords of INLINE_ARRAY per batch */

/* Texture stage 0, decoded from what the title programmed.
 *
 * Only stage 0: it is the only one the dashboard configures, and a stage
 * nothing writes to is a stage nothing can be sampled from. The rest arrive
 * as unhandled methods and are counted as such, which is how the next title
 * that needs them will say so. */
typedef struct {
    uint32_t offset;                    /* guest address of texel (0,0)  */
    uint32_t width, height;             /* from IMAGE_RECT               */
    uint32_t pitch;                     /* bytes per row, from CONTROL1  */
    uint32_t color;                     /* NV097 colour-format code      */
    uint32_t addr_u, addr_v;            /* wrap mode per axis            */
    int      valid;
    /* The registers as written, for the D3D11 backend's sampler and mip
     * chain. */
    uint32_t format_raw, address_raw, filter, border;
} Texture;

static struct {
    VertexAttr attr[NV_VERTEX_ATTRS];
    uint32_t   prim;                    /* SET_BEGIN_END parameter, 0 = ended */
    uint32_t   idx[NV_MAX_INDICES];
    uint32_t   idx_count;
    /* INLINE_ARRAY payload: vertices written straight into the pushbuffer
     * instead of into a buffer the title points at. Same vertex format, a
     * different place to read them from. */
    uint32_t   inline_buf[NV_MAX_INLINE];
    uint32_t   inline_count;
    /* Current values of the immediate-mode attributes, and how many complete
     * vertices they have produced in this batch. */
    float      imm_pos[4];
    uint32_t   imm_diffuse;
    float      imm_tex[2];
    uint32_t   imm_count;
    int        inline_active;
    uint32_t   draws, verts, nonzero_draws;
    float      min_x, max_x, min_y, max_y;
    uint32_t color_offset, color_base, pitch, format;
    /* The surface the last batch actually drew into. A double-buffered title
     * has already pointed color_offset at the next buffer and cleared it by
     * the time the flip arrives, so dumping the current one dumps the frame
     * that has not been drawn yet -- which is how a correctly rendered
     * sequence came out as 12 black BMPs. */
    uint32_t drawn_offset;
    uint64_t pixels;
    uint32_t pixel_max;   /* brightest value any pixel write carried */
    uint32_t clip_x, clip_w, clip_y, clip_h;
    uint32_t clear_color;
    uint32_t clears, unhandled_total;
    uint32_t flip_read, flip_write, flip_modulo, flips;
    uint32_t tris_drawn, tris_skipped_offscreen, batches_untransformed;
    /* Why a batch came out flat. "Untextured" has two causes that look
     * identical on screen and want opposite fixes: the batch carried no
     * texture coordinates, or it did and the stage was not usable. */
    uint32_t batches_textured, batches_no_uv, batches_no_tex;
    uint32_t blend_enable, blend_sfactor, blend_dfactor;
    Texture  tex;
    /* Depth and alpha test, as the title programs them. The depth buffer
     * itself is host-side (see zbuf_current), keyed by the zeta offset. */
    uint32_t zeta_offset, depth_test, depth_func, depth_mask, zclear;
    uint32_t alpha_test, alpha_func, alpha_ref;
    uint32_t zbuf_rejects, alpha_rejects;
    /* SET_COLOR_MASK, one byte per channel (A R G B), each 0 or 1. A depth-
     * only pass -- a shadow map, a Z pre-pass -- is drawn with this at 0 and
     * a colour offset of 0, because there is no colour target; honouring it
     * is what keeps such a pass from painting physical page 0. */
    uint32_t color_mask;
    /* State only the D3D11 backend acts on so far. */
    uint32_t zpitch, blend_eq, blend_color;
    uint32_t fog_enable, fog_mode;
    float    fog_param[3];
    uint32_t cull_enable, cull_face, front_face;
    uint32_t stencil_test, stencil_mask, stencil_func, stencil_ref,
             stencil_func_mask, stencil_fail, stencil_zfail, stencil_zpass;
} s_gpu = { .depth_func = 0x0201, .depth_mask = 1, .alpha_func = 0x0207,
            .color_mask = 0x01010101u, .blend_eq = 0x8006,
            .cull_face = 0x405, .front_face = 0x901,
            .stencil_mask = 0xFF, .stencil_func = 0x0207, .stencil_func_mask = 0xFF,
            .stencil_fail = 0x1E00, .stencil_zfail = 0x1E00, .stencil_zpass = 0x1E00 };

/* Unhandled methods, ranked. The interesting output is not that something was
 * skipped but which things dominate, because that is the order to implement
 * them in. */
#define PB_EXEC_MAX_UNHANDLED 2048
typedef struct { uint32_t method, count, last_param; } PbUnhandled;
static PbUnhandled s_unhandled[PB_EXEC_MAX_UNHANDLED];
static int s_unhandled_count;

/* Every texture-stage register, as the title last set it.
 * Texturing is not implemented yet; knowing which formats and sizes a title
 * actually programs is what decides which ones are worth implementing. */
#define NV_TEX_FIRST 0x1B00
#define NV_TEX_LAST  0x1BFC
static uint32_t s_tex_reg[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];
static uint8_t  s_tex_set[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];

/* Formats whose dimensions come from the format word and whose coordinates
 * arrive normalised, rather than from a pitch and SET_TEXTURE_IMAGE_RECT with
 * coordinates in texels. Swizzled and block-compressed are both in this group,
 * and every place that used to test only for swizzled needs the pair. */
static int tex_size_from_format(uint32_t fmt)
{
    return d3d8_format_is_swizzled(fmt) || d3d8_format_dxt_block_bytes(fmt);
}

/* All four texture stages, for the register combiners.
 *
 * s_gpu.tex is stage 0 as the single-texture path has always seen it; this
 * is the same decoding for every stage, kept apart so that path is not
 * disturbed. A stage's registers are 0x40 apart from 0x1B00. */
static Texture  s_stex[4];
static uint32_t s_stex_enable[4];

static uint32_t dma_resolve(uint32_t offset);

static void stage_tex_method(uint32_t method, uint32_t param)
{
    uint32_t stage = (method - NV_TEX_FIRST) / 0x40, reg = (method - NV_TEX_FIRST) % 0x40;
    Texture *t;

    if (stage >= 4)
        return;
    t = &s_stex[stage];
    switch (reg) {
    case 0x00: t->offset = dma_resolve(param); break;          /* OFFSET */
    case 0x04:                                                 /* FORMAT */
        t->format_raw = param;
        t->color = (param >> 8) & 0xFF;
        if (tex_size_from_format(t->color)) {
            t->width  = 1u << ((param >> 20) & 0xF);
            t->height = 1u << ((param >> 24) & 0xF);
        }
        break;
    case 0x08:                                                 /* ADDRESS */
        t->address_raw = param;
        t->addr_u =  param        & 0xF;
        t->addr_v = (param >>  8) & 0xF;
        break;
    case 0x0C: s_stex_enable[stage] = (param >> 30) & 1u; break; /* CONTROL0 */
    case 0x10: t->pitch = param >> 16; break;                  /* CONTROL1 */
    case 0x1C: t->width = param >> 16; t->height = param & 0xFFFF; break;
    case 0x14: t->filter = param; break;                       /* FILTER */
    case 0x24: t->border = param; break;                       /* BORDER_COLOR */
    default: break;
    }
    t->valid = t->offset && t->width && t->height
            && (tex_size_from_format(t->color) || t->pitch);
}

static void record_tex_reg(uint32_t method, uint32_t param)
{
    stage_tex_method(method, param);
    s_tex_reg[(method - NV_TEX_FIRST) / 4] = param;
    s_tex_set[(method - NV_TEX_FIRST) / 4] = 1;
    /* A pitch is a linear texture's property. A swizzled one has no rows and
     * so no pitch, and requiring one here refused every swizzled texture --
     * which is nearly all of them, since swizzled is the Xbox default. That
     * left the title's own textures unsampled and every textured quad drawn in
     * flat vertex colour. */
    s_gpu.tex.valid = s_gpu.tex.offset && s_gpu.tex.width && s_gpu.tex.height
                   && (tex_size_from_format(s_gpu.tex.color)
                       || s_gpu.tex.pitch);
}

/* Every distinct texture a batch was drawn with, and how many batches used it.
 *
 * The per-draw verbose print shows the first few draws of the first frame,
 * which is enough to see that texturing works at all and not enough to answer
 * "is a font page ever bound". This is the same shape as the unhandled-method
 * table below it: a small set, ranked, printed with the rest of the report. */
#define PB_EXEC_MAX_TEXTURES 1024
typedef struct {
    uint32_t offset, color, width, height, batches;
} PbTexUse;
static PbTexUse s_tex_use[PB_EXEC_MAX_TEXTURES];
static int s_tex_use_count;

/* Defined below, next to the sampler it goes through. */
static void dump_texture_bmp(uint32_t seq);

/* Repeat dumps are numbered from well past the first-use sequence, so a
 * listing sorts them after the textures they came from and no first-use file
 * is ever overwritten by one. */
#define TEX_DUMP_SEQ_BASE 1000u
#define TEX_DUMP_SEQ_MAX  40u

static void note_texture_use(void)
{
    int i;

    if (!s_gpu.tex.valid)
        return;
    for (i = 0; i < s_tex_use_count; i++) {
        if (s_tex_use[i].offset == s_gpu.tex.offset
         && s_tex_use[i].color  == s_gpu.tex.color) {
            s_tex_use[i].batches++;
            /* Dump a surface that is redrawn, every Nth time it is bound.
             *
             * First use alone cannot tell a decode error that is wrong in
             * every frame from one that accumulates across them. A block
             * transform that is wrong is wrong on its own, in the keyframe
             * as much as anywhere; motion compensation that is wrong starts
             * from a clean keyframe and smears further with each predicted
             * frame after it. In a single frame the two look identical, and
             * in a sequence they look nothing alike -- so the sequence is
             * what has to be captured.
             *
             * It belongs on this side of the return: a video surface keeps
             * one address for the whole film, so after the first frame it is
             * only ever found here, and the first-use dump below never fires
             * for it again. RECOMP_TEX_DUMP_EVERY=<n> sets the interval, and
             * RECOMP_TEX_DUMP still names the files. */
            {
                static int every = -1;
                static unsigned binds, seq;
                if (every < 0) {
                    const char *e = getenv("RECOMP_TEX_DUMP_EVERY");
                    every = e ? atoi(e) : 0;
                }
                if (every > 0 && ++binds % (unsigned)every == 0
                    && seq < TEX_DUMP_SEQ_MAX)
                    dump_texture_bmp(TEX_DUMP_SEQ_BASE + seq++);
            }
            return;
        }
    }
    if (s_tex_use_count < PB_EXEC_MAX_TEXTURES) {
        s_tex_use[s_tex_use_count].offset  = s_gpu.tex.offset;
        s_tex_use[s_tex_use_count].color   = s_gpu.tex.color;
        s_tex_use[s_tex_use_count].width   = s_gpu.tex.width;
        s_tex_use[s_tex_use_count].height  = s_gpu.tex.height;
        s_tex_use[s_tex_use_count].batches = 1;
        s_tex_use_count++;
        dump_texture_bmp((uint32_t)s_tex_use_count - 1);
    }
}

static void note_unhandled(uint32_t method, uint32_t param)
{
    int i;

    s_gpu.unhandled_total++;
    for (i = 0; i < s_unhandled_count; i++) {
        if (s_unhandled[i].method == method) {
            s_unhandled[i].count++;
            s_unhandled[i].last_param = param;
            return;
        }
    }
    if (s_unhandled_count < PB_EXEC_MAX_UNHANDLED) {
        s_unhandled[s_unhandled_count].method = method;
        s_unhandled[s_unhandled_count].count = 1;
        s_unhandled[s_unhandled_count].last_param = param;
        s_unhandled_count++;
    }
}

/* Read attribute `a` of vertex `index` as floats. Only the float and the
 * normalised-byte types appear in practice; anything else returns 0 so a
 * caller sees a degenerate vertex rather than reading past the array. */
static float s_cur_attr[NV_VERTEX_ATTRS][4];
static uint32_t s_cur_attr_set;

static int fetch_attr(const VertexAttr *a, uint32_t index, float out[4])
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint8_t *p;
    uint32_t i;

    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (!a->size || !a->stride) {
        /* No array: the attribute's current value (see cur_attr_method). */
        if (a >= s_gpu.attr && a < s_gpu.attr + NV_VERTEX_ATTRS)
            memcpy(out, s_cur_attr[a - s_gpu.attr], 4 * sizeof(float));
        return 0;
    }
    if (s_gpu.inline_active) {
        /* The batch arrived as INLINE_ARRAY, so `offset` is a byte offset into
         * the buffered payload rather than a guest address -- and 0 is a legal
         * one there, which is why the offset test is on the other side of this
         * branch. */
        size_t at = (size_t)a->offset + (size_t)index * a->stride;
        if (at + 4 > (size_t)s_gpu.inline_count * 4)
            return 0;
        p = (const uint8_t *)s_gpu.inline_buf + at;
    } else {
        if (!a->offset)
            return 0;
        p = mem + a->offset + (size_t)index * a->stride;
    }

    switch (a->type) {
    case 0:                                  /* D3DCOLOR */
        /* A DWORD 0xAARRGGBB, so little-endian bytes are B,G,R,A -- not the
         * component order of every other format here. Returned as R,G,B,A so
         * callers need not know which format the title chose. */
        out[0] = (float)p[2] / 255.0f;
        out[1] = (float)p[1] / 255.0f;
        out[2] = (float)p[0] / 255.0f;
        out[3] = (float)p[3] / 255.0f;
        return 1;
    case 2:                                  /* float */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = ((const float *)p)[i];
        return 1;
    case 4:                                  /* unsigned byte, normalised */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)p[i] / 255.0f;
        return 1;
    case 1:                                  /* S1: signed short, normalised */
        for (i = 0; i < a->size && i < 4; i++) {
            float v = (float)((const int16_t *)p)[i] / 32767.0f;
            out[i] = v < -1.0f ? -1.0f : v;
        }
        return 1;
    case 5:                                  /* S32K: signed short, as is */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)((const int16_t *)p)[i];
        return 1;
    case 6: {                                /* CMP: packed 11:11:10 normal */
        /* Xbox D3D's NORMPACKED3, the usual way a title stores normals. It
         * read as absent -- 0,0,0 -- so every lit vertex program normalised
         * a zero vector, took rsq(0) = inf, and multiplied it by zero: NaN
         * diffuse, and the combiners drew every lit model black. */
        uint32_t v;
        memcpy(&v, p, 4);
        out[0] = (float)((int32_t)(v << 21) >> 21) / 1023.0f;
        out[1] = (float)((int32_t)(v << 10) >> 21) / 1023.0f;
        out[2] = (float)((int32_t)v >> 22) / 511.0f;
        return 1;
    }
    default:
        return 0;
    }
}

static uint32_t surface_bpp(void)
{
    /* The pitch and the clip width together give the pixel size, which is more
     * reliable than decoding the format field: the format's colour code is
     * only meaningful alongside a type the title also sets, while the pitch is
     * always exactly how many bytes a row occupies. */
    if (!s_gpu.clip_w)
        return 0;
    return s_gpu.pitch / s_gpu.clip_w;
}


/* Write the current surface out as a 24-bit BMP.
 *
 * A framebuffer window needs someone watching it. A file does not, which makes
 * this the only way to check what a title actually rendered on a machine you
 * are not sitting at -- and the only way to put a picture in a bug report.
 *
 * ponytail: bottom-up 24bpp BMP, no palette, no compression. That is the one
 * format every viewer reads and it is 30 lines; PNG would need a dependency.
 */
static void dump_surface_bmp(void)
{
    const char *prefix = getenv("RECOMP_FB_DUMP");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    static int seq;
    char path[512];
    uint32_t w = s_gpu.clip_w, h = s_gpu.clip_h, y, x;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    FILE *f;

    if (!prefix || !w || !h || (bpp != 2 && bpp != 4) || !s_gpu.color_offset)
        return;

    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    snprintf(path, sizeof path, "%s%03d.bmp", prefix, seq++);
    f = fopen(path, "wb");
    if (!f)
        return;

    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54;
    hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1;
    hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    /* BMP rows run bottom-up. */
    for (y = h; y-- > 0; ) {
        const uint8_t *row = mem + dma_resolve(s_gpu.drawn_offset
                                                ? s_gpu.drawn_offset
                                                : s_gpu.color_offset)
                           + (size_t)(s_gpu.clip_y + y) * s_gpu.pitch;
        for (x = 0; x < w; x++) {
            uint8_t bgr[3];
            if (bpp == 4) {
                uint32_t v = ((const uint32_t *)row)[s_gpu.clip_x + x];
                bgr[0] = (uint8_t)(v);
                bgr[1] = (uint8_t)(v >> 8);
                bgr[2] = (uint8_t)(v >> 16);
            } else {
                uint16_t v = ((const uint16_t *)row)[s_gpu.clip_x + x];
                bgr[0] = (uint8_t)(( v        & 0x1F) << 3);
                bgr[1] = (uint8_t)(((v >>  5) & 0x3F) << 2);
                bgr[2] = (uint8_t)(((v >> 11) & 0x1F) << 3);
            }
            fwrite(bgr, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    if (seq == 1)
        fprintf(stderr, "  [GPU] framebuffer dump: %s (%ux%u from 0x%08X %ubpp)\n",
                path, w, h, s_gpu.color_offset, bpp);
}

/* Defined below, next to the rest of the rasteriser; the clear path uses it
 * for RECOMP_RASTER_TEST. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2]);
static void clear_depth(void);

static void hw_fill_targets(Nv2aHwDraw *d);

static void clear_surface(uint32_t param)
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    uint32_t y, x;

    if (env_cached(&s_env_batch_trace, "RECOMP_BATCH_TRACE"))
        fprintf(stderr, "  [BATCH] clear %02X surf %08X zeta %08X\n",
                param, s_gpu.color_offset, s_gpu.zeta_offset);
    if (nv2a_d3d11_active()) {
        Nv2aHwDraw d;
        hw_fill_targets(&d);
        nv2a_d3d11_clear(&d, param, s_gpu.clear_color, s_gpu.zclear);
        if (param & NV097_CLEAR_COLOR_MASK) {
            s_gpu.clears++;
            if (s_gpu.flips == 0 && d.color_va)
                xbox_FramebufferWindowSet(d.color_va, s_gpu.pitch);
            xbox_FramebufferWindowStart();
        }
        return;
    }
    if (param & 0x01u)                     /* Z: the host-side depth buffer */
        clear_depth();
    if (!(param & NV097_CLEAR_COLOR_MASK))
        return;                            /* depth/stencil only */
    if (!s_gpu.color_offset || !s_gpu.pitch || !s_gpu.clip_h || bpp == 0)
        return;
    {
        uint32_t base = dma_resolve(s_gpu.color_offset);
        if (surface_write_refused(base,
                                  (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch,
                                  "clear"))
            return;
        s_gpu.color_base = base;
    }

    for (y = 0; y < s_gpu.clip_h; y++) {
        uint8_t *row = mem + s_gpu.color_base
                     + (size_t)(s_gpu.clip_y + y) * s_gpu.pitch;
        if (bpp == 4) {
            uint32_t *p = (uint32_t *)row + s_gpu.clip_x;
            for (x = 0; x < s_gpu.clip_w; x++)
                p[x] = s_gpu.clear_color;
        } else if (bpp == 2) {
            /* The clear value is always given as A8R8G8B8; a 16-bit surface
             * takes the same colour reduced to 5:6:5. */
            uint16_t v = (uint16_t)(((s_gpu.clear_color >> 8) & 0xF800)
                                  | ((s_gpu.clear_color >> 5) & 0x07E0)
                                  | ((s_gpu.clear_color >> 3) & 0x001F));
            uint16_t *p = (uint16_t *)row + s_gpu.clip_x;
            for (x = 0; x < s_gpu.clip_w; x++)
                p[x] = v;
        }
    }
    s_gpu.clears++;
    /* Progress markers, interleaved with everything else in the log. The
     * summary says drawing stopped; only a marker next to the surrounding
     * activity says what the title was doing when it stopped. */
    if ((s_gpu.clears % 100) == 0)
        fprintf(stderr, "  [GPU] clear #%u\n", s_gpu.clears);
    /* Distinct clear colours actually used. "Cleared to black" and "the clear
     * never ran" look identical in the framebuffer, and only one of them is a
     * bug -- so record what was asked for, not just how often. */
    {
        static uint32_t seen[8];
        static int n;
        int i;
        for (i = 0; i < n; i++)
            if (seen[i] == s_gpu.clear_color) break;
        if (i == n && n < 8) {
            seen[n++] = s_gpu.clear_color;
            fprintf(stderr, "  [GPU] clear colour 0x%08X -> surface 0x%08X"
                            " (%ubpp)\n",
                    s_gpu.clear_color, s_gpu.color_offset, surface_bpp());
        }
    }

    /* Prove the pixel path end to end, independent of whether the title has
     * given us any geometry yet.
     *
     * "Nothing on screen" has three very different causes -- the surface
     * address or pitch is wrong, the rasteriser is broken, or the title's
     * vertex buffers are empty -- and they are indistinguishable from a black
     * window. RECOMP_RASTER_TEST draws one known triangle into the surface
     * just cleared, so a visible triangle rules out the first two and leaves
     * only the third. On Wreckless it is the third: attribute 0 decodes
     * correctly (float, size 2, stride 16) and the buffer it points at stays
     * zero.
     *
     * ponytail: bring-up aid, not a feature. It costs one branch per clear. */
    if (getenv("RECOMP_RASTER_TEST")) {
        static int announced;
        /* Every clear, not once: the title clears each frame and double-buffers,
         * so a triangle drawn a single time is erased before anyone sees it. */
        if (s_gpu.clip_w && s_gpu.clip_h) {
            float a[2], b[2], c[2];
            a[0] = s_gpu.clip_w * 0.5f; a[1] = s_gpu.clip_h * 0.15f;
            b[0] = s_gpu.clip_w * 0.85f; b[1] = s_gpu.clip_h * 0.85f;
            c[0] = s_gpu.clip_w * 0.15f; c[1] = s_gpu.clip_h * 0.85f;
            raster_triangle(a, b, c, 0xFFFF00FFu, NULL);  /* magenta: never a clear colour */
            if (announced++ == 0)
            fprintf(stderr, "  [GPU] raster self-test: triangle (%.0f,%.0f)"
                            " (%.0f,%.0f) (%.0f,%.0f) into 0x%08X %ubpp\n",
                    a[0], a[1], b[0], b[1], c[0], c[1],
                    s_gpu.color_offset, surface_bpp());
        }
    }

    /* Show the surface actually being drawn into. A title that double-buffers
     * renders into the back buffer, so following AvSetDisplayMode's address
     * would show the one nothing is writing. */
    /* The window has to read where the pixels actually are, which is the
     * resolved address rather than the DMA-object offset. */
    /* Only until the title flips. Following the draw surface on every scan
     * shows the buffer being written right now, half a frame at a time; past
     * the first flip the window is repointed at the finished one instead. */
    if (s_gpu.flips == 0)
        xbox_FramebufferWindowSet(dma_resolve(s_gpu.color_offset), s_gpu.pitch);

    /* And open the window, rather than waiting for AvSetDisplayMode to do it.
     *
     * That was the only caller, so a title which draws before setting a display
     * mode -- or never sets one at all -- got no window however much it
     * rendered. The Xbox Dashboard clears a 1280x960 surface at 0x00088000 on
     * its first frame and had not called AvSetDisplayMode by then, so
     * RECOMP_FB_WINDOW=1 was set, the executor knew the address and the pitch,
     * and nothing appeared.
     *
     * Here is the better trigger anyway: this runs when a surface address is
     * known to be real, because a clear just used it. Idempotent and gated on
     * RECOMP_FB_WINDOW, so the cost is one interlocked compare per clear. */
    xbox_FramebufferWindowStart();
}


/* ── Rasteriser ──────────────────────────────────────────────────────────
 *
 * Fills triangles straight into the guest framebuffer, the same memory
 * clear_surface() writes and the framebuffer window already shows. That is the
 * whole reason it is done on the CPU rather than through D3D: nothing new has
 * to be plumbed for the result to be visible.
 *
 * ponytail: flat-shaded, no depth buffer, no texturing, no perspective
 * correction, and only batches whose attribute 0 is already in screen space.
 * A title running a vertex program hands over object-space positions that mean
 * nothing without executing the program, so those batches are counted and
 * skipped rather than drawn somewhere wrong. Upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c once vertex programs are
 * translated; this exists to get the first geometry on screen for every title,
 * which in practice is UI, HUD and 2D overlays -- all pre-transformed.
 */

/* One texel, in the title's own format.
 *
 * The codes are the NV097 colour field, which is the Xbox D3DFMT_ enum --
 * src/d3d/d3d8_xbox.h is the table, and it is the table to check against
 * rather than recollection: 0x1E is LIN_X8R8G8B8 and not, as this first read
 * it, a byte-reversed BGRA. Getting that one wrong turned an opaque black
 * render target into a screen of pure blue, which is the kind of wrong that
 * looks like content.
 *
 * Only the linear (LIN_) formats are read. A swizzled texture stores its
 * texels in Morton order rather than in rows, so reading one as if it had a
 * pitch does not give a slightly wrong colour, it gives a different image --
 * and inventing that image is exactly what this is not for. An unsupported
 * format samples nothing and the caller keeps the vertex colour, which is
 * visibly wrong rather than quietly wrong.
 *
 * ponytail: nearest texel, no filtering, whatever SET_TEXTURE_FILTER asked
 * for. Bilinear when a title's output actually depends on it.
 */
/* Off the edge of the texture, the way the title asked for.
 *
 * Refusing to sample instead is not neutral: it hands the caller back the
 * vertex colour, so a pass whose coordinates reach the last texel by half a
 * texel gets a bright line down the edge of the screen. The dashboard's
 * resolve does exactly that -- its last column and last row, 1119 pixels of
 * white on a black frame, from a rounding step at the boundary.
 */
static uint32_t wrap_coord(uint32_t c, uint32_t size, uint32_t mode)
{
    if (!size)
        return 0;
    if (mode == 1)                         /* wrap */
        return c % size;
    return c >= size ? size - 1 : c;       /* clamp, and everything else */
}

static uint32_t expand(uint32_t v, uint32_t bits)
{
    return d3d8_expand_channel(v, bits);
}

/* The linear format that decodes the same texels as a swizzled one.
 *
 * Swizzling changes where a texel lives, not what it says: A8R8G8B8 (0x06) and
 * LIN_A8R8G8B8 (0x12) are the same four bytes in the same order. So the whole
 * difference is the address calculation, and one of those lets every format
 * below serve both. Pairs read off the table in d3d8_xbox.h rather than
 * recalled -- the comment above this one is about getting exactly that wrong. */
static uint32_t linear_twin(uint32_t fmt)
{
    switch (fmt) {
    case 0x00: return 0x13;                /* L8        -> LIN_L8        */
    case 0x02: return 0x10;                /* A1R5G5B5  -> LIN_A1R5G5B5  */
    case 0x03: return 0x1C;                /* X1R5G5B5  -> LIN_X1R5G5B5  */
    case 0x04: return 0x1D;                /* A4R4G4B4  -> LIN_A4R4G4B4  */
    case 0x05: return 0x11;                /* R5G6B5    -> LIN_R5G6B5    */
    case 0x06: return 0x12;                /* A8R8G8B8  -> LIN_A8R8G8B8  */
    case 0x07: return 0x1E;                /* X8R8G8B8  -> LIN_X8R8G8B8  */
    case 0x19: return 0x1F;                /* A8        -> LIN_A8        */
    case 0x1A: return 0x20;                /* A8L8      -> LIN_A8L8      */
    case 0x3A: return 0x3F;                /* A8B8G8R8  -> LIN_A8B8G8R8  */
    case 0x3B: return 0x40;                /* B8G8R8A8  -> LIN_B8G8R8A8  */
    case 0x3C: return 0x41;                /* R8G8B8A8  -> LIN_R8G8B8A8  */
    default:   return fmt;                 /* already linear, or unhandled */
    }
}

/* Texel `u` of a row in linear format `fmt`, as A8R8G8B8. The one decoder
 * both paths share: the software sampler below, and the D3D11 backend's
 * texture upload through nv2a_pb_decode_level. */
static int decode_texel(uint32_t fmt, const uint8_t *p, uint32_t u, uint32_t *argb)
{
    switch (fmt) {

    /* 32-bit, alpha-red-green-blue in the dword. */
    case 0x12:                                      /* LIN_A8R8G8B8 */
        *argb = ((const uint32_t *)p)[u];
        return 1;
    case 0x1E:                                      /* LIN_X8R8G8B8 */
        *argb = ((const uint32_t *)p)[u] | 0xFF000000u;
        return 1;

    /* 32-bit, other channel orders. The name gives the byte order from the
     * top of the dword down, so each is a permutation of the same four. */
    case 0x3F: {                                    /* LIN_A8B8G8R8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = (t & 0xFF00FF00u) | ((t & 0xFF) << 16) | ((t >> 16) & 0xFF);
        return 1;
    }
    case 0x40: {                                    /* LIN_B8G8R8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24)                 /* A, from the bottom */
              | (((t >>  8) & 0xFFu) << 16)         /* R */
              | (((t >> 16) & 0xFFu) <<  8)         /* G */
              |  ((t >> 24) & 0xFFu);               /* B, from the top */
        return 1;
    }
    case 0x41: {                                    /* LIN_R8G8B8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24) | (t >> 8);
        return 1;
    }

    /* 16-bit. */
    case 0x10: {                                    /* LIN_A1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = ((t & 0x8000u) ? 0xFF000000u : 0u)
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1C: {                                    /* LIN_X1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x11: {                                    /* LIN_R5G6B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 11) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x3F, 6) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1D: {                                    /* LIN_A4R4G4B4 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = (expand((t >> 12) & 0x0F, 4) << 24)
              | (expand((t >>  8) & 0x0F, 4) << 16)
              | (expand((t >>  4) & 0x0F, 4) <<  8)
              |  expand( t        & 0x0F, 4);
        return 1;
    }

    /* 8-bit. */
    case 0x13: {                                    /* LIN_L8 */
        uint32_t t = p[u];
        *argb = 0xFF000000u | (t << 16) | (t << 8) | t;
        return 1;
    }
    case 0x1F:                                      /* LIN_A8 */
        *argb = ((uint32_t)p[u] << 24) | 0x00FFFFFFu;
        return 1;
    case 0x20: {                                    /* LIN_A8L8 */
        uint32_t t = ((const uint16_t *)p)[u], l = t & 0xFFu;
        *argb = ((t >> 8) << 24) | (l << 16) | (l << 8) | l;
        return 1;
    }

    /* 4:2:2 packed YUV, two texels per four bytes.
     *
     * This is how a title hands over a decoded video frame, and without it
     * the frame falls through to `default` -- which returns 0, so the caller
     * paints the quad's vertex colour and the movie is a flat rectangle.
     *
     * The chroma pair is shared between an even texel and the one after it,
     * so the group is found by masking the bottom bit of the index. BT.601,
     * the same coefficients the D3D8 upload path converts with, so the two
     * paths agree rather than each having its own idea of the colour. */
    case 0x24:                                      /* LC_CR8YB8CB8YA8, YUY2 */
    case 0x25: {                                    /* LC_YB8CR8YA8CB8, UYVY */
        uint32_t yoff = (fmt == 0x24) ? 0u : 1u;
        const uint8_t *g = p + (size_t)(u & ~1u) * 2;
        int c  = (int)g[(u & 1u) ? 2 + yoff : yoff] - 16;
        int cu = (int)g[1 - yoff] - 128;
        int cv = (int)g[3 - yoff] - 128;
        int r = (298 * c + 409 * cv + 128) >> 8;
        int gg = (298 * c - 100 * cu - 208 * cv + 128) >> 8;
        int b = (298 * c + 516 * cu + 128) >> 8;
        if (r < 0) r = 0;
        if (r > 255) r = 255;
        if (gg < 0) gg = 0;
        if (gg > 255) gg = 255;
        if (b < 0) b = 0;
        if (b > 255) b = 255;
        *argb = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)gg << 8)
              | (uint32_t)b;
        return 1;
    }

    default:
        return 0;
    }
}

static int sample_tex(const Texture *t, uint32_t u, uint32_t v, uint32_t *argb)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint8_t *p;
    uint32_t fmt;

    if (!t->valid)
        return 0;
    u = wrap_coord(u, t->width,  t->addr_u);
    v = wrap_coord(v, t->height, t->addr_v);

    fmt = t->color;
    if (d3d8_format_dxt_block_bytes(fmt))
        return d3d8_dxt_decode_texel(mem + t->offset, fmt, u, v,
                                     t->width, argb);
    if (d3d8_format_is_swizzled(fmt)) {
        /* Morton order: a texel's index is interleaved from x and y instead of
         * v*pitch + u, so index from the base of the image. decode_texel
         * casts to each format's own width, which makes that index a texel
         * index for every one of them. */
        fmt = linear_twin(fmt);
        p = mem + t->offset;
        u = swizzle_offset(u, v, t->width, t->height);
    } else {
        p = mem + t->offset + (size_t)v * t->pitch;
    }
    return decode_texel(fmt, p, u, argb);
}

/* Bytes per texel of an uncompressed format this decoder reads, 0 if none. */
uint32_t nv2a_pb_texel_bytes(uint32_t color)
{
    switch (linear_twin(color)) {
    case 0x12: case 0x1E: case 0x3F: case 0x40: case 0x41:
        return 4;
    case 0x10: case 0x1C: case 0x11: case 0x1D: case 0x20: case 0x24: case 0x25:
        return 2;
    case 0x13: case 0x1F:
        return 1;
    default:
        return 0;
    }
}

/* One whole texture level as A8R8G8B8 rows, w x h, for the D3D11 backend.
 * `pitch` is the row pitch of a linear format and ignored otherwise. */
int nv2a_pb_decode_level(uint32_t color, const uint8_t *src, uint32_t w,
                         uint32_t h, uint32_t pitch, uint32_t *out)
{
    static uint8_t *tmp;
    static size_t tmp_n;
    uint32_t x, y, bpp, fmt = color;
    const uint8_t *base = src;

    if (d3d8_format_dxt_block_bytes(color)) {
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++)
                if (!d3d8_dxt_decode_texel(src, color, x, y, w, &out[(size_t)y * w + x]))
                    return 0;
        return 1;
    }
    bpp = nv2a_pb_texel_bytes(color);
    if (!bpp)
        return 0;
    if (d3d8_format_is_swizzled(color)) {
        size_t need = (size_t)w * h * bpp;
        if (need > tmp_n) {
            free(tmp);
            tmp = (uint8_t *)malloc(need);
            tmp_n = tmp ? need : 0;
            if (!tmp)
                return 0;
        }
        xbox_unswizzle_rect(tmp, src, w, h, bpp);
        base = tmp;
        pitch = w * bpp;
        fmt = linear_twin(color);
    }
    for (y = 0; y < h; y++) {
        const uint8_t *row = base + (size_t)y * pitch;
        for (x = 0; x < w; x++)
            if (!decode_texel(fmt, row, x, &out[(size_t)y * w + x]))
                return 0;
    }
    return 1;
}

static int sample_texture(uint32_t u, uint32_t v, uint32_t *argb)
{
    return sample_tex(&s_gpu.tex, u, v, argb);
}

/* Write a bound texture out as a BMP, through the sampler rather than around it.
 *
 * "Which texture is this" is not answerable from an address and a format, and
 * it is the question behind most of the ones that matter -- is that a font
 * page or an icon atlas, did the swizzle decode, is the alpha inverted. Going
 * through sample_texture means the file shows exactly what the rasteriser
 * sees, so a decode bug appears here rather than only as a wrong-looking
 * triangle.
 *
 * ponytail: RGB only, alpha dropped. A glyph page is alpha and would come out
 * black, so alpha is composited onto mid-grey to stay legible; that is a
 * viewing choice, not a decode. One file per distinct texture, first use only.
 */
static void dump_texture_bmp(uint32_t seq)
{
    const char *prefix = getenv("RECOMP_TEX_DUMP");
    uint32_t w = s_gpu.tex.width, h = s_gpu.tex.height, x, y;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    char path[512];
    FILE *f;

    if (!prefix || !w || !h || w > 4096 || h > 4096)
        return;
    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    snprintf(path, sizeof path, "%s%02u_%08X_fmt%02X.bmp",
             prefix, seq, s_gpu.tex.offset, s_gpu.tex.color);
    f = fopen(path, "wb");
    if (!f)
        return;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uint32_t argb = 0, a;
            uint8_t px[3];
            if (!sample_texture(x, h - 1 - y, &argb))
                argb = 0;
            a = (argb >> 24) & 0xFFu;
            /* over mid-grey, so an alpha-only page is visible either way */
            px[0] = (uint8_t)(((argb & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[1] = (uint8_t)((((argb >> 8) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[2] = (uint8_t)((((argb >> 16) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            fwrite(px, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    fprintf(stderr, "  [TEXDUMP] %s (%ux%u fmt 0x%02X)\n",
            path, w, h, s_gpu.tex.color);
    fflush(stderr);
}

/* The surface, resolved once per batch.
 *
 * dma_resolve consults the contiguous arena's high-water mark and
 * surface_hits_image walks the image range; both were being done per pixel
 * -- dma_resolve twice -- which cost more than the rasterisation they
 * guarded. Neither answer can change inside a batch, because the colour
 * offset arrives as a method and a method cannot arrive mid-triangle.
 *
 * This is not a micro-optimisation for its own sake: the loader's video
 * paces on frames actually presented, so the rasteriser's throughput is the
 * playback rate. */
static uint8_t *s_surface;          /* host address of surface row 0 */

static int surface_begin_batch(const uint8_t *mem)
{
    uint32_t base = dma_resolve(s_gpu.color_offset);

    if (surface_hits_image(base, (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch))
        return 0;
    /* A surface at the very bottom of the contiguous window. Nothing a
     * title renders to normally lives in its first 64 KB, and a driver's
     * DMA structures often do; say which surface it was, once per offset. */
    if (base >= XBOX_CONTIG_BASE && base < XBOX_CONTIG_BASE + 0x10000u) {
        static uint32_t said[8];
        static int nsaid;
        int k, fresh = 1;
        for (k = 0; k < nsaid; k++)
            if (said[k] == s_gpu.color_offset) fresh = 0;
        if (fresh && nsaid < 8) {
            said[nsaid++] = s_gpu.color_offset;
            fprintf(stderr, "  [GPU] batch renders to the bottom of contiguous"
                    " memory: colour offset 0x%08X -> 0x%08X, pitch %u,"
                    " clip %ux%u+%u+%u, format 0x%X, zeta 0x%08X, colour mask 0x%08X\n",
                    s_gpu.color_offset, base, s_gpu.pitch, s_gpu.clip_w,
                    s_gpu.clip_h, s_gpu.clip_x, s_gpu.clip_y, s_gpu.format,
                    s_gpu.zeta_offset, s_gpu.color_mask);
            fflush(stderr);
        }
    }
    s_surface = (uint8_t *)mem + base;
    return 1;
}

/* A blend factor, per colour channel, from the source and destination
 * (R, G, B, A in [0,1]). NV097 uses the GL enum values. Returns 0 for a
 * factor not modelled (the constant-colour ones), and the caller then writes
 * the source unblended, as it always did. */
static int blend_factors(uint32_t f, const float s[4], const float d[4],
                         float out[3])
{
    int k;
    for (k = 0; k < 3; k++) {
        switch (f) {
        case 0x0000: out[k] = 0.0f; break;                  /* ZERO */
        case 0x0001: out[k] = 1.0f; break;                  /* ONE */
        case 0x0300: out[k] = s[k]; break;                  /* SRC_COLOR */
        case 0x0301: out[k] = 1.0f - s[k]; break;           /* ONE_MINUS_SRC_COLOR */
        case 0x0302: out[k] = s[3]; break;                  /* SRC_ALPHA */
        case 0x0303: out[k] = 1.0f - s[3]; break;           /* ONE_MINUS_SRC_ALPHA */
        case 0x0304: out[k] = d[3]; break;                  /* DST_ALPHA */
        case 0x0305: out[k] = 1.0f - d[3]; break;           /* ONE_MINUS_DST_ALPHA */
        case 0x0306: out[k] = d[k]; break;                  /* DST_COLOR */
        case 0x0307: out[k] = 1.0f - d[k]; break;           /* ONE_MINUS_DST_COLOR */
        case 0x0308: {                                      /* SRC_ALPHA_SATURATE */
            float m = 1.0f - d[3];
            out[k] = s[3] < m ? s[3] : m;
            break;
        }
        default:
            return 0;
        }
    }
    return 1;
}

static void put_pixel_unmasked(uint8_t *mem, uint32_t bpp, int x, int y,
                               uint32_t argb);

/* The colour write, through SET_COLOR_MASK. Fully masked writes nothing --
 * the depth-only pass case -- and a partial mask keeps the masked channels of
 * what was there. */
static void put_pixel(uint8_t *mem, uint32_t bpp, int x, int y, uint32_t argb)
{
    uint32_t m = s_gpu.color_mask;
    uint32_t keep32 = 0, keep16 = 0, old = 0;
    uint8_t *px;

    if (!(m & 0x01010101u))
        return;
    if ((m & 0x01010101u) == 0x01010101u) {
        put_pixel_unmasked(mem, bpp, x, y, argb);
        return;
    }
    if (x < (int)s_gpu.clip_x || x >= (int)(s_gpu.clip_x + s_gpu.clip_w)
            || y < (int)s_gpu.clip_y || y >= (int)(s_gpu.clip_y + s_gpu.clip_h))
        return;
    px = s_surface + (size_t)y * s_gpu.pitch + (size_t)x * bpp;
    if (!(m & 0x01000000u)) keep32 |= 0xFF000000u;
    if (!(m & 0x00010000u)) { keep32 |= 0x00FF0000u; keep16 |= 0xF800u; }
    if (!(m & 0x00000100u)) { keep32 |= 0x0000FF00u; keep16 |= 0x07E0u; }
    if (!(m & 0x00000001u)) { keep32 |= 0x000000FFu; keep16 |= 0x001Fu; }
    if (bpp == 4)
        old = *(uint32_t *)px;
    else if (bpp == 2)
        old = *(uint16_t *)px;
    put_pixel_unmasked(mem, bpp, x, y, argb);
    if (bpp == 4)
        *(uint32_t *)px = (*(uint32_t *)px & ~keep32) | (old & keep32);
    else if (bpp == 2)
        *(uint16_t *)px = (uint16_t)((*(uint16_t *)px & ~keep16)
                                     | (old & keep16));
}

static void put_pixel_unmasked(uint8_t *mem, uint32_t bpp, int x, int y,
                               uint32_t argb)
{
    uint8_t *row;

    (void)mem;
    if (x < (int)s_gpu.clip_x || x >= (int)(s_gpu.clip_x + s_gpu.clip_w))
        return;
    if (y < (int)s_gpu.clip_y || y >= (int)(s_gpu.clip_y + s_gpu.clip_h))
        return;
    s_gpu.pixels++;
    /* RECOMP_PIXEL_PROBE=x,y[,from]: every write to one pixel from batch
     * <from> on, with the state that drew it. The quickest way from "this
     * part of the picture is wrong" to the batch responsible. */
    {
        static int px = -2, py, pfrom, shown;
        if (px == -2) {
            const char *e = getenv("RECOMP_PIXEL_PROBE");
            px = -1;
            if (e)
                sscanf(e, "%d,%d,%d", &px, &py, &pfrom);
        }
        if (x == px && y == py && s_batch_no >= (unsigned)pfrom && shown < 800) {
            shown++;
            fprintf(stderr, "  [PROBE] batch %u surf %08X argb %08X | mode 0x%X prim %u"
                    " z %u/%X/%u a %u/%X/%u blend %u %X/%X mask %08X | t0 %d %ux%u"
                    " fmt %X @%08X | t1 %d %ux%u fmt %X | stages %08X comb %u\n",
                    s_batch_no, s_gpu.color_offset, argb, s_vp.execution_mode,
                    s_gpu.prim, s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
                    s_gpu.alpha_test, s_gpu.alpha_func, s_gpu.alpha_ref,
                    s_gpu.blend_enable, s_gpu.blend_sfactor, s_gpu.blend_dfactor,
                    s_gpu.color_mask, s_stex[0].valid && s_stex_enable[0],
                    s_stex[0].width, s_stex[0].height, s_stex[0].color,
                    s_stex[0].offset, s_stex[1].valid && s_stex_enable[1],
                    s_stex[1].width, s_stex[1].height, s_stex[1].color,
                    s_comb.stage_program, s_comb.control & 0xFFu);
        }
    }
    if ((argb & 0x00FFFFFFu) > (s_gpu.pixel_max & 0x00FFFFFFu))
        s_gpu.pixel_max = argb;
    row = s_surface + (size_t)y * s_gpu.pitch;

    /* src*srcAlpha + dst*(1-srcAlpha), and only that pair.
     *
     * Any other factor combination falls through to an opaque write rather
     * than being approximated: a wrong blend is harder to recognise on
     * screen than no blend, and this is the only pair this title sets.
     *
     * Fully opaque is left alone deliberately. It is the same arithmetic,
     * but skipping it keeps the full-screen quads -- which are drawn with
     * blending enabled and alpha 255 -- on exactly the path they were on
     * before, so this cannot change what they produce. */
    /* Every other factor pair, generically: out = src*sf + dst*df, clamped.
     * The fast path below keeps SRC_ALPHA/INV_SRC_ALPHA exactly as it was;
     * this is for the rest, which used to be written opaque -- additive
     * glows (ONE/ONE, SRC_ALPHA/ONE) then came out as solid bands. */
    if (s_gpu.blend_enable
        && !(s_gpu.blend_sfactor == NV_BLEND_SRC_ALPHA
             && s_gpu.blend_dfactor == NV_BLEND_ONE_MINUS_SRC_ALPHA)) {
        uint32_t dst = 0xFF000000u;
        float s[4], d[4], sf[3], df[3];
        int k;
        if (bpp == 4) {
            dst = ((const uint32_t *)row)[x];
        } else if (bpp == 2) {
            uint32_t t = ((const uint16_t *)row)[x];
            dst = 0xFF000000u | (((t & 0xF800u) << 8) | ((t & 0x07E0u) << 5)
                                 | ((t & 0x001Fu) << 3));
        }
        for (k = 0; k < 4; k++) {
            s[k] = (float)((argb >> (16 - 8 * (k < 3 ? k : -1))) & 0xFF) / 255.0f;
            d[k] = (float)((dst  >> (16 - 8 * (k < 3 ? k : -1))) & 0xFF) / 255.0f;
        }
        /* k = 0..2 is R,G,B (shifts 16, 8, 0); index 3 is alpha (shift 24). */
        if (blend_factors(s_gpu.blend_sfactor, s, d, sf)
                && blend_factors(s_gpu.blend_dfactor, s, d, df)) {
            uint32_t ch[3];
            for (k = 0; k < 3; k++) {
                float v = s[k] * sf[k] + d[k] * df[k];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                ch[k] = (uint32_t)(v * 255.0f + 0.5f);
            }
            argb = 0xFF000000u | (ch[0] << 16) | (ch[1] << 8) | ch[2];
        }
    } else if (s_gpu.blend_enable
        && s_gpu.blend_sfactor == NV_BLEND_SRC_ALPHA
        && s_gpu.blend_dfactor == NV_BLEND_ONE_MINUS_SRC_ALPHA
        && (argb >> 24) != 0xFF) {
        uint32_t sa = argb >> 24;
        uint32_t dst = 0;
        if (sa == 0)
            return;                        /* nothing of the source survives */
        if (bpp == 4) {
            dst = ((const uint32_t *)row)[x];
        } else if (bpp == 2) {
            uint32_t t = ((const uint16_t *)row)[x];
            dst = (((t & 0xF800u) << 8) | ((t & 0x07E0u) << 5)
                 | ((t & 0x001Fu) << 3));
        }
        {
            uint32_t r = (((argb >> 16) & 0xFF) * sa
                        + ((dst >> 16) & 0xFF) * (255u - sa) + 127u) / 255u;
            uint32_t g = (((argb >>  8) & 0xFF) * sa
                        + ((dst >>  8) & 0xFF) * (255u - sa) + 127u) / 255u;
            uint32_t b = (((argb      ) & 0xFF) * sa
                        + ((dst      ) & 0xFF) * (255u - sa) + 127u) / 255u;
            argb = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }

    if (bpp == 4) {
        ((uint32_t *)row)[x] = argb;
    } else if (bpp == 2) {
        ((uint16_t *)row)[x] = (uint16_t)(((argb >> 8) & 0xF800)
                                        | ((argb >> 5) & 0x07E0)
                                        | ((argb >> 3) & 0x001F));
    }
}

/* One rasteriser vertex. x, y are surface pixels; z is window depth in the
 * zeta surface's units (0..2^24-1 for Z24S8, 0..65535 for Z16 -- the same
 * units Xbox D3D scales oPos.z into); w is clip-space w, used only to make
 * the other attributes perspective-correct (1 for pre-transformed vertices);
 * u, v are texel coordinates; c is R, G, B, A in [0,1]. */
typedef struct {
    float x, y, z, w;
    float u, v;
    float c[4];
    /* For the register combiners: the specular colour (oD1) and all four
     * texture coordinates as the vertex produced them -- s, t, r, q, not yet
     * scaled to texels, since a pass-through stage reads them as a colour.
     * Every field is a float: the near clip interpolates the struct as an
     * array. */
    float c1[4];
    float t[4][4];
} RVert;

/* The depth buffer, host-side.
 *
 * The zeta surface lives in guest memory as well, but nothing on the host
 * reads it and titles do not read it back in practice, so a float buffer per
 * zeta offset is kept here instead of packing Z24S8. Keyed by offset because a
 * title renders to more than one target (the menus draw the boarder previews
 * off screen), and each target has its own depth. */
#define ZBUF_SLOTS 6
static struct { uint32_t zeta, w, h; float *z; } s_zbuf[ZBUF_SLOTS];

static float zeta_max(void)
{
    return ((s_gpu.format >> 4) & 0xFu) == 1u ? 65535.0f : 16777215.0f;
}

static float *zbuf_current(uint32_t *out_w)
{
    uint32_t w = s_gpu.clip_x + s_gpu.clip_w, h = s_gpu.clip_y + s_gpu.clip_h;
    int i, victim = 0;

    if (!s_gpu.zeta_offset || !w || !h || w > 4096 || h > 4096)
        return NULL;
    for (i = 0; i < ZBUF_SLOTS; i++) {
        if (s_zbuf[i].z && s_zbuf[i].zeta == s_gpu.zeta_offset) {
            if (s_zbuf[i].w < w || s_zbuf[i].h < h) {
                float *n = (float *)realloc(s_zbuf[i].z, sizeof(float) * w * h);
                uint32_t k;
                if (!n)
                    return NULL;
                s_zbuf[i].z = n;
                s_zbuf[i].w = w;
                s_zbuf[i].h = h;
                for (k = 0; k < w * h; k++)
                    n[k] = zeta_max();
            }
            *out_w = s_zbuf[i].w;
            return s_zbuf[i].z;
        }
        if (!s_zbuf[i].z)
            victim = i;
    }
    if (s_zbuf[victim].z)
        free(s_zbuf[victim].z);
    s_zbuf[victim].z = (float *)malloc(sizeof(float) * w * h);
    if (!s_zbuf[victim].z)
        return NULL;
    s_zbuf[victim].zeta = s_gpu.zeta_offset;
    s_zbuf[victim].w = w;
    s_zbuf[victim].h = h;
    {
        uint32_t k;
        for (k = 0; k < w * h; k++)
            s_zbuf[victim].z[k] = zeta_max();
    }
    *out_w = w;
    return s_zbuf[victim].z;
}

/* CLEAR_SURFACE with the Z bit: the clear value's depth field, in the same
 * units as z above (Z24S8 keeps depth in the top 24 bits). */
static void clear_depth(void)
{
    uint32_t bw, x, y;
    float *zb = zbuf_current(&bw), v;

    if (!zb)
        return;
    v = ((s_gpu.format >> 4) & 0xFu) == 1u ? (float)(s_gpu.zclear & 0xFFFFu)
                                           : (float)(s_gpu.zclear >> 8);
    for (y = s_gpu.clip_y; y < s_gpu.clip_y + s_gpu.clip_h; y++)
        for (x = s_gpu.clip_x; x < s_gpu.clip_x + s_gpu.clip_w; x++)
            zb[(size_t)y * bw + x] = v;
}

/* GL comparison enums, as NV097 uses them for depth and alpha. */
static int nv_compare(uint32_t func, float a, float b)
{
    switch (func) {
    case 0x0200: return 0;             /* NEVER */
    case 0x0201: return a <  b;        /* LESS */
    case 0x0202: return a == b;        /* EQUAL */
    case 0x0203: return a <= b;        /* LEQUAL */
    case 0x0204: return a >  b;        /* GREATER */
    case 0x0205: return a != b;        /* NOTEQUAL */
    case 0x0206: return a >= b;        /* GEQUAL */
    default:     return 1;             /* ALWAYS, and anything unknown */
    }
}

static uint32_t pack_argb(const float c[4])
{
    uint32_t ch[4];
    int k;
    for (k = 0; k < 4; k++) {
        float v = c[k] < 0.0f ? 0.0f : (c[k] > 1.0f ? 1.0f : c[k]);
        ch[k] = (uint32_t)(v * 255.0f + 0.5f);
    }
    return (ch[3] << 24) | (ch[0] << 16) | (ch[1] << 8) | ch[2];
}

/* Half-space fill. Barycentric edge functions rather than scanline slopes:
 * the same test decides both windings, so a title that emits clockwise
 * triangles does not silently render nothing.
 *
 * Per pixel: depth (linear in screen space, which window z is), colour and
 * texture coordinates perspective-correct (interpolated as value/w with 1/w),
 * then the alpha test, the depth test, texture x colour, blending. `smooth`
 * selects Gouraud colour; flat uses vertex a's. */
static void raster_tri(const RVert *a, const RVert *b, const RVert *c,
                       int textured, int smooth)
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp(), zb_w = 0;
    float area, iwa, iwb, iwc;
    float *zb = NULL;
    int minx, maxx, miny, maxy, x, y;
    int depth = 0, use_comb;
    static int modulate = -1, no_comb = -1;

    if (modulate < 0)
        modulate = getenv("RECOMP_TEX_REPLACE") == NULL;
    if (no_comb < 0)
        no_comb = getenv("RECOMP_NO_COMBINERS") != NULL;
    textured = textured && s_gpu.tex.valid;
    /* The register combiners, once the title has programmed them -- which
     * Xbox D3D does for every draw. RECOMP_NO_COMBINERS=1 keeps the old
     * texture-times-diffuse path for comparison. */
    use_comb = s_comb.programmed && !no_comb;

    if (bpp != 4 && bpp != 2)
        return;

    area = (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x);
    if (area == 0.0f)
        return;                            /* degenerate */

    /* Where this batch writes. The same check the per-pixel path made, made
     * once: a surface address landing on the title's own image is no safer
     * one pixel at a time than 4.9 MB at once. */
    if (!surface_begin_batch(mem))
        return;

    minx = (int)floorf(fminf(a->x, fminf(b->x, c->x)));
    maxx = (int)ceilf (fmaxf(a->x, fmaxf(b->x, c->x)));
    miny = (int)floorf(fminf(a->y, fminf(b->y, c->y)));
    maxy = (int)ceilf (fmaxf(a->y, fmaxf(b->y, c->y)));

    if (minx < (int)s_gpu.clip_x) minx = (int)s_gpu.clip_x;
    if (miny < (int)s_gpu.clip_y) miny = (int)s_gpu.clip_y;
    if (maxx > (int)(s_gpu.clip_x + s_gpu.clip_w)) maxx = (int)(s_gpu.clip_x + s_gpu.clip_w);
    if (maxy > (int)(s_gpu.clip_y + s_gpu.clip_h)) maxy = (int)(s_gpu.clip_y + s_gpu.clip_h);
    if (minx >= maxx || miny >= maxy) {
        s_gpu.tris_skipped_offscreen++;
        return;
    }

    /* RECOMP_NO_DEPTH=1 draws as the executor did before it had a depth
     * buffer: nothing is rejected and nothing is written. For telling a depth
     * bug from a missing-geometry one with a single run. */
    {
        static int no_depth = -1;
        if (no_depth < 0)
            no_depth = getenv("RECOMP_NO_DEPTH") != NULL;
        if (!no_depth && (s_gpu.depth_test || s_gpu.depth_mask)) {
            zb = zbuf_current(&zb_w);
            depth = zb != NULL;
        }
    }
    iwa = a->w > 0.0f ? 1.0f / a->w : 1.0f;
    iwb = b->w > 0.0f ? 1.0f / b->w : 1.0f;
    iwc = c->w > 0.0f ? 1.0f / c->w : 1.0f;

    for (y = miny; y < maxy; y++) {
        for (x = minx; x < maxx; x++) {
            float px = (float)x + 0.5f, py = (float)y + 0.5f;
            float w0 = (b->x - a->x) * (py - a->y) - (b->y - a->y) * (px - a->x);
            float w1 = (c->x - b->x) * (py - b->y) - (c->y - b->y) * (px - b->x);
            float w2 = (a->x - c->x) * (py - c->y) - (a->y - c->y) * (px - c->x);
            float la, lb, lc, q, pa, pb, pc, col[4];
            uint32_t out;
            int k;

            if (!((w0 >= 0 && w1 >= 0 && w2 >= 0)
               || (w0 <= 0 && w1 <= 0 && w2 <= 0)))
                continue;
            /* w1 is the area opposite a, w2 opposite b, w0 opposite c. */
            la = w1 / area;
            lb = w2 / area;
            lc = w0 / area;

            if (depth) {
                float z = la * a->z + lb * b->z + lc * c->z;
                float *zp = &zb[(size_t)y * zb_w + x];
                if (s_gpu.depth_test && !nv_compare(s_gpu.depth_func, z, *zp)) {
                    s_gpu.zbuf_rejects++;
                    continue;
                }
                /* Written below, once the pixel survives the alpha test. */
                (void)zp;
            }

            /* Perspective-correct weights. */
            q  = la * iwa + lb * iwb + lc * iwc;
            pa = la * iwa / q;
            pb = lb * iwb / q;
            pc = lc * iwc / q;

            if (smooth)
                for (k = 0; k < 4; k++)
                    col[k] = pa * a->c[k] + pb * b->c[k] + pc * c->c[k];
            else
                memcpy(col, a->c, sizeof col);
            out = pack_argb(col);

            if (use_comb) {
                float v1[4], tx[4][4], res[4];
                int st, killed = 0;

                for (k = 0; k < 4; k++)
                    v1[k] = pa * a->c1[k] + pb * b->c1[k] + pc * c->c1[k];
                memset(tx, 0, sizeof tx);
                for (st = 0; st < 4; st++) {
                    uint32_t mode = nv2a_comb_stage_mode(&s_comb, st), texel;
                    const Texture *t = &s_stex[st];
                    float cs = pa * a->t[st][0] + pb * b->t[st][0] + pc * c->t[st][0];
                    float ct = pa * a->t[st][1] + pb * b->t[st][1] + pc * c->t[st][1];
                    float cr = pa * a->t[st][2] + pb * b->t[st][2] + pc * c->t[st][2];
                    float cq = pa * a->t[st][3] + pb * b->t[st][3] + pc * c->t[st][3];
                    int ix, iy;

                    if (mode == NV2A_TEXMODE_NONE)
                        continue;
                    if (mode == NV2A_TEXMODE_PASS_THROUGH) {
                        tx[st][0] = cs; tx[st][1] = ct;
                        tx[st][2] = cr; tx[st][3] = cq;
                        continue;
                    }
                    if (mode == NV2A_TEXMODE_CLIP_PLANE) {
                        if (cs < 0.0f || ct < 0.0f || cr < 0.0f || cq < 0.0f)
                            killed = 1;
                        continue;
                    }
                    if (!t->valid || !s_stex_enable[st])
                        continue;
                    if ((mode == NV2A_TEXMODE_2D_PROJECTIVE
                         || mode == NV2A_TEXMODE_3D_PROJECTIVE)
                            && cq != 0.0f && cq != 1.0f) {
                        cs /= cq;
                        ct /= cq;
                    }
                    if (tex_size_from_format(t->color)) {
                        cs *= (float)t->width;
                        ct *= (float)t->height;
                    }
                    ix = (int)floorf(cs);
                    iy = (int)floorf(ct);
                    /* Wrap works on the two's-complement value for the
                     * power-of-two sizes a swizzled texture has; clamp needs
                     * the negative side pinned to 0 first. */
                    if (ix < 0 && t->addr_u != 1) ix = 0;
                    if (iy < 0 && t->addr_v != 1) iy = 0;
                    if (sample_tex(t, (uint32_t)ix, (uint32_t)iy, &texel)) {
                        tx[st][0] = (float)((texel >> 16) & 0xFF) / 255.0f;
                        tx[st][1] = (float)((texel >>  8) & 0xFF) / 255.0f;
                        tx[st][2] = (float)( texel        & 0xFF) / 255.0f;
                        tx[st][3] = (float)( texel >> 24        ) / 255.0f;
                    }
                }
                if (killed)
                    continue;
                for (k = 0; k < 4; k++)
                    col[k] = col[k] < 0.0f ? 0.0f : (col[k] > 1.0f ? 1.0f : col[k]);
                nv2a_comb_eval(&s_comb, col, v1, 1.0f, (const float (*)[4])tx, res);
                out = pack_argb(res);
            } else if (textured) {
                uint32_t texel;
                float su = pa * a->u + pb * b->u + pc * c->u;
                float sv = pa * a->v + pb * b->v + pc * c->v;
                if (su < 0.0f) su = 0.0f;
                if (sv < 0.0f) sv = 0.0f;
                if (sample_texture((uint32_t)su, (uint32_t)sv, &texel)) {
                    /* Texture times vertex colour, D3D's default stage-0
                     * op, rather than the texel alone: a translucent or
                     * tinted layer carries its fade in the vertex colour,
                     * and replacing it drew every such layer opaque.
                     * RECOMP_TEX_REPLACE=1 keeps the old behaviour. */
                    if (modulate) {
                        uint32_t sh, m = 0;
                        for (sh = 0; sh < 32; sh += 8) {
                            uint32_t t = (texel >> sh) & 0xFFu;
                            uint32_t v = (out   >> sh) & 0xFFu;
                            m |= ((t * v + 127u) / 255u) << sh;
                        }
                        texel = m;
                    }
                    out = texel;
                }
            }

            /* Alpha test: the cut-out of foliage, fences and glyph edges. */
            if (s_gpu.alpha_test
                    && !nv_compare(s_gpu.alpha_func, (float)(out >> 24),
                                   (float)(s_gpu.alpha_ref & 0xFFu))) {
                s_gpu.alpha_rejects++;
                continue;
            }
            if (depth && s_gpu.depth_mask)
                zb[(size_t)y * zb_w + x] = la * a->z + lb * b->z + lc * c->z;
            put_pixel(mem, bpp, x, y, out);
        }
    }
    s_gpu.tris_drawn++;
    s_gpu.drawn_offset = s_gpu.color_offset;
}

/* The flat, affine entry point the older paths use: screen-space positions,
 * one colour, optional texel coordinates. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2])
{
    RVert v[3];
    const float *p[3] = { a, b, c };
    int i, k;

    memset(v, 0, sizeof v);
    for (i = 0; i < 3; i++) {
        v[i].x = p[i][0];
        v[i].y = p[i][1];
        v[i].z = 0.0f;
        v[i].w = 1.0f;
        v[i].u = uv ? uv[i][0] : 0.0f;
        v[i].v = uv ? uv[i][1] : 0.0f;
        for (k = 0; k < 4; k++)
            v[i].c[k] = (float)((argb >> (k < 3 ? 16 - 8 * k : 24)) & 0xFFu) / 255.0f;
    }
    raster_tri(&v[0], &v[1], &v[2], uv != NULL, 0);
}

/* Attribute 3 is diffuse colour in every NV2A layout that sets one. Absent it,
 * white -- a visible wrong colour beats an invisible correct one during
 * bring-up. */
/* Which attribute carries the colour.
 *
 * Slot 3 is diffuse by convention and titles that follow it are read straight
 * from there. Half-Life 2 does not: its vertex is position, colour, texcoord
 * at stride 24, and the colour arrives in slot 5. So fall back to the format
 * rather than the slot number -- D3DCOLOR is the one attribute type that is
 * only ever a colour, which makes it a stronger signal than the convention. */
static const VertexAttr *color_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[3].offset && s_gpu.attr[3].stride)
        return &s_gpu.attr[3];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 0 && s_gpu.attr[a].size == 4
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[3];
}

/* Attribute 9 is texture coordinate 0 in the NV2A vertex layout, the same way
 * 0 is position and 3 is diffuse -- for a title that follows the convention.
 *
 * Half-Life 2 does not, in either place. Its menu and HUD vertex is position,
 * colour, texcoord at stride 24, with the colour in slot 5 and the texcoords
 * in slot 7, so reading slot 9 found nothing and every batch drew untextured.
 * That is invisible rather than wrong-looking: the menu paints a full-screen
 * quad and then draws its text over it, and with no sampling both come out
 * white, so the screen is blank white and nothing suggests the text was ever
 * drawn.
 *
 * Falling back to the format works because the three attributes of such a
 * vertex are distinguishable: position is float3, colour is D3DCOLOR, and a
 * float2 is a texture coordinate and nothing else.
 *
 * ponytail: takes the first float2 it finds, so a title with two texcoord sets
 * gets stage 0's -- which is what this single-texture rasteriser samples
 * anyway. Multi-texture wants the D3D11 translator, not another heuristic. */
static const VertexAttr *texcoord_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[9].offset && s_gpu.attr[9].stride)
        return &s_gpu.attr[9];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 2 && s_gpu.attr[a].size == 2
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[9];
}

/* Texel coordinates, whichever convention the title used.
 *
 * The two are not interchangeable and the format decides which is in force: a
 * swizzled texture is addressed in [0,1], a linear one in texels. Both are
 * scaled to texels here so that everything downstream -- the barycentric
 * interpolation and the sampler -- works in one unit.
 *
 * This mattered the moment swizzled formats became samplable. Normalised
 * coordinates truncated to a texel index land on texel 0 for any coordinate
 * below 1.0, so a whole quad sampled a single texel and came out flat: the
 * background painted one near-black colour, which looks like a texture that
 * decoded wrong rather than one that was never indexed. */
static int fetch_texcoord(uint32_t index, float out[2])
{
    float t[4];

    if (!fetch_attr(texcoord_attr(), index, t))
        return 0;
    out[0] = t[0];
    out[1] = t[1];
    if (tex_size_from_format(s_gpu.tex.color)) {
        out[0] *= (float)s_gpu.tex.width;
        out[1] *= (float)s_gpu.tex.height;
    }
    return 1;
}

static uint32_t vertex_color(uint32_t index)
{
    float c[4];

    if (!fetch_attr(color_attr(), index, c))
        return 0xFFFFFFFFu;
    return ((uint32_t)(c[3] * 255.0f) << 24)
         | ((uint32_t)(c[0] * 255.0f) << 16)
         | ((uint32_t)(c[1] * 255.0f) <<  8)
         |  (uint32_t)(c[2] * 255.0f);
}

/* An untransformed batch drawn as if it were screen space smears a few pixels
 * into the corner, so the batch has to be classified before it is rasterised.
 *
 * This used to demand that every vertex land inside the surface, which is a
 * different question and the wrong one: geometry that extends past the
 * viewport is ordinary, and clipping it is raster_triangle's job (it clamps
 * its span to the clip rect). The dashboard is exactly the case that exposed
 * it -- a full-screen pass drawn as one oversized triangle, vertices at
 * (-0.5,-0.5), (2*w,-0.5), (-0.5,2*h), all correct and all rejected.
 *
 * What actually separates the two is scale. Object-space positions are model
 * units, a handful either side of the origin; screen-space ones are measured
 * in pixels of a surface hundreds of pixels wide. So: the batch has to be able
 * to touch the surface at all, and it has to be bigger than object space.
 *
 * ponytail: a genuinely tiny screen-space sprite reads as object space and is
 * skipped. It is counted as skipped rather than silently dropped, and the
 * unambiguous answer needs the vertex-program state, which is not tracked yet.
 */
#define OBJECT_SPACE_SPAN 8.0f

static int batch_is_screen_space(void)
{
    float p[4], lo_x, hi_x, lo_y, hi_y;
    uint32_t i;

    if (!s_gpu.clip_w || !s_gpu.clip_h || !s_gpu.idx_count)
        return 0;
    if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p))
        return 0;
    lo_x = hi_x = p[0];
    lo_y = hi_y = p[1];
    for (i = 1; i < s_gpu.idx_count; i++) {
        if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], p))
            return 0;
        if (p[0] < lo_x) lo_x = p[0];
        if (p[0] > hi_x) hi_x = p[0];
        if (p[1] < lo_y) lo_y = p[1];
        if (p[1] > hi_y) hi_y = p[1];
    }

    /* Entirely off the surface: nothing to draw under either reading. */
    if (hi_x < (float)s_gpu.clip_x
     || lo_x > (float)(s_gpu.clip_x + s_gpu.clip_w)
     || hi_y < (float)s_gpu.clip_y
     || lo_y > (float)(s_gpu.clip_y + s_gpu.clip_h))
        return 0;

    /* Small enough to be model units rather than pixels. */
    if (hi_x - lo_x < OBJECT_SPACE_SPAN && hi_y - lo_y < OBJECT_SPACE_SPAN)
        return 0;

    return 1;
}

/* NV097 primitive types.
 *
 * These are the operand of SET_BEGIN_END, where 0 is END and the list starts
 * at 1. They were each one too low, so every title's geometry was decomposed
 * as the primitive below the one it asked for -- a strip as a fan, a fan as
 * quads, and TRIANGLES, the one case whose vertex count must be a multiple
 * of three, as a strip.
 *
 * The vertex order says which numbering is right without taking a table on
 * trust: a strip arrives in Z order and a fan in cyclic order, and they only
 * line up with the primitive under this one. */
#define NV_PRIM_POINTS         1
#define NV_PRIM_LINES          2
#define NV_PRIM_LINE_LOOP      3
#define NV_PRIM_LINE_STRIP     4
#define NV_PRIM_TRIANGLES      5
#define NV_PRIM_TRIANGLE_STRIP 6
#define NV_PRIM_TRIANGLE_FAN   7
#define NV_PRIM_QUADS          8
#define NV_PRIM_QUAD_STRIP     9
#define NV_PRIM_POLYGON        10

/* How many post-draw captures to keep: enough to see whether the geometry
 * is stable from frame to frame, few enough not to fill a directory. */
#define FB_DUMP_AFTER_DRAW 8
static int s_drawn_dumps;

static void dump_surface_bmp(void);
static void find_float_matrices(void);

/* One triangle by vertex index: gather position and, if the batch has one,
 * texture coordinate 0. A vertex whose position cannot be read is not drawn;
 * a batch whose texcoords cannot be read is drawn untextured rather than not
 * at all, so a missing coordinate stream costs the colour and not the shape.
 */
/* One vertex through the title's vertex program: v0..v15 from the attribute
 * arrays (fetch_attr leaves an absent one at 0,0,0,1), then oPos, oT0 and oD0.
 *
 * Xbox D3D appends the viewport transform and the divide by w to every
 * program, so oPos.xy comes out in surface pixels -- the space the rasteriser
 * already works in. oT0 follows the same convention as a raw texcoord
 * attribute: normalised for swizzled and block-compressed textures, texels
 * for linear ones. */
static int vp_vertex(uint32_t index, RVert *v)
{
    float in[NV2A_VP_INPUTS][4], out[NV2A_VP_OUTPUTS][4];
    uint32_t a;
    int k;

    for (a = 0; a < NV2A_VP_INPUTS; a++)
        fetch_attr(&s_gpu.attr[a], index, in[a]);
    s_vp_vertices++;
    if (!nv2a_vp_run(&s_vp, (const float (*)[4])in, out)) {
        s_vp_failed++;
        return 0;
    }
    /* RECOMP_VP_TRACE=<n>: the first n vertices in and out, which is the only
     * way to tell a decode error from a transform the rasteriser misreads. */
    {
        static int trace = -1, shown;
        static unsigned from, seen;
        static uint32_t dumped_hash[8];
        static int dumped;
        uint32_t hash = 2166136261u;
        if (trace < 0) {
            const char *e = getenv("RECOMP_VP_TRACE");
            const char *f = getenv("RECOMP_VP_TRACE_FROM");
            trace = e ? atoi(e) : 0;
            from = f ? (unsigned)atoi(f) : 0;
        }
        if (trace > 0 && seen++ >= from && shown < trace) {
            /* A program not dumped before gets its listing and constants,
             * wherever in the run it first appears. */
            uint32_t pc;
            int d, fresh = 1;
            for (pc = s_vp.program_start; pc < NV2A_VP_SLOTS; pc++) {
                uint32_t k;
                for (k = 1; k < 4; k++)
                    hash = (hash ^ s_vp.program[pc][k]) * 16777619u;
                if (s_vp.program[pc][3] & 1u)
                    break;
            }
            for (d = 0; d < dumped; d++)
                if (dumped_hash[d] == hash)
                    fresh = 0;
            if (fresh && dumped < 8)
                dumped_hash[dumped++] = hash;
            else
                fresh = 0;
            shown++;
            fprintf(stderr, "  [VP] #%d start %u v0=(%g %g %g %g) v3=(%g %g %g %g)"
                    " v9=(%g %g)\n       oPos=(%g %g %g %g) oD0=(%g %g %g %g)"
                    " oT0=(%g %g %g %g)\n", shown, s_vp.program_start,
                    in[0][0], in[0][1], in[0][2], in[0][3],
                    in[3][0], in[3][1], in[3][2], in[3][3], in[9][0], in[9][1],
                    out[NV2A_VP_OPOS][0], out[NV2A_VP_OPOS][1],
                    out[NV2A_VP_OPOS][2], out[NV2A_VP_OPOS][3],
                    out[NV2A_VP_OD0][0], out[NV2A_VP_OD0][1],
                    out[NV2A_VP_OD0][2], out[NV2A_VP_OD0][3],
                    out[NV2A_VP_OT0][0], out[NV2A_VP_OT0][1],
                    out[NV2A_VP_OT0][2], out[NV2A_VP_OT0][3]);
            if (fresh) {
                uint32_t k;
                fprintf(stderr, "       program %08X:\n", hash);
                for (pc = s_vp.program_start; pc < NV2A_VP_SLOTS; pc++) {
                    fprintf(stderr, "       slot %3u: %08X %08X %08X %08X\n", pc,
                            s_vp.program[pc][0], s_vp.program[pc][1],
                            s_vp.program[pc][2], s_vp.program[pc][3]);
                    if (s_vp.program[pc][3] & 1u)
                        break;
                }
                for (k = 0; k < NV2A_VP_CONSTANTS; k++)
                    if (s_vp.constants[k][0] != 0.0f || s_vp.constants[k][1] != 0.0f
                     || s_vp.constants[k][2] != 0.0f || s_vp.constants[k][3] != 0.0f)
                        fprintf(stderr, "       c[%3u] = (%g %g %g %g)\n", k,
                                s_vp.constants[k][0], s_vp.constants[k][1],
                                s_vp.constants[k][2], s_vp.constants[k][3]);
            }
            fflush(stderr);
        }
    }
    /* oPos.xyz is already in surface pixels and zeta units; oPos.w is clip
     * w, which a vertex behind the eye has <= 0 (see raster_indexed). */
    v->x = out[NV2A_VP_OPOS][0];
    v->y = out[NV2A_VP_OPOS][1];
    v->z = out[NV2A_VP_OPOS][2];
    v->w = out[NV2A_VP_OPOS][3];
    if (!isfinite(v->x) || !isfinite(v->y)
            || fabsf(v->x) > 1e6f || fabsf(v->y) > 1e6f)
        return 0;
    if (!isfinite(v->z))
        v->z = 0.0f;
    v->u = out[NV2A_VP_OT0][0];
    v->v = out[NV2A_VP_OT0][1];
    if (tex_size_from_format(s_gpu.tex.color)) {
        v->u *= (float)s_gpu.tex.width;
        v->v *= (float)s_gpu.tex.height;
    }
    for (k = 0; k < 4; k++) {
        v->c[k]  = out[NV2A_VP_OD0][k];
        v->c1[k] = out[NV2A_VP_OD1][k];
        v->t[0][k] = out[NV2A_VP_OT0][k];
        v->t[1][k] = out[NV2A_VP_OT1][k];
        v->t[2][k] = out[NV2A_VP_OT2][k];
        v->t[3][k] = out[NV2A_VP_OT3][k];
    }
    return 1;
}

/* Near-plane clipping for program-transformed triangles.
 *
 * The Xbox D3D epilogue leaves oPos.xy = clip.xy / w * scale + offset, with
 * scale and offset in constant memory at XFCTX_VPSCL/VPOFF (c[58], c[59])
 * and oPos.w = clip w. That inverts exactly, so clip space is recovered per
 * vertex, the triangle is clipped against w = VP_NEAR_W (Sutherland-Hodgman,
 * one plane), texture coordinates are interpolated linearly in clip space --
 * which is the perspective-correct place to do it -- and the 3 or 4 vertex
 * result is re-projected and drawn as a fan. */
#define VP_NEAR_W 1e-3f

/* Per-batch cache of vertex-program results, keyed by vertex index. */
#define VCACHE_SIZE 1024
static RVert    s_vcache[VCACHE_SIZE];
static uint32_t s_vcache_idx[VCACHE_SIZE], s_vcache_gen[VCACHE_SIZE];
static uint8_t  s_vcache_ok[VCACHE_SIZE];
static uint32_t s_vcache_cur = 1;

static void clip_near_and_raster(const RVert in[3], int textured)
{
    const float *scl = s_vp.constants[0x3A], *off = s_vp.constants[0x3B];
    RVert cv[3], out[4];                   /* x, y, z in clip space */
    int i, n = 0;

    if (scl[0] == 0.0f || scl[1] == 0.0f)
        return;                            /* no viewport: cannot invert */
    for (i = 0; i < 3; i++) {
        cv[i] = in[i];
        cv[i].x = (in[i].x - off[0]) / scl[0] * in[i].w;
        cv[i].y = (in[i].y - off[1]) / scl[1] * in[i].w;
        /* Without a z scale there is nothing to invert; carry window z
         * through as it is, which only misplaces depth near the cut. */
        if (scl[2] != 0.0f)
            cv[i].z = (in[i].z - off[2]) / scl[2] * in[i].w;
    }
    for (i = 0; i < 3; i++) {
        const RVert *a = &cv[i], *b = &cv[(i + 1) % 3];
        int ain = a->w > VP_NEAR_W, bin = b->w > VP_NEAR_W;
        if (ain)
            out[n++] = *a;
        if (ain != bin && n < 4) {
            const float *fa = &a->x, *fb = &b->x;
            float *fo = &out[n].x;
            float t = (a->w - VP_NEAR_W) / (a->w - b->w);
            int k;
            /* Every RVert field is a float, in declaration order. */
            for (k = 0; k < (int)(sizeof(RVert) / sizeof(float)); k++)
                fo[k] = fa[k] + t * (fb[k] - fa[k]);
            n++;
        }
    }
    if (n < 3)
        return;                            /* entirely behind the eye */
    for (i = 0; i < n; i++) {
        out[i].x = out[i].x / out[i].w * scl[0] + off[0];
        out[i].y = out[i].y / out[i].w * scl[1] + off[1];
        if (scl[2] != 0.0f)
            out[i].z = out[i].z / out[i].w * scl[2] + off[2];
    }
    for (i = 1; i + 1 < n; i++)
        raster_tri(&out[0], &out[i], &out[i + 1], textured, 1);
}

static void unpack_argb(uint32_t argb, float c[4])
{
    c[0] = (float)((argb >> 16) & 0xFFu) / 255.0f;
    c[1] = (float)((argb >>  8) & 0xFFu) / 255.0f;
    c[2] = (float)( argb        & 0xFFu) / 255.0f;
    c[3] = (float)( argb >> 24         ) / 255.0f;
}

/* One triangle of a batch. `argb` is the provoking vertex's colour, which
 * RECOMP_FLAT_SHADE=1 uses for the whole triangle (the old behaviour);
 * otherwise colour is Gouraud-interpolated from each vertex's own. */
static void raster_indexed(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t argb)
{
    const uint32_t idx[3] = { i0, i1, i2 };
    static int smooth = -1;
    RVert v[3];
    int i, textured = 1;

    if (smooth < 0)
        smooth = getenv("RECOMP_FLAT_SHADE") == NULL;

    if (s_batch_vp) {
        for (i = 0; i < 3; i++) {
            /* A strip or an indexed list names most vertices three times,
             * and the program's inputs cannot change inside a batch, so its
             * output for an index is computed once per batch. */
            uint32_t slot = idx[i] & (VCACHE_SIZE - 1);
            if (s_vcache_gen[slot] == s_vcache_cur && s_vcache_idx[slot] == idx[i]) {
                if (!s_vcache_ok[slot])
                    return;
                v[i] = s_vcache[slot];
                continue;
            }
            s_vcache_gen[slot] = s_vcache_cur;
            s_vcache_idx[slot] = idx[i];
            s_vcache_ok[slot] = (uint8_t)vp_vertex(idx[i], &s_vcache[slot]);
            if (!s_vcache_ok[slot])
                return;
            v[i] = s_vcache[slot];
        }
        if (!smooth)
            for (i = 1; i < 3; i++)
                memcpy(v[i].c, v[0].c, sizeof v[0].c);
        /* A vertex behind the eye. The program has already divided by w, so
         * such a vertex arrives mirrored through the centre of the screen,
         * and the triangle it belongs to becomes a band across the whole
         * surface -- the grey streaks over Amped 2's menus, from background
         * layers rotating past the camera. Dropping those triangles instead
         * left holes in a background the title relies on to cover the last
         * frame (it clears rarely), so old screens showed through. So clip,
         * as the hardware does: see clip_near_and_raster. */
        if (v[0].w <= VP_NEAR_W || v[1].w <= VP_NEAR_W || v[2].w <= VP_NEAR_W) {
            s_vp_behind_eye++;
            clip_near_and_raster(v, s_gpu.tex.valid);
            return;
        }
        raster_tri(&v[0], &v[1], &v[2], s_gpu.tex.valid, 1);
        return;
    }

    /* Pre-transformed (XYZRHW) vertices: x, y in pixels, z as given, and
     * 1/rhw as w so texture and colour stay perspective-correct. */
    memset(v, 0, sizeof v);
    for (i = 0; i < 3; i++) {
        float p[4], uv[2];
        int k;
        if (!fetch_attr(&s_gpu.attr[0], idx[i], p))
            return;
        /* The combiners' inputs: texcoords 0-3 are attributes 9-12,
         * specular is attribute 4. An absent one reads as 0,0,0,1. */
        for (k = 0; k < 4; k++)
            fetch_attr(&s_gpu.attr[9 + k], idx[i], v[i].t[k]);
        fetch_attr(&s_gpu.attr[4], idx[i], v[i].c1);
        v[i].x = p[0];
        v[i].y = p[1];
        v[i].z = p[2];
        v[i].w = p[3] > 0.0f && isfinite(p[3]) ? 1.0f / p[3] : 1.0f;
        if (textured && fetch_texcoord(idx[i], uv)) {
            v[i].u = uv[0];
            v[i].v = uv[1];
        } else {
            textured = 0;
            v[i].u = v[i].v = 0.0f;
        }
        unpack_argb(smooth ? vertex_color(idx[i]) : argb, v[i].c);
    }
    raster_tri(&v[0], &v[1], &v[2], textured, 1);
}

/* ── The D3D11 backend ───────────────────────────────────────────────
 *
 * With RECOMP_GPU=d3d11 a batch is described to nv2a_d3d11.c instead of
 * rasterised here. Everything above -- decoding, the attribute table, the
 * texture stages, the combiner and vertex-program state -- is the same state
 * either way; only what happens at a draw changes. */

/* Targets and raster state, shared by draws and clears. */
static void hw_fill_targets(Nv2aHwDraw *d)
{
    uint32_t cbase = s_gpu.color_offset ? dma_resolve(s_gpu.color_offset) : 0;

    memset(d, 0, sizeof *d);
    if (cbase && surface_hits_image(cbase, (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch))
        cbase = 0;
    d->color_va = cbase;
    d->zeta_va = s_gpu.zeta_offset ? dma_resolve(s_gpu.zeta_offset) : 0;
    d->pitch = s_gpu.pitch;
    d->zpitch = s_gpu.zpitch;
    d->format = s_gpu.format;
    d->clip_x = s_gpu.clip_x;
    d->clip_y = s_gpu.clip_y;
    d->clip_w = s_gpu.clip_w;
    d->clip_h = s_gpu.clip_h;
    d->depth_test = s_gpu.depth_test;
    d->depth_func = s_gpu.depth_func;
    d->depth_mask = s_gpu.depth_mask;
    d->stencil_test = s_gpu.stencil_test;
    d->stencil_mask = s_gpu.stencil_mask;
    d->stencil_func = s_gpu.stencil_func;
    d->stencil_ref = s_gpu.stencil_ref;
    d->stencil_func_mask = s_gpu.stencil_func_mask;
    d->stencil_fail = s_gpu.stencil_fail;
    d->stencil_zfail = s_gpu.stencil_zfail;
    d->stencil_zpass = s_gpu.stencil_zpass;
    d->alpha_test = s_gpu.alpha_test;
    d->alpha_func = s_gpu.alpha_func;
    d->alpha_ref = s_gpu.alpha_ref;
    d->blend_enable = s_gpu.blend_enable;
    d->blend_sfactor = s_gpu.blend_sfactor;
    d->blend_dfactor = s_gpu.blend_dfactor;
    d->blend_eq = s_gpu.blend_eq;
    d->blend_color = s_gpu.blend_color;
    d->color_mask = s_gpu.color_mask;
    d->cull_enable = s_gpu.cull_enable;
    d->cull_face = s_gpu.cull_face;
    d->front_face = s_gpu.front_face;
    d->fog_enable = s_gpu.fog_enable;
    d->fog_mode = s_gpu.fog_mode;
    memcpy(d->fog_param, s_gpu.fog_param, sizeof d->fog_param);
    d->vp = &s_vp;
    d->comb = &s_comb;
}

static void hw_draw(void)
{
    static Nv2aHwDraw d;
    uint32_t a;
    int i;

    hw_fill_targets(&d);
    d.prim = s_gpu.prim;
    d.idx = s_gpu.idx;
    d.idx_count = s_gpu.idx_count;
    for (a = 0; a < NV_VERTEX_ATTRS; a++) {
        d.attr[a].offset = s_gpu.attr[a].offset;
        d.attr[a].type = s_gpu.attr[a].type;
        d.attr[a].size = s_gpu.attr[a].size;
        d.attr[a].stride = s_gpu.attr[a].stride;
    }
    if (s_gpu.inline_active) {
        d.inline_base = (const uint8_t *)s_gpu.inline_buf;
        d.inline_bytes = s_gpu.inline_count * 4;
    }
    d.use_vp = s_batch_vp;
    d.cur_attr = (const float (*)[4])s_cur_attr;
    d.cur_attr_set = s_cur_attr_set;
    d.diffuse_attr = (uint32_t)(color_attr() - s_gpu.attr);
    d.tex0_attr = (uint32_t)(texcoord_attr() - s_gpu.attr);
    for (i = 0; i < 4; i++) {
        const Texture *t = &s_stex[i];
        d.tex[i].va = t->offset;
        d.tex[i].format = t->format_raw;
        d.tex[i].color = t->color;
        d.tex[i].width = t->width;
        d.tex[i].height = t->height;
        d.tex[i].pitch = t->pitch;
        d.tex[i].address = t->address_raw;
        d.tex[i].filter = t->filter;
        d.tex[i].border = t->border;
        /* Before the combiners are programmed the software path samples
         * stage 0 whatever its enable bit says; so does this. */
        d.tex[i].enabled = t->valid && (s_stex_enable[i] || (!s_comb.programmed && i == 0));
    }
    nv2a_d3d11_draw(&d);
    if (d.color_va && (s_gpu.color_mask & 0x01010101u))
        s_gpu.drawn_offset = s_gpu.color_offset;
}

static void raster_batch(void)
{
    s_batch_no++;
    s_vcache_cur++;                        /* a new batch: forget the vertices */
    uint32_t i;
    uint32_t before = s_gpu.tris_drawn;

    if (s_gpu.idx_count < 3)
        return;
    /* A running vertex program decides where the vertices land, so the
     * screen-space guess below does not apply: run the program instead. */
    s_batch_vp = nv2a_vp_active(&s_vp) && !env_cached(&s_env_vp_off, "RECOMP_VP_OFF");

    /* RECOMP_FIND_FLOAT: scan once the menu's composite matrix has had the
     * x-row-is-(0,0,s,0) shape for a few hundred batches, i.e. in steady
     * state rather than on the first frame of a transition. */
    if (s_batch_vp && s_vp.constants[96][0] == 0.0f
            && s_vp.constants[96][1] == 0.0f && s_vp.constants[96][2] != 0.0f) {
        static unsigned seen_shape;
        if (++seen_shape == 400)
            find_float_matrices();
    }

    /* RECOMP_BATCH_TRACE=<n>[,<from>]: one line per batch -- the state it
     * draws with and where its first vertex lands. What a frame is made of
     * has to be known before any of it can be fixed. */
    {
        static int trace = -1;
        static unsigned from, stride = 1, seen, shown;
        if (trace < 0) {
            const char *e = getenv("RECOMP_BATCH_TRACE");
            char *end = NULL;
            trace = e ? (int)strtoul(e, &end, 0) : 0;
            if (end && *end == ',')
                from = (unsigned)strtoul(end + 1, &end, 0);
            if (end && *end == ',')
                stride = (unsigned)strtoul(end + 1, &end, 0);
            if (!stride)
                stride = 1;
        }
        if (trace > 0 && seen++ >= from && (seen - 1 - from) % stride == 0
                && shown < (unsigned)trace) {
            float p[4] = { 0, 0, 0, 0 };
            RVert v;
            shown++;
            fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p);
            memset(&v, 0, sizeof v);
            if (s_batch_vp)
                vp_vertex(s_gpu.idx[0], &v);
            fprintf(stderr, "  [BATCH] #%u mode 0x%X prim %u n %u surf %08X"
                    " fmt %X zeta %08X | z %u/%X/%u a %u/%X/%u blend %u %X/%X"
                    " tex %d %ux%u fmt %X | v0 (%g %g %g %g) -> (%g %g %g %g)"
                    " c (%.2f %.2f %.2f %.2f)\n",
                    seen - 1, s_vp.execution_mode, s_gpu.prim, s_gpu.idx_count,
                    s_gpu.color_offset, s_gpu.format, s_gpu.zeta_offset,
                    s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
                    s_gpu.alpha_test, s_gpu.alpha_func, s_gpu.alpha_ref,
                    s_gpu.blend_enable, s_gpu.blend_sfactor, s_gpu.blend_dfactor,
                    s_gpu.tex.valid, s_gpu.tex.width, s_gpu.tex.height,
                    s_gpu.tex.color, p[0], p[1], p[2], p[3],
                    v.x, v.y, v.z, v.w, v.c[0], v.c[1], v.c[2], v.c[3]);
            if (s_batch_vp) {
                const float (*c)[4] = (const float (*)[4])&s_vp.constants[96];
                fprintf(stderr, "          c96 (%g %g %g %g) (%g %g %g %g)"
                        " (%g %g %g %g) (%g %g %g %g)\n",
                        c[0][0], c[0][1], c[0][2], c[0][3],
                        c[1][0], c[1][1], c[1][2], c[1][3],
                        c[2][0], c[2][1], c[2][2], c[2][3],
                        c[3][0], c[3][1], c[3][2], c[3][3]);
            }
        }
    }

    if (s_batch_vp) {
        s_vp_batches++;
        if (s_gpu.tex.valid) {
            s_gpu.batches_textured++;
            note_texture_use();
        } else {
            s_gpu.batches_no_tex++;
        }
    } else if (!batch_is_screen_space()) {
        s_gpu.batches_untransformed++;
        return;
    }

    /* Count why, once per batch: the texture stage cannot change inside one. */
    if (!s_batch_vp) {
        const VertexAttr *tc = texcoord_attr();

        if (!(tc->offset && tc->stride))
            s_gpu.batches_no_uv++;
        else if (!s_gpu.tex.valid)
            s_gpu.batches_no_tex++;
        else {
            s_gpu.batches_textured++;
            note_texture_use();
        }
    }

    /* RECOMP_VP_BAD=<n>: the first n program batches whose first vertices
     * come out of the interpreter non-finite or far off the surface, with
     * their attribute layout -- geometry that is drawn every frame and never
     * seen. */
    {
        static int want = -1, shown;
        if (want < 0) {
            const char *e = getenv("RECOMP_VP_BAD");
            want = e ? atoi(e) : 0;
        }
        if (want > 0 && shown < want && s_batch_vp && s_gpu.idx_count >= 9) {
            /* Collapsed: the first nine vertices all within a pixel of each
             * other -- a mesh whose transform put every vertex in one place.
             * Logged once per program and layout. */
            static uint32_t seen[64];
            static int nseen;
            RVert v, v0;
            uint32_t k, bad = 0, sig = s_vp.program_start * 2654435761u;
            float spread = 0.0f;
            for (k = 0; k < NV_VERTEX_ATTRS; k++)
                sig = (sig ^ (s_gpu.attr[k].type | s_gpu.attr[k].size << 4
                              | s_gpu.attr[k].stride << 8)) * 16777619u;
            memset(&v0, 0, sizeof v0);
            vp_vertex(s_gpu.idx[0], &v0);
            for (k = 1; k < 9; k++) {
                memset(&v, 0, sizeof v);
                vp_vertex(s_gpu.idx[k], &v);
                spread = fmaxf(spread, fabsf(v.x - v0.x) + fabsf(v.y - v0.y));
            }
            bad = spread < 1.0f;
            for (k = 0; k < (uint32_t)nseen; k++)
                if (seen[k] == sig)
                    bad = 0;
            /* RECOMP_VP_SKIN: skinned batches regardless, for comparing one
             * that renders with one that does not. */
            {
                static int skin = -1;
                if (skin < 0)
                    skin = getenv("RECOMP_VP_SKIN") != NULL;
                if (skin && s_gpu.attr[2].size && s_gpu.attr[3].size
                        && s_gpu.attr[3].type == 4 && s_gpu.idx_count > 100) {
                    bad = 1;
                    fprintf(stderr, "  [VPBAD] (skinned, spread %.1f px)\n", spread);
                }
            }
            if (bad && nseen < 64)
                seen[nseen++] = sig;
            if (bad) {
                uint32_t a;
                shown++;
                float in0[NV2A_VP_INPUTS][4];
                for (k = 0; k < NV2A_VP_INPUTS; k++)
                    fetch_attr(&s_gpu.attr[k], s_gpu.idx[0], in0[k]);
                fprintf(stderr, "  [VPBAD] batch %u prim %u n %u start %u: collapsed at (%g %g %g %g)\n",
                        s_batch_no, s_gpu.prim, s_gpu.idx_count, s_vp.program_start,
                        v0.x, v0.y, v0.z, v0.w);
                for (k = 0; k < NV2A_VP_INPUTS; k++)
                    if (s_gpu.attr[k].size || in0[k][0] != 0.0f || in0[k][1] != 0.0f
                            || in0[k][2] != 0.0f || in0[k][3] != 1.0f)
                        fprintf(stderr, "  [VPBAD]   in v%u = (%g %g %g %g)\n", k,
                                in0[k][0], in0[k][1], in0[k][2], in0[k][3]);
                /* Skinned layouts (weights in v2, indices in v3) get the whole
                 * program and constant state printed. */
                if (s_gpu.attr[2].size && s_gpu.attr[3].size
                        && s_gpu.attr[3].type == 4 && s_gpu.idx_count > 100) {
                    uint32_t pc;
                    for (pc = s_vp.program_start; pc < NV2A_VP_SLOTS; pc++) {
                        fprintf(stderr, "  [VPBAD]   slot %3u: %08X %08X %08X\n", pc,
                                s_vp.program[pc][1], s_vp.program[pc][2], s_vp.program[pc][3]);
                        if (s_vp.program[pc][3] & 1u)
                            break;
                    }
                    for (k = 0; k < NV2A_VP_CONSTANTS; k++)
                        fprintf(stderr, "  [VPBAD]   c[%3u] = (%g %g %g %g)\n", k,
                                s_vp.constants[k][0], s_vp.constants[k][1],
                                s_vp.constants[k][2], s_vp.constants[k][3]);
                    /* Where the palette came from: guest RAM holding the
                     * bits of the bone row this vertex reads, a0+1 (the
                     * pushbuffer's copy included). */
                    {
                        const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                        float probe[4][4];
                        uint32_t a0 = (uint32_t)floorf(in0[3][0] * s_vp.constants[168][0]);
                        static const uint32_t ranges[2][2] = {
                            { 0x00010000u, 0x04000000u },
                            { XBOX_CONTIG_BASE, XBOX_CONTIG_BASE + 0x04000000u } };
                        int r, hits = 0;
                        if (a0 + 1 < NV2A_VP_CONSTANTS) {
                            memcpy(probe[0], s_vp.constants[a0 + 1], 16);
                            fprintf(stderr, "  [VPBAD]   searching for c[%u] bits\n", a0 + 1);
                            for (r = 0; r < 2 && hits < 16; r++) {
                                uint32_t va;
                                for (va = ranges[r][0]; va + 16 <= ranges[r][1] && hits < 16; va += 4)
                                    if (!memcmp(mem + va, probe[0], 16)) {
                                        hits++;
                                        fprintf(stderr, "  [VPBAD]   row found at 0x%08X\n", va);
                                    }
                            }
                        }
                    }
                }
                for (a = 0; a < NV_VERTEX_ATTRS; a++)
                    if (s_gpu.attr[a].size)
                        fprintf(stderr, "  [VPBAD]   v%u type %u size %u stride %u @%08X\n", a,
                                s_gpu.attr[a].type, s_gpu.attr[a].size, s_gpu.attr[a].stride,
                                s_gpu.attr[a].offset);
                fflush(stderr);
            }
        }
    }

    if (nv2a_d3d11_active()) {
        hw_draw();
        return;
    }

    switch (s_gpu.prim) {
    case NV_PRIM_TRIANGLES:
        for (i = 0; i + 2 < s_gpu.idx_count; i += 3)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_STRIP:
        for (i = 0; i + 2 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:
        for (i = 1; i + 1 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[0], s_gpu.idx[i], s_gpu.idx[i+1],
                           vertex_color(s_gpu.idx[0]));
        break;
    case NV_PRIM_QUADS:
        /* Independent quads, four vertices each. A batch of eight is two
         * quads, not one six-triangle fan around the first vertex; with
         * exactly four the two agreed, which is why sharing the fan arm
         * looked right. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 4) {
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+2], s_gpu.idx[i+3],
                           vertex_color(s_gpu.idx[i]));
        }
        break;
    case NV_PRIM_QUAD_STRIP:
        /* Each vertex pair past the first closes another quad against the
         * pair before it. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 2) {
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+3],
                           vertex_color(s_gpu.idx[i]));
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+3], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        }
        break;
    default:
        break;                             /* points and lines: not yet */
    }

    if (s_gpu.tris_drawn && (s_gpu.tris_drawn % 500) == 0)
        fprintf(stderr, "  [GPU] %u triangles rasterised\n", s_gpu.tris_drawn);

    /* Capture the surface while the geometry is still on it.
     *
     * The periodic report dumps too, but a title clears every frame and draws
     * in only some of them, so a report almost always lands on a surface that
     * was wiped a moment ago -- which reads as "nothing was drawn" when the
     * triangles went down correctly just before it. A few frames that actually
     * contain geometry are worth more than any number of clears. */
    if (s_gpu.tris_drawn != before && s_drawn_dumps < FB_DUMP_AFTER_DRAW) {
        s_drawn_dumps++;
        dump_surface_bmp();
    }
}

/* What a batch actually contains. Before anything can be rasterised, the
 * question is what space attribute 0 arrives in: a title running a vertex
 * program hands over object-space positions that mean nothing without running
 * it, while pre-transformed screen-space coordinates can be drawn directly. */
static void draw_primitive(void)
{
    float v[4];
    uint32_t i;

    if (!s_gpu.prim || !s_gpu.idx_count)
        return;
    s_gpu.draws++;
    /* Every 200 was a line every few milliseconds once draws got cheap. */
    if ((s_gpu.draws % 50000) == 0)
        fprintf(stderr, "  [GPU] draw #%u\n", s_gpu.draws);
    s_gpu.verts += s_gpu.idx_count;

    /* How many batches carry coordinates at all, and what range they span.
     * A pipeline that decodes perfectly and draws nothing is indistinguishable
     * from one that never ran, unless the vertices themselves are measured. */
    {
        float p[4];
        if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p)) {
            if (p[0] != 0.0f || p[1] != 0.0f || p[2] != 0.0f) {
                s_gpu.nonzero_draws++;
                if (p[0] < s_gpu.min_x) s_gpu.min_x = p[0];
                if (p[0] > s_gpu.max_x) s_gpu.max_x = p[0];
                if (p[1] < s_gpu.min_y) s_gpu.min_y = p[1];
                if (p[1] > s_gpu.max_y) s_gpu.max_y = p[1];
            }
        }
    }

    raster_batch();

    if (env_cached(&s_env_verbose, "RECOMP_PB_EXEC_VERBOSE")) {
        static int shown;
        if (shown++ < 6) {
            fprintf(stderr, "  [GPU] prim %u, %u indices, pos attr:"
                            " off 0x%08X type %u size %u stride %u\n",
                    s_gpu.prim, s_gpu.idx_count, s_gpu.attr[0].offset,
                    s_gpu.attr[0].type, s_gpu.attr[0].size, s_gpu.attr[0].stride);
            /* The texture stage, for either kind of batch. This used to print
             * only for inline batches, which meant a title drawing through
             * vertex arrays -- Half-Life 2's menu, for one -- showed no
             * texture state at all, and the reason a quad sampled flat was
             * invisible. */
            fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                            " colour 0x%02X swizzled %d valid %d\n",
                    s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                    s_gpu.tex.pitch, s_gpu.tex.color,
                    d3d8_format_is_swizzled(s_gpu.tex.color), s_gpu.tex.valid);
            {
                uint32_t k;
                for (k = 0; k < s_gpu.idx_count && k < 3; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
            }
            /* An inline batch has no guest buffer to go and look at -- the
             * vertices are the payload -- so print the payload too. */
            if (s_gpu.inline_active) {
                uint32_t k;
                fprintf(stderr, "  [GPU]   inline %u dwords:", s_gpu.inline_count);
                for (k = 0; k < s_gpu.inline_count && k < 16; k++)
                    fprintf(stderr, " %08X", s_gpu.inline_buf[k]);
                fprintf(stderr, "\n");
                for (k = 0; k < s_gpu.idx_count && k < 4; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
                fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                                " colour 0x%02X valid %d\n",
                        s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                        s_gpu.tex.pitch, s_gpu.tex.color, s_gpu.tex.valid);
            }
            {
                /* Every attribute the batch has, not just position. If the
                 * other streams carry data and position does not, the problem
                 * is one buffer rather than the whole vertex path. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t a, k;
                for (a = 0; a < NV_VERTEX_ATTRS; a++) {
                    const VertexAttr *at = &s_gpu.attr[a];
                    uint32_t nz = 0;
                    if (!at->offset || !at->size)
                        continue;
                    for (k = 0; k < 64; k++)
                        if (mem[at->offset + k]) nz++;
                    fprintf(stderr, "  [GPU]   attr%-2u off 0x%08X type %u"
                                    " size %u stride %-3u  %u/64 bytes set\n",
                            a, at->offset, at->type, at->size, at->stride, nz);
                }
            }
            {
                /* Raw bytes at the array, in case the values read as zero:
                 * that looks the same whether the offset is wrong or the
                 * buffer genuinely has not been filled yet. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t k;
                fprintf(stderr, "  [GPU]   bytes @0x%08X:", s_gpu.attr[0].offset);
                for (k = 0; k < 32; k++)
                    fprintf(stderr, " %02X", mem[s_gpu.attr[0].offset + k]);
                fprintf(stderr, "\n");
                fprintf(stderr, "  [GPU]   indices:");
                for (k = 0; k < s_gpu.idx_count && k < 8; k++)
                    fprintf(stderr, " %u", s_gpu.idx[k]);
                fprintf(stderr, "\n");
            }
            for (i = 0; i < s_gpu.idx_count && i < 3; i++) {
                if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], v))
                    fprintf(stderr, "  [GPU]   v[%u] = %.3f %.3f %.3f %.3f\n",
                            s_gpu.idx[i], v[0], v[1], v[2], v[3]);
            }
        }
    }
}

/* Draw the vertices the title wrote straight into the pushbuffer.
 *
 * INLINE_ARRAY carries no offsets and no indices: the dwords between BEGIN and
 * END *are* the vertex buffer, packed in attribute order using the same
 * SET_VERTEX_DATA_ARRAY_FORMAT registers an ordinary array would use. So the
 * whole batch is describable as a vertex array whose base happens to be that
 * payload, which means synthesising the layout and handing it to the existing
 * path -- rather than a second copy of the topology and rasterisation code.
 *
 * The title's own attribute table is saved and put back: these offsets and
 * strides are ours, and it has not stopped using its.
 *
 * ponytail: each attribute is padded to a whole dword. That is exact for the
 * float and D3DCOLOR formats every inline batch actually uses; a packed
 * sub-dword attribute would need the unpadded layout.
 */
static void draw_inline_array(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t off = 0, a, i, vsize, count;

    memcpy(saved, s_gpu.attr, sizeof saved);

    for (a = 0; a < NV_VERTEX_ATTRS; a++) {
        uint32_t bytes;
        if (!s_gpu.attr[a].size)
            continue;
        switch (s_gpu.attr[a].type) {
        case 0:  bytes = 4;                        break;  /* D3DCOLOR   */
        case 2:  bytes = 4 * s_gpu.attr[a].size;   break;  /* float      */
        case 4:  bytes = s_gpu.attr[a].size;       break;  /* ubyte norm */
        default: bytes = 4 * s_gpu.attr[a].size;   break;
        }
        s_gpu.attr[a].offset = off;
        off += (bytes + 3u) & ~3u;
    }
    vsize = off;
    if (!vsize)
        goto out;

    count = (s_gpu.inline_count * 4) / vsize;
    if (count < 3 || count > NV_MAX_INDICES)
        goto out;
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].size)
            s_gpu.attr[a].stride = vsize;

    for (i = 0; i < count; i++)
        s_gpu.idx[i] = i;
    s_gpu.idx_count = count;

    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

out:
    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
}

/* Draw the vertices SET_VERTEX3F/4F completed.
 *
 * Same trick as draw_inline_array: rather than a second copy of the topology
 * and rasterisation code, describe what was accumulated as an ordinary vertex
 * array and hand it to the existing path. The layout is ours and fixed, so
 * the attribute table is written out here rather than derived from the
 * title's format registers.
 *
 * The title's own table is saved and put back -- it has not stopped using it.
 *
 * ponytail: position, diffuse and texcoord0 only. That is what a 2D quad
 * carries and what this rasteriser samples; a second texcoord set or a normal
 * would need the D3D11 translator, not more slots here.
 */
static void draw_immediate(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t i;

    if (s_gpu.imm_count < 3)
        return;

    memcpy(saved, s_gpu.attr, sizeof saved);
    memset(s_gpu.attr, 0, sizeof s_gpu.attr);
    /* Offsets are byte offsets into inline_buf here, not guest addresses --
     * fetch_attr reads them that way while inline_active is set, which is
     * also why 0 is a legal offset for position. */
    s_gpu.attr[0].type = 2; s_gpu.attr[0].size = 4;   /* position float4  */
    s_gpu.attr[0].offset = 0;
    s_gpu.attr[3].type = 0; s_gpu.attr[3].size = 4;   /* diffuse D3DCOLOR */
    s_gpu.attr[3].offset = 16;
    s_gpu.attr[9].type = 2; s_gpu.attr[9].size = 2;   /* texcoord0 float2 */
    s_gpu.attr[9].offset = 20;
    s_gpu.attr[0].stride = s_gpu.attr[3].stride = s_gpu.attr[9].stride =
        IMM_VERTEX_DWORDS * 4;

    for (i = 0; i < s_gpu.imm_count && i < NV_MAX_INDICES; i++)
        s_gpu.idx[i] = i;
    s_gpu.idx_count = i;

    /* fetch_attr bounds-checks against inline_count dwords. */
    s_gpu.inline_count = s_gpu.imm_count * IMM_VERTEX_DWORDS;
    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
    s_gpu.inline_count = 0;
}

/* A vertex is complete: append it in the layout draw_immediate describes. */
static void imm_emit_vertex(void)
{
    uint32_t at = s_gpu.imm_count * IMM_VERTEX_DWORDS;

    if (!s_gpu.prim || at + IMM_VERTEX_DWORDS > NV_MAX_INLINE)
        return;
    memcpy(&s_gpu.inline_buf[at],     s_gpu.imm_pos, 4 * sizeof(float));
    memcpy(&s_gpu.inline_buf[at + 4], &s_gpu.imm_diffuse, sizeof(uint32_t));
    memcpy(&s_gpu.inline_buf[at + 5], s_gpu.imm_tex, 2 * sizeof(float));
    s_gpu.imm_count++;
}

/* The immediate-mode writes. Returns 1 if `method` was one of them.
 *
 * Split out because it is a range test against five separate bases, and that
 * reads better than five more cases in an already long switch.
 */
/* The current value of every vertex attribute, as SET_VERTEX_DATA* leaves it.
 *
 * An attribute with no array is not zero on NV2A: it is whatever the title
 * last set it to, and a title leans on that -- a rigid piece of a skinned
 * character can carry its bone index as a per-batch constant rather than a
 * per-vertex stream. Both draw paths read absent attributes from here. */
static float s_cur_attr[NV_VERTEX_ATTRS][4] = {
    {0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},
    {0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1},{0,0,0,1}
};
/* s_cur_attr_set: attributes a method has written (declared above). */

#define NV097_SET_VERTEX_DATA2S           0x1900   /* + attr*4,  2 shorts  */
#define NV097_SET_VERTEX_DATA4S_M         0x1980   /* + attr*8,  4 shorts  */

static int cur_attr_method(uint32_t method, uint32_t param)
{
    union { uint32_t u; float f; } v;
    uint32_t a, c;

    v.u = param;
    if (method >= NV097_SET_VERTEX_DATA2F_M
            && method < NV097_SET_VERTEX_DATA2F_M + NV_VERTEX_ATTRS * 8) {
        a = (method - NV097_SET_VERTEX_DATA2F_M) / 8;
        c = ((method - NV097_SET_VERTEX_DATA2F_M) % 8) / 4;
        s_cur_attr[a][c] = v.f;
        if (c == 1) { s_cur_attr[a][2] = 0.0f; s_cur_attr[a][3] = 1.0f; }
    } else if (method >= NV097_SET_VERTEX_DATA4F_M
            && method < NV097_SET_VERTEX_DATA4F_M + NV_VERTEX_ATTRS * 16) {
        a = (method - NV097_SET_VERTEX_DATA4F_M) / 16;
        s_cur_attr[a][((method - NV097_SET_VERTEX_DATA4F_M) % 16) / 4] = v.f;
    } else if (method >= NV097_SET_VERTEX_DATA2S
            && method < NV097_SET_VERTEX_DATA2S + NV_VERTEX_ATTRS * 4) {
        a = (method - NV097_SET_VERTEX_DATA2S) / 4;
        s_cur_attr[a][0] = (float)(int16_t)(param & 0xFFFFu);
        s_cur_attr[a][1] = (float)(int16_t)(param >> 16);
        s_cur_attr[a][2] = 0.0f;
        s_cur_attr[a][3] = 1.0f;
    } else if (method >= NV097_SET_VERTEX_DATA4UB
            && method < NV097_SET_VERTEX_DATA4UB + NV_VERTEX_ATTRS * 4) {
        /* Bytes in order, normalised: x is the low byte (xemu's reading). */
        a = (method - NV097_SET_VERTEX_DATA4UB) / 4;
        for (c = 0; c < 4; c++)
            s_cur_attr[a][c] = (float)((param >> (8 * c)) & 0xFFu) / 255.0f;
    } else if (method >= NV097_SET_VERTEX_DATA4S_M
            && method < NV097_SET_VERTEX_DATA4S_M + NV_VERTEX_ATTRS * 8) {
        a = (method - NV097_SET_VERTEX_DATA4S_M) / 8;
        c = ((method - NV097_SET_VERTEX_DATA4S_M) % 8) / 4 * 2;
        s_cur_attr[a][c]     = (float)(int16_t)(param & 0xFFFFu);
        s_cur_attr[a][c + 1] = (float)(int16_t)(param >> 16);
    } else {
        return 0;
    }
    s_cur_attr_set |= 1u << a;
    return 1;
}

static int imm_vertex_method(uint32_t method, uint32_t param)
{
    union { uint32_t u; float f; } v;
    v.u = param;

    if (cur_attr_method(method, param)
            && (method >= NV097_SET_VERTEX_DATA2S && method < NV097_SET_VERTEX_DATA2S + NV_VERTEX_ATTRS * 4
                || method >= NV097_SET_VERTEX_DATA4S_M && method < NV097_SET_VERTEX_DATA4S_M + NV_VERTEX_ATTRS * 8))
        return 1;                        /* no immediate-mode meaning of its own */

    if (method >= NV097_SET_VERTEX4F && method < NV097_SET_VERTEX4F + 16) {
        uint32_t c = (method - NV097_SET_VERTEX4F) / 4;
        s_gpu.imm_pos[c] = v.f;
        if (c == 3)                      /* w completes the vertex */
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX3F && method < NV097_SET_VERTEX3F + 12) {
        uint32_t c = (method - NV097_SET_VERTEX3F) / 4;
        s_gpu.imm_pos[c] = v.f;
        if (c == 2) {                    /* z completes it, w is implicitly 1 */
            s_gpu.imm_pos[3] = 1.0f;
            imm_emit_vertex();
        }
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA2F_M
            && method < NV097_SET_VERTEX_DATA2F_M + NV_VERTEX_ATTRS * 8) {
        uint32_t off = method - NV097_SET_VERTEX_DATA2F_M;
        if (off / 8 == 9)                /* attribute 9 is texture coord 0 */
            s_gpu.imm_tex[(off % 8) / 4] = v.f;
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4F_M
            && method < NV097_SET_VERTEX_DATA4F_M + NV_VERTEX_ATTRS * 16) {
        uint32_t off = method - NV097_SET_VERTEX_DATA4F_M;
        uint32_t attr = off / 16, c = (off % 16) / 4;
        if (attr == 0) {
            s_gpu.imm_pos[c] = v.f;
            if (c == 3)
                imm_emit_vertex();
        } else if (attr == 9 && c < 2) {
            s_gpu.imm_tex[c] = v.f;
        }
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4UB
            && method < NV097_SET_VERTEX_DATA4UB + NV_VERTEX_ATTRS * 4) {
        if ((method - NV097_SET_VERTEX_DATA4UB) / 4 == 3)   /* diffuse */
            s_gpu.imm_diffuse = param;
        return 1;
    }
    return 0;
}
/* Frame pacing on the title's own thread, for a title whose present call is
 * hooked (Amped 2: D3DDevice_Swap, sub_0031C4B0). Waiting at FLIP_STALL
 * instead holds the executor, and the title then waits on the executor's
 * flush acknowledgements too, so every frame paid the wait twice -- about
 * 50 fps in game where it runs 64 unpaced. Waiting before the swap leaves
 * the executor draining while the title sleeps. The first call turns the
 * FLIP_STALL wait off. RECOMP_FLIP_NOWAIT still disables pacing. */
static volatile int s_present_paced;

void xbox_PresentPace(void)
{
    static LARGE_INTEGER qf;
    static LONGLONG last;
    static int off = -1;
    LARGE_INTEGER now;
    LONGLONG due;

    if (off < 0) {
        off = getenv("RECOMP_FLIP_NOWAIT") != NULL;
        QueryPerformanceFrequency(&qf);
    }
    s_present_paced = 1;
    if (off)
        return;
    QueryPerformanceCounter(&now);
    due = last + qf.QuadPart / 60;
    while (last && now.QuadPart < due) {
        if (due - now.QuadPart > qf.QuadPart / 500)
            Sleep(1);
        else
            YieldProcessor();
        QueryPerformanceCounter(&now);
    }
    /* A frame late does not bank time for the next one. */
    last = (last && now.QuadPart - due < qf.QuadPart / 60) ? due : now.QuadPart;
    /* One line per 600 presents: the rate, and how much of it was waiting. */
    {
        static unsigned n;
        static LONGLONG t0, waited;
        static LARGE_INTEGER before;
        if (!t0)
            t0 = now.QuadPart;
        (void)before;
        if (n >= 600) {
            n = 0;
            t0 = now.QuadPart;
        }
        if (++n == 600 && getenv("RECOMP_PACE_TRACE")) {
            fprintf(stderr, "  [PACE] 600 presents in %.2f s (%.1f/s)\n",
                    (double)(now.QuadPart - t0) / (double)qf.QuadPart,
                    600.0 * (double)qf.QuadPart / (double)(now.QuadPart - t0));
            t0 = now.QuadPart;
            (void)waited;
        }
    }
}

void nv2a_pb_exec_method(uint32_t subch, uint32_t method, uint32_t param)
{
    static int inited;
    if (!inited) {
        inited = 1;
        s_gpu.min_x = s_gpu.min_y = 1e30f;
        s_gpu.max_x = s_gpu.max_y = -1e30f;
    }
    /* Bring-up: the first parameters each surface method carries. A wrong
     * pitch or clip is indistinguishable from a method never arriving unless
     * the values are visible. */
    if (env_cached(&s_env_verbose, "RECOMP_PB_EXEC_VERBOSE")) {
        static int shown[8];
        int slot = -1;
        switch (method) {
        case NV097_SET_SURFACE_CLIP_HORIZONTAL: slot = 0; break;
        case NV097_SET_SURFACE_CLIP_VERTICAL:   slot = 1; break;
        case NV097_SET_SURFACE_FORMAT:          slot = 2; break;
        case NV097_SET_SURFACE_PITCH:           slot = 3; break;
        case NV097_SET_SURFACE_COLOR_OFFSET:    slot = 4; break;
        case NV097_SET_COLOR_CLEAR_VALUE:       slot = 5; break;
        case NV097_CLEAR_SURFACE:               slot = 6; break;
        default: break;
        }
        if (slot >= 0 && shown[slot]++ < 4)
            fprintf(stderr, "  [GPU] subch %u method 0x%04X param 0x%08X\n",
                    subch, method, param);
    }

    if (subch != 0) {                      /* 3D class lives on subchannel 0 */
        note_unhandled(method, param);
        return;
    }
    if (nv2a_comb_method(&s_comb, method, param))
        return;
    switch (method) {
    case NV097_SET_SURFACE_CLIP_HORIZONTAL:
        s_gpu.clip_x = param & 0xFFFF;
        s_gpu.clip_w = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_CLIP_VERTICAL:
        s_gpu.clip_y = param & 0xFFFF;
        s_gpu.clip_h = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_FORMAT:
        s_gpu.format = param;
        break;
    case NV097_SET_SURFACE_PITCH:
        s_gpu.pitch = param & 0xFFFF;      /* colour pitch; zeta is the top half */
        s_gpu.zpitch = param >> 16;
        break;
    case NV097_SET_SURFACE_COLOR_OFFSET:
        /* A frame, as far as the executor can tell: the colour target moving
         * to a different full-screen buffer. Counted with the wall clock so
         * the report can say how fast the title is actually running. */
        if (param != s_gpu.color_offset && param
                && s_gpu.clip_w >= 640 && s_gpu.clip_h >= 480) {
            s_frames_seen++;
            if (!s_frames_t0)
                s_frames_t0 = GetTickCount64();
        }
        s_gpu.color_offset = param;
        break;
    case NV097_SET_COLOR_CLEAR_VALUE:
        s_gpu.clear_color = param;
        break;
    case NV097_SET_BLEND_ENABLE:
        s_gpu.blend_enable = param;
        break;
    /* Depth and alpha test state; the tests themselves are in raster_tri. */
    case 0x0214:                           /* SET_SURFACE_ZETA_OFFSET */
        s_gpu.zeta_offset = param;
        break;
    case 0x030C:                           /* SET_DEPTH_TEST_ENABLE */
        s_gpu.depth_test = param;
        break;
    case 0x0354:                           /* SET_DEPTH_FUNC */
        s_gpu.depth_func = param;
        break;
    case 0x035C:                           /* SET_DEPTH_MASK */
        s_gpu.depth_mask = param;
        break;
    case 0x1D8C:                           /* SET_ZSTENCIL_CLEAR_VALUE */
        s_gpu.zclear = param;
        break;
    case 0x0300:                           /* SET_ALPHA_TEST_ENABLE */
        s_gpu.alpha_test = param;
        break;
    case 0x033C:                           /* SET_ALPHA_FUNC */
        s_gpu.alpha_func = param;
        break;
    case 0x0340:                           /* SET_ALPHA_REF */
        s_gpu.alpha_ref = param;
        break;
    case 0x0358:                           /* SET_COLOR_MASK: A R G B bytes */
        s_gpu.color_mask = param;
        break;
    case 0x02A4: s_gpu.fog_enable = param; break;         /* SET_FOG_ENABLE */
    case 0x029C: s_gpu.fog_mode = param; break;           /* SET_FOG_MODE */
    case 0x09C0: case 0x09C4: case 0x09C8: {              /* SET_FOG_PARAMS */
        union { uint32_t u; float f; } fv;
        fv.u = param;
        s_gpu.fog_param[(method - 0x09C0) / 4] = fv.f;
        break;
    }
    case 0x0308: s_gpu.cull_enable = param; break;        /* SET_CULL_FACE_ENABLE */
    case 0x039C: s_gpu.cull_face = param; break;          /* SET_CULL_FACE */
    case 0x03A0: s_gpu.front_face = param; break;         /* SET_FRONT_FACE */
    case 0x032C: s_gpu.stencil_test = param; break;       /* SET_STENCIL_TEST_ENABLE */
    case 0x0360: s_gpu.stencil_mask = param; break;       /* SET_STENCIL_MASK */
    case 0x0364: s_gpu.stencil_func = param; break;       /* SET_STENCIL_FUNC */
    case 0x0368: s_gpu.stencil_ref = param; break;        /* SET_STENCIL_FUNC_REF */
    case 0x036C: s_gpu.stencil_func_mask = param; break;  /* SET_STENCIL_FUNC_MASK */
    case 0x0370: s_gpu.stencil_fail = param; break;       /* SET_STENCIL_OP_FAIL */
    case 0x0374: s_gpu.stencil_zfail = param; break;      /* SET_STENCIL_OP_ZFAIL */
    case 0x0378: s_gpu.stencil_zpass = param; break;      /* SET_STENCIL_OP_ZPASS */
    case 0x034C: s_gpu.blend_color = param; break;        /* SET_BLEND_COLOR */
    case 0x0350: s_gpu.blend_eq = param; break;           /* SET_BLEND_EQUATION */
    case NV097_SET_BLEND_FUNC_SFACTOR:
        s_gpu.blend_sfactor = param;
        break;
    case NV097_SET_BLEND_FUNC_DFACTOR:
        s_gpu.blend_dfactor = param;
        break;
    case NV097_CLEAR_SURFACE:
        clear_surface(param);
        break;

    case NV097_SET_BEGIN_END:
        if (param) {
            s_gpu.prim = param;
            s_gpu.idx_count = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        } else {
            /* Three ways a batch can have arrived, and only one is in use at
             * a time: vertices completed by SET_VERTEX4F, a payload written
             * with INLINE_ARRAY, or indices into the title's own arrays. */
            if (s_gpu.imm_count)
                draw_immediate();
            else if (s_gpu.inline_count)
                draw_inline_array();
            else
                draw_primitive();
            s_gpu.prim = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        }
        break;

    case NV097_INLINE_ARRAY:
        /* Vertex data, not a pointer to it. Buffered rather than decoded here
         * because the format is only fully known at END. */
        if (s_gpu.prim && s_gpu.inline_count < NV_MAX_INLINE)
            s_gpu.inline_buf[s_gpu.inline_count++] = param;
        break;

    case NV097_DRAW_ARRAYS: {
        /* The method this title actually draws with, and the reason the
         * executor reported zero draws while geometry was being submitted the
         * whole time: BEGIN_END arrived, END arrived, and in between came a
         * run description rather than the index list the draw path wanted, so
         * every batch ended with idx_count == 0 and was dropped in silence.
         *
         * Expanded into indices because that is what the rasteriser consumes,
         * and an implicit run is just the indices start..start+count-1. */
        uint32_t start = param & 0x00FFFFFFu;
        uint32_t count = ((param >> 24) & 0xFFu) + 1u;
        uint32_t i;

        if (!s_gpu.prim)
            break;
        for (i = 0; i < count && s_gpu.idx_count < NV_MAX_INDICES; i++)
            s_gpu.idx[s_gpu.idx_count++] = start + i;
        break;
    }

    case 0x1808:                           /* ARRAY_ELEMENT32 */
        if (s_gpu.prim && s_gpu.idx_count < NV_MAX_INDICES)
            s_gpu.idx[s_gpu.idx_count++] = param;
        break;

    case NV097_ARRAY_ELEMENT16:
        /* Two 16-bit indices per parameter word. */
        if (s_gpu.prim && s_gpu.idx_count + 2 <= NV_MAX_INDICES) {
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param & 0xFFFF);
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param >> 16);
        }
        break;
    case NV097_SET_TEXTURE_OFFSET:
        /* A texture offset is a DMA-object offset, exactly like a surface or a
         * vertex array offset -- physical, and reachable only through the
         * contiguous window when it names contiguous memory. */
        s_gpu.tex.offset = dma_resolve(param);
        record_tex_reg(method, param);
        break;

    case NV097_SET_FLIP_READ:
        s_gpu.flip_read = param;
        return;

    case NV097_SET_FLIP_WRITE:
        s_gpu.flip_write = param;
        return;

    case NV097_SET_FLIP_MODULO:
        s_gpu.flip_modulo = param;
        return;

    case NV097_FLIP_INCREMENT_WRITE:
        s_gpu.flip_write = s_gpu.flip_modulo
                         ? (s_gpu.flip_write + 1) % s_gpu.flip_modulo
                         : s_gpu.flip_write + 1;
        s_gpu.flips++;
        return;

    case NV097_FLIP_STALL:
        /* The stall ends when the buffer being read is the one just finished,
         * which on the console is the next vblank. Waiting for it is what
         * paces the title: the software rasteriser was slow enough never to
         * notice, and on the GPU the title ran its frame loop at 90 Hz.
         * RECOMP_FLIP_INTERVAL=<n> waits n vblanks instead of one, and
         * RECOMP_FLIP_NOWAIT=1 flips at once, as before. */
        {
            extern volatile long g_xbox_vblank_count;
            static long last_vb;
            static int nowait = -1;
            static long interval = 1;
            if (nowait < 0) {
                const char *iv = getenv("RECOMP_FLIP_INTERVAL");
                nowait = getenv("RECOMP_FLIP_NOWAIT") != NULL || !getenv("RECOMP_VBLANK");
                if (iv && atoi(iv) > 0)
                    interval = atoi(iv);
            }
            /* Once per presented frame. FLIP_STALL arrives several times per
             * frame, and pacing every one of them waited out a sixtieth each
             * time: 23 fps in game where the title manages 64 unpaced, and
             * the menus held at 44. Only a stall that follows a new
             * FLIP_INCREMENT_WRITE -- a frame actually queued -- waits. */
            static uint32_t paced_flips;
            if (!nowait && !s_present_paced && s_gpu.flips != paced_flips) {
                paced_flips = s_gpu.flips;
                /* At least `interval` sixtieths of a second since the last
                 * flip, measured on the clock rather than waited out to the
                 * next vblank. Nothing scans out here -- frames reach the
                 * window by readback -- so aligning to vblank bought no
                 * smoothness, and it cost a frame that ran 20 ms a wait to
                 * 33, which averaged about 40 fps. The vblank interrupt keeps
                 * its own 60 Hz for the title. RECOMP_FLIP_VBLANK=1 waits for
                 * vblanks as before. */
                static int by_vblank = -1;
                static LARGE_INTEGER qf, last_t;
                if (by_vblank < 0) {
                    by_vblank = getenv("RECOMP_FLIP_VBLANK") != NULL;
                    QueryPerformanceFrequency(&qf);
                }
                if (by_vblank) {
                    ULONGLONG t0 = GetTickCount64();
                    while (g_xbox_vblank_count - last_vb < interval
                           && GetTickCount64() - t0 < 100)
                        Sleep(1);
                } else {
                    LARGE_INTEGER now;
                    LONGLONG due = last_t.QuadPart + qf.QuadPart * interval / 60;
                    QueryPerformanceCounter(&now);
                    extern void nv2a_pb_scan_progress(void);
                    /* Let D3D have the space already consumed while the
                     * flip waits, as the stalled pusher would. */
                    nv2a_pb_scan_progress();
                    while (last_t.QuadPart && now.QuadPart < due) {
                        nv2a_pb_scan_progress();
                        if (due - now.QuadPart > qf.QuadPart / 500)
                            Sleep(1);             /* more than 2 ms to go */
                        else
                            YieldProcessor();
                        QueryPerformanceCounter(&now);
                    }
                    last_t = now;
                }
                last_vb = g_xbox_vblank_count;
            }
        }
        s_gpu.flip_read = s_gpu.flip_write;
        /* And this is a completed swap, which is what a title's own swap
         * counter counts -- see xbox_Nv2aFrameCounterFlip. */
        xbox_Nv2aFrameCounterFlip();
        /* Hand the window a copy of the frame just finished.
         *
         * The buffer the title has finished is the one the last batch drew
         * into, which is what drawn_offset holds and why it exists: by the
         * flip, color_offset has already moved to the next buffer. Copying
         * here, rather than letting the window read guest memory on its own
         * clock, is what stops it showing a surface the rasteriser is still
         * writing. */
        if (s_gpu.pitch) {
            extern void xbox_FramebufferWindowPresent(uint32_t, uint32_t);
            uint32_t done = s_gpu.drawn_offset ? s_gpu.drawn_offset
                                               : s_gpu.color_offset;
            if (done) {
                xbox_FramebufferWindowSet(dma_resolve(done), s_gpu.pitch);
                if (nv2a_d3d11_active())
                    nv2a_d3d11_flip(dma_resolve(done));
                else
                    xbox_FramebufferWindowPresent(dma_resolve(done), s_gpu.pitch);
            }
        }
        if (env_cached(&s_env_verbose, "RECOMP_PB_EXEC_VERBOSE")) {
            static unsigned n;
            if (n++ < 8) {
                fprintf(stderr, "  [GPU] flip %u: read=%u write=%u\n",
                        s_gpu.flips, s_gpu.flip_read, s_gpu.flip_write);
                fflush(stderr);
            }
        }
        return;

    case NV097_SET_TEXTURE_FORMAT:
        s_gpu.tex.color = (param >> 8) & 0xFF;
        /* A swizzled texture carries its own dimensions here, as log2 in
         * BASE_SIZE_U/V (nv2a_regs.h: 0x00F00000 / 0x0F000000). It has to:
         * SET_TEXTURE_IMAGE_RECT describes a linear image, and a title that
         * only uses swizzled textures never sends one -- this title sends it
         * once and sets a format 3,176 times. Without this the width and
         * height stayed zero and nothing was ever sampled. */
        if (tex_size_from_format((param >> 8) & 0xFF)) {
            s_gpu.tex.width  = 1u << ((param >> 20) & 0xF);
            s_gpu.tex.height = 1u << ((param >> 24) & 0xF);
        }
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_ADDRESS:
        /* Four bits per axis. 1 is wrap, 3 is clamp-to-edge; the rest (mirror,
         * border) fall back to clamp, which is wrong at an edge rather than
         * wrong everywhere. */
        s_gpu.tex.addr_u =  param        & 0xF;
        s_gpu.tex.addr_v = (param >>  8) & 0xF;
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_CONTROL1:
        /* Pitch lives in the top half. Only meaningful for a linear format; a
         * swizzled texture has no pitch because it has no rows. */
        s_gpu.tex.pitch = param >> 16;
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_IMAGE_RECT:
        s_gpu.tex.width  = param >> 16;
        s_gpu.tex.height = param & 0xFFFF;
        record_tex_reg(method, param);
        break;

    default:
        if (method >= NV_TEX_FIRST && method <= NV_TEX_LAST)
            record_tex_reg(method, param);
        if (method >= NV097_SET_VERTEX_DATA_ARRAY_OFFSET
                && method < NV097_SET_VERTEX_DATA_ARRAY_OFFSET + NV_VERTEX_ATTRS * 4) {
            /* Resolved here, once, so every consumer -- the rasteriser's
             * attribute reads and the diagnostics alike -- sees the same
             * address. A vertex array offset is a DMA-object offset exactly
             * like a surface offset: physical, and addressable only through
             * the window when it names contiguous memory. */
            s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4].offset =
                dma_resolve(param);
        } else if (method >= NV097_SET_VERTEX_DATA_ARRAY_FORMAT
                && method < NV097_SET_VERTEX_DATA_ARRAY_FORMAT + NV_VERTEX_ATTRS * 4) {
            VertexAttr *a = &s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4];
            a->type   =  param        & 0x0F;
            a->size   = (param >> 4)  & 0x0F;
            a->stride = (param >> 8)  & 0xFF;
        } else if (nv2a_vp_method(&s_vp, method, param)) {
            /* vertex program / constant upload, or its control registers */
        } else if (!imm_vertex_method(method, param)) {
            note_unhandled(method, param);
        }
        break;
    }
}

/* Find where the title actually wrote its quad.
 *
 * The GPU is pointed at a buffer that stays zero, which says the data went
 * somewhere else -- and the only way to find somewhere else is to look for the
 * data. A screen-space quad for a 640x480 target contains 640.0f and 480.0f as
 * floats, which is a distinctive enough pair to search guest RAM for. Whatever
 * address that turns up is where the title's writes are landing, and the
 * difference from the programmed offset is the bug. */
static void find_quad_vertices(void)
{
    const uint32_t W = 0x44200000u;   /* 640.0f */
    const uint32_t H = 0x43F00000u;   /* 480.0f */
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for 640.0f/480.0f pairs...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 8 && hits < 12; i++) {
        if (ram[i] != W && ram[i] != H)
            continue;
        /* Both values within a few words of each other: a lone 640.0f is
         * common, the pair much less so. */
        {
            int has_w = 0, has_h = 0;
            uint32_t k;
            for (k = 0; k < 8; k++) {
                if (ram[i + k] == W) has_w = 1;
                if (ram[i + k] == H) has_h = 1;
            }
            if (!has_w || !has_h)
                continue;
        }
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 8; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 8;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none found -- the quad is not in RAM in"
                        " that form\n");
    fflush(stderr);
}

/* Locate NaN-filled transform matrices in guest RAM.
 *
 * A matrix arriving as NaN says the maths went wrong somewhere upstream, and
 * the only way to find where is to find the matrix and watch who writes it.
 * Three consecutive real-indefinite values is a distinctive enough signature:
 * ordinary data does not contain runs of 0xFFC00000. */
static void find_nan_matrices(void)
{
    const uint32_t NAN_NEG = 0xFFC00000u;
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for NaN matrices...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 20 && hits < 10; i++) {
        if (ram[i] != NAN_NEG || ram[i + 1] != NAN_NEG || ram[i + 2] != NAN_NEG)
            continue;
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 16; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 16;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none in RAM -- the NaNs are computed into"
                        " registers, not stored\n");
    fflush(stderr);
}

/* RECOMP_FIND_FLOAT=<value>: guest RAM windows of 16 floats holding at least
 * four of +/-value, printed as floats -- a matrix the title keeps in memory
 * is found by the numbers it is made of, and its address leads back to the
 * code that builds it. Called once, from the batch that first shows the
 * shape being chased. */
static void find_float_matrices(void)
{
    const char *spec = getenv("RECOMP_FIND_FLOAT");
    const float *ram = (const float *)xbox_GetMemoryOffset();
    float want;
    uint32_t i, hits = 0;

    if (!spec)
        return;
    want = (float)atof(spec);
    fprintf(stderr, "[GPU] searching guest RAM for matrices of +/-%g...\n", want);
    for (i = 0x1000 / 4; i + 16 < 0x04000000u / 4 && hits < 40; i += 4) {
        int k, n = 0;
        for (k = 0; k < 16; k++) {
            float v = fabsf(ram[i + k]);
            if (isfinite(v) && fabsf(v - want) < 2e-4f)
                n++;
        }
        if (n < 4)
            continue;
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        for (k = 0; k < 16; k++)
            fprintf(stderr, "%s%g", (k % 4) ? " " : " | ", ram[i + k]);
        fprintf(stderr, "\n");
        /* RECOMP_FIND_FLOAT_CTX=<rows>: the 16-byte rows either side too,
         * because a hit window rarely starts where the matrix does. */
        {
            static int ctx = -1;
            int r;
            if (ctx < 0) {
                const char *c = getenv("RECOMP_FIND_FLOAT_CTX");
                ctx = c ? atoi(c) : 0;
            }
            for (r = -ctx; r < ctx + 4 && hits <= 8; r++) {
                uint32_t row = i + (uint32_t)(r * 4);
                if (row < 0x1000 / 4 || row + 4 >= 0x04000000u / 4)
                    continue;
                fprintf(stderr, "            %08X: %g %g %g %g\n", row * 4,
                        ram[row], ram[row + 1], ram[row + 2], ram[row + 3]);
            }
        }
        i += 12;
    }
    fprintf(stderr, "  [GPU]   %u windows\n", hits);
    fflush(stderr);
}

/* Print guest dwords named by RECOMP_PEEK, as hex and as float.
 *
 * Chasing a value backwards means reading it, and a value that is only wrong
 * for one frame in a thousand cannot be caught by stopping. Both
 * interpretations are printed because the question is usually "is this a
 * pointer or a number", and guessing wrong costs a run. */
static void peek_addresses(void)
{
    const char *spec = getenv("RECOMP_PEEK");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256];
    char *tok, *ctx = NULL;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t va = (uint32_t)strtoul(tok, NULL, 0);
        uint32_t v;
        float f;
        if (va < 0x1000u || va >= 0x04000000u)
            continue;
        v = *(const uint32_t *)(mem + va);
        memcpy(&f, &v, 4);
        fprintf(stderr, "  [PEEK] 0x%08X = %08X  (%g)\n", va, v, f);
    }
    fflush(stderr);
}

/* Walk a pointer chain and print every step.
 *
 * RECOMP_PEEK_CHAIN="0x1315A8,8,0x10,0" starts at that address, and for each
 * offset dereferences the current pointer and adds it. Following a chain by
 * hand costs one run per level; this costs one run for the whole chain, and
 * prints where it goes wrong when a level is null.
 */
static void peek_chain(void)
{
    const char *spec = getenv("RECOMP_PEEK_CHAIN");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256], *tok, *ctx = NULL;
    uint32_t cur = 0;
    int step = 0;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t off = (uint32_t)strtoul(tok, NULL, 0);
        if (step == 0) {
            cur = off;
            fprintf(stderr, "  [CHAIN] start 0x%08X\n", cur);
        } else {
            if (cur < 0x1000u || cur + 4 >= 0x04000000u) {
                fprintf(stderr, "  [CHAIN] step %d: 0x%08X is not a guest"
                                " pointer -- chain ends\n", step, cur);
                return;
            }
            cur = *(const uint32_t *)(mem + cur) + off;
            fprintf(stderr, "  [CHAIN] step %d: deref +0x%X -> 0x%08X\n",
                    step, off, cur);
        }
        step++;
    }
    if (cur >= 0x1000u && cur + 4 < 0x04000000u)
        fprintf(stderr, "  [CHAIN] final value at 0x%08X = 0x%08X\n",
                cur, *(const uint32_t *)(mem + cur));
    fflush(stderr);
}

void nv2a_pb_exec_report(void)
{
    nv2a_d3d11_report();
    peek_addresses();
    peek_chain();
    if (getenv("RECOMP_FIND_NAN")) {
        /* Every report, not once: the matrix is fine early on and only turns
         * to NaN later, so a single scan at startup finds nothing and says
         * nothing. */
        find_nan_matrices();
    }
    if (getenv("RECOMP_FIND_QUAD")) {
        static int done;
        if (!done) { done = 1; find_quad_vertices(); }
    }
    int i, j;

    fprintf(stderr, "[GPU] surface 0x%08X pitch %u clip %ux%u+%u+%u"
                    " clears %u | %u unhandled methods (%d distinct)\n",
            s_gpu.color_offset, s_gpu.pitch, s_gpu.clip_w, s_gpu.clip_h,
            s_gpu.clip_x, s_gpu.clip_y, s_gpu.clears,
            s_gpu.unhandled_total, s_unhandled_count);
    fprintf(stderr, "[GPU] draws %u (%u with coordinates), %u indices;"
                    " x %.1f..%.1f  y %.1f..%.1f\n",
            s_gpu.draws, s_gpu.nonzero_draws, s_gpu.verts,
            s_gpu.min_x, s_gpu.max_x, s_gpu.min_y, s_gpu.max_y);
    /* One picture per report rather than per clear: a title clears hundreds of
     * times a second and nobody wants that many files. */
    dump_surface_bmp();

    /* Drawn and skipped separately: "nothing appeared" and "every batch needed
     * a vertex program we do not run" look identical on screen, and only one
     * of them means the rasteriser is broken. */
    fprintf(stderr, "[GPU] brightest pixel written 0x%08X\n", s_gpu.pixel_max);
    fprintf(stderr, "[GPU] %llu pixels written; draw surface 0x%08X"
                    " -> 0x%08X, clear surface 0x%08X -> 0x%08X\n",
            (unsigned long long)s_gpu.pixels, s_gpu.drawn_offset,
            dma_resolve(s_gpu.drawn_offset), s_gpu.color_offset,
            dma_resolve(s_gpu.color_offset));
    fprintf(stderr, "[GPU] rasterised %u triangles; %u batches skipped as not"
                    " screen-space, %u triangles fully off-surface\n",
            s_gpu.tris_drawn, s_gpu.batches_untransformed,
            s_gpu.tris_skipped_offscreen);
    fprintf(stderr, "[GPU] vertex programs: mode 0x%X, start %u, %u program"
                    " writes; %u batches, %u vertices run, %u ran off the end,"
                    " %u triangles behind the eye\n",
            s_vp.execution_mode, s_vp.program_start, s_vp.program_writes,
            s_vp_batches, s_vp_vertices, s_vp_failed, s_vp_behind_eye);
    if (s_frames_seen > 1) {
        double secs = (double)(GetTickCount64() - s_frames_t0) / 1000.0;
        fprintf(stderr, "[GPU] frames: %u in %.0f s (%.2f fps)\n", s_frames_seen,
                secs, secs > 0 ? s_frames_seen / secs : 0.0);
    }
    fprintf(stderr, "[GPU] depth: test %u func 0x%X mask %u zeta 0x%X;"
                    " %u pixels failed depth, %u failed alpha (func 0x%X ref %u)\n",
            s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
            s_gpu.zeta_offset, s_gpu.zbuf_rejects, s_gpu.alpha_rejects,
            s_gpu.alpha_func, s_gpu.alpha_ref);

    /* And of the batches that did rasterise, how many sampled anything. A menu
     * that draws its background from one texture and its text from another
     * shows both as flat colour if either half is missing, so the split is
     * what says which half. */
    fprintf(stderr, "[GPU] batches: %u textured, %u with no texcoords,"
                    " %u with texcoords but no usable stage\n",
            s_gpu.batches_textured, s_gpu.batches_no_uv, s_gpu.batches_no_tex);
    /* The table is large enough for a whole level; the report shows its
     * start, and RECOMP_TEXUSE_ALL=1 the rest. */
    for (i = 0; i < s_tex_use_count
                && (i < 12 || getenv("RECOMP_TEXUSE_ALL")); i++)
        fprintf(stderr, "  [TEXUSE] 0x%08X %ux%u fmt 0x%02X%s: %u batches\n",
                s_tex_use[i].offset, s_tex_use[i].width, s_tex_use[i].height,
                s_tex_use[i].color,
                d3d8_format_dxt_block_bytes(s_tex_use[i].color) ? " dxt"
                    : d3d8_format_is_swizzled(s_tex_use[i].color) ? " swz" : " lin",
                s_tex_use[i].batches);

    if (getenv("RECOMP_TEX_STATE")) {
        uint32_t k;
        for (k = 0; k < sizeof s_tex_set / sizeof s_tex_set[0]; k++)
            if (s_tex_set[k])
                fprintf(stderr, "  [TEX] 0x%04X = 0x%08X\n",
                        (unsigned)(NV_TEX_FIRST + k * 4), s_tex_reg[k]);
    }

    /* Top ten by frequency: selection sort over a small table, once every few
     * seconds, is not worth a better algorithm.
     *
     * RECOMP_PB_UNHANDLED_ALL lists every one instead. Ten is the right
     * default -- the tail is a long list of state registers nobody needs to
     * read -- but when a title stops and the question is which method it
     * stopped on, the answer is as likely to be the one seen twice as the
     * one seen a thousand times, and ten hides it. */
    {
        int shown = getenv("RECOMP_PB_UNHANDLED_ALL") ? s_unhandled_count : 10;
    for (i = 0; i < shown && i < s_unhandled_count; i++) {
        int best = i;
        for (j = i + 1; j < s_unhandled_count; j++)
            if (s_unhandled[j].count > s_unhandled[best].count)
                best = j;
        if (best != i) {
            PbUnhandled t = s_unhandled[i];
            s_unhandled[i] = s_unhandled[best];
            s_unhandled[best] = t;
        }
        {
            /* The value as well as the count. A method nobody decoded is
             * a guess until you see what it carried: screen coordinates,
             * a 0..1 texcoord and a packed colour are told apart at a
             * glance, and that is what says which vertex encoding a title
             * is using. */
            union { uint32_t u; float f; } v;
            v.u = s_unhandled[i].last_param;
            fprintf(stderr, "  [GPU]   0x%04X x%-8u last=0x%08X (%.4f)\n",
                    s_unhandled[i].method, s_unhandled[i].count,
                    v.u, v.f);
        }
    }
    }    fflush(stderr);
}
