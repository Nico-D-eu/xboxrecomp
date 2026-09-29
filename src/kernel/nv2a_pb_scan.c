/*
 * Read-only survey of the pushbuffer a title submits.
 *
 * The title builds NV2A commands in guest RAM and advances DMA_PUT; nothing
 * here executes them, so the framebuffer stays black however far the game
 * gets. Before any of that can be made to draw, the question is what it
 * actually asks for -- which methods, on which object classes, how many of
 * them -- because that is the difference between "the existing PGRAPH
 * translator nearly covers this" and "this needs a real one".
 *
 * Purely a reader: it walks the buffer and counts, and never writes to guest
 * memory or to the GPU state. Enabled with RECOMP_PB_SCAN.
 *
 * Pushbuffer encoding (NV20/NV2A), one dword per command header:
 *   (w & 0xE0030003) == 0x00000000  increasing methods
 *   (w & 0xE0030003) == 0x40000000  non-increasing (same method, count params)
 *   (w & 0x00000003) == 0x00000001  jump
 *   (w & 0x00000003) == 0x00000002  call
 *   (w & 0xFFFF0003) == 0x00020000  return
 * For a method header: count = (w >> 18) & 0x7FF, subchannel = (w >> 13) & 7,
 * method = w & 0x1FFC.
 */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>   /* ptrdiff_t */
#include <stdlib.h>
#include <string.h>
#include "kernel.h"   /* XBOX_CONTIG_BASE */
#include <windows.h>

extern ptrdiff_t xbox_GetMemoryOffset(void);

#define PB_MAX_METHODS 4096

static struct { uint32_t method, subch, count; } s_seen[PB_MAX_METHODS];
static int s_seen_count;

/* Parse health. An inventory is only worth reading if the walk stayed in step
 * with the command stream: a decoder that desynchronises produces plausible
 * looking method numbers out of parameter data, and the counts then describe
 * nothing. Unrecognised words are the tell. */
static uint32_t s_tot_words, s_tot_unknown, s_tot_jumps, s_tot_segments;

/* Executing is opt-in separately from surveying: a survey is read-only, while
 * the executor writes to guest memory. */
extern void nv2a_pb_exec_method(uint32_t subch, uint32_t method, uint32_t param);
extern void nv2a_pb_exec_report(void);
static int s_exec_enabled = -1;
static int s_scan_on = -1;
/* The pushbuffer ring's bounds, as the caller has learned them from PUT, and
 * how many jumps and calls out of it were followed. */
static uint32_t s_ring_lo, s_ring_hi, s_tot_followed;

/* GET as the executor goes. The caller used to set GET to PUT only after a
 * whole segment was executed, so while a big segment ran -- or the executor
 * waited out a flip -- D3D saw no space freed, kicked and spun on the flush
 * bit, and the title's thread stood still behind the GPU's. Hardware
 * advances GET as it consumes; this does too, and acknowledges the flush
 * request with it, since the flush is D3D asking "how far have you got". */
static volatile uint32_t *s_get_reg, *s_flush_reg;
static uint32_t s_cur_va;
static int s_cur_in_ring;

void nv2a_pb_scan_set_regs(volatile uint32_t *get, volatile uint32_t *flush)
{
    s_get_reg = get;
    s_flush_reg = flush;
}

static void pb_scan_get(int ack_flush)
{
    if (!s_get_reg || !s_cur_in_ring || !s_cur_va)
        return;
    *s_get_reg = s_cur_va & 0x0FFFFFFFu;
    if (ack_flush && s_flush_reg)
        *s_flush_reg &= ~0x00010000u;
}

/* From the flip wait: the pusher is stalled at FLIP_STALL, everything before
 * it is consumed, so GET moves there and a flush request is answered.
 * Answering flushes mid-walk instead hung boot after 9 frames -- D3D reads
 * the flush as "idle", as the old fast-ack experiment found. */
void nv2a_pb_scan_progress(void)
{
    pb_scan_get(getenv("RECOMP_PB_FLUSH_IN_WAIT") != NULL);
}

void nv2a_pb_scan_set_ring(uint32_t lo_va, uint32_t hi_va)
{
    s_ring_lo = lo_va;
    s_ring_hi = hi_va;
}

static void note(uint32_t subch, uint32_t method)
{
    for (int i = 0; i < s_seen_count; i++) {
        if (s_seen[i].method == method && s_seen[i].subch == subch) {
            s_seen[i].count++;
            return;
        }
    }
    if (s_seen_count >= PB_MAX_METHODS) {
        /* Silently dropping past the cap is how a truncated inventory reads as
         * "the title never does that" -- exactly the wrong conclusion when the
         * inventory is being used to decide what to implement. */
        static int warned;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[PB] method table full at %d -- inventory is"
                            " truncated\n", PB_MAX_METHODS);
        }
    }
    if (s_seen_count < PB_MAX_METHODS) {
        s_seen[s_seen_count].method = method;
        s_seen[s_seen_count].subch  = subch;
        s_seen[s_seen_count].count  = 1;
        s_seen_count++;
    }
}

/* NV097 (Kelvin 3D class) methods worth naming. The point of the survey is to
 * decide what a translator has to implement, and a bare method number does not
 * answer that -- "0x1808 x412" only means something once it reads
 * INLINE_ARRAY. Unnamed ones still get counted. */
static const struct { uint32_t m; const char *name; } NV097_NAMES[] = {
    { 0x0000, "SET_OBJECT" },
    { 0x0100, "NO_OPERATION" },
    { 0x0104, "SET_WARNING_ENABLE" },
    { 0x0130, "SET_FLIP_READ" },
    { 0x0200, "SET_SURFACE_CLIP_HORIZONTAL" },
    { 0x0204, "SET_SURFACE_CLIP_VERTICAL" },
    { 0x0208, "SET_SURFACE_FORMAT" },
    { 0x020C, "SET_SURFACE_PITCH" },
    { 0x0210, "SET_SURFACE_COLOR_OFFSET" },
    { 0x0214, "SET_SURFACE_ZETA_OFFSET" },
    { 0x0300, "SET_ALPHA_TEST_ENABLE" },
    { 0x0304, "SET_BLEND_ENABLE" },
    { 0x030C, "SET_DEPTH_TEST_ENABLE" },
    { 0x0310, "SET_DITHER_ENABLE" },
    { 0x0314, "SET_LIGHTING_ENABLE" },
    { 0x033C, "SET_CULL_FACE_ENABLE" },
    { 0x0340, "SET_DEPTH_MASK" },
    { 0x0350, "SET_CLEAR_DEPTH_VALUE" },
    { 0x1D8C, "SET_CLEAR_DEPTH" },
    { 0x1D90, "SET_COLOR_CLEAR_VALUE" },
    { 0x1D94, "CLEAR_SURFACE" },
    { 0x1D6C, "SET_ZSTENCIL_CLEAR" },
    { 0x0B80, "SET_TRANSFORM_PROGRAM" },
    { 0x0B00, "SET_TRANSFORM_CONSTANT" },
    { 0x1720, "SET_VERTEX_DATA_ARRAY_OFFSET" },
    { 0x1760, "SET_VERTEX_DATA_ARRAY_FORMAT" },
    { 0x17FC, "SET_BEGIN_END" },
    { 0x1800, "ARRAY_ELEMENT16" },
    { 0x1808, "INLINE_ARRAY" },
    { 0x1810, "DRAW_ARRAYS" },
    { 0x1B00, "SET_TEXTURE_OFFSET" },
    { 0x1B04, "SET_TEXTURE_FORMAT" },
    { 0x1B08, "SET_TEXTURE_ADDRESS" },
    { 0x1B0C, "SET_TEXTURE_CONTROL0" },
    { 0x1B14, "SET_TEXTURE_IMAGE_RECT" },
    { 0x0FD8, "SET_COMBINER_*" },
    { 0x0000, NULL },
};

static const char *nv097_name(uint32_t m)
{
    int i;
    for (i = 0; NV097_NAMES[i].name; i++)
        if (NV097_NAMES[i].m == m)
            return NV097_NAMES[i].name;
    return "";
}

void nv2a_pb_scan_report(void)
{
    int i;

    if (s_exec_enabled > 0)
        nv2a_pb_exec_report();
    if (s_tot_followed) {
        static uint32_t said;
        if (s_tot_followed != said) {
            said = s_tot_followed;
            fprintf(stderr, "[PB] %u jumps/calls out of the ring followed\n",
                    s_tot_followed);
        }
    }
    if (!s_seen_count || !getenv("RECOMP_PB_SCAN"))
        return;
    fprintf(stderr, "[PB] %u segments, %u words, %u jumps, %u unrecognised"
                    " -- %d distinct (subchannel, method) pairs\n",
            s_tot_segments, s_tot_words, s_tot_jumps, s_tot_unknown,
            s_seen_count);
    for (i = 0; i < s_seen_count; i++)
        fprintf(stderr, "  [PB]   subch %u  method 0x%04X  x%-6u %s\n",
                s_seen[i].subch, s_seen[i].method, s_seen[i].count,
                nv097_name(s_seen[i].method));
    fflush(stderr);
}

/* Time spent walking and executing, for the per-frame report. */
volatile long long g_nv2a_scan_ticks;

static void pb_scan_body(uint32_t start_va, uint32_t end_va);

void nv2a_pb_scan(uint32_t start_va, uint32_t end_va)
{
    LARGE_INTEGER a, b;
    QueryPerformanceCounter(&a);
    pb_scan_body(start_va, end_va);
    QueryPerformanceCounter(&b);
    g_nv2a_scan_ticks += b.QuadPart - a.QuadPart;
}

static void pb_scan_body(uint32_t start_va, uint32_t end_va)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t va = start_va;
    uint32_t words = 0, jumps = 0, unknown = 0;

    if (s_exec_enabled < 0)
        s_exec_enabled = getenv("RECOMP_PB_EXEC") != NULL;
    if (s_scan_on < 0)
        s_scan_on = getenv("RECOMP_PB_SCAN") != NULL;
    if (!(s_scan_on || s_exec_enabled) || end_va <= start_va)
        return;
    if (end_va - start_va > 0x400000u)        /* a sane single-frame bound */
        end_va = start_va + 0x400000u;

    /* Jumps and calls. A jump inside the ring is its wrap, which the caller
     * handles from PUT, so it ends the segment as it always did. A jump out
     * of the ring is D3D running another pushbuffer (RunPushBuffer, a
     * recorded buffer); that has to be followed until it jumps back, or
     * everything in it -- draws, constant uploads -- is never executed. It
     * was not: a rider's skinning palette, uploaded correctly by the CPU,
     * never reached the vertex program. Calls (one level, as on NV2A) the
     * same. */
    int outside = 0, in_call = 0;
    uint32_t call_ret = 0, followed = 0;

    for (;;) {
        uint32_t limit = (outside || in_call) ? XBOX_CONTIG_BASE + 0x04000000u : end_va;
        uint32_t w;

        if (va >= limit || words >= 0x100000u)
            break;
        s_cur_in_ring = !outside && !in_call;
        if (s_cur_in_ring) {
            s_cur_va = va;                    /* everything before is consumed */
            if ((words & 1023u) == 0)
                pb_scan_get(0);
        }
        w = *(const uint32_t *)(mem + va);
        va += 4;
        words++;

        if ((w & 3u) == 1u || (w & 0xE0000003u) == 0x20000000u) {
            uint32_t target = ((w & 3u) == 1u) ? (w & 0xFFFFFFFCu) : (w & 0x1FFFFFFCu);
            int in_ring;
            target = XBOX_CONTIG_BASE | (target & 0x0FFFFFFFu);
            /* The ring's bounds are learned from PUT, and its wrap lands on
             * the ring's true start, which can sit below the lowest PUT
             * seen -- so the ring is taken generously. Treating a wrap as a
             * jump out of it re-executed stale commands up to garbage. */
            in_ring = s_ring_hi && target + 0x10000u >= s_ring_lo
                   && target <= s_ring_hi + 0x10000u;
            jumps++;
            if (!outside) {
                if (in_ring || !s_ring_hi)
                    break;                    /* the ring's own wrap */
                if (target >= XBOX_CONTIG_BASE + 0x04000000u || ++followed > 256)
                    break;                    /* not memory, or runaway */
                outside = 1;
                s_tot_followed++;
                va = target;
                continue;
            }
            if (in_ring)
                outside = 0;                  /* back into the ring */
            else if (target >= XBOX_CONTIG_BASE + 0x04000000u || ++followed > 256)
                break;
            va = target;
            continue;
        }
        if ((w & 3u) == 2u) {                 /* call */
            uint32_t target = XBOX_CONTIG_BASE | ((w & 0xFFFFFFFCu) & 0x0FFFFFFFu);
            if (!in_call && target < XBOX_CONTIG_BASE + 0x04000000u && ++followed <= 256) {
                in_call = 1;
                call_ret = va;
                s_tot_followed++;
                va = target;
            }
            continue;
        }
        if ((w & 0xFFFF0003u) == 0x00020000u) {   /* return */
            if (in_call) {
                in_call = 0;
                va = call_ret;
            }
            continue;
        }
        if ((w & 0x00030003u) == 0u) {
            uint32_t count  = (w >> 18) & 0x7FFu;
            uint32_t subch  = (w >> 13) & 7u;
            uint32_t method =  w & 0x1FFCu;
            int noninc = (w & 0xE0000000u) == 0x40000000u;

            for (uint32_t i = 0; i < count && va < limit; i++) {
                uint32_t m = noninc ? method : method + i * 4;
                /* The survey's table is a linear search per method word --
                 * half the executor thread's time once the GPU backend made
                 * everything else cheap -- so only when a survey was asked
                 * for. */
                if (s_scan_on)
                    note(subch, m);
                /* Same walk, two consumers: the survey counts, the executor
                 * acts. Keeping them on one decode means they can never
                 * disagree about what the stream said. */
                if (s_exec_enabled)
                    nv2a_pb_exec_method(subch, m,
                                        *(const uint32_t *)(mem + va));
                va += 4;
                words++;
            }
            continue;
        }
        unknown++;
    }

    s_cur_va = 0;
    s_tot_words += words;
    s_tot_unknown += unknown;
    s_tot_jumps += jumps;
    s_tot_segments++;
}
