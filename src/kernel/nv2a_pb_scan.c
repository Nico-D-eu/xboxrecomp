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

static void pb_run_body(uint32_t put_va);

/* The method header being executed and where the walk was (0 ring,
 * 1 followed jump, 2 followed call, with that target), for reports. */
uint32_t g_pb_hdr_va, g_pb_hdr, g_pb_hdr_where, g_pb_hdr_from;
/* The walk's first jump or call out of the ring. */
uint32_t g_pb_entry_src, g_pb_entry_word, g_pb_entry_target;
/* The last 32 words the walker decoded as headers or control flow. */
uint32_t g_pb_hist[32][2], g_pb_hist_pos;
uint32_t g_pb_entry_hist[32][2], g_pb_entry_hist_pos;   /* as it was at the entry */
static uint32_t g_pb_jump_target, g_pb_call_target;
/* The last 16 segments walked: start, end asked for, where the walk stopped
 * and why (0 end, 1 ring jump, 2 bad jump target, 3 word cap, 4 pending
 * carried, 5 pending finished first). */
uint32_t g_pb_seg[16][4], g_pb_seg_pos;
/* Where the ring jump that ended the last segment went, or 0. */
uint32_t g_pb_wrap_target;

/* Run the ring up to PUT, from wherever the last run stopped -- the GPU's
 * GET. */
void nv2a_pb_run(uint32_t put_va)
{
    LARGE_INTEGER a, b;
    QueryPerformanceCounter(&a);
    pb_run_body(put_va);
    QueryPerformanceCounter(&b);
    g_nv2a_scan_ticks += b.QuadPart - a.QuadPart;
}

/* A method whose data ran past the end of a segment.
 *
 * D3D may set PUT in the middle of a long command -- a batch of inline
 * indices, say -- and the hardware simply waits for the rest. This walk ends
 * each segment at PUT, and it used to drop the remaining count there, so the
 * next segment began by decoding index pairs as headers: 0x00D100D0 is
 * nothing, 0x00D200D1 reads as a jump and 0x00D200D2 as a call, and the
 * walker went off through zeroed memory into texture data, where a word like
 * 0xFF000000 sets 1984 consecutive methods from 0. In Amped 2 that was the
 * in-game terrain drawn without ground and the other state garbage (a depth
 * function of 0xBC2F3242). The unfinished method is kept here and finished
 * first when the walk resumes where it stopped. */
static struct {
    int      active, noninc;
    uint32_t subch, method, remaining, resume_va;
} s_pend;

/* Where the walk has got to: the GPU's GET, as a guest VA. 0 before the
 * first PUT. */
static uint32_t s_get;
/* The walk stopped past PUT: a jump back into the ring landed beyond the PUT
 * sampled, so there is nothing to run until PUT gets there. */
static int s_get_ahead;

/* The ring, run the way the GPU runs it.
 *
 * This used to walk segments: [old PUT, new PUT) going forward, and on a wrap
 * [old PUT, highest PUT seen) then [lowest PUT seen, new PUT). Neither bound
 * is where the title wraps. Amped 2 writes on past the old PUT and jumps back
 * to the ring's start wherever it runs out, often beyond the highest PUT
 * seen, so the commands in between were dropped; and the jump back can come
 * from a recorded buffer (RunPushBuffer) the walk had followed out of the
 * ring, which the segment walk treated as a return and ran on from the ring's
 * start up to the 1 MB bound, a lap of old commands. Both showed in game as
 * lost or garbage state.
 *
 * Now there is one position, GET. A run starts there, follows every jump --
 * the wrap included, from the ring or from a followed buffer -- and stops at
 * PUT once it has wrapped if it needed to. The next run resumes where it
 * stopped. */
static void pb_run_body(uint32_t end_va)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t start_va, va;
    uint32_t words = 0, jumps = 0, unknown = 0;
    int outside = 0, in_call = 0, need_wrap, stop = 0;
    uint32_t call_ret = 0, followed = 0, left_ring = 0;

    if (s_exec_enabled < 0)
        s_exec_enabled = getenv("RECOMP_PB_EXEC") != NULL;
    if (s_scan_on < 0)
        s_scan_on = getenv("RECOMP_PB_SCAN") != NULL;
    if (!(s_scan_on || s_exec_enabled))
        return;
    if (!s_get) {                             /* the first PUT: start there */
        s_get = end_va;
        return;
    }
    if (s_get == end_va)
        return;
    if (s_get_ahead) {
        if (end_va < s_get && s_get - end_va < 0x400000u)
            return;                           /* PUT not there yet */
        s_get_ahead = 0;
    }
    start_va = va = s_get;
    /* GET past PUT: the title has wrapped, and the walk has to as well
     * before PUT means anything. */
    need_wrap = start_va > end_va;

    g_pb_wrap_target = 0;
    g_pb_seg[g_pb_seg_pos % 16][0] = start_va;
    g_pb_seg[g_pb_seg_pos % 16][1] = end_va;
    g_pb_seg[g_pb_seg_pos % 16][3] = 0;
    if (s_pend.active) {
        /* Only where it stopped; a run starting anywhere else cannot be the
         * rest of that method. */
        if (s_pend.resume_va == va) {
            while (s_pend.remaining && (need_wrap || va < end_va)) {
                if (s_scan_on)
                    note(s_pend.subch, s_pend.method);
                if (s_exec_enabled)
                    nv2a_pb_exec_method(s_pend.subch, s_pend.method,
                                        *(const uint32_t *)(mem + va));
                if (!s_pend.noninc)
                    s_pend.method += 4;
                s_pend.remaining--;
                va += 4;
                words++;
            }
            s_pend.resume_va = va;
            g_pb_seg[g_pb_seg_pos % 16][3] = 5;
            if (s_pend.remaining) {           /* still more to come */
                s_tot_words += words;
                s_get = va;
                g_pb_seg[g_pb_seg_pos % 16][2] = va;
                g_pb_seg[g_pb_seg_pos % 16][3] = 4;
                g_pb_seg_pos++;
                return;
            }
        }
        s_pend.active = 0;
    }

    for (;;) {
        int ring = !outside && !in_call;
        uint32_t limit = (ring && !need_wrap) ? end_va : XBOX_CONTIG_BASE + 0x04000000u;
        uint32_t w;

        if (va >= limit)
            break;
        /* Bounds on one run: a word budget (the rest waits for the next
         * run), and a wrap that never comes within 4 MB. */
        if (ring && words >= 0x100000u) {
            stop = 3;
            break;
        }
        if (ring && need_wrap && va - start_va > 0x400000u) {
            stop = 2;
            break;
        }
        s_cur_in_ring = ring;
        if (ring) {
            s_cur_va = va;                    /* everything before is consumed */
            if ((words & 1023u) == 0)
                pb_scan_get(0);
        }
        w = *(const uint32_t *)(mem + va);
        g_pb_hist[g_pb_hist_pos % 32][0] = va;      /* control/header words */
        g_pb_hist[g_pb_hist_pos % 32][1] = w;
        g_pb_hist_pos++;
        va += 4;
        words++;

        if ((w & 3u) == 1u || (w & 0xE0000003u) == 0x20000000u) {
            uint32_t target = ((w & 3u) == 1u) ? (w & 0xFFFFFFFCu) : (w & 0x1FFFFFFCu);
            int in_ring;
            target = XBOX_CONTIG_BASE | (target & 0x0FFFFFFFu);
            /* The ring's bounds are learned from PUT, and its wrap lands on
             * the ring's true start, which can sit below the lowest PUT
             * seen -- so the ring is taken generously. */
            in_ring = !s_ring_hi || (target + 0x10000u >= s_ring_lo
                                     && target <= s_ring_hi + 0x10000u);
            jumps++;
            /* A jump to itself is where the title parks the GPU until it
             * writes the real command over it; the GPU spins re-reading it.
             * So wait there, and read it again next run. */
            if (target == va - 4 && !outside && !in_call) {
                va -= 4;
                stop = 4;
                break;
            }
            /* The follow budget is per trip out of the ring: a frame runs
             * hundreds of recorded buffers, and counting them all against
             * one budget called a long run a runaway. Jumps within the ring
             * are bounded by the word budget instead. */
            if (target >= XBOX_CONTIG_BASE + 0x04000000u
                    || ((outside || in_call || !in_ring) && ++followed > 256)) {
                stop = 2;                     /* not memory, or runaway */
                break;
            }
            if (!outside && !in_call) {
                if (in_ring) {
                    /* Within the ring: its wrap when backwards. */
                    if (target < va - 4) {
                        g_pb_wrap_target = target;
                        need_wrap = 0;
                    }
                    va = target;
                    continue;
                }
                /* Out of the ring: D3D running another pushbuffer
                 * (RunPushBuffer, a recorded buffer). Followed until it
                 * jumps back, or everything in it -- draws, constant uploads,
                 * a rider's skinning palette -- is never executed. */
                outside = 1;
                left_ring = va - 4;
                s_tot_followed++;
                g_pb_jump_target = target;
                g_pb_entry_src = va - 4; g_pb_entry_word = w; g_pb_entry_target = target;
                memcpy(g_pb_entry_hist, g_pb_hist, sizeof g_pb_hist);
                g_pb_entry_hist_pos = g_pb_hist_pos;
                va = target;
                continue;
            }
            if (outside && in_ring) {
                /* Back into the ring; behind where it left, that is the
                 * wrap as well. */
                {
                    static uint32_t n_next, n_other;
                    static int tr = -1;
                    if (tr < 0)
                        tr = getenv("RECOMP_PB_RET_TRACE") != NULL;
                    if (target == left_ring + 4) n_next++; else n_other++;
                    if (tr && n_other && target != left_ring + 4 && n_other <= 12)
                        fprintf(stderr, "  [NV2A] return from %08X (left ring at %08X) to %08X;"
                                " %u returns to the next word, %u elsewhere\n",
                                g_pb_jump_target, left_ring, target, n_next, n_other);
                }
                outside = 0;
                followed = 0;
                if (target < left_ring) {
                    g_pb_wrap_target = target;
                    need_wrap = 0;
                }
            }
            va = target;
            continue;
        }
        if ((w & 3u) == 2u) {                 /* call, one level as on NV2A */
            uint32_t target = XBOX_CONTIG_BASE | ((w & 0xFFFFFFFCu) & 0x0FFFFFFFu);
            {
                static unsigned ncall;
                static int tr = -1;
                if (tr < 0)
                    tr = getenv("RECOMP_PB_RET_TRACE") != NULL;
                if (tr && ncall++ < 16)
                    fprintf(stderr, "  [NV2A] call %08X at %08X (%s)\n", w, va - 4,
                            outside ? "in a followed buffer" : in_call ? "in a call" : "in the ring");
            }
            /* D3D does not use calls (every one seen in Amped 2 was a pair
             * of 16-bit indices read as a command), so one means the walk
             * is reading data: stop rather than execute it.
             * RECOMP_PB_CALLS=1 follows them as before. */
            static int calls_ok = -1;
            if (calls_ok < 0)
                calls_ok = getenv("RECOMP_PB_CALLS") != NULL;
            if (!calls_ok) {
                stop = 5;
                break;
            }
            if (!in_call && target < XBOX_CONTIG_BASE + 0x04000000u && ++followed <= 256) {
                in_call = 1;
                call_ret = va;
                s_tot_followed++;
                g_pb_call_target = target;
                if (!outside) {
                    g_pb_entry_src = va - 4; g_pb_entry_word = w; g_pb_entry_target = target;
                    memcpy(g_pb_entry_hist, g_pb_hist, sizeof g_pb_hist);
                    g_pb_entry_hist_pos = g_pb_hist_pos;
                }
                va = target;
            }
            continue;
        }
        if ((w & 0xFFFF0003u) == 0x00020000u) {   /* return */
            if (in_call) {
                in_call = 0;
                va = call_ret;
                if (!outside)
                    followed = 0;
            }
            continue;
        }
        if ((w & 0x00030003u) == 0u) {
            uint32_t count  = (w >> 18) & 0x7FFu;
            g_pb_hdr_va = va - 4;             /* for the executor's reports */
            g_pb_hdr = w;
            g_pb_hdr_where = in_call ? 2 : outside ? 1 : 0;
            g_pb_hdr_from = in_call ? g_pb_call_target : outside ? g_pb_jump_target : 0;
            uint32_t subch  = (w >> 13) & 7u;
            uint32_t method =  w & 0x1FFCu;
            int noninc = (w & 0xE0000000u) == 0x40000000u;

            uint32_t i;
            for (i = 0; i < count && va < limit; i++) {
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
            if (i < count && !outside && !in_call) {
                s_pend.active = 1;
                s_pend.noninc = noninc;
                s_pend.subch = subch;
                s_pend.method = noninc ? method : method + i * 4;
                s_pend.remaining = count - i;
                s_pend.resume_va = va;
            }
            continue;
        }
        /* Low bits 3 is no command at all: the walk is in data. */
        unknown++;
        stop = 5;
        break;
    }

    if (stop == 4) {
        /* Parked. Two seconds on one word is not the title about to patch
         * it; log it and go on from PUT rather than stall for good. */
        static uint32_t park_va;
        static ULONGLONG park_t0;
        ULONGLONG now = GetTickCount64();
        if (va != park_va) {
            park_va = va;
            park_t0 = now;
        }
        if (now - park_t0 > 2000) {
            static unsigned said;
            if (said++ < 8)
                fprintf(stderr, "  [NV2A] ring walk parked on the jump-to-self at %08X"
                        " for 2 s; resuming at PUT %08X\n", va, end_va);
            park_va = 0;
            s_get = end_va;
            s_pend.active = 0;
        } else {
            s_get = va;
            s_get_ahead = !need_wrap && va > end_va;
        }
    } else if (stop == 3) {
        s_get = va;                           /* budget: carry on next run */
        s_get_ahead = !need_wrap && va > end_va;
    } else if (stop || outside || in_call || need_wrap) {
        /* Lost: a runaway, or no wrap where one had to be. Start again at
         * PUT rather than walk on through whatever is there. */
        static unsigned said;
        if (said++ < 8) {
            uint32_t k;
            fprintf(stderr, "  [NV2A] ring walk %08X -> PUT %08X lost at %08X (%s);"
                    " resuming at PUT\n", start_va, end_va, va,
                    stop == 2 ? "runaway" : stop == 5 ? "not a command"
                    : need_wrap ? "no wrap" : "in a followed buffer");
            fprintf(stderr, "  [NV2A]   last words:");
            for (k = 0; k < 12; k++) {
                const uint32_t *h = g_pb_hist[(g_pb_hist_pos - 12 + k) % 32];
                fprintf(stderr, " %08X:%08X", h[0], h[1]);
            }
            fprintf(stderr, "\n  [NV2A]   ring %08X..%08X, %u words this run\n",
                    s_ring_lo, s_ring_hi, words);
        }
        s_get = end_va;
        s_pend.active = 0;
    } else {
        s_get = va;
        s_get_ahead = va > end_va;
    }

    g_pb_seg[g_pb_seg_pos % 16][2] = va;
    if (s_pend.active && s_pend.resume_va == va && !g_pb_seg[g_pb_seg_pos % 16][3])
        g_pb_seg[g_pb_seg_pos % 16][3] = 4;
    if (stop)
        g_pb_seg[g_pb_seg_pos % 16][3] = stop;
    g_pb_seg_pos++;
    s_cur_va = 0;
    s_tot_words += words;
    s_tot_unknown += unknown;
    s_tot_jumps += jumps;
    s_tot_segments++;
}
