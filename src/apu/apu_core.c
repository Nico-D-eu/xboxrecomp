/*
 * MCPX APU Core - Standalone extraction from xemu
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2018-2019 Jannik Vogel
 * Copyright (c) 2019-2025 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "apu_state.h"
#include "apu.h"
#include "apu_xaudio2.h"
#include "fpconv.h"

/* ============================================================
 * Globals
 * ============================================================ */

uint8_t *g_apu_ram_ptr = NULL;
uint8_t *g_apu_lowram_ptr = NULL;

MCPXAPUState *g_state = NULL;

/* Forward declarations for software mixer */
static void mixer_init(void);
static void mixer_render(int16_t frame_buf[][2], int num_samples);
static APUMixerVoice g_mixer_voices[APU_MIXER_MAX_VOICES];
static volatile int g_mixer_active_count = 0;
/* Guest MMIO handlers queued on d->lock (voice_lock in apu_vp.c). */
volatile long g_apu_lock_waiters;
static CRITICAL_SECTION g_mixer_cs;
static bool g_mixer_initialized = false;
struct McpxApuDebug g_dbg;
struct McpxApuDebug g_dbg_cache;
int g_dbg_voice_monitor = -1;
uint64_t g_dbg_muted_voices[4] = { 0 };

/* Global audio mute — disables all AWD/mixer sound playback */
volatile int g_audio_muted = 0;  /* 0 = audio enabled */

/* ============================================================
 * Debug frame markers (minimal stubs)
 * ============================================================ */

void mcpx_debug_begin_frame(void) {}
void mcpx_debug_end_frame(void) {}

/* ============================================================
 * IRQ handling (stubbed - no PCI bus in standalone)
 * ============================================================ */

/* The APU's interrupt line (vector 5 on the console), for the kernel's timer
 * thread to deliver: DirectSound's ISR learns from it that a voice has ended
 * or a method has been trapped. It was a no-op PCI stub, so DirectSound never
 * heard a voice finish, XACT never freed a finished cue, and Amped 2 ran its
 * sound heap out of memory about three minutes into a level. */
extern volatile long g_xbox_apu_irq_line;          /* kernel_bridge.c */
extern void (*volatile g_xbox_apu_irq_poll)(void);

static void update_irq(MCPXAPUState *d);

/* A pending set_irq from any path (a frame, a trapped method), folded into
 * the line. Called by the kernel's timer thread before it looks at the
 * line, so no path has to remember to. */
void mcpx_apu_poll_irq(void)
{
    MCPXAPUState *d = g_state;
    if (d && d->set_irq) {
        d->set_irq = false;
        update_irq(d);
    }
}

static void update_irq(MCPXAPUState *d)
{
    if (d->regs[NV_PAPU_FECTL] & NV_PAPU_FECTL_FEMETHMODE_TRAPPED) {
        qatomic_or(&d->regs[NV_PAPU_ISTS], NV_PAPU_ISTS_FETINTSTS);
    }
    if ((d->regs[NV_PAPU_IEN] & NV_PAPU_ISTS_GINTSTS) &&
        ((d->regs[NV_PAPU_ISTS] & ~NV_PAPU_ISTS_GINTSTS) &
         d->regs[NV_PAPU_IEN])) {
        qatomic_or(&d->regs[NV_PAPU_ISTS], NV_PAPU_ISTS_GINTSTS);
        /* In standalone mode we don't raise a PCI IRQ; the game's kernel
         * stub will poll ISTS directly or we'll signal via a flag. */
        g_xbox_apu_irq_line = 1;
    } else {
        qatomic_and(&d->regs[NV_PAPU_ISTS], ~NV_PAPU_ISTS_GINTSTS);
        g_xbox_apu_irq_line = 0;
    }
}

/* ============================================================
 * MMIO Read / Write
 * ============================================================ */

uint64_t mcpx_apu_read(void *opaque, hwaddr addr, unsigned int size)
{
    MCPXAPUState *d = (MCPXAPUState *)opaque;
    uint64_t r = 0;

    switch (addr) {
    case NV_PAPU_XGSCNT:
        r = (uint64_t)(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) / 100);
        break;
    default:
        if (addr < 0x20000) {
            r = qatomic_read(&d->regs[addr]);
        }
        break;
    }

    /* Uncomment for register tracing:
     * fprintf(stderr, "[APU] read  [0x%05llX] size=%u -> 0x%08llX\n",
     *         (unsigned long long)addr, size, (unsigned long long)r);
     */
    (void)size;
    return r;
}

void mcpx_apu_write(void *opaque, hwaddr addr, uint64_t val,
                     unsigned int size)
{
    MCPXAPUState *d = (MCPXAPUState *)opaque;

    /* Uncomment for register tracing:
     * fprintf(stderr, "[APU] write [0x%05llX] size=%u <- 0x%08llX\n",
     *         (unsigned long long)addr, size, (unsigned long long)val);
     */
    (void)size;

    switch (addr) {
    case NV_PAPU_ISTS:
        qatomic_and(&d->regs[NV_PAPU_ISTS], ~(uint32_t)val);
        update_irq(d);
        qemu_cond_broadcast(&d->cond);
        break;
    case NV_PAPU_FECTL:
    case NV_PAPU_SECTL:
        qatomic_set(&d->regs[addr], (uint32_t)val);
        /* Starting the APU has to start the frame thread.
         *
         * The thread idles on pause_requested, which init sets and only the
         * test tone ever cleared -- so a title that enabled the APU through
         * these registers got an APU that stayed asleep. Nothing then advanced
         * the front end, and a title waiting on a notify completion (the
         * FEMEMDATA magic write, which is how completion reaches guest memory)
         * waited forever. Wreckless hangs exactly there during DirectSound
         * init, and because it initialises its whole engine behind a
         * successful DirectSound create, that hang is not confined to audio.
         *
         * Resume whenever the write is not switching the block off; the thread
         * re-checks FECTL itself and idles again if it is halted or trapped. */
        {
            uint32_t sectl = qatomic_read(&d->regs[NV_PAPU_SECTL]);
            uint32_t fectl = qatomic_read(&d->regs[NV_PAPU_FECTL]);
            bool running =
                ((sectl & NV_PAPU_SECTL_XCNTMODE) != NV_PAPU_SECTL_XCNTMODE_OFF)
                && ((fectl & NV_PAPU_FECTL_FEMETHMODE)
                    != NV_PAPU_FECTL_FEMETHMODE_HALTED);
            if (running && d->pause_requested) {
                d->pause_requested = false;
                fprintf(stderr, "[APU] started by the title"
                                " (SECTL=%08X FECTL=%08X)\n", sectl, fectl);
            }
        }
        qemu_cond_broadcast(&d->cond);
        break;
    case NV_PAPU_FEMEMDATA:
        /* 'magic write' - value written to FEMEMADDR on notify completion */
        stl_le_phys(address_space_memory, d->regs[NV_PAPU_FEMEMADDR], (uint32_t)val);
        qatomic_set(&d->regs[addr], (uint32_t)val);
        break;
    default:
        if (addr < 0x20000) {
            qatomic_set(&d->regs[addr], (uint32_t)val);
        }
        break;
    }
}

/* ============================================================
 * Test tone state (used by monitor and test tone functions)
 * ============================================================ */

static struct {
    bool active;
    double phase;
    double phase_inc;
    int16_t amplitude;
} g_test_tone = { false, 0.0, 0.0, 0 };

/* ============================================================
 * Monitor - Audio output (XAudio2 primary, waveOut fallback)
 * ============================================================ */

#if defined(_WIN32)
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif
/* On Linux, waveOut* are inert stubs from win32_compat.h: the APU's
 * waveOut fallback path stays inactive and never produces audio. */

/* Ring of waveOut buffers for double-buffering */
#define WAVEOUT_NUM_BUFS 4
#define WAVEOUT_BUF_SAMPLES 2048  /* ~42.7ms at 48kHz, matches 8-frame delivery rate */
#define MIXER_FRAME_SAMPLES 256  /* Internal mixing frame size (matches frame_buf) */

typedef struct {
    HWAVEOUT hwo;
    WAVEHDR  hdrs[WAVEOUT_NUM_BUFS];
    int16_t  bufs[WAVEOUT_NUM_BUFS][WAVEOUT_BUF_SAMPLES][2];
    int      next_buf;
    bool     initialized;
    int      frames_written;
} WaveOutState;

static WaveOutState g_waveout = { 0 };

void mcpx_apu_monitor_init(MCPXAPUState *d, Error **errp)
{
    (void)errp;
    d->monitor.stream = NULL;
    d->monitor.queued_bytes_low = 1024;
    d->monitor.queued_bytes_high = 3072;

    /* Try XAudio2 first (lower latency) */
    if (xa2_init()) {
        fprintf(stderr, "[APU] Using XAudio2 audio backend\n");
        return;
    }
    fprintf(stderr, "[APU] XAudio2 unavailable, falling back to waveOut\n");

    WAVEFORMATEX wfx = { 0 };
    wfx.wFormatTag      = WAVE_FORMAT_PCM;
    wfx.nChannels       = 2;
    wfx.nSamplesPerSec  = 48000;
    wfx.wBitsPerSample  = 16;
    wfx.nBlockAlign     = wfx.nChannels * wfx.wBitsPerSample / 8;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

    MMRESULT mr = waveOutOpen(&g_waveout.hwo, WAVE_MAPPER, &wfx,
                               0, 0, CALLBACK_NULL);
    if (mr != MMSYSERR_NOERROR) {
        fprintf(stderr, "[APU] waveOutOpen failed (error %u)\n", mr);
        g_waveout.initialized = false;
        return;
    }

    /* Prepare all headers */
    for (int i = 0; i < WAVEOUT_NUM_BUFS; i++) {
        memset(&g_waveout.hdrs[i], 0, sizeof(WAVEHDR));
        g_waveout.hdrs[i].lpData = (LPSTR)g_waveout.bufs[i];
        g_waveout.hdrs[i].dwBufferLength = WAVEOUT_BUF_SAMPLES * 2 * sizeof(int16_t);
        waveOutPrepareHeader(g_waveout.hwo, &g_waveout.hdrs[i], sizeof(WAVEHDR));
    }

    g_waveout.next_buf = 0;
    g_waveout.initialized = true;
    g_waveout.frames_written = 0;

    fprintf(stderr, "[APU] waveOut audio output initialized (48kHz stereo 16-bit, %d buffers)\n",
            WAVEOUT_NUM_BUFS);
}

void mcpx_apu_monitor_finalize(MCPXAPUState *d)
{
    (void)d;
    if (xa2_is_active()) {
        xa2_shutdown();
        return;
    }
    if (!g_waveout.initialized) return;

    waveOutReset(g_waveout.hwo);
    for (int i = 0; i < WAVEOUT_NUM_BUFS; i++) {
        waveOutUnprepareHeader(g_waveout.hwo, &g_waveout.hdrs[i], sizeof(WAVEHDR));
    }
    waveOutClose(g_waveout.hwo);
    g_waveout.initialized = false;
    fprintf(stderr, "[APU] waveOut audio output shut down (%d frames written)\n",
            g_waveout.frames_written);
}

/* Time the frame thread spent handing the lock to guest writes (RECOMP_APU_STATS). */
static int64_t g_apu_handoff_us;
static unsigned g_apu_handoffs;
static int64_t g_apu_frame_us, g_apu_frame_max_us;   /* in se_frame */
static int64_t g_apu_relock_us;                      /* re-taking the lock */
static unsigned g_apu_underruns, g_apu_overruns;     /* output queue empty / full */
static unsigned g_apu_stalls;                        /* waits on a trapped front end */

void mcpx_apu_monitor_frame(MCPXAPUState *d)
{
    if ((d->ep_frame_div + 1) % 8) {
        return;
    }

    {
        extern volatile uint32_t g_apu_stat_on, g_apu_stat_off,
                                 g_apu_stat_pio_off, g_apu_stat_notify;
        static int stats = -1;
        static unsigned n;
        if (stats < 0)
            stats = getenv("RECOMP_APU_STATS") != NULL;
        if (stats && ++n % 938 == 0)            /* ~5 s of 256-sample frames */
            fprintf(stderr, "  [APU] voices on %u, off %u (%u by PIO), notifications %u; "
                    "ISTS %08X IEN %08X FENADDR %08X; %d frames/s (1500 is real "
                    "time), %.0f%% busy; SECTL %08X FECTL %08X; list tops 2D %04X 3D %04X MP %04X\n",
                    g_apu_stat_on, g_apu_stat_off, g_apu_stat_pio_off, g_apu_stat_notify,
                    d->regs[NV_PAPU_ISTS], d->regs[NV_PAPU_IEN], d->regs[NV_PAPU_FENADDR],
                    g_dbg.frames_processed, g_dbg.utilization * 100.0f,
                    d->regs[NV_PAPU_SECTL], d->regs[NV_PAPU_FECTL],
                    d->regs[NV_PAPU_TVL2D] & 0xFFFF, d->regs[NV_PAPU_TVL3D] & 0xFFFF,
                    d->regs[NV_PAPU_TVLMP] & 0xFFFF);
        if (stats && n % 938 == 0) {
            fprintf(stderr, "  [APU]   last 5 s: %.0f ms mixing (longest frame %.2f ms),"
                    " %.0f ms re-taking the lock, %u hand-offs %.1f ms\n",
                    (double)g_apu_frame_us / 1000.0, (double)g_apu_frame_max_us / 1000.0,
                    (double)g_apu_relock_us / 1000.0,
                    g_apu_handoffs, (double)g_apu_handoff_us / 1000.0);
            g_apu_handoffs = 0;
            g_apu_handoff_us = g_apu_frame_us = g_apu_frame_max_us = g_apu_relock_us = 0;
            fprintf(stderr, "  [APU]   output queue ran dry %u times, was full %u times\n",
                    g_apu_underruns, g_apu_overruns);
            g_apu_underruns = g_apu_overruns = 0;
            fprintf(stderr, "  [APU]   %u waits on a trapped or halted front end\n", g_apu_stalls);
            g_apu_stalls = 0;
            /* Every voice on the three lists: bins (CFG_VBIN), format
             * (stream bit 24, persist 23), state (active 21, paused 18),
             * offsets and the first two target volumes. Then the level of
             * each mix bin over the last 5 s (g_apu_bin_rms, apu_dsp.c). */
            extern double g_apu_bin_sq[32];
            extern unsigned long g_apu_bin_n;
            static const uint32_t tops[3] = { NV_PAPU_TVL2D, NV_PAPU_TVL3D, NV_PAPU_TVLMP };
            static const char *names[3] = { "2D", "3D", "MP" };
            int l, b;
            for (l = 0; l < 3; l++) {
                uint32_t v = d->regs[tops[l]] & 0xFFFF;
                int guard = 0;
                while (v != 0xFFFF && v < 256 && guard++ < 64) {
                    hwaddr vb = d->regs[NV_PAPU_VPVADDR] + v * NV_PAVS_SIZE;
                    uint32_t fmt = ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_CFG_FMT);
                    uint32_t st  = ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_PAR_STATE);
                    fprintf(stderr, "  [APU]   %s voice %02X: vbin %08X fmt %08X%s%s state %08X%s%s"
                            " offset %08X next %08X vol %08X link %08X\n",
                            names[l], v,
                            ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_CFG_VBIN),
                            fmt, (fmt & (1u << 24)) ? " STREAM" : "", (fmt & (1u << 23)) ? " PERSIST" : "",
                            st, (st & (1u << 21)) ? " ACTIVE" : "", (st & (1u << 18)) ? " PAUSED" : "",
                            ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_PAR_OFFSET),
                            ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_PAR_NEXT),
                            ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_TAR_VOLA),
                            ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_TAR_PITCH_LINK));
                    if ((fmt & (1u << 24)) && v < MCPX_HW_MAX_VOICES) {
                        /* A stream voice: its current segment (SSL entry),
                         * what the APU reads there, and what the other store
                         * holds at the same number. */
                        int si = d->vp.ssl[v].ssl_index;
                        uint32_t pg = d->vp.ssl[v].base[si] + d->vp.ssl[v].ssl_seg;
                        uint32_t so = ldl_le_phys(address_space_memory,
                                                  d->regs[NV_PAPU_VPSSLADDR] + pg * 8);
                        uint32_t sl = ldl_le_phys(address_space_memory,
                                                  d->regs[NV_PAPU_VPSSLADDR] + pg * 8 + 4);
                        extern void *xbox_GetMemoryBase(void);
                        const uint8_t *a = apu_phys(so);
                        const uint8_t *c = g_apu_ram_ptr + (so & 0x03FFFFFF);
                        const uint8_t *lo = (const uint8_t *)xbox_GetMemoryBase() + (so & 0x03FFFFFF);
                        fprintf(stderr, "  [APU]     stream ssl %d seg %d count %d -> phys %08X len %08X;"
                                " read %02X%02X%02X%02X%02X%02X%02X%02X window %02X%02X%02X%02X"
                                " low RAM %02X%02X%02X%02X\n", si, d->vp.ssl[v].ssl_seg,
                                d->vp.ssl[v].count[si], so, sl,
                                a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                                c[0], c[1], c[2], c[3], lo[0], lo[1], lo[2], lo[3]);
                    }
                    if (!(fmt & (1u << 24)) && !(fmt & (1u << 21))) {
                        /* A buffer voice: where its data is, and what each of
                         * the two stores holds there. Physical addresses are
                         * ambiguous here -- low RAM and the contiguous window
                         * are separate storage -- so the answer to "which
                         * one did the title mean" is whichever is not
                         * garbage. */
                        extern void *xbox_GetMemoryBase(void);
                        uint32_t ba = ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_CUR_PSL_START)
                                    & NV_PAVS_VOICE_CUR_PSL_START_BA;
                        uint32_t pa = ldl_le_phys(address_space_memory,
                                                  d->regs[NV_PAPU_VPSGEADDR] + (ba / 4096) * 8)
                                    + ba % 4096;
                        const uint8_t *c = g_apu_ram_ptr + (pa & 0x03FFFFFF);
                        const uint8_t *lo = (const uint8_t *)xbox_GetMemoryBase() + (pa & 0x03FFFFFF);
                        fprintf(stderr, "  [APU]     buffer ba %06X -> phys %08X; contig"
                                " %02X%02X%02X%02X%02X%02X%02X%02X low RAM"
                                " %02X%02X%02X%02X%02X%02X%02X%02X\n", ba, pa,
                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7]);
                    }
                    v = ldl_le_phys(address_space_memory, vb + NV_PAVS_VOICE_TAR_PITCH_LINK) & 0xFFFF;
                }
            }
            {
                /* The six voices that produced the most, with their DC and
                 * Nyquist content relative to their level. */
                extern float g_apu_voice_rate[256];
                extern double g_apu_voice_dc[256], g_apu_voice_nyq[256], g_apu_voice_sq[256];
                extern unsigned long g_apu_voice_n[256];
                int shown[6] = { -1, -1, -1, -1, -1, -1 }, k, j;
                for (k = 0; k < 6; k++) {
                    int best = -1;
                    for (j = 0; j < 256; j++) {
                        int dup = 0, q;
                        for (q = 0; q < k; q++) dup |= shown[q] == j;
                        if (!dup && g_apu_voice_n[j]
                                && (best < 0 || g_apu_voice_sq[j] > g_apu_voice_sq[best]))
                            best = j;
                    }
                    shown[k] = best;
                }
                fprintf(stderr, "  [APU]   voices by output (rms, dc, nyquist @rate):");
                for (k = 0; k < 6 && shown[k] >= 0; k++) {
                    j = shown[k];
                    fprintf(stderr, " %02X:%.3f/%.3f/%.3f@%.2f", j,
                            sqrt(g_apu_voice_sq[j] / g_apu_voice_n[j]),
                            g_apu_voice_dc[j] / g_apu_voice_n[j],
                            g_apu_voice_nyq[j] / g_apu_voice_n[j], g_apu_voice_rate[j]);
                }
                fprintf(stderr, "\n");
                memset(g_apu_voice_dc, 0, sizeof g_apu_voice_dc);
                memset(g_apu_voice_nyq, 0, sizeof g_apu_voice_nyq);
                memset(g_apu_voice_sq, 0, sizeof g_apu_voice_sq);
                memset(g_apu_voice_n, 0, sizeof g_apu_voice_n);
            }
            {
                extern float g_apu_mix_peak;
                fprintf(stderr, "  [APU]   mix peak before clamp %.2f\n", g_apu_mix_peak);
                g_apu_mix_peak = 0;
            }
            if (g_apu_bin_n) {
                fprintf(stderr, "  [APU]   bin rms x1000:");
                for (b = 0; b < 32; b++) {
                    fprintf(stderr, " %d:%.0f", b, 1000.0 * sqrt(g_apu_bin_sq[b] / g_apu_bin_n));
                    g_apu_bin_sq[b] = 0;
                }
                fprintf(stderr, "\n");
                g_apu_bin_n = 0;
            }
        }
    }

    /* XAudio2 path: the guest's audio, then the host mixer on top.
     *
     * monitor.frame_buf holds what the VP and the DSP stage accumulated over
     * the last 8 frames (256 samples). This path used to clear it and render
     * only the test tone and the host-side mixer, so everything the title
     * played through DirectSound was thrown away -- and it submitted 1024
     * samples for 256 samples' worth of time. Now it submits exactly the
     * frames that elapsed, with the mixer and tone added (mixer_render
     * accumulates with clamping). */
    if (xa2_is_active()) {
        int16_t out[256][2];

        /* RECOMP_APU_GUEST_AUDIO=0 mutes the guest's audio. (A clipping
         * 750 Hz buzz seen here was the APU reading its voices from low RAM
         * instead of the contiguous window -- a ram_ptr problem in the
         * caller, fixed in Amped 2's main.c.) */
        static int guest = -1;
        if (guest < 0) {
            const char *e = getenv("RECOMP_APU_GUEST_AUDIO");
            guest = !(e && *e == '0');
        }
        if (guest)
            memcpy(out, d->monitor.frame_buf, sizeof out);
        else
            memset(out, 0, sizeof out);
        memset(d->monitor.frame_buf, 0, sizeof(d->monitor.frame_buf));
        if (g_audio_muted) {
            memset(out, 0, sizeof out);
        } else {
            if (g_test_tone.active) {
                for (int i = 0; i < 256; i++) {
                    int32_t s = (int32_t)(sin(g_test_tone.phase) * g_test_tone.amplitude);
                    int32_t l = out[i][0] + s, r = out[i][1] + s;
                    out[i][0] = (int16_t)(l > 32767 ? 32767 : (l < -32768 ? -32768 : l));
                    out[i][1] = (int16_t)(r > 32767 ? 32767 : (r < -32768 ? -32768 : r));
                    g_test_tone.phase += g_test_tone.phase_inc;
                    if (g_test_tone.phase >= 2.0 * M_PI)
                        g_test_tone.phase -= 2.0 * M_PI;
                }
            }
            mixer_render(out, 256);
        }
        xa2_submit_samples((const int16_t *)out, 256);
        return;
    }

    if (!g_waveout.initialized) return;

    int idx = g_waveout.next_buf;
    WAVEHDR *hdr = &g_waveout.hdrs[idx];

    /* Wait if this buffer is still playing (with timeout) */
    int wait_loops = 0;
    while (!(hdr->dwFlags & WHDR_DONE) && (hdr->dwFlags & WHDR_INQUEUE)) {
        qemu_mutex_unlock(&d->lock);
        Sleep(1);
        qemu_mutex_lock(&d->lock);
        if (++wait_loops > 50) break;
    }

    /* Fill the large waveOut buffer by rendering multiple 256-sample frames */
    int16_t *out = (int16_t *)g_waveout.bufs[idx];
    int remaining = WAVEOUT_BUF_SAMPLES;
    int out_offset = 0;

    while (remaining > 0) {
        int chunk = (remaining < MIXER_FRAME_SAMPLES) ? remaining : MIXER_FRAME_SAMPLES;

        memset(d->monitor.frame_buf, 0, sizeof(d->monitor.frame_buf));

        /* Test tone (skip if muted) */
        if (g_test_tone.active && !g_audio_muted) {
            for (int i = 0; i < chunk; i++) {
                int16_t s = (int16_t)(sin(g_test_tone.phase) * g_test_tone.amplitude);
                d->monitor.frame_buf[i][0] = s;
                d->monitor.frame_buf[i][1] = s;
                g_test_tone.phase += g_test_tone.phase_inc;
                if (g_test_tone.phase >= 2.0 * M_PI)
                    g_test_tone.phase -= 2.0 * M_PI;
            }
        }

        /* Mix software voices (skip if muted) */
        if (!g_audio_muted)
            mixer_render(d->monitor.frame_buf, chunk);

        /* Copy to waveOut buffer */
        memcpy(out + out_offset * 2, d->monitor.frame_buf, chunk * 2 * sizeof(int16_t));
        out_offset += chunk;
        remaining -= chunk;
    }

    /* Submit to waveOut */
    hdr->dwFlags &= ~WHDR_DONE;
    waveOutWrite(g_waveout.hwo, hdr, sizeof(WAVEHDR));

    g_waveout.next_buf = (idx + 1) % WAVEOUT_NUM_BUFS;
    g_waveout.frames_written++;
}

/* ============================================================
 * Throttle (timing control for frame pacing)
 * ============================================================ */

static void throttle(MCPXAPUState *d)
{
    if (d->ep_frame_div % 8) {
        return;
    }

    int64_t now_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int queued = xa2_queued();

    /* Paced by the wall clock, kept honest by the output queue. This used
     * to restart the schedule whenever it ran more than one frame late, so
     * a late wake-up or a slow frame (up to 7 ms seen) was simply lost: on
     * Amped 2's night stage the APU ran at 1360-1450 frames a second of
     * 1500, the 85 ms queue ran dry and the sound broke up. Now only a
     * stall of 100 ms gives up on the time; anything less is made up. And
     * the queue has the last word, since the device's clock is the one the
     * sound plays at: three buffers or fewer left, render now; twelve or
     * more, wait a frame longer (a full queue drops what is submitted). */
    if (d->next_frame_time_us == 0 ||
        now_us - d->next_frame_time_us > 100000) {
        d->next_frame_time_us = now_us;
    }
    if (queued >= 0 && queued <= 3 && d->next_frame_time_us > now_us)
        d->next_frame_time_us = now_us;
    if (queued >= 12)
        d->next_frame_time_us += EP_FRAME_US;
    if (queued == 0)
        g_apu_underruns++;
    if (queued >= 16)
        g_apu_overruns++;

    while (!d->pause_requested) {
        now_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        int64_t remaining_ms = (d->next_frame_time_us - now_us) / 1000;
        if (remaining_ms > 0) {
            qemu_cond_timedwait(&d->cond, &d->lock, (int)remaining_ms);
        } else {
            break;
        }
    }
    d->next_frame_time_us += EP_FRAME_US;

    d->sleep_acc_us += (int)(qemu_clock_get_us(QEMU_CLOCK_REALTIME) - now_us);
}

/* ============================================================
 * se_frame - Process one audio frame (VP -> GP -> EP pipeline)
 * ============================================================ */

static void se_frame(MCPXAPUState *d)
{
    mcpx_apu_update_dsp_preference(d);
    mcpx_debug_begin_frame();
    g_dbg.gp_realtime = d->gp.realtime;
    g_dbg.ep_realtime = d->ep.realtime;

    int64_t now_ms = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    int64_t elapsed_ms = now_ms - d->frame_count_time_ms;
    if (elapsed_ms >= 1000) {
        g_dbg.utilization = 1.0f - d->sleep_acc_us / (elapsed_ms * 1000.0f);
        g_dbg.frames_processed = (int)(d->frame_count * 1000.0 / elapsed_ms + 0.5);
        d->frame_count_time_ms = now_ms;
        d->frame_count = 0;
        d->sleep_acc_us = 0;
    }
    d->frame_count++;

    /* Buffer for all mixbins for this frame */
    float mixbins[NUM_MIXBINS][NUM_SAMPLES_PER_FRAME];
    memset(mixbins, 0, sizeof(mixbins));

    mcpx_apu_vp_frame(d, mixbins);
    mcpx_apu_dsp_frame(d, mixbins);
    mcpx_apu_monitor_frame(d);

    d->ep_frame_div++;

    mcpx_debug_end_frame();
}

/* ============================================================
 * APU frame thread (background processing)
 * ============================================================ */

static void *mcpx_apu_frame_thread(void *arg)
{
    MCPXAPUState *d = MCPX_APU_DEVICE(arg);
    bool stalled = false;     /* waiting out a trapped front end */
    int64_t trap_since_us = 0;  /* when the current trap began (0: none) */
    qemu_mutex_lock(&d->lock);

    while (!qatomic_read(&d->exiting)) {
        if (d->pause_requested && !g_test_tone.active && !g_mixer_active_count) {
            d->is_idle = true;
            qemu_cond_signal(&d->idle_cond);
            qemu_cond_wait(&d->cond, &d->lock);
            d->is_idle = false;
            continue;
        }

        /* Always run the audio output loop — the software mixer and test tone
         * need continuous frame delivery regardless of APU register state.
         * The VP/DSP pipeline (se_frame) only runs when registers allow it. */
        if (!stalled)
            throttle(d);
        stalled = false;

        int xcntmode = GET_MASK(qatomic_read(&d->regs[NV_PAPU_SECTL]),
                                NV_PAPU_SECTL_XCNTMODE);
        uint32_t fectl = qatomic_read(&d->regs[NV_PAPU_FECTL]);
        bool apu_active = (xcntmode != NV_PAPU_SECTL_XCNTMODE_OFF) &&
                          !(fectl & NV_PAPU_FECTL_FEMETHMODE_TRAPPED) &&
                          !(fectl & NV_PAPU_FECTL_FEMETHMODE_HALTED);

        if (apu_active)
            trap_since_us = 0;
        if (apu_active && !g_test_tone.active) {
            /* Full pipeline: VP voices → DSP → monitor → waveOut */
            int64_t f0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME), f1;
            se_frame(d);
            f1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME) - f0;
            g_apu_frame_us += f1;
            if (f1 > g_apu_frame_max_us) g_apu_frame_max_us = f1;
            /* A voice event from the frame raises the line (xemu does this
             * in its frame thread; the port had dropped it). */
            if (d->set_irq) {
                d->set_irq = false;
                update_irq(d);
            }
        } else if (xcntmode != NV_PAPU_SECTL_XCNTMODE_OFF && xa2_is_active()
                   && !g_test_tone.active
                   && (!trap_since_us
                       || qemu_clock_get_us(QEMU_CLOCK_REALTIME) - trap_since_us < 5000)) {
            /* Front end trapped or halted: DirectSound's idle-voice method
             * (SE2FE_IDLE_VOICE) traps it until the title's ISR and DPC have
             * run, a timer-thread pass or more here. This used to emit the
             * sub-frame anyway, silent, and on Amped 2's night stage 7 of
             * every 8 sub-frames around a trap came out as zeros -- 224-
             * sample holes in the output, dozens a second, heard as the
             * sound breaking up. The VP produces nothing while trapped (as
             * xemu's frame thread does); wait for the title to release it,
             * without moving the schedule, and the output queue's pacing
             * makes the time up. */
            /* Bounded: a front end held for 5 ms on end (once, 1 run in 12,
             * the title left it trapped through most of a level and the APU
             * ran at a third of real time) gets the old treatment below --
             * silent sub-frames on schedule -- until it is released. */
            if (!trap_since_us)
                trap_since_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            if (d->set_irq) {
                d->set_irq = false;
                update_irq(d);
            }
            qemu_cond_timedwait(&d->cond, &d->lock, 1);
            stalled = true;
            g_apu_stalls++;
            continue;
        } else {
            /* Lightweight: just monitor frame (test tone + software mixer) */
            mcpx_apu_monitor_frame(d);
            d->ep_frame_div++;
        }

        /* Hand the lock to a queued guest write. throttle() only releases
         * it while it sleeps, and a frame thread that has fallen behind real
         * time never sleeps: a Critical Section is not fair, so it took the
         * lock straight back every frame, and DirectSound's VOICE_LOCK
         * write waited for good while holding the title's own DirectSound
         * lock -- the whole game stopped (Amped 2 ML1, 1-4 min in, once its
         * sounds actually played). */
        if (qatomic_read(&g_apu_lock_waiters)) {
            int64_t t0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            int64_t until = t0 + 2000;
            qemu_mutex_unlock(&d->lock);
            while (qatomic_read(&g_apu_lock_waiters) &&
                   qemu_clock_get_us(QEMU_CLOCK_REALTIME) < until)
                SwitchToThread();
            qemu_mutex_lock(&d->lock);
            g_apu_handoff_us += qemu_clock_get_us(QEMU_CLOCK_REALTIME) - t0;
            g_apu_handoffs++;
        }
    }

    qemu_mutex_unlock(&d->lock);
    return NULL;
}

/* ============================================================
 * Wait for idle / resume helpers
 * ============================================================ */

static void mcpx_apu_wait_for_idle(MCPXAPUState *d)
{
    d->pause_requested = true;
    qemu_cond_signal(&d->cond);
    while (!d->is_idle) {
        qemu_cond_wait(&d->idle_cond, &d->lock);
    }
}

static void mcpx_apu_resume(MCPXAPUState *d)
{
    d->pause_requested = false;
    qemu_cond_signal(&d->cond);
}

/* ============================================================
 * Reset
 * ============================================================ */

static void mcpx_apu_reset_locked(MCPXAPUState *d)
{
    memset(d->regs, 0, sizeof(d->regs));
    mcpx_apu_vp_reset(d);

    if (d->gp.dsp) {
        memset((void *)d->gp.dsp->core.pram_opcache, 0,
               sizeof(d->gp.dsp->core.pram_opcache));
    }
    if (d->ep.dsp) {
        memset((void *)d->ep.dsp->core.pram_opcache, 0,
               sizeof(d->ep.dsp->core.pram_opcache));
    }
    d->set_irq = false;
}

/* ============================================================
 * Public API: Init / Shutdown
 * ============================================================ */

MCPXAPUState *mcpx_apu_init_standalone(uint8_t *ram_ptr)
{
    MCPXAPUState *d = (MCPXAPUState *)calloc(1, sizeof(MCPXAPUState));
    if (!d) {
        fprintf(stderr, "[APU] Failed to allocate MCPXAPUState\n");
        return NULL;
    }

    g_apu_ram_ptr = ram_ptr;
    {
        /* Low RAM, for the pages MmGetPhysicalAddress says came from
         * ordinary memory (apu_phys). */
        extern void *xbox_GetMemoryBase(void);
        g_apu_lowram_ptr = (uint8_t *)xbox_GetMemoryBase();
        if (g_apu_lowram_ptr == ram_ptr)
            g_apu_lowram_ptr = NULL;        /* the old single-store setup */
    }
    g_state = d;
    d->ram_ptr = ram_ptr;
    g_xbox_apu_irq_poll = mcpx_apu_poll_irq;

    d->set_irq = false;
    d->exiting = false;
    d->is_idle = false;
    d->pause_requested = true;

    qemu_mutex_init(&d->lock);
    qemu_mutex_lock(&d->lock);
    qemu_cond_init(&d->cond);
    qemu_cond_init(&d->idle_cond);

    /* Init VP (voice processor) */
    mcpx_apu_vp_init(d);

    /* Init DSP (GP/EP - stubbed) */
    mcpx_apu_dsp_init(d);

    /* Init software mixer for DirectSound bridge */
    mixer_init();

    /* Init monitor (waveOut output) */
    Error *local_err = NULL;
    mcpx_apu_monitor_init(d, &local_err);
    if (local_err) {
        warn_reportf_err(local_err, "mcpx_apu_monitor_init failed: ");
    }

    /* Start background frame thread */
    qemu_thread_create(&d->apu_thread, "mcpx.apu_thread",
                       mcpx_apu_frame_thread, d, QEMU_THREAD_JOINABLE);
    mcpx_apu_wait_for_idle(d);
    qemu_mutex_unlock(&d->lock);

    fprintf(stderr, "[APU] MCPX APU initialized (standalone)\n");
    fprintf(stderr, "[APU]   RAM pointer: %p\n", (void *)ram_ptr);
    fprintf(stderr, "[APU]   MMIO base: 0xFE800000 (512KB)\n");
    fprintf(stderr, "[APU]   VP: %d max voices, %d samples/frame\n",
            MCPX_HW_MAX_VOICES, NUM_SAMPLES_PER_FRAME);
    return d;
}

void mcpx_apu_shutdown(MCPXAPUState *d)
{
    if (!d) return;

    fprintf(stderr, "[APU] Shutting down MCPX APU...\n");

    qemu_mutex_lock(&d->lock);
    mcpx_apu_wait_for_idle(d);
    qatomic_set(&d->exiting, true);
    qemu_cond_signal(&d->cond);
    qemu_mutex_unlock(&d->lock);

    qemu_thread_join(&d->apu_thread);
    mcpx_apu_vp_finalize(d);
    mcpx_apu_monitor_finalize(d);

    free(d);
    g_state = NULL;
    fprintf(stderr, "[APU] Shutdown complete\n");
}

/* ============================================================
 * VP MMIO handlers (sub-region at +0x20000)
 *
 * These are called when the game writes to the VP PIO registers
 * to configure voices, SSL, etc.
 * ============================================================ */

uint64_t mcpx_apu_vp_read(void *opaque, hwaddr addr, unsigned int size);
void mcpx_apu_vp_write(void *opaque, hwaddr addr, uint64_t val, unsigned int size);

/* Dispatch a VP-region access (offset 0x20000-0x2FFFF from APU base) */
void mcpx_apu_dispatch_mmio(MCPXAPUState *d, hwaddr addr, uint64_t val,
                             unsigned int size, bool is_write)
{
    if (addr >= 0x20000 && addr < 0x30000) {
        /* VP region */
        hwaddr vp_addr = addr - 0x20000;
        if (is_write) {
            mcpx_apu_vp_write(d, vp_addr, val, size);
        }
        /* VP reads handled by caller if needed */
    } else if (addr < 0x20000) {
        /* Main APU registers */
        if (is_write) {
            mcpx_apu_write(d, addr, val, size);
        }
    }
    /* GP (0x30000) and EP (0x50000) regions ignored for now */
}

/* ============================================================
 * Public MMIO API (called from VEH or MMIO hook)
 * addr is offset from APU base (0xFE800000)
 * ============================================================ */

uint64_t mcpx_apu_mmio_read(MCPXAPUState *d, uint64_t addr, unsigned int size)
{
    if (!d) return 0;
    if (addr >= 0x20000 && addr < 0x30000) {
        return mcpx_apu_vp_read(d, addr - 0x20000, size);
    } else if (addr < 0x20000) {
        return mcpx_apu_read(d, (hwaddr)addr, size);
    }
    return 0;
}

void mcpx_apu_mmio_write(MCPXAPUState *d, uint64_t addr, uint64_t val, unsigned int size)
{
    if (!d) return;
    mcpx_apu_dispatch_mmio(d, (hwaddr)addr, val, size, true);
}

/* ============================================================
 * APU Test Tone - Direct waveOut sine generator
 *
 * Bypasses the VP pipeline entirely and writes a 440Hz sine wave
 * directly to the monitor frame_buf. This verifies that waveOut
 * output works correctly.
 * ============================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void mcpx_apu_play_test_tone(MCPXAPUState *d)
{
    if (!d) {
        fprintf(stderr, "[APU-TEST] No APU state\n");
        return;
    }

    if (g_test_tone.active) {
        /* Toggle off */
        g_test_tone.active = false;
        fprintf(stderr, "[APU-TEST] Test tone OFF\n");
        return;
    }

    /* 440Hz at 48kHz sample rate */
    g_test_tone.phase = 0.0;
    g_test_tone.phase_inc = 2.0 * M_PI * 440.0 / 48000.0;
    g_test_tone.amplitude = 6000;  /* ~18% of full scale */
    g_test_tone.active = true;

    /* Make sure waveOut is running - enable SECTL and resume APU thread */
    qemu_mutex_lock(&d->lock);
    d->regs[NV_PAPU_SECTL] = NV_PAPU_SECTL_XCNTMODE & ~NV_PAPU_SECTL_XCNTMODE_OFF;
    d->regs[NV_PAPU_FECTL] = NV_PAPU_FECTL_FEMETHMODE_FREE_RUNNING;
    /* Initialize empty voice lists so VP frame doesn't crash */
    d->regs[NV_PAPU_TVL2D] = 0xFFFF;
    d->regs[NV_PAPU_TVL3D] = 0xFFFF;
    d->regs[NV_PAPU_TVLMP] = 0xFFFF;
    mcpx_apu_resume(d);
    qemu_mutex_unlock(&d->lock);

    fprintf(stderr, "[APU-TEST] Test tone ON - 440Hz sine, amplitude=%d\n",
            g_test_tone.amplitude);
}

/* ============================================================
 * Software mixer - mixes DirectSound buffers to waveOut
 *
 * This bypasses the VP hardware voice pipeline entirely.
 * DirectSound buffers register PCM data here, and the APU
 * frame thread mixes them into the monitor frame_buf.
 * ============================================================ */

static void mixer_init(void)
{
    if (g_mixer_initialized) return;
    InitializeCriticalSection(&g_mixer_cs);
    memset(g_mixer_voices, 0, sizeof(g_mixer_voices));
    g_mixer_initialized = true;
}

int apu_mixer_alloc_voice(void)
{
    if (!g_mixer_initialized) mixer_init();
    EnterCriticalSection(&g_mixer_cs);
    for (int i = 0; i < APU_MIXER_MAX_VOICES; i++) {
        if (!g_mixer_voices[i].active && !g_mixer_voices[i].pcm_data) {
            g_mixer_voices[i].volume = 1.0f;
            g_mixer_voices[i].sample_rate = 44100;
            g_mixer_voices[i].num_channels = 2;
            LeaveCriticalSection(&g_mixer_cs);
            return i;
        }
    }
    LeaveCriticalSection(&g_mixer_cs);
    return -1;
}

void apu_mixer_free_voice(int slot)
{
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return;
    EnterCriticalSection(&g_mixer_cs);
    apu_mixer_stop(slot);
    g_mixer_voices[slot].pcm_data = NULL;
    g_mixer_voices[slot].pcm_bytes = 0;
    g_mixer_voices[slot].play_offset = 0;
    LeaveCriticalSection(&g_mixer_cs);
}

APUMixerVoice *apu_mixer_get_voice(int slot)
{
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return NULL;
    return &g_mixer_voices[slot];
}

int apu_mixer_set_position(int slot, uint32_t byte_offset)
{
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return 0;
    EnterCriticalSection(&g_mixer_cs);
    APUMixerVoice *v = &g_mixer_voices[slot];
    uint32_t frame_bytes = v->num_channels * sizeof(int16_t);
    int valid = (v->num_channels == 1 || v->num_channels == 2) &&
                byte_offset < v->pcm_bytes &&
                byte_offset / frame_bytes < v->pcm_bytes / frame_bytes;
    if (valid) v->play_offset = ((uint64_t)(byte_offset / frame_bytes)) << 16;
    LeaveCriticalSection(&g_mixer_cs);
    return valid;
}

void apu_mixer_get_state(int slot, uint32_t *byte_offset, int *active, int *looping)
{
    if (byte_offset) *byte_offset = 0;
    if (active) *active = 0;
    if (looping) *looping = 0;
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return;
    EnterCriticalSection(&g_mixer_cs);
    APUMixerVoice *v = &g_mixer_voices[slot];
    if (byte_offset) *byte_offset = (uint32_t)(v->play_offset >> 16) *
                                    v->num_channels * sizeof(int16_t);
    if (active) *active = v->active;
    if (looping) *looping = v->active && v->looping;
    LeaveCriticalSection(&g_mixer_cs);
}

void apu_mixer_play(int slot, int looping)
{
    if (g_audio_muted) return;
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return;
    EnterCriticalSection(&g_mixer_cs);
    APUMixerVoice *v = &g_mixer_voices[slot];
    if (!v->pcm_data || (v->num_channels != 1 && v->num_channels != 2) ||
        v->pcm_bytes < v->num_channels * sizeof(int16_t)) {
        LeaveCriticalSection(&g_mixer_cs);
        return;
    }
    v->looping = looping;
    if (!v->active) InterlockedIncrement((volatile LONG *)&g_mixer_active_count);
    v->active = 1;

    static int play_log_count = 0;
    if (play_log_count < 20) {
        fprintf(stderr, "[APU-MIX] Play voice %d: %u bytes, %u ch, %u Hz, vol=%.2f, loop=%d\n",
                slot, v->pcm_bytes, v->num_channels, v->sample_rate, v->volume, looping);
        play_log_count++;
    }
    LeaveCriticalSection(&g_mixer_cs);

    /* The frame thread takes the APU lock before the mixer lock. */
    extern MCPXAPUState *g_state;
    if (g_state) {
        qemu_mutex_lock(&g_state->lock);
        g_state->pause_requested = false;
        qemu_cond_signal(&g_state->cond);
        qemu_mutex_unlock(&g_state->lock);
    }
}

void apu_mixer_stop(int slot)
{
    if (slot < 0 || slot >= APU_MIXER_MAX_VOICES) return;
    EnterCriticalSection(&g_mixer_cs);
    if (g_mixer_voices[slot].active) {
        g_mixer_voices[slot].active = 0;
        InterlockedDecrement((volatile LONG *)&g_mixer_active_count);
    }
    LeaveCriticalSection(&g_mixer_cs);
}

/* Mix all active voices into frame_buf. Called from mcpx_apu_monitor_frame.
 * Keep 16 fractional bits in a wide offset so buffers can exceed 65536 frames. */
static void mixer_render(int16_t frame_buf[][2], int num_samples)
{
    if (!g_mixer_initialized) return;
    EnterCriticalSection(&g_mixer_cs);

    for (int v = 0; v < APU_MIXER_MAX_VOICES; v++) {
        APUMixerVoice *voice = &g_mixer_voices[v];
        if (!voice->active || !voice->pcm_data || voice->pcm_bytes == 0)
            continue;

        uint32_t total_frames = voice->pcm_bytes / sizeof(int16_t);
        if (voice->num_channels == 2) total_frames /= 2;
        if (total_frames == 0) continue;

        /* Fixed-point 16.16 increment per output sample */
        uint64_t inc = ((uint64_t)voice->sample_rate << 16) / 48000;
        uint64_t pos = voice->play_offset;
        uint64_t end = (uint64_t)total_frames << 16;
        float vol = voice->volume;

        for (int i = 0; i < num_samples; i++) {
            uint32_t src_frame = pos >> 16;

            if (src_frame >= total_frames) {
                if (voice->looping) {
                    pos %= end;
                    src_frame = (uint32_t)(pos >> 16);
                } else {
                    pos = 0;
                    voice->active = 0;
                    InterlockedDecrement((volatile LONG *)&g_mixer_active_count);
                    break;
                }
            }

            int32_t left, right;
            if (voice->num_channels >= 2) {
                left  = (int32_t)(voice->pcm_data[src_frame * 2] * vol);
                right = (int32_t)(voice->pcm_data[src_frame * 2 + 1] * vol);
            } else {
                left = right = (int32_t)(voice->pcm_data[src_frame] * vol);
            }

            /* Accumulate (mix) into frame_buf with clamping */
            int32_t mixed_l = frame_buf[i][0] + left;
            int32_t mixed_r = frame_buf[i][1] + right;
            if (mixed_l > 32767) mixed_l = 32767;
            if (mixed_l < -32768) mixed_l = -32768;
            if (mixed_r > 32767) mixed_r = 32767;
            if (mixed_r < -32768) mixed_r = -32768;
            frame_buf[i][0] = (int16_t)mixed_l;
            frame_buf[i][1] = (int16_t)mixed_r;

            pos += inc;
        }

        voice->play_offset = pos;
        uint32_t end_frame = pos >> 16;
        if (end_frame >= total_frames) {
            if (voice->looping) {
                voice->play_offset = pos % end;
            } else if (voice->active) {
                voice->play_offset = 0;
                voice->active = 0;
                InterlockedDecrement((volatile LONG *)&g_mixer_active_count);
            }
        }
    }
    LeaveCriticalSection(&g_mixer_cs);
}
