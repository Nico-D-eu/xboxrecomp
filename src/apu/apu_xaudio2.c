/**
 * XAudio2 Audio Output Backend
 *
 * Provides low-latency audio output via XAudio2 (Win7+).
 * Called from the APU monitor frame to submit mixed samples.
 * Falls back gracefully if XAudio2 is unavailable.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "apu_xaudio2.h"

/* The XAudio2 backend is Windows-only. On Linux all xa2_* functions are
 * stubbed to report inactive; real audio output via SDL2 comes later. */
#if defined(_WIN32)

#define COBJMACROS
#include <windows.h>
#include <xaudio2.h>

#pragma comment(lib, "xaudio2.lib")
#pragma comment(lib, "ole32.lib")

#define XA2_SAMPLE_RATE   48000
#define XA2_CHANNELS      2
#define XA2_BUF_SAMPLES   1024   /* ~21ms per submission */
/* Submissions are 256 samples (5.3 ms) now, one per 8 APU frames; 16 in the
 * queue is 85 ms of slack against the frame thread's jitter. */
#define XA2_NUM_BUFS      16

static IXAudio2               *g_xa2 = NULL;
static IXAudio2MasteringVoice *g_xa2_master = NULL;
static IXAudio2SourceVoice    *g_xa2_source = NULL;
static int16_t                 g_xa2_bufs[XA2_NUM_BUFS][XA2_BUF_SAMPLES][2];
static int                     g_xa2_next_buf = 0;
static int                     g_xa2_initialized = 0;
static int                     g_xa2_frames_written = 0;

int xa2_init(void)
{
    HRESULT hr;
    int com_initialized;
    WAVEFORMATEX wfx = { 0 };

    if (g_xa2_initialized) return 1;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "[XA2] CoInitializeEx failed: 0x%08lX\n", hr);
        return 0;
    }
    com_initialized = SUCCEEDED(hr);

    hr = XAudio2Create(&g_xa2, 0, XAUDIO2_DEFAULT_PROCESSOR);
    if (FAILED(hr) || !g_xa2) {
        fprintf(stderr, "[XA2] XAudio2Create failed: 0x%08lX\n", hr);
        goto fail;
    }

    hr = IXAudio2_CreateMasteringVoice(g_xa2, &g_xa2_master,
        XA2_CHANNELS, XA2_SAMPLE_RATE, 0, NULL, NULL, 0);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] CreateMasteringVoice failed: 0x%08lX\n", hr);
        goto fail;
    }
    /* RECOMP_AUDIO_VOLUME=<0..100>: the master volume, in percent. */
    {
        const char *e = getenv("RECOMP_AUDIO_VOLUME");
        if (e && *e) {
            int pct = atoi(e);
            if (pct < 0) pct = 0;
            if (pct > 100) pct = 100;
            IXAudio2MasteringVoice_SetVolume(g_xa2_master, (float)pct / 100.0f, 0);
        }
    }

    wfx.wFormatTag      = WAVE_FORMAT_PCM;
    wfx.nChannels       = XA2_CHANNELS;
    wfx.nSamplesPerSec  = XA2_SAMPLE_RATE;
    wfx.wBitsPerSample  = 16;
    wfx.nBlockAlign     = XA2_CHANNELS * 2;
    wfx.nAvgBytesPerSec = XA2_SAMPLE_RATE * wfx.nBlockAlign;

    hr = IXAudio2_CreateSourceVoice(g_xa2, &g_xa2_source,
        &wfx, 0, XAUDIO2_DEFAULT_FREQ_RATIO, NULL, NULL, NULL);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] CreateSourceVoice failed: 0x%08lX\n", hr);
        goto fail;
    }

    hr = IXAudio2SourceVoice_Start(g_xa2_source, 0, XAUDIO2_COMMIT_NOW);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] Start failed: 0x%08lX\n", hr);
        goto fail;
    }

    g_xa2_next_buf = 0;
    g_xa2_initialized = 1;
    g_xa2_frames_written = 0;

    fprintf(stderr, "[XA2] XAudio2 initialized (%d Hz stereo 16-bit, %d x %d-sample buffers)\n",
            XA2_SAMPLE_RATE, XA2_NUM_BUFS, XA2_BUF_SAMPLES);
    return 1;

fail:
    xa2_shutdown();
    /* Failed initialization still runs on the COM-initializing thread. */
    if (com_initialized) CoUninitialize();
    return 0;
}

void xa2_shutdown(void)
{
    if (g_xa2_source) {
        IXAudio2SourceVoice_Stop(g_xa2_source, 0, XAUDIO2_COMMIT_NOW);
        IXAudio2SourceVoice_FlushSourceBuffers(g_xa2_source);
        g_xa2_source->lpVtbl->DestroyVoice(g_xa2_source);
        g_xa2_source = NULL;
    }
    if (g_xa2_master) {
        g_xa2_master->lpVtbl->DestroyVoice(g_xa2_master);
        g_xa2_master = NULL;
    }
    if (g_xa2) {
        IXAudio2_Release(g_xa2);
        g_xa2 = NULL;
    }

    if (g_xa2_initialized)
        fprintf(stderr, "[XA2] Shut down (%d frames written)\n", g_xa2_frames_written);
    g_xa2_initialized = 0;
}

int xa2_is_active(void)
{
    return g_xa2_initialized;
}

/* Submit a buffer of mixed samples to XAudio2.
 * Called from APU frame thread. Returns 1 if buffer was submitted. */
int xa2_submit_samples(const int16_t *samples, int num_samples)
{
    XAUDIO2_VOICE_STATE state;
    XAUDIO2_BUFFER xbuf;
    int idx;
    int copy_samples;
    HRESULT hr;

    if (!g_xa2_initialized || !g_xa2_source) return 0;

    IXAudio2SourceVoice_GetState(g_xa2_source, &state, XAUDIO2_VOICE_NOSAMPLESPLAYED);
    if ((int)state.BuffersQueued >= XA2_NUM_BUFS) return 0;

    idx = g_xa2_next_buf;
    copy_samples = (num_samples > XA2_BUF_SAMPLES) ? XA2_BUF_SAMPLES : num_samples;
    memcpy(g_xa2_bufs[idx], samples, copy_samples * XA2_CHANNELS * sizeof(int16_t));

    memset(&xbuf, 0, sizeof(xbuf));
    xbuf.AudioBytes = copy_samples * XA2_CHANNELS * sizeof(int16_t);
    xbuf.pAudioData = (const BYTE *)g_xa2_bufs[idx];

    hr = IXAudio2SourceVoice_SubmitSourceBuffer(g_xa2_source, &xbuf, NULL);
    if (FAILED(hr)) return 0;

    g_xa2_next_buf = (idx + 1) % XA2_NUM_BUFS;
    g_xa2_frames_written++;

    /* RECOMP_AUDIO_WAV=<file>: the first 60 s of output as a 48 kHz stereo
     * WAV, to listen to what a headless run played. */
    {
        static FILE *wav;
        static int state = -1;
        static uint32_t written;
        if (state < 0) {
            const char *path = getenv("RECOMP_AUDIO_WAV");
            state = 0;
            if (path && (wav = fopen(path, "wb")) != NULL) {
                static const uint8_t hdr[44] = {
                    'R','I','F','F', 0,0,0,0, 'W','A','V','E', 'f','m','t',' ',
                    16,0,0,0, 1,0, 2,0, 0x80,0xBB,0,0, 0x00,0xEE,0x02,0, 4,0, 16,0,
                    'd','a','t','a', 0,0,0,0 };
                fwrite(hdr, 1, sizeof hdr, wav);
                state = 1;
            }
        }
        static uint64_t skipped, skip = (uint64_t)-1;
        if (skip == (uint64_t)-1) {
            const char *e = getenv("RECOMP_AUDIO_WAV_SKIP");     /* seconds */
            skip = e ? (uint64_t)atoi(e) * 48000u : 0;
        }
        if (state == 1 && skipped < skip) {
            skipped += (uint64_t)copy_samples;
        } else if (state == 1) {
            uint32_t bytes = (uint32_t)copy_samples * XA2_CHANNELS * 2u;
            fwrite(samples, 1, bytes, wav);
            written += bytes;
            if (written >= 48000u * 4u * 60u) {
                uint32_t riff = 36 + written;
                fseek(wav, 4, SEEK_SET);
                fwrite(&riff, 4, 1, wav);
                fseek(wav, 40, SEEK_SET);
                fwrite(&written, 4, 1, wav);
                fclose(wav);
                state = 2;
                fprintf(stderr, "  [AUDIO] 60 s of output written to %s\n",
                        getenv("RECOMP_AUDIO_WAV"));
            }
        }
    }

    /* RECOMP_AUDIO_METER=1: the output's peak and how often it was non-silent,
     * every ~5 s -- whether anything the title plays reaches the speakers,
     * on a machine nobody is listening to. */
    {
        static int meter = -1;
        static int peak;
        static unsigned bufs, loud;
        int i, p = 0;
        if (meter < 0)
            meter = getenv("RECOMP_AUDIO_METER") != NULL;
        if (meter) {
            for (i = 0; i < copy_samples * XA2_CHANNELS; i++) {
                int v = samples[i] < 0 ? -samples[i] : samples[i];
                if (v > p) p = v;
            }
            if (p > peak) peak = p;
            if (p > 64) loud++;
            if (++bufs >= 48000u * 5u / (unsigned)(copy_samples ? copy_samples : 1)) {
                fprintf(stderr, "  [AUDIO] 5 s: peak %d, %u of %u buffers audible\n",
                        peak, loud, bufs);
                fflush(stderr);
                peak = 0;
                bufs = loud = 0;
            }
        }
    }
    return 1;
}

int xa2_get_buffer_size(void)
{
    return XA2_BUF_SAMPLES;
}

int xa2_queued(void)
{
    XAUDIO2_VOICE_STATE state;
    if (!g_xa2_initialized || !g_xa2_source) return -1;
    IXAudio2SourceVoice_GetState(g_xa2_source, &state, XAUDIO2_VOICE_NOSAMPLESPLAYED);
    return (int)state.BuffersQueued;
}

#else /* !_WIN32 -- POSIX stubs (no audio output yet) */

int  xa2_init(void)                                   { return 0; }
void xa2_shutdown(void)                               {}
int  xa2_is_active(void)                              { return 0; }
int  xa2_submit_samples(const int16_t *s, int n)      { (void)s; (void)n; return 0; }
int  xa2_get_buffer_size(void)                        { return 0; }
int  xa2_queued(void)                                 { return -1; }

#endif /* _WIN32 */
