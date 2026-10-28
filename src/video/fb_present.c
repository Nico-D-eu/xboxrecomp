/**
 * Show the guest framebuffer in a window.
 *
 * The title renders into its own framebuffer in guest RAM and tells the kernel
 * where it is through AvSetDisplayMode; on hardware the CRTC scans that memory
 * out. Nothing here scans anything out, so however much of the GPU is
 * implemented, none of it is observable. This is the other half: a window that
 * reads that memory and puts it on screen.
 *
 * Deliberately plain GDI rather than the D3D8 layer. The point is to display
 * whatever the guest actually wrote, so the fewer stages between guest memory
 * and the screen the better -- and it must keep working while the D3D8 layer
 * is busy with something else, such as the FMV player's own window.
 * RECOMP_D3D_PRESENT=1 shows the same frames through a D3D11 swap chain of
 * the window's own instead (gpu_present.c): scaled on the GPU, vsync-paced.
 *
 * The window opens at the frame's size, or RECOMP_WINDOW_SIZE=<w>x<h>, or
 * RECOMP_WINDOW_SCALE times the frame (made smaller until it fits the
 * desktop), and can be resized; the frame is letterboxed to its 4:3 shape.
 * F11 or Alt+Enter toggles borderless fullscreen; RECOMP_WINDOW_MODE=
 * borderless or RECOMP_FULLSCREEN=1 starts in it. Closing the window ends the
 * process, as the SDL window does. With RECOMP_GPU=d3d11 the GPU backend
 * presents on this window through its own swap chain (RECOMP_GPU_PRESENT=gdi
 * keeps GDI) and the window only keeps s_rgb current for the dumps.
 *
 * Off unless RECOMP_FB_WINDOW is set.
 */
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Which keys are down, for the pad stand-in in src/input: this window
 * records them in the shared key table (src/input/keyboard.c).
 *
 * GetAsyncKeyState looked like the cheaper way to ask and does not work
 * here: it reads a state Wine keeps for the X server, and a guest process
 * drawing through GDI never sees it change. The window that has the focus
 * is the thing that receives the keys, so that is what has to remember
 * them. */
#include "keyboard.h"   /* src/input */
#include "gpu_present.h"
#include "recomp_exit_hook.h"   /* src/platform */
#include <stdbool.h>
#include "pvideo_scanout.h"
#include "../nv2a/nv2a_mmio_hook.h"

extern ptrdiff_t xbox_GetMemoryOffset(void);

static volatile LONG s_fb_running;
static uint32_t      s_fb_va, s_fb_pitch, s_fb_width = 640, s_fb_height = 480;
static uint32_t     *s_rgb;           /* converted 32-bit copy for GDI */

/* A finished frame, taken at the flip and shown until the next one.
 *
 * The window used to convert straight out of guest memory every 16 ms. Even
 * pointed at the buffer the title had just finished, that races the executor
 * drawing the next frame into the other one and, whenever the two swap, puts
 * a half-drawn image on the screen -- which is the flicker. Copying the
 * finished frame once per flip means the window never reads memory the
 * rasteriser is writing, so what it shows cannot be half of anything.
 *
 * Two buffers and an index, swapped after the copy completes, so the window
 * thread is never reading the one being filled. */
static uint32_t     *s_present[2];
static volatile LONG s_present_idx = -1;   /* -1 until the first flip */

/* The last PVIDEO frame shown. A frame stays on screen until the next one or
 * STOP, but once its bank is consumed the title may reuse the memory, so the
 * pixels are kept here rather than read again. */
static uint8_t      *s_pvideo_source;
static size_t        s_pvideo_bytes;
static unsigned      s_pvideo_bank;
static uint32_t      s_pvideo_regs[0x1000 / 4];

/* Draw the video overlay over s_rgb: a newly submitted bank if there is one
 * (copied, drawn, then handed back to the title), else the last frame. */
static void fb_overlay(void)
{
    uint32_t regs[0x1000 / 4];
    unsigned bank;
    bool presented = false;

    if (!s_rgb || !nv2a_hook_pvideo_snapshot(regs)) return;
    if ((regs[NV_PVIDEO_STOP / 4] & 1u) ||
        (s_pvideo_source &&
         regs[(NV_PVIDEO_SIZE_IN + s_pvideo_bank * 4) / 4] == UINT32_MAX)) {
        free(s_pvideo_source);
        s_pvideo_source = NULL;
        if (regs[NV_PVIDEO_STOP / 4] & 1u) return;
    }
    for (bank = 0; bank < 2; ++bank) {
        uint64_t physical = (uint64_t)regs[(NV_PVIDEO_BASE + bank * 4) / 4] +
                            regs[(NV_PVIDEO_OFFSET + bank * 4) / 4];
        unsigned pitch = regs[(NV_PVIDEO_FORMAT + bank * 4) / 4] & NV_PVIDEO_FORMAT_PITCH;
        unsigned height = (regs[(NV_PVIDEO_SIZE_IN + bank * 4) / 4] >> 16) & 0x7ff;
        uint64_t bytes = (uint64_t)pitch * height;
        uintptr_t va;
        MEMORY_BASIC_INFORMATION mapping;
        uint8_t *snapshot;

        if (!(regs[NV_PVIDEO_BUFFER / 4] & (1u << (bank * 4u)))) continue;
        if (!bytes || physical + bytes > 0x04000000u) continue;
        va = (uintptr_t)(0x80000000u + (uint32_t)physical) +
             (uintptr_t)xbox_GetMemoryOffset();
        if (!VirtualQuery((const void *)va, &mapping, sizeof(mapping)) ||
            mapping.State != MEM_COMMIT ||
            (mapping.Protect & (PAGE_GUARD | PAGE_NOACCESS)) ||
            bytes > mapping.RegionSize - (size_t)(va - (uintptr_t)mapping.BaseAddress))
            continue;
        snapshot = (uint8_t *)malloc((size_t)bytes);
        if (!snapshot) continue;
        memcpy(snapshot, (const void *)va, (size_t)bytes);
        if (!pvideo_scanout(regs, snapshot, (size_t)bytes, s_rgb,
                            s_fb_width, s_fb_height, bank)) {
            free(snapshot);
            continue;
        }
        free(s_pvideo_source);
        s_pvideo_source = snapshot;
        s_pvideo_bytes = (size_t)bytes;
        s_pvideo_bank = bank;
        memcpy(s_pvideo_regs, regs, sizeof(s_pvideo_regs));
        presented = true;
        nv2a_hook_pvideo_consume(bank);
    }
    if (!presented && s_pvideo_source)
        pvideo_scanout(s_pvideo_regs, s_pvideo_source, s_pvideo_bytes, s_rgb,
                       s_fb_width, s_fb_height, s_pvideo_bank);
}

void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch)
{
    /* RECOMP_FB_VA pins the window to one guest address instead of following
     * whichever surface is being drawn into. A black window cannot distinguish
     * "the read path is broken" from "the title rendered black", and pointing
     * it at memory known to have content settles that. */
    const char *pin = getenv("RECOMP_FB_VA");

    s_fb_va = pin ? (uint32_t)strtoul(pin, NULL, 0) : fb_va;
    if (pitch)
        s_fb_pitch = pitch;
}

/* Vertical anti-aliasing factor of the surface being shown: 2 for 2x2, where
 * the surface has twice the rows as well as twice the columns, and 1 for
 * everything else (2x1 included). Only the pushbuffer executor knows it. */
static uint32_t s_fb_aa_sy = 1;

void xbox_FramebufferWindowSetAA(uint32_t sy)
{
    s_fb_aa_sy = sy ? sy : 1;
}

/* Called by the pushbuffer executor when the title flips. */
void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch)
{
    const uint8_t *src;
    LONG next;
    uint32_t bpp, x, y;

    if (!s_fb_running || !fb_va || !pitch)
        return;
    if (getenv("RECOMP_FB_VA"))
        return;                       /* pinned: leave the old path alone */
    next = (s_present_idx == 0) ? 1 : 0;
    if (!s_present[next]) {
        s_present[next] = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height,
                                             4);
        if (!s_present[next])
            return;
    }
    bpp = pitch / s_fb_width;
    src = (const uint8_t *)((uintptr_t)fb_va + xbox_GetMemoryOffset());
    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * pitch * (bpp == 8 ? s_fb_aa_sy : 1);
        uint32_t *dst = s_present[next] + (size_t)y * s_fb_width;

        if (bpp == 4) {
            memcpy(dst, row, (size_t)s_fb_width * 4);
        } else if (bpp == 8) {
            /* 2x-wide anti-aliased 32-bit surface: every other pixel, as in
             * fb_convert. */
            const uint32_t *p = (const uint32_t *)row;
            for (x = 0; x < s_fb_width; x++)
                dst[x] = p[x * 2];
        } else if (bpp == 2) {
            const uint16_t *p = (const uint16_t *)row;
            for (x = 0; x < s_fb_width; x++) {
                uint16_t v = p[x];
                uint32_t r = (uint32_t)((v >> 11) & 0x1F) * 255u / 31u;
                uint32_t g = (uint32_t)((v >>  5) & 0x3F) * 255u / 63u;
                uint32_t b = (uint32_t)( v        & 0x1F) * 255u / 31u;
                dst[x] = (r << 16) | (g << 8) | b;
            }
        } else {
            memset(dst, 0, (size_t)s_fb_width * 4);
        }
    }
    /* Published only once it is whole. */
    InterlockedExchange(&s_present_idx, next);
}

/* A finished frame from the D3D11 backend: 32-bit A8R8G8B8 rows, already
 * out of the GPU, so the window, its dumps and everything downstream see the
 * GPU's picture through the same buffers the software path fills. */
void xbox_FramebufferWindowPresentArgb(const void *px, uint32_t w, uint32_t h,
                                       uint32_t row_pitch)
{
    LONG next;
    uint32_t y, cw, ch;

    if (!s_fb_running || !px)
        return;
    next = (s_present_idx == 0) ? 1 : 0;
    if (!s_present[next]) {
        s_present[next] = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);
        if (!s_present[next])
            return;
    }
    cw = w < s_fb_width ? w : s_fb_width;
    ch = h < s_fb_height ? h : s_fb_height;
    for (y = 0; y < ch; y++)
        memcpy(s_present[next] + (size_t)y * s_fb_width,
               (const uint8_t *)px + (size_t)y * row_pitch, (size_t)cw * 4);
    InterlockedExchange(&s_present_idx, next);
}

/* Which keys are down lives in src/input/keyboard.c now (upstream): this
 * window records them there with xbox_InputKeySet, and
 * xbox_FramebufferKeyDown reads that table. */

/* Title bar, the way ps3recomp's window shows it. Written by the flip,
 * read once a second by the window thread; a torn read shows one stale
 * number for a second, which nobody can tell apart from a real one. */
static wchar_t       s_title[48] = L"Xbox Recomp";
static volatile LONG s_flips;
static volatile LONG s_frame_draws;
/* RECOMP_WINDOW_TITLE set: that caption stays, without the FPS line. */
static int           s_fixed_title;

void xbox_FramebufferWindowSetTitle(const uint16_t *name, int max_chars)
{
    int i;
    for (i = 0; i < max_chars && i < 47 && name[i]; i++)
        s_title[i] = (wchar_t)name[i];
    if (i)
        s_title[i] = 0;
}

void xbox_FramebufferWindowFrameStats(uint32_t draws)
{
    InterlockedIncrement(&s_flips);
    InterlockedExchange(&s_frame_draws, (LONG)draws);
}

/* The mouse cursor over the game: hidden from the start, shown while the
 * mouse moves, hidden again once it has been still for
 * RECOMP_CURSOR_HIDE_MS (2000 by default; 0 keeps it always visible). Only
 * over the client area -- the title bar and borders keep theirs. Windows
 * sends WM_MOUSEMOVE without a move as well (the window appearing under the
 * cursor, a resize, F11), so a move is the screen position changing. All of
 * this runs on the window thread. */
static int   s_cursor_hidden = 1;
static DWORD s_cursor_moved;
static POINT s_cursor_pos;
static DWORD s_cursor_hide_ms = 2000;

static void fb_cursor_init(void)
{
    const char *e = getenv("RECOMP_CURSOR_HIDE_MS");
    if (e)
        s_cursor_hide_ms = (DWORD)strtoul(e, NULL, 10);
    s_cursor_hidden = s_cursor_hide_ms != 0;
    GetCursorPos(&s_cursor_pos);
    s_cursor_moved = GetTickCount();
}

/* Is the cursor over this window's client area? */
static int fb_cursor_over_client(HWND h)
{
    POINT p;
    RECT r;
    if (!GetCursorPos(&p) || WindowFromPoint(p) != h)
        return 0;
    ScreenToClient(h, &p);
    GetClientRect(h, &r);
    return PtInRect(&r, p);
}

static void fb_cursor_moved(void)
{
    POINT p;
    if (!s_cursor_hide_ms || !GetCursorPos(&p)
            || (p.x == s_cursor_pos.x && p.y == s_cursor_pos.y))
        return;
    s_cursor_pos = p;
    s_cursor_moved = GetTickCount();
    if (s_cursor_hidden) {
        s_cursor_hidden = 0;
        SetCursor(LoadCursorA(NULL, IDC_ARROW));
    }
}

/* Once per pass of the window loop. */
static void fb_cursor_tick(HWND h)
{
    if (s_cursor_hidden || !s_cursor_hide_ms
            || GetTickCount() - s_cursor_moved < s_cursor_hide_ms)
        return;
    s_cursor_hidden = 1;
    if (fb_cursor_over_client(h))
        SetCursor(NULL);
}

/* Borderless fullscreen: the window as a popup over its whole monitor, with
 * the windowed placement kept to come back to. F11 or Alt+Enter toggles it;
 * RECOMP_WINDOW_MODE=borderless starts in it. The D3D11 presenter follows
 * the client size, so nothing else has to know. */
/* Set by the project: flush its log on quit, quickly and without stdio
 * (see WM_CLOSE in fb_wndproc). NULL: fprintf + g_xbox_flush_logs. */
void (*g_xbox_quit_logs)(void);

static int s_borderless;
static WINDOWPLACEMENT s_windowed_place = { sizeof(WINDOWPLACEMENT) };

static void fb_set_borderless(HWND h, int on)
{
    if (on == s_borderless)
        return;
    if (on) {
        MONITORINFO mi = { sizeof mi };
        GetWindowPlacement(h, &s_windowed_place);
        GetMonitorInfoA(MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST), &mi);
        SetWindowLongA(h, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(h, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                     mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
        SetWindowLongA(h, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
        SetWindowPlacement(h, &s_windowed_place);
        SetWindowPos(h, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
    s_borderless = on;
}

static LRESULT CALLBACK fb_wndproc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    /* F11, or Alt+Enter: borderless fullscreen (fb_set_borderless). First
     * press only (bit 30: the key was already down), and with Alt held
     * (bit 29) for Enter; neither reaches the key table, so Alt+Enter is not
     * also START, its auto-repeat included. (DXGI's own Alt+Enter, exclusive
     * mode at the frame's 640x480, is off -- gpu_present.c; both presenters
     * letterbox to any size.) */
    if ((m == WM_SYSKEYDOWN && w == VK_RETURN && (l & (1 << 29)) && !(l & (1 << 30)))
        || (m == WM_KEYDOWN && w == VK_F11 && !(l & (1 << 30)))) {
        fb_set_borderless(h, !s_borderless);
        return 0;
    }
    if (m == WM_SYSKEYDOWN && w == VK_RETURN)
        return 0;                       /* auto-repeat of Alt+Enter */
    switch (m) {
    case WM_CLOSE:
    case WM_DESTROY:
        /* Closing the window is quitting the game. Clearing s_fb_running
         * alone left the title running with no window -- a process only the
         * task manager could end. There is nothing to shut down in order:
         * the title saves as it goes, as a console that is switched off
         * would. Flush the host's buffered log first. TerminateProcess
         * rather than ExitProcess: the guest threads are still running, and
         * ExitProcess's cleanup can wait on a lock one of them held.
         * RECOMP_CLOSE_KEEPS_RUNNING=1 (upstream's) only closes the window. */
        InterlockedExchange(&s_fb_running, 0);
        if (getenv("RECOMP_CLOSE_KEEPS_RUNNING"))
            return 0;
        /* Gone from the screen at once: what follows (the log drain, then
         * Windows releasing the D3D device and the guest's address space)
         * can take seconds, and a window that lingers that long reads as a
         * game that did not quit. */
        ShowWindow(h, SW_HIDE);
        /* A project with an asynchronous log hands us a bounded, stdio-free
         * flush (g_xbox_quit_logs): a full flush takes the stdio lock and
         * waits for the log to go quiet, which while other threads print
         * held Amped 2's quit for 8 s. Without one, the old way, plus the
         * exit hook (the kernel's missing-file summary), which prints. */
        if (g_xbox_quit_logs) {
            g_xbox_quit_logs();
        } else {
            extern void (*g_xbox_flush_logs)(void);
            fprintf(stderr, "  [WINDOW] closed -- exiting\n");
            recomp_exit_hook_run("window");
            if (g_xbox_flush_logs)
                g_xbox_flush_logs();
            fflush(stdout);
            fflush(stderr);
        }
        TerminateProcess(GetCurrentProcess(), 0);
        return 0;

    case WM_APP + 0x11:
        /* xbox_SetFullscreen, from another thread. */
        fb_set_borderless(h, (int)w);
        return 0;

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        /* F3: the debug overlay (nv2a_d3d11.c), not passed to the title. */
        if (w == VK_F3 && !(l & (1 << 30))) {
            extern volatile long g_xbox_hud_on;
            InterlockedExchange(&g_xbox_hud_on, !g_xbox_hud_on);
            return 0;
        }
        xbox_InputKeySet((int)w, 1);
        /* A Shift, Ctrl or Alt is also its side's key, for bindings to
         * "lshift" and the like; wParam stays the either-side one. */
        {
            int side = xbox_InputWin32SideVk((int)w, (uint32_t)l);
            if (side)
                xbox_InputKeySet(side, 1);
        }
        /* RECOMP_KEY_TRACE: each key as it arrives, edge-triggered.
         *
         * The obvious diagnostic -- sampling which keys are held, once a
         * second, from the input path -- cannot tell a key that was never
         * pressed from one that was tapped: a 100 ms press is caught about
         * one time in ten. That ambiguity is expensive when the only way
         * to test is to ask someone to press a key and describe what
         * happened. This answers "did it arrive" on its own. */
        if (getenv("RECOMP_KEY_TRACE")) {
            static unsigned n;
            if (n++ < 40) {
                fprintf(stderr, "  [KEY] down vk=0x%02X\n", (unsigned)w);
                fflush(stderr);
            }
        }
        /* System keys still go to Windows, or Alt+F4 stops closing us. */
        return m == WM_SYSKEYDOWN ? DefWindowProcA(h, m, w, l) : 0;

    case WM_KEYUP:
    case WM_SYSKEYUP:
        xbox_InputKeySet((int)w, 0);
        {
            int side = xbox_InputWin32SideVk((int)w, (uint32_t)l);
            if (side)
                xbox_InputKeySet(side, 0);
            /* With both Shifts down Windows sends one key-up, for the
             * last released, so the other side would stay held. Autorepeat
             * sets a Shift still down again. */
            if (w == VK_SHIFT) {
                xbox_InputKeySet(VK_LSHIFT, 0);
                xbox_InputKeySet(VK_RSHIFT, 0);
            }
        }
        return m == WM_SYSKEYUP ? DefWindowProcA(h, m, w, l) : 0;

    case WM_MOUSEMOVE:
        fb_cursor_moved();
        return 0;

    case WM_SETCURSOR:
        if (LOWORD(l) == HTCLIENT && s_cursor_hidden) {
            SetCursor(NULL);
            return TRUE;
        }
        return DefWindowProcA(h, m, w, l);

    /* Mouse buttons go in the same table, for bindings that name them.
     * Capturing the mouse while one is down delivers its release even
     * outside the window. */
    case WM_LBUTTONDOWN: case WM_LBUTTONUP:
    case WM_RBUTTONDOWN: case WM_RBUTTONUP:
    case WM_MBUTTONDOWN: case WM_MBUTTONUP:
    case WM_XBUTTONDOWN: case WM_XBUTTONUP:
        {
            int down = 0;
            int vk = xbox_InputWin32MouseVk(m, (uint32_t)w, &down);
            if (vk)
                xbox_InputKeySet(vk, down);
            if (down)
                SetCapture(h);
            else if (!(w & (MK_LBUTTON | MK_RBUTTON | MK_MBUTTON | MK_XBUTTON1 | MK_XBUTTON2)))
                ReleaseCapture();
        }
        return m == WM_XBUTTONDOWN || m == WM_XBUTTONUP ? TRUE : 0;

    /* Alt-tabbing away with a key held would leave it held for ever. */
    case WM_KILLFOCUS:
        xbox_InputKeysClear();
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

/* The display formats an Xbox front buffer is actually set to. The pitch says
 * how wide a row is in bytes, so pitch/width gives the pixel size; the exact
 * component layout only matters for 16-bit, where 5:6:5 and 1:5:5:5 differ. */
static void fb_convert(const uint8_t *src, uint32_t bpp)
{
    uint32_t x, y;

    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * s_fb_pitch
                           * (bpp == 8 ? s_fb_aa_sy : 1);
        uint32_t *dst = s_rgb + (size_t)y * s_fb_width;

        if (bpp == 4) {
            memcpy(dst, row, (size_t)s_fb_width * 4);
        } else if (bpp == 8) {
            /* A 2x-wide anti-aliased 32-bit surface (pitch = 2 * width * 4):
             * take every other pixel. X-Men Legends renders its 640-wide
             * frame into a 1280-wide surface, which read as "8 bytes per
             * pixel" and showed black. */
            const uint32_t *p = (const uint32_t *)row;
            for (x = 0; x < s_fb_width; x++)
                dst[x] = p[x * 2];
        } else if (bpp == 2) {
            const uint16_t *p = (const uint16_t *)row;
            for (x = 0; x < s_fb_width; x++) {
                uint16_t v = p[x];
                uint32_t r = (uint32_t)((v >> 11) & 0x1F) * 255u / 31u;
                uint32_t g = (uint32_t)((v >>  5) & 0x3F) * 255u / 63u;
                uint32_t b = (uint32_t)( v        & 0x1F) * 255u / 31u;
                dst[x] = (r << 16) | (g << 8) | b;
            }
        } else {
            memset(dst, 0, (size_t)s_fb_width * 4);
        }
    }
}

/* Write what the window is currently showing to a 24-bit BMP.
 *
 * A black window is ambiguous: it means either that the read path is wrong or
 * that the title really did render black. Dumping the same converted pixels
 * the window draws settles which, and does it without a screenshot. */
int xbox_FramebufferDumpBmp(const char *path)
{
    FILE *f;
    uint32_t row = ((s_fb_width * 3u) + 3u) & ~3u;
    uint32_t img = row * s_fb_height, total = 54u + img, y, x;
    uint8_t hdr[54], *line;

    if (!s_rgb || !s_fb_va)
        return -1;
    f = fopen(path, "wb");
    if (!f)
        return -1;
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &total, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &s_fb_width, 4);
    memcpy(hdr + 22, &s_fb_height, 4);
    hdr[26] = 1; hdr[28] = 24;
    memcpy(hdr + 34, &img, 4);
    fwrite(hdr, 1, sizeof(hdr), f);

    line = (uint8_t *)calloc(1, row);
    for (y = 0; y < s_fb_height; y++) {
        const uint32_t *src = s_rgb + (size_t)(s_fb_height - 1 - y) * s_fb_width;
        for (x = 0; x < s_fb_width; x++) {
            line[x * 3 + 0] = (uint8_t)(src[x] & 0xFF);
            line[x * 3 + 1] = (uint8_t)((src[x] >> 8) & 0xFF);
            line[x * 3 + 2] = (uint8_t)((src[x] >> 16) & 0xFF);
        }
        fwrite(line, 1, row, f);
    }
    free(line);
    fclose(f);
    fprintf(stderr, "  [FBWIN] wrote %s (%ux%u from 0x%08X)\n",
            path, s_fb_width, s_fb_height, s_fb_va);
    return 0;
}

/* The window, for a presenter that draws into it directly (the D3D11
 * backend's swap chain). Once one claims it, GDI stops painting: the two
 * would fight over the same client area. */
static volatile HWND s_fb_hwnd;
static volatile LONG s_fb_gdi_off;

HWND xbox_FramebufferWindowHwnd(void)
{
    return s_fb_hwnd;
}

/* Quit as the window's close button does (WM_CLOSE -> fb_wndproc), for a
 * project that offers it from inside the title -- Amped 2's menu mod has a
 * QUIT item. Callable from any thread; no window, no-op. */
void xbox_RequestQuit(void)
{
    HWND h = s_fb_hwnd;
    if (h)
        PostMessageA(h, WM_CLOSE, 0, 0);
}

/* Borderless fullscreen on or off, as F11 does, for a project that offers it
 * in the title's own options (Amped 2's Game Options). Callable from any
 * thread: the change is made on the window's thread. */
void xbox_SetFullscreen(int on)
{
    HWND h = s_fb_hwnd;
    if (h)
        PostMessageA(h, WM_APP + 0x11, (WPARAM)(on != 0), 0);
}

int xbox_IsFullscreen(void)
{
    return s_borderless;
}

void xbox_FramebufferWindowClaim(void)
{
    InterlockedExchange(&s_fb_gdi_off, 1);
}

/* GDI: the frame letterboxed into the client area, like the D3D11
 * presenter; the borders are cleared when the size changes. */
static void gdi_blit(HWND hwnd, HDC hdc, const BITMAPINFO *bi)
{
    static LONG lw, lh;
    RECT c;
    LONG cw, ch, w, h;
    GetClientRect(hwnd, &c);
    cw = c.right - c.left; ch = c.bottom - c.top;
    if (cw <= 0 || ch <= 0)
        return;
    if (cw != lw || ch != lh) {
        FillRect(hdc, &c, (HBRUSH)GetStockObject(BLACK_BRUSH));
        lw = cw; lh = ch;
    }
    if ((long long)cw * s_fb_height <= (long long)ch * s_fb_width) {
        w = cw; h = (LONG)((long long)cw * s_fb_height / s_fb_width);
    } else {
        h = ch; w = (LONG)((long long)ch * s_fb_width / s_fb_height);
    }
    SetStretchBltMode(hdc, HALFTONE);
    SetBrushOrgEx(hdc, 0, 0, NULL);     /* HALFTONE wants it after the mode */
    StretchDIBits(hdc, (cw - w) / 2, (ch - h) / 2, w, h,
                  0, 0, (int)s_fb_width, (int)s_fb_height,
                  s_rgb, bi, DIB_RGB_COLORS, SRCCOPY);
}

/* The converted frame in s_rgb, on screen: D3D11 when it is up (which waits
 * for vsync, so returns 1 when the loop is already paced), else GDI. */
static int fb_show(HWND hwnd, HDC hdc, const BITMAPINFO *bi)
{
    if (gpu_present_active())
        return gpu_present_frame(s_rgb);
    gdi_blit(hwnd, hdc, bi);
    return 0;
}

static DWORD WINAPI fb_thread(LPVOID unused)
{
    HWND hwnd;
    HDC hdc;
    BITMAPINFO bi;
    RECT r;
    int fb_pinned;

    (void)unused;
    {
        extern void xbox_PinHostThread(void);
        xbox_PinHostThread();           /* not the guest's CPU */
    }

    {
        WNDCLASSA wc;
        memset(&wc, 0, sizeof(wc));
        wc.lpfnWndProc   = fb_wndproc;
        wc.hInstance     = GetModuleHandleA(NULL);
        wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
        wc.hIcon         = LoadIconA(wc.hInstance, MAKEINTRESOURCEA(1));  /* the game's, if it has one */
        wc.lpszClassName = "XboxRecompFramebuffer";
        RegisterClassA(&wc);
    }
    /* RECOMP_WINDOW_SIZE=<w>x<h>: the client area (the guest's own size by
     * default); RECOMP_WINDOW_TITLE: the caption.
     *
     * RECOMP_WINDOW_SCALE (1-8, upstream's, as in the SDL window), when
     * RECOMP_WINDOW_SIZE is not set: the client area is the frame times this,
     * made smaller until the window fits the desktop work area, and centred
     * on it. Neither set: the frame's own size where Windows puts it, as this
     * window has always opened. Any size is letterboxed to the frame's shape
     * (gdi_blit, or the D3D11 presenter). */
    {
        const char *sz = getenv("RECOMP_WINDOW_SIZE");
        const char *sc = getenv("RECOMP_WINDOW_SCALE");
        const char *title = getenv("RECOMP_WINDOW_TITLE");
        unsigned cw = s_fb_width, ch = s_fb_height;
        int sized = 0, x = CW_USEDEFAULT, y = CW_USEDEFAULT;
        if (sz) {
            unsigned a = 0, b = 0;
            if (sscanf(sz, "%ux%u", &a, &b) == 2 && a >= 320 && b >= 240
                    && a <= 16384 && b <= 16384) {
                cw = a;
                ch = b;
                sized = 1;
            }
        }
        if (!sized && sc) {
            int scale = atoi(sc);
            RECT work;
            if (scale < 1 || scale > 8)
                scale = 2;
            if (!SystemParametersInfoA(SPI_GETWORKAREA, 0, &work, 0)) {
                work.left = work.top = 0; work.right = 1920; work.bottom = 1080;
            }
            for (;; scale--) {
                r.left = 0; r.top = 0;
                r.right = (LONG)s_fb_width * scale; r.bottom = (LONG)s_fb_height * scale;
                AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
                if (scale <= 1 || (r.right - r.left <= work.right - work.left
                                   && r.bottom - r.top <= work.bottom - work.top))
                    break;
            }
            x = work.left + ((work.right - work.left) - (r.right - r.left)) / 2;
            y = work.top + ((work.bottom - work.top) - (r.bottom - r.top)) / 2;
        } else {
            r.left = 0; r.top = 0; r.right = (LONG)cw; r.bottom = (LONG)ch;
            AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        }
        s_fixed_title = title && *title;
        hwnd = CreateWindowExA(0, "XboxRecompFramebuffer",
                               s_fixed_title ? title : "Xbox Recomp - Framebuffer",
                               WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                               x, y,
                               r.right - r.left, r.bottom - r.top,
                               NULL, NULL, GetModuleHandleA(NULL), NULL);
    }
    if (!hwnd) {
        InterlockedExchange(&s_fb_running, 0);
        return 0;
    }
    {
        const char *mode = getenv("RECOMP_WINDOW_MODE");
        if (mode && (!_stricmp(mode, "borderless") || !_stricmp(mode, "fullscreen")))
            fb_set_borderless(hwnd, 1);
    }
    hdc = GetDC(hwnd);
    s_fb_hwnd = hwnd;
    fb_cursor_init();
    {
        /* RECOMP_FULLSCREEN=1 (upstream's switch) starts borderless too, as
         * RECOMP_WINDOW_MODE=borderless does. */
        const char *fs = getenv("RECOMP_FULLSCREEN");
        if (fs && *fs && *fs != '0')
            fb_set_borderless(hwnd, 1);
    }

    memset(&bi, 0, sizeof(bi));
    bi.bmiHeader.biSize        = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth       = (LONG)s_fb_width;
    bi.bmiHeader.biHeight      = -(LONG)s_fb_height;   /* top-down */
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    s_rgb = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);
    /* A window pinned to one address shows that memory and nothing else. */
    fb_pinned = getenv("RECOMP_FB_VA") != NULL;

    fprintf(stderr, "  [FBWIN] framebuffer window open (%ux%u)\n",
            s_fb_width, s_fb_height);
    /* Two swap chains cannot share a window: the D3D11 GPU backend
     * (RECOMP_GPU=d3d11) presents on it itself unless RECOMP_GPU_PRESENT=gdi
     * (xbox_FramebufferWindowClaim), so then RECOMP_D3D_PRESENT stands
     * aside. */
    if (gpu_present_enabled()) {
        const char *g = getenv("RECOMP_GPU"), *gp = getenv("RECOMP_GPU_PRESENT");
        if (g && !_stricmp(g, "d3d11") && !(gp && !_stricmp(gp, "gdi")))
            fprintf(stderr, "  [FBWIN] RECOMP_D3D_PRESENT ignored: the D3D11 "
                            "backend presents on this window\n");
        else
            gpu_present_init(hwnd, s_fb_width, s_fb_height);
    }

    while (InterlockedCompareExchange(&s_fb_running, 1, 1)) {
        MSG msg;
        int paced = 0;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        fb_cursor_tick(hwnd);
        if (s_fb_gdi_off && s_present_idx >= 0 && s_rgb) {
            /* A swap chain owns the client area; keep s_rgb current for the
             * dumps, paint nothing. */
            LONG idx = s_present_idx;
            if (s_present[idx])
                memcpy(s_rgb, s_present[idx], (size_t)s_fb_width * s_fb_height * 4);
        } else if (s_present_idx >= 0 && s_rgb) {
            /* A finished frame, published by the flip. Copied into s_rgb so
             * the dump path and GDI see one consistent image even if the
             * next flip lands mid-blit. */
            LONG idx = s_present_idx;
            if (s_present[idx])
                memcpy(s_rgb, s_present[idx],
                       (size_t)s_fb_width * s_fb_height * 4);
            if (!fb_pinned)
                fb_overlay();
            paced = fb_show(hwnd, hdc, &bi);
        } else if (s_fb_va && s_fb_pitch && s_rgb) {
            /* No flip yet, or pinned with RECOMP_FB_VA: read guest memory as
             * before, which is also what a title that never flips needs. */
            const uint8_t *src =
                (const uint8_t *)((uintptr_t)s_fb_va + xbox_GetMemoryOffset());
            fb_convert(src, s_fb_pitch / s_fb_width);
            if (!fb_pinned)
                fb_overlay();
            paced = fb_show(hwnd, hdc, &bi);
        }
        {
            /* One dump a few seconds in, so the title has had time to render
             * something rather than catching the first blank frame.
             *
             * RECOMP_FB_WINDOW_DUMP_EVERY=<frames> dumps repeatedly instead.
             * This window follows the address AvSetDisplayMode gave, which is
             * what the CRTC scans and therefore what a person sees; the
             * pushbuffer executor's own dump follows its draw surface. With
             * double buffering those are different buffers, and measuring
             * progress from the executor's dump reports a blank screen while
             * the window is showing the title's logo. Ask the window. */
            const char *dump = getenv("RECOMP_FB_DUMP");
            const char *every = getenv("RECOMP_FB_WINDOW_DUMP_EVERY");
            static int frames;
            int period = every ? atoi(every) : 0;
            frames++;
            if (dump && period > 0) {
                if (frames % period == 0)
                    xbox_FramebufferDumpBmp(dump);
            } else if (dump && frames == 600) {
                xbox_FramebufferDumpBmp(dump);
            }
        }
        {
            static DWORD t0;
            static LONG f0;
            DWORD now = GetTickCount();
            if (now - t0 >= 1000 && !s_fixed_title) {
                LONG f = s_flips;
                wchar_t tb[128];
                _snwprintf(tb, 127, L"%ls | FPS: %.1f | draws: %ld", s_title,
                           t0 ? (f - f0) * 1000.0 / (now - t0) : 0.0,
                           (long)s_frame_draws);
                tb[127] = 0;
                SetWindowTextW(hwnd, tb);
                t0 = now; f0 = f;
            }
        }
        if (!paced)
            Sleep(16);                  /* D3D11's Present waited for vsync */
    }

    gpu_present_shutdown();
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    free(s_rgb);
    s_rgb = NULL;
    free(s_pvideo_source);
    s_pvideo_source = NULL;
    return 0;
}

void xbox_FramebufferWindowStart(void)
{
    HANDLE th;

    if (!getenv("RECOMP_FB_WINDOW"))
        return;
    if (InterlockedCompareExchange(&s_fb_running, 1, 0) != 0)
        return;
    th = CreateThread(NULL, 0, fb_thread, NULL, 0, NULL);
    if (th)
        CloseHandle(th);
    else
        InterlockedExchange(&s_fb_running, 0);
}

#endif /* _WIN32: POSIX hosts get the SDL2 window in fb_present_sdl.c */
