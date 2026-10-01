/*
 * usb_gamepad.c -- an Xbox controller, as the console's USB stack expects it.
 *
 * Descriptors and the input report, from the USB 2.0 specification for the
 * standard requests and from the device's own published interface class for
 * the rest. The gamepad is not a HID device: it reports interface class 0x58
 * subclass 0x42, which is Microsoft's own, and its report has a fixed layout
 * rather than one described by a HID report descriptor. That is why there is
 * no report descriptor here and why nothing asks for one.
 *
 * Input comes from the host through the existing xbox_input layer, so a real
 * pad plugged into the PC drives this one.
 */
#include "usb_gamepad.h"

#include <stdio.h>
#include <stdlib.h>

#include <string.h>

/* ---- descriptors ------------------------------------------------------- */

static const uint8_t s_device_desc[18] = {
    18,             /* bLength                                    */
    0x01,           /* bDescriptorType: DEVICE                    */
    0x10, 0x01,     /* bcdUSB 1.10                                */
    0x00,           /* bDeviceClass: per interface                */
    0x00,           /* bDeviceSubClass                            */
    0x00,           /* bDeviceProtocol                            */
    0x08,           /* bMaxPacketSize0: 8                         */
    0x5E, 0x04,     /* idVendor  0x045E Microsoft                 */
    0x89, 0x02,     /* idProduct 0x0289 Controller S              */
    0x21, 0x01,     /* bcdDevice                                  */
    0x00,           /* iManufacturer: none                        */
    0x00,           /* iProduct: none                             */
    0x00,           /* iSerialNumber: none                        */
    0x01            /* bNumConfigurations                         */
};

/* Configuration, interface and both endpoints, in the one block a
 * GET_DESCRIPTOR(CONFIGURATION) returns. wTotalLength covers all of it. */
static const uint8_t s_config_desc[32] = {
    /* configuration */
    9, 0x02, 32, 0x00, 0x01, 0x01, 0x00, 0x80, 50,
    /* interface: class 0x58 subclass 0x42, the Xbox gamepad's own */
    9, 0x04, 0x00, 0x00, 0x02, 0x58, 0x42, 0x00, 0x00,
    /* endpoint 0x81 IN, interrupt, 32 bytes, 4 ms */
    7, 0x05, 0x81, 0x03, 0x20, 0x00, 0x04,
    /* endpoint 0x02 OUT, interrupt, 32 bytes, 4 ms -- rumble */
    7, 0x05, 0x02, 0x03, 0x20, 0x00, 0x04
};

static uint8_t s_address;
static uint8_t s_configuration;

uint8_t usb_gamepad_address(void) { return s_address; }

/* ---- control transfers ------------------------------------------------- */

#define REQ_GET_STATUS         0x00
#define REQ_CLEAR_FEATURE      0x01
#define REQ_SET_FEATURE        0x03
#define REQ_SET_ADDRESS        0x05
#define REQ_GET_DESCRIPTOR     0x06
#define REQ_GET_CONFIGURATION  0x08
#define REQ_SET_CONFIGURATION  0x09
#define REQ_SET_INTERFACE      0x0B

#define DESC_DEVICE            0x01
#define DESC_CONFIGURATION     0x02
#define DESC_STRING            0x03

static int copy_out(uint8_t *out, int max, const uint8_t *src, int len,
                    uint16_t wLength)
{
    /* A device sends the smaller of what was asked for and what it has. */
    if (len > (int)wLength) len = (int)wLength;
    if (len > max)          len = max;
    if (len > 0)            memcpy(out, src, (size_t)len);
    return len;
}

int usb_gamepad_control(const UsbSetup *setup, uint8_t *out, int max)
{
    int is_in = (setup->bmRequestType & 0x80) != 0;
    int type  = (setup->bmRequestType >> 5) & 3;   /* 0 standard, 1 class */

    if (type == 0) {
        switch (setup->bRequest) {
        case REQ_GET_DESCRIPTOR:
            switch (setup->wValue >> 8) {
            case DESC_DEVICE:
                return copy_out(out, max, s_device_desc,
                                (int)sizeof s_device_desc, setup->wLength);
            case DESC_CONFIGURATION:
                return copy_out(out, max, s_config_desc,
                                (int)sizeof s_config_desc, setup->wLength);
            case DESC_STRING:
                /* No string descriptors. Stalling is the correct answer and
                 * the one a host expects; returning an empty descriptor gets
                 * read as a malformed one. */
                return -1;
            default:
                return -1;
            }

        case REQ_SET_ADDRESS:
            s_address = (uint8_t)(setup->wValue & 0x7F);
            return 0;                    /* zero-length status stage */

        case REQ_SET_CONFIGURATION:
            s_configuration = (uint8_t)(setup->wValue & 0xFF);
            return 0;

        case REQ_GET_CONFIGURATION:
            if (!is_in || max < 1) return -1;
            out[0] = s_configuration;
            return 1;

        case REQ_GET_STATUS:
            /* Bus-powered, no remote wakeup. */
            if (!is_in || max < 2) return -1;
            out[0] = 0; out[1] = 0;
            return 2;

        case REQ_CLEAR_FEATURE:
        case REQ_SET_FEATURE:
        case REQ_SET_INTERFACE:
            return 0;

        default:
            return -1;
        }
    }

    /* Class requests on the interface -- GET_REPORT.
     *
     * XAPI reads the pad through the interrupt endpoint and also asks for
     * the same report over the control pipe. Stalling that is not a small
     * omission: a stalled control transfer reads to the driver as a broken
     * device, and because the stall also halts the endpoint, the pad stops
     * being polled for good.
     *
     * Measured on Shin Megami Tensei: Nine, the periodic list runs at about
     * a hundred descriptors a second and then collapses to nothing the
     * moment one `A1 01 value 0100 len 20` arrives.
     *
     * The answer is the report the interrupt endpoint would have sent.
     */
    if ((setup->bmRequestType & 0x60u) == 0x20u        /* class */
        && setup->bRequest == 0x01u                    /* GET_REPORT */
        && is_in) {
        uint8_t report[20];
        int n = usb_gamepad_report(report, (int)sizeof report);
        if (n <= 0)
            return -1;
        return copy_out(out, max, report, n, setup->wLength);
    }

    /* Vendor requests on the interface -- the XID protocol.
     *
     * This is how XAPI tells a controller from any other USB device. The
     * standard descriptors say "interface class 0x58", which gets the device
     * enumerated and no further: XAPI then asks for the XID descriptor to
     * learn what kind of controller it is and how big its reports are, and a
     * stall there means "not a controller", so the device is enumerated,
     * configured, and then ignored. Which is exactly what it looked like --
     * a clean enumeration and a title that still saw no gamepad.
     */
    if ((setup->bmRequestType & 0x60u) == 0x40u) {   /* vendor */
        static const uint8_t xid_desc[16] = {
            0x10,           /* bLength                                  */
            0x42,           /* bDescriptorType: XID                     */
            0x00, 0x01,     /* bcdXid 1.00                              */
            0x01,           /* bType: gamepad                           */
            0x02,           /* bSubType: gamepad S                      */
            20,             /* bMaxInputReportSize                      */
            6,              /* bMaxOutputReportSize                     */
            0xFF, 0xFF, 0xFF, 0xFF,   /* wAlternateProductIds[0..1]     */
            0xFF, 0xFF, 0xFF, 0xFF    /* wAlternateProductIds[2..3]     */
        };
        /* Capabilities are the report with every supported field set to all
         * ones: the same layout, read as a mask. A gamepad S supports every
         * field of both, so both are filled in apart from the id and length
         * bytes, which are values rather than flags. */
        static const uint8_t caps_in[20] = {
            0x00, 20,
            0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
        };
        static const uint8_t caps_out[6] = {
            0x00, 6, 0xFF, 0xFF, 0xFF, 0xFF
        };

        if (!is_in)
            return 0;                     /* accept, nothing to send back */
        if (setup->bRequest == REQ_GET_DESCRIPTOR
                && (setup->wValue >> 8) == 0x42)
            return copy_out(out, max, xid_desc, (int)sizeof xid_desc,
                            setup->wLength);
        if (setup->bRequest == 0x01) {    /* GET_CAPABILITIES */
            if ((setup->wValue >> 8) == 0x01)
                return copy_out(out, max, caps_in, (int)sizeof caps_in,
                                setup->wLength);
            if ((setup->wValue >> 8) == 0x02)
                return copy_out(out, max, caps_out, (int)sizeof caps_out,
                                setup->wLength);
        }
        return -1;
    }

    /* Class requests: not ours to guess at. */
    return -1;
}

/* ---- the input report -------------------------------------------------- */

/* The host's own pad, through the layer that already maps one to XInput.
 * A real controller plugged into the PC drives this emulated one. */
#include "../input/xinput_xbox.h"

/*
 * The Xbox report is 20 bytes and fixed:
 *
 *   0      report id, always 0
 *   1      length, always 20
 *   2      digital buttons: dpad, start, back, thumb clicks
 *   3      reserved
 *   4..11  analog buttons A B X Y Black White, then the two triggers
 *   12..19 four signed 16-bit stick axes, little endian
 */
/* A synthetic press, for bringing a title up without a pad on the desk.
 *
 * RECOMP_PAD_PRESS=0x10 holds Start for a quarter of a second every two
 * seconds. A title sitting on a "press Start" screen needs an edge, not a
 * level, so this pulses rather than latching -- and it repeats because the
 * moment the title starts reading input is not knowable from here.
 *
 * Bit values are the Xbox digital button mask: 0x01/02/04/08 dpad
 * up/down/left/right, 0x10 Start, 0x20 Back, 0x40/0x80 thumb clicks.
 * Bits 8..15 press the analog buttons fully, in report order: 0x100 A,
 * 0x200 B, 0x400 X, 0x800 Y, 0x1000 Black, 0x2000 White, 0x4000/0x8000 the
 * left/right triggers. A menu that confirms with A rather than Start could
 * not be walked through before.
 *
 * This is a bring-up probe and nothing else. It is off unless the variable
 * is set, and a real pad on the host is always the better input.
 */
/* The pad's output report: rumble. Six bytes -- report id 0, length 6, then
 * the left (low-frequency) and right (high-frequency) motor speeds as 16-bit
 * little-endian values -- passed to the host pad. RECOMP_RUMBLE=0 turns it
 * off. Sent only when it changes: the title repeats it every frame or so,
 * and a host call per report is wasted work. */
void usb_gamepad_output(const uint8_t *r, int len)
{
    static int on = -1;
    static XBOX_VIBRATION last = { 0xFFFF, 0xFFFF };
    XBOX_VIBRATION v;

    if (on < 0) {
        const char *e = getenv("RECOMP_RUMBLE");
        on = !(e && *e == '0');
    }
    if (!on || len < 6 || r[1] < 6)
        return;
    v.wLeftMotorSpeed  = (WORD)(r[2] | (r[3] << 8));
    v.wRightMotorSpeed = (WORD)(r[4] | (r[5] << 8));
    if (v.wLeftMotorSpeed == last.wLeftMotorSpeed
            && v.wRightMotorSpeed == last.wRightMotorSpeed)
        return;
    last = v;
    xbox_InputSetState(0, &v);
    if (getenv("RECOMP_INPUT_DIAG"))
        fprintf(stderr, "  [INPUT] rumble %u / %u\n",
                (unsigned)v.wLeftMotorSpeed, (unsigned)v.wRightMotorSpeed);
}

/* Reports handed to the title, for RECOMP_INPUT_DIAG (ohci.c prints it). */
unsigned long s_reports;

static unsigned synthetic_buttons(void)
{
    static int      configured = -1;
    static unsigned mask, period_ms, hold_ms;
    static unsigned long t0;
    unsigned long now, phase;

    if (configured < 0) {
        const char *spec = getenv("RECOMP_PAD_PRESS");
        configured = 0;
        if (spec && *spec) {
            char *end;
            mask = (unsigned)strtoul(spec, &end, 0) & 0xFFFFu;
            period_ms = (*end == ',') ? (unsigned)strtoul(end + 1, &end, 0)
                                      : 2000u;
            hold_ms   = (*end == ',') ? (unsigned)strtoul(end + 1, &end, 0)
                                      : 250u;
            if (!period_ms) period_ms = 2000u;
            if (!hold_ms || hold_ms >= period_ms) hold_ms = period_ms / 4u;
            if (mask) {
                configured = 1;
                t0 = (unsigned long)GetTickCount();
                fprintf(stderr, "  PAD: synthesising button mask 0x%04X for "
                        "%u ms every %u ms\n", mask, hold_ms, period_ms);
                fflush(stderr);
            }
        }
    }
    /* RECOMP_PAD_SCRIPT="ms:mask[:hold],...": at ms after the first poll,
     * hold mask for hold ms (default 200). A mask of 0 is a quiet spell:
     * RECOMP_PAD_PRESS's pulses are held off for it. Outside the script's
     * entries the pulses run as usual, so a run can be walked to a menu by
     * the pulses, steered there by the script, and carried on after. Bits
     * 16-19 push the left stick fully down, up, left and right. */
    {
        static int nscript = -1;
        static struct { unsigned long at, hold; unsigned mask; } script[64];
        static unsigned long s0;
        int i;
        if (nscript < 0) {
            const char *p = getenv("RECOMP_PAD_SCRIPT");
            nscript = 0;
            while (p && *p && nscript < 64) {
                char *end;
                script[nscript].at = strtoul(p, &end, 0);
                if (*end != ':')
                    break;
                script[nscript].mask = (unsigned)strtoul(end + 1, &end, 0) & 0xFFFFFu;
                script[nscript].hold = (*end == ':') ? strtoul(end + 1, &end, 0) : 200u;
                nscript++;
                p = (*end == ',') ? end + 1 : NULL;
            }
            s0 = (unsigned long)GetTickCount();
            if (nscript)
                fprintf(stderr, "  PAD: script of %d steps\n", nscript);
        }
        int quiet = 0;
        now = (unsigned long)GetTickCount();
        for (i = 0; i < nscript; i++) {
            if (now - s0 >= script[i].at && now - s0 < script[i].at + script[i].hold) {
                if (script[i].mask)
                    return script[i].mask;     /* a press wins over a quiet spell */
                quiet = 1;
            }
        }
        if (quiet)
            return 0;
    }
    if (!configured)
        return 0;
    now = (unsigned long)GetTickCount();
    phase = (now - t0) % period_ms;
    return (phase < hold_ms) ? mask : 0u;
}

int usb_gamepad_report(uint8_t *out, int max)
{
    XBOX_INPUT_STATE state;
    const XBOX_GAMEPAD *g;
    unsigned synth_all = synthetic_buttons();
    uint8_t synth = (uint8_t)(synth_all & 0xFFu);
    int i;

    if (max < 20)
        return 0;
    s_reports++;
    memset(out, 0, 20);
    out[0] = 0;
    out[1] = 20;

    /* A disconnected host pad is not an error here: the device is present on
     * the bus either way, it just reports nothing pressed. */
    /* RECOMP_INPUT_DIAG: the whole chain on one line, once a second.
     *
     * "Nothing happens when I press a key" has several candidate causes and
     * guessing between them costs a rebuild each: the variable not read,
     * XInput claiming a pad so the keyboard fallback never runs, the window
     * not receiving the key, or the report going out without it. Printing
     * all four together answers it in one run.
     *
     * It samples the held state, so pair it with the window's own
     * RECOMP_KEY_TRACE: that answers "did the key arrive", this answers
     * "is the chain wired". */
    {
        static int diag = -1;
        if (diag < 0)
            diag = getenv("RECOMP_INPUT_DIAG") != NULL;
        if (diag) {
            extern int xbox_FramebufferKeyDown(int vk);
            static unsigned long last;
            unsigned long now = (unsigned long)GetTickCount();
            if (now - last > 1000) {
                XBOX_INPUT_STATE probe;
                DWORD rc = xbox_InputGetState(0, &probe);
                last = now;
                static unsigned long last_reports;
                fprintf(stderr, "  [INPUT] kbd_env=%d window_has_RETURN=%d "
                        "InputGetState=%lu buttons=0x%04X A=%u LX=%d; "
                        "%lu reports to the title in the last second\n",
                        getenv("RECOMP_KEYBOARD") ? 1 : 0,
                        xbox_FramebufferKeyDown(0x0D),
                        (unsigned long)rc,
                        rc == 0 ? probe.Gamepad.wButtons : 0,
                        rc == 0 ? probe.Gamepad.bAnalogButtons[0] : 0,
                        rc == 0 ? probe.Gamepad.sThumbLX : 0,
                        s_reports - last_reports);
                last_reports = s_reports;
                fflush(stderr);
            }
        }
    }

    if (xbox_InputGetState(0, &state) != 0) {
        out[2] = synth;
        for (i = 0; i < 8; i++)
            if (synth_all & (0x100u << i))
                out[4 + i] = 0xFF;
        state.Gamepad.sThumbLX = state.Gamepad.sThumbLY = 0;
        goto stick;
    }

    g = &state.Gamepad;
    out[2] = (uint8_t)((g->wButtons & 0xFF) | synth);
    out[3] = (uint8_t)((g->wButtons >> 8) & 0xFF);
    for (i = 0; i < 8; i++)
        out[4 + i] = (synth_all & (0x100u << i)) ? 0xFF : g->bAnalogButtons[i];
    out[16] = (uint8_t)(g->sThumbRX & 0xFF);
    out[17] = (uint8_t)((g->sThumbRX >> 8) & 0xFF);
    out[18] = (uint8_t)(g->sThumbRY & 0xFF);
    out[19] = (uint8_t)((g->sThumbRY >> 8) & 0xFF);
stick:
    {
        int lx = state.Gamepad.sThumbLX, ly = state.Gamepad.sThumbLY;
        if (synth_all & 0x10000u) ly = -32767;
        if (synth_all & 0x20000u) ly = 32767;
        if (synth_all & 0x40000u) lx = -32767;
        if (synth_all & 0x80000u) lx = 32767;
        out[12] = (uint8_t)(lx & 0xFF);
        out[13] = (uint8_t)((lx >> 8) & 0xFF);
        out[14] = (uint8_t)(ly & 0xFF);
        out[15] = (uint8_t)((ly >> 8) & 0xFF);
    }
    return 20;
}

