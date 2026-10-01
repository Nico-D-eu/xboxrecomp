/**
 * Xbox Input Compatibility Layer
 *
 * Translates the Xbox controller API to a host gamepad backend.
 * Handles the structural differences between the Xbox gamepad
 * (analog buttons as bytes, separate trigger channels) and the host
 * (digital face buttons, trigger axes).
 *
 *   _WIN32 -> Windows XInput
 *   POSIX  -> SDL2 GameController
 */

#include "xinput_xbox.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================== */
#if defined(_WIN32)
/* ====================  XInput backend  ================================== */
/* ======================================================================== */

#include <xinput.h>
#pragma comment(lib, "xinput.lib")

static BOOL  g_controller_connected[XBOX_MAX_CONTROLLERS] = { FALSE };
static DWORD g_last_packet[XBOX_MAX_CONTROLLERS] = { 0 };

/* ---- keyboard, when there is no pad --------------------------------------
 *
 * A title that waits on PRESS START is unreachable on a machine with no
 * controller plugged in, which is most machines someone brings this up on.
 * The whole point of a recompilation is to be able to look at the thing
 * running, and a build nobody can press a button in cannot be looked at.
 *
 * Off by default, because a keyboard silently acting as player 1 is
 * surprising when a real pad is what you meant to use. RECOMP_KEYBOARD=1
 * turns it on. It only ever answers for port 0, and is merged on top of a
 * pad connected there, so a real controller keeps working.
 *
 * The keys are the ones a Dreamcast or Saturn emulator would pick, which is
 * the closest thing to a convention here:
 *
 *   arrows        d-pad             Enter      START
 *   Z X A S       A B X Y           Backspace  BACK
 *   Q E           white black       1 3        triggers
 *   numpad 8/2/4/6 left thumb       I/K/J/L    right thumb
 *
 * Keys come from the framebuffer window, which only receives them while it
 * has the focus, so typing in another window does not drive the game.
 */
static BOOL keyboard_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECOMP_KEYBOARD");
        on = (v && *v && *v != '0') ? 1 : 0;
    }
    return on ? TRUE : FALSE;
}

/* The framebuffer window records what is held; see src/video/fb_present.c.
 *
 * GetAsyncKeyState was the first attempt and reads nothing here -- under
 * Wine it answers about a state a GDI-drawing guest never touches. Going
 * through the window is also the better gate: a window only receives keys
 * while it has the focus, so there is no separate focus check to get wrong
 * and no way for typing in another application to drive the game.
 *
 * Declared rather than included so src/input does not depend on the video
 * module's include path. */
extern int xbox_FramebufferKeyDown(int vk);

static BOOL key_down(int vk)
{
    return xbox_FramebufferKeyDown(vk) ? TRUE : FALSE;
}

/* The keyboard's bindings: up to three keys for each pad input.
 *
 * The defaults are the original layout (arrows = d-pad, Z X A S = A B X Y,
 * numeric pad = left stick, I J K L = right stick). RECOMP_KEYMAP_PRESET=wasd
 * puts the left stick on W A S D and moves the buttons out of its way;
 * RECOMP_KEYMAP="A=SPACE,START=RETURN|ESCAPE,LX-=A,..." then changes single
 * bindings (names below; a key is a letter or digit, a name such as SPACE,
 * F1 or NUMPAD4, or a code like 0x41). */
enum {
    KB_UP, KB_DOWN, KB_LEFT, KB_RIGHT, KB_START, KB_BACK, KB_LTHUMB, KB_RTHUMB,
    KB_A, KB_B, KB_X, KB_Y, KB_WHITE, KB_BLACK, KB_LT, KB_RT,
    KB_LX_NEG, KB_LX_POS, KB_LY_NEG, KB_LY_POS,
    KB_RX_NEG, KB_RX_POS, KB_RY_NEG, KB_RY_POS, KB_COUNT
};
static const char *const kb_names[KB_COUNT] = {
    "UP", "DOWN", "LEFT", "RIGHT", "START", "BACK", "LTHUMB", "RTHUMB",
    "A", "B", "X", "Y", "WHITE", "BLACK", "LT", "RT",
    "LX-", "LX+", "LY-", "LY+", "RX-", "RX+", "RY-", "RY+"
};
static unsigned char kb_map[KB_COUNT][3];

static int kb_key_code(const char *s, size_t n)
{
    static const struct { const char *name; int vk; } names[] = {
        { "UP", VK_UP }, { "DOWN", VK_DOWN }, { "LEFT", VK_LEFT }, { "RIGHT", VK_RIGHT },
        { "RETURN", VK_RETURN }, { "ENTER", VK_RETURN }, { "BACK", VK_BACK },
        { "BACKSPACE", VK_BACK }, { "SPACE", VK_SPACE }, { "TAB", VK_TAB },
        { "ESCAPE", VK_ESCAPE }, { "ESC", VK_ESCAPE }, { "SHIFT", VK_SHIFT },
        { "LSHIFT", VK_SHIFT }, { "CTRL", VK_CONTROL }, { "CONTROL", VK_CONTROL },
        { "ALT", VK_MENU }, { "INSERT", VK_INSERT }, { "DELETE", VK_DELETE },
        { "HOME", VK_HOME }, { "END", VK_END }, { "PAGEUP", VK_PRIOR },
        { "PAGEDOWN", VK_NEXT }, { "COMMA", VK_OEM_COMMA }, { "PERIOD", VK_OEM_PERIOD },
    };
    char buf[16];
    size_t i;
    if (!n || n >= sizeof buf)
        return 0;
    for (i = 0; i < n; i++)
        buf[i] = (char)((s[i] >= 'a' && s[i] <= 'z') ? s[i] - 32 : s[i]);
    buf[n] = 0;
    if (n == 1 && ((buf[0] >= 'A' && buf[0] <= 'Z') || (buf[0] >= '0' && buf[0] <= '9')))
        return buf[0];
    if (buf[0] == '0' && buf[1] == 'X')
        return (int)strtol(buf, NULL, 16) & 0xFF;
    if (buf[0] == 'F' && n <= 3 && atoi(buf + 1) >= 1 && atoi(buf + 1) <= 24)
        return VK_F1 + atoi(buf + 1) - 1;
    if (!strncmp(buf, "NUMPAD", 6) && n == 7 && buf[6] >= '0' && buf[6] <= '9')
        return VK_NUMPAD0 + (buf[6] - '0');
    for (i = 0; i < sizeof names / sizeof names[0]; i++)
        if (!strcmp(buf, names[i].name))
            return names[i].vk;
    return 0;
}

static void kb_bind(int target, const char *keys, size_t n)
{
    size_t start = 0, i;
    int k = 0;
    memset(kb_map[target], 0, sizeof kb_map[target]);
    for (i = 0; i <= n && k < 3; i++) {
        if (i == n || keys[i] == '|') {
            int vk = kb_key_code(keys + start, i - start);
            if (vk)
                kb_map[target][k++] = (unsigned char)vk;
            start = i + 1;
        }
    }
}

static void kb_init(void)
{
    static const char *const defaults =
        "UP=UP,DOWN=DOWN,LEFT=LEFT,RIGHT=RIGHT,START=RETURN,BACK=BACK,"
        "LTHUMB=SHIFT,RTHUMB=CTRL,A=Z,B=X,X=A,Y=S,WHITE=Q,BLACK=E,LT=1,RT=3,"
        "LX-=NUMPAD4,LX+=NUMPAD6,LY-=NUMPAD2,LY+=NUMPAD8,RX-=J,RX+=L,RY-=K,RY+=I";
    static const char *const wasd =
        "LX-=A,LX+=D,LY-=S,LY+=W,A=SPACE,B=J,X=K,Y=L,WHITE=U,BLACK=O,LT=Q,RT=E,"
        "START=RETURN|ESCAPE,BACK=BACK|TAB,LTHUMB=SHIFT,RTHUMB=CTRL,"
        "RX-=LEFT,RX+=RIGHT,RY-=DOWN,RY+=UP,UP=NUMPAD8,DOWN=NUMPAD2,"
        "LEFT=NUMPAD4,RIGHT=NUMPAD6";
    const char *lists[3];
    int l;
    lists[0] = defaults;
    {
        const char *p = getenv("RECOMP_KEYMAP_PRESET");
        lists[1] = (p && !_stricmp(p, "wasd")) ? wasd : NULL;
    }
    lists[2] = getenv("RECOMP_KEYMAP");
    for (l = 0; l < 3; l++) {
        const char *p = lists[l];
        while (p && *p) {
            const char *eq = strchr(p, '='), *end = strchr(p, ',');
            int t;
            if (!end)
                end = p + strlen(p);
            if (eq && eq < end) {
                for (t = 0; t < KB_COUNT; t++)
                    if (strlen(kb_names[t]) == (size_t)(eq - p)
                            && !_strnicmp(p, kb_names[t], (size_t)(eq - p)))
                        kb_bind(t, eq + 1, (size_t)(end - eq - 1));
            }
            p = *end ? end + 1 : end;
        }
    }
}

static BOOL kb_down(int target)
{
    int k;
    for (k = 0; k < 3 && kb_map[target][k]; k++)
        if (key_down(kb_map[target][k]))
            return TRUE;
    return FALSE;
}

static SHORT kb_axis(int negative, int positive)
{
    int v = 0;
    if (kb_down(negative)) v -= 32767;
    if (kb_down(positive)) v += 32767;
    return (SHORT)v;
}

static void keyboard_state(XBOX_INPUT_STATE *pState)
{
    static DWORD packet;
    static int inited;
    WORD b = 0;

    if (!inited) {
        kb_init();
        inited = 1;
    }
    memset(pState, 0, sizeof(*pState));

    if (kb_down(KB_UP))     b |= XBOX_GAMEPAD_DPAD_UP;
    if (kb_down(KB_DOWN))   b |= XBOX_GAMEPAD_DPAD_DOWN;
    if (kb_down(KB_LEFT))   b |= XBOX_GAMEPAD_DPAD_LEFT;
    if (kb_down(KB_RIGHT))  b |= XBOX_GAMEPAD_DPAD_RIGHT;
    if (kb_down(KB_START))  b |= XBOX_GAMEPAD_START;
    if (kb_down(KB_BACK))   b |= XBOX_GAMEPAD_BACK;
    if (kb_down(KB_LTHUMB)) b |= XBOX_GAMEPAD_LEFT_THUMB;
    if (kb_down(KB_RTHUMB)) b |= XBOX_GAMEPAD_RIGHT_THUMB;
    pState->Gamepad.wButtons = b;

    /* Analog on the console, so a key is 255 rather than a flag -- a title
     * that reads these as a pressure never sees a press if they are 1. */
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_A]        = kb_down(KB_A) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_B]        = kb_down(KB_B) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_X]        = kb_down(KB_X) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_Y]        = kb_down(KB_Y) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_WHITE]    = kb_down(KB_WHITE) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_BLACK]    = kb_down(KB_BLACK) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_LTRIGGER] = kb_down(KB_LT) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_RTRIGGER] = kb_down(KB_RT) ? 255 : 0;

    pState->Gamepad.sThumbLX = kb_axis(KB_LX_NEG, KB_LX_POS);
    pState->Gamepad.sThumbLY = kb_axis(KB_LY_NEG, KB_LY_POS);
    pState->Gamepad.sThumbRX = kb_axis(KB_RX_NEG, KB_RX_POS);
    pState->Gamepad.sThumbRY = kb_axis(KB_RY_NEG, KB_RY_POS);

    /* The title's input layer looks for button edges, so the packet number
     * has to move whenever the state does or a press is never noticed. */
    pState->dwPacketNumber = ++packet;
}

/* RECOMP_PAD_DEADZONE=<0..0.9>: a radial dead zone on the host pad's sticks,
 * as a fraction of full deflection, with the rest rescaled so the edge of the
 * zone reads as 0 and full deflection is still full. Unset or 0: the stick
 * as XInput reports it (the title applies its own). */
static void stick_deadzone(SHORT *x, SHORT *y)
{
    static double dz = -1.0;
    double fx, fy, m, k;
    if (dz < 0.0) {
        const char *e = getenv("RECOMP_PAD_DEADZONE");
        dz = e ? atof(e) : 0.0;
        if (dz < 0.0 || dz > 0.9)
            dz = 0.0;
    }
    if (dz <= 0.0)
        return;
    fx = *x / 32767.0;
    fy = *y / 32767.0;
    m = sqrt(fx * fx + fy * fy);
    if (m <= dz) {
        *x = *y = 0;
        return;
    }
    k = (m > 1.0 ? 1.0 : (m - dz) / (1.0 - dz)) / m;
    fx *= k;
    fy *= k;
    *x = (SHORT)(fx > 1.0 ? 32767 : fx < -1.0 ? -32767 : fx * 32767.0);
    *y = (SHORT)(fy > 1.0 ? 32767 : fy < -1.0 ? -32767 : fy * 32767.0);
}

void xbox_InputInit(void)
{
    for (DWORD i = 0; i < XBOX_MAX_CONTROLLERS; i++) {
        XINPUT_STATE state;
        DWORD result = XInputGetState(i, &state);
        g_controller_connected[i] = (result == ERROR_SUCCESS);
    }
}

DWORD xbox_InputGetState(DWORD dwPort, XBOX_INPUT_STATE *pState)
{
    XINPUT_STATE xi_state;
    DWORD result;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pState)
        return ERROR_DEVICE_NOT_CONNECTED;

    result = XInputGetState(dwPort, &xi_state);
    if (result != ERROR_SUCCESS) {
        g_controller_connected[dwPort] = FALSE;
        if (dwPort == 0 && keyboard_enabled()) {
            keyboard_state(pState);
            return ERROR_SUCCESS;
        }
        return result;
    }

    g_controller_connected[dwPort] = TRUE;
    g_last_packet[dwPort] = xi_state.dwPacketNumber;

    memset(pState, 0, sizeof(XBOX_INPUT_STATE));
    pState->dwPacketNumber = xi_state.dwPacketNumber;
    pState->Gamepad.wButtons = xi_state.Gamepad.wButtons & 0x00FF;

    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_A] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_A) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_B] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_B) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_X] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_X) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_Y] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_Y) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_BLACK] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_WHITE] =
        (xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_LTRIGGER] = xi_state.Gamepad.bLeftTrigger;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_RTRIGGER] = xi_state.Gamepad.bRightTrigger;

    pState->Gamepad.sThumbLX = xi_state.Gamepad.sThumbLX;
    pState->Gamepad.sThumbLY = xi_state.Gamepad.sThumbLY;
    pState->Gamepad.sThumbRX = xi_state.Gamepad.sThumbRX;
    pState->Gamepad.sThumbRY = xi_state.Gamepad.sThumbRY;
    stick_deadzone(&pState->Gamepad.sThumbLX, &pState->Gamepad.sThumbLY);
    stick_deadzone(&pState->Gamepad.sThumbRX, &pState->Gamepad.sThumbRY);

    /* Merge the keyboard on top rather than only standing in for a missing
     * pad.
     *
     * The first version put the keyboard behind XInput's failure, on the
     * assumption that with nothing plugged in the call would fail. It does
     * not: under Wine XInputGetState returns ERROR_SUCCESS and a gamepad
     * with every button at rest, so the fallback was unreachable and
     * pressing a key did nothing at all.
     *
     * Merging is also the better rule. A real pad keeps working -- its
     * buttons are already in pState and the keyboard only adds to them --
     * and there is no special case left to get wrong. */
    if (dwPort == 0 && keyboard_enabled()) {
        XBOX_INPUT_STATE kb;
        int i;
        keyboard_state(&kb);
        pState->Gamepad.wButtons |= kb.Gamepad.wButtons;
        for (i = 0; i < 8; i++)
            if (kb.Gamepad.bAnalogButtons[i] > pState->Gamepad.bAnalogButtons[i])
                pState->Gamepad.bAnalogButtons[i] = kb.Gamepad.bAnalogButtons[i];
        if (kb.Gamepad.sThumbLX) pState->Gamepad.sThumbLX = kb.Gamepad.sThumbLX;
        if (kb.Gamepad.sThumbLY) pState->Gamepad.sThumbLY = kb.Gamepad.sThumbLY;
        if (kb.Gamepad.sThumbRX) pState->Gamepad.sThumbRX = kb.Gamepad.sThumbRX;
        if (kb.Gamepad.sThumbRY) pState->Gamepad.sThumbRY = kb.Gamepad.sThumbRY;
        /* The input layer records edges, so an unchanged packet number is
         * read as the same state and the press never happens. */
        pState->dwPacketNumber = kb.dwPacketNumber;
    }

    return ERROR_SUCCESS;
}

DWORD xbox_InputSetState(DWORD dwPort, const XBOX_VIBRATION *pVibration)
{
    XINPUT_VIBRATION xi_vib;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pVibration)
        return ERROR_DEVICE_NOT_CONNECTED;

    xi_vib.wLeftMotorSpeed = pVibration->wLeftMotorSpeed;
    xi_vib.wRightMotorSpeed = pVibration->wRightMotorSpeed;
    return XInputSetState(dwPort, &xi_vib);
}

BOOL xbox_InputIsConnected(DWORD dwPort)
{
    if (dwPort >= XBOX_MAX_CONTROLLERS) return FALSE;
    return g_controller_connected[dwPort];
}

DWORD xbox_InputGetCapabilities(DWORD dwPort, DWORD dwFlags, XBOX_INPUT_CAPABILITIES *pCaps)
{
    XINPUT_CAPABILITIES xi_caps;
    DWORD result;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pCaps)
        return ERROR_DEVICE_NOT_CONNECTED;

    result = XInputGetCapabilities(dwPort, dwFlags, &xi_caps);
    if (result != ERROR_SUCCESS) return result;

    memset(pCaps, 0, sizeof(XBOX_INPUT_CAPABILITIES));
    pCaps->Type = xi_caps.Type;
    pCaps->SubType = xi_caps.SubType;
    pCaps->Flags = xi_caps.Flags;
    return ERROR_SUCCESS;
}

/* ======================================================================== */
#else /* !_WIN32 */
/* ====================  SDL2 GameController backend  ===================== */
/* ======================================================================== */

#include <SDL.h>

static SDL_GameController *g_pads[XBOX_MAX_CONTROLLERS];
static BOOL  g_controller_connected[XBOX_MAX_CONTROLLERS];
static DWORD g_packet[XBOX_MAX_CONTROLLERS];

/* Open up to XBOX_MAX_CONTROLLERS attached game controllers. */
static void open_controllers(void)
{
    int slot = 0;
    for (int i = 0; i < SDL_NumJoysticks() && slot < XBOX_MAX_CONTROLLERS; i++) {
        if (!SDL_IsGameController(i))
            continue;
        if (!g_pads[slot]) {
            g_pads[slot] = SDL_GameControllerOpen(i);
            g_controller_connected[slot] = (g_pads[slot] != NULL);
        }
        slot++;
    }
}

void xbox_InputInit(void)
{
    if (!SDL_WasInit(SDL_INIT_GAMECONTROLLER))
        SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER);
    open_controllers();
}

DWORD xbox_InputGetState(DWORD dwPort, XBOX_INPUT_STATE *pState)
{
    if (dwPort >= XBOX_MAX_CONTROLLERS || !pState)
        return ERROR_DEVICE_NOT_CONNECTED;

    SDL_GameController *c = g_pads[dwPort];
    if (!c || !SDL_GameControllerGetAttached(c)) {
        g_controller_connected[dwPort] = FALSE;
        return ERROR_DEVICE_NOT_CONNECTED;
    }

    SDL_GameControllerUpdate();
    g_controller_connected[dwPort] = TRUE;

    memset(pState, 0, sizeof(XBOX_INPUT_STATE));
    pState->dwPacketNumber = ++g_packet[dwPort];

    WORD btn = 0;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_DPAD_UP))    btn |= XBOX_GAMEPAD_DPAD_UP;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_DPAD_DOWN))  btn |= XBOX_GAMEPAD_DPAD_DOWN;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_DPAD_LEFT))  btn |= XBOX_GAMEPAD_DPAD_LEFT;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) btn |= XBOX_GAMEPAD_DPAD_RIGHT;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_START))      btn |= XBOX_GAMEPAD_START;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_BACK))       btn |= XBOX_GAMEPAD_BACK;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_LEFTSTICK))  btn |= XBOX_GAMEPAD_LEFT_THUMB;
    if (SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_RIGHTSTICK)) btn |= XBOX_GAMEPAD_RIGHT_THUMB;
    pState->Gamepad.wButtons = btn;

    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_A] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_A) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_B] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_B) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_X] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_X) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_Y] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_Y) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_BLACK] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_LEFTSHOULDER) ? 255 : 0;
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_WHITE] =
        SDL_GameControllerGetButton(c, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER) ? 255 : 0;

    /* SDL trigger axes are 0..32767 -> Xbox analog button 0..255 */
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_LTRIGGER] =
        (BYTE)(SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_TRIGGERLEFT) >> 7);
    pState->Gamepad.bAnalogButtons[XBOX_BUTTON_RTRIGGER] =
        (BYTE)(SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) >> 7);

    /* SDL Y axis points down; the Xbox Y axis points up -- invert.
     * Use (-1 - v) so v = -32768 does not overflow SHORT. */
    pState->Gamepad.sThumbLX = SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_LEFTX);
    pState->Gamepad.sThumbLY =
        (SHORT)(-1 - SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_LEFTY));
    pState->Gamepad.sThumbRX = SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_RIGHTX);
    pState->Gamepad.sThumbRY =
        (SHORT)(-1 - SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_RIGHTY));

    return ERROR_SUCCESS;
}

DWORD xbox_InputSetState(DWORD dwPort, const XBOX_VIBRATION *pVibration)
{
    if (dwPort >= XBOX_MAX_CONTROLLERS || !pVibration)
        return ERROR_DEVICE_NOT_CONNECTED;

    SDL_GameController *c = g_pads[dwPort];
    if (!c) return ERROR_DEVICE_NOT_CONNECTED;

    /* SDL rumble needs a duration; refresh for ~1s on each call (the game
     * polls vibration continuously). */
    SDL_GameControllerRumble(c, pVibration->wLeftMotorSpeed,
                             pVibration->wRightMotorSpeed, 1000);
    return ERROR_SUCCESS;
}

BOOL xbox_InputIsConnected(DWORD dwPort)
{
    if (dwPort >= XBOX_MAX_CONTROLLERS) return FALSE;
    return g_controller_connected[dwPort];
}

DWORD xbox_InputGetCapabilities(DWORD dwPort, DWORD dwFlags, XBOX_INPUT_CAPABILITIES *pCaps)
{
    (void)dwFlags;
    if (dwPort >= XBOX_MAX_CONTROLLERS || !pCaps)
        return ERROR_DEVICE_NOT_CONNECTED;
    if (!g_pads[dwPort])
        return ERROR_DEVICE_NOT_CONNECTED;

    memset(pCaps, 0, sizeof(XBOX_INPUT_CAPABILITIES));
    pCaps->Type    = 1;   /* XINPUT_DEVTYPE_GAMEPAD */
    pCaps->SubType = 1;   /* XINPUT_DEVSUBTYPE_GAMEPAD */
    pCaps->Flags   = 0;
    return ERROR_SUCCESS;
}

#endif /* _WIN32 */
