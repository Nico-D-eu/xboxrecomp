/*
 * A project's replacement for the video overlay's picture.
 *
 * Optional: a project sets g_xbox_overlay_source to draw its own picture in
 * place of the video overlay's (PVIDEO) surface -- for example a
 * higher-resolution copy of the movie the title is playing. The renderer
 * calls acquire once per presented frame while the overlay is on. It returns
 * 32-bit BGRA pixels (alpha unused) and their size and row pitch, or NULL for
 * the title's own picture. The pixels must stay valid until the matching
 * release, which follows at once. They are drawn into the overlay's
 * destination rectangle, with its colour key. Called on the render thread.
 * The D3D11 and OpenGL renderers honour it; the software rasteriser still
 * shows the title's own picture.
 *
 * Its own header, because the renderers do not include kernel.h.
 */
#ifndef XBOX_OVERLAY_SOURCE_H
#define XBOX_OVERLAY_SOURCE_H

#include <stdint.h>

typedef struct {
    const void *(*acquire)(uint32_t *w, uint32_t *h, uint32_t *pitch);
    void (*release)(void);
} XboxOverlaySource;

extern const XboxOverlaySource *volatile g_xbox_overlay_source;

#endif
