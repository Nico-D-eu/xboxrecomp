/*
 * epoxy/gl.h for Android: the system's OpenGL ES 3.2 headers in place of
 * libepoxy. On Android every GLES 3.2 entry point is exported by
 * libGLESv3.so, so no loader is needed, and libepoxy would be one more
 * library to cross-compile and ship.
 *
 * The GL back end (nv2a_gl.c) runs in its OpenGL ES mode here (s_es). Its
 * desktop-only paths are behind run-time checks but still have to compile,
 * so the few desktop enums and calls they name are defined below; none of
 * them is reached on an ES context.
 */
#ifndef RECOMP_ANDROID_EPOXY_GL_H
#define RECOMP_ANDROID_EPOXY_GL_H

#include <GLES3/gl32.h>
#include <GLES2/gl2ext.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* Desktop-only enums named on paths ES never takes. */
#ifndef GL_BGRA
#define GL_BGRA                                 0x80E1
#endif
#ifndef GL_UNSIGNED_INT_8_8_8_8_REV
#define GL_UNSIGNED_INT_8_8_8_8_REV             0x8367
#endif
#ifndef GL_SAMPLES_PASSED
#define GL_SAMPLES_PASSED                       0x8914
#endif
#ifndef GL_DEPTH_CLAMP
#define GL_DEPTH_CLAMP                          0x864F
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT           0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT       0x84FF
#endif
#ifndef GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT        0x83F1
#endif
#ifndef GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT        0x83F2
#endif
#ifndef GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT        0x83F3
#endif
#ifndef GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT  0x8C4D
#endif
#ifndef GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT  0x8C4E
#endif
#ifndef GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT  0x8C4F
#endif
#ifndef GL_COMPRESSED_RGBA_BPTC_UNORM
#define GL_COMPRESSED_RGBA_BPTC_UNORM           0x8E8C
#endif
#ifndef GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM
#define GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM     0x8E8D
#endif

#ifndef GL_COMPRESSED_RED_RGTC1
#define GL_COMPRESSED_RED_RGTC1                 0x8DBB
#endif
#ifndef GL_COMPRESSED_RG_RGTC2
#define GL_COMPRESSED_RG_RGTC2                  0x8DBD
#endif
#ifndef GL_MAP_PERSISTENT_BIT
#define GL_MAP_PERSISTENT_BIT                   0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT
#define GL_MAP_COHERENT_BIT                     0x0080
#endif

/* Desktop calls with an ES equivalent. */
static inline void glClearDepth(double d) { glClearDepthf((float)d); }
static inline void glDepthRange(double n, double f) { glDepthRangef((float)n, (float)f); }
static inline void glGetQueryObjectiv(GLuint id, GLenum pname, GLint *v)
{
    GLuint r = 0;
    glGetQueryObjectuiv(id, pname, &r);
    *v = (GLint)r;
}
static inline void glGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *v)
{
    GLuint r = 0;
    glGetQueryObjectuiv(id, pname, &r);
    *v = r;
}
/* Only used when the context reports GL 4.4 or ARB_buffer_storage, which
 * an ES context never does. */
static inline void glBufferStorage(GLenum t, GLsizeiptr n, const void *d, GLbitfield f)
{
    (void)f;
    glBufferData(t, n, d, GL_DYNAMIC_DRAW);
}

/* libepoxy's two queries: the version as 10 * major + minor ("OpenGL ES
 * 3.2 ..." -> 32), and an extension test through glGetStringi. */
static inline int epoxy_gl_version(void)
{
    const char *v = (const char *)glGetString(GL_VERSION);
    const char *p = v;
    if (!v)
        return 0;
    while (*p && (*p < '0' || *p > '9'))
        p++;
    if (!*p)
        return 0;
    return atoi(p) * 10 + (strchr(p, '.') ? atoi(strchr(p, '.') + 1) % 10 : 0);
}

static inline bool epoxy_has_gl_extension(const char *name)
{
    GLint n = 0, i;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (i = 0; i < n; i++) {
        const char *e = (const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && strcmp(e, name) == 0)
            return true;
    }
    return false;
}

#endif /* RECOMP_ANDROID_EPOXY_GL_H */
