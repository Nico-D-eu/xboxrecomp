/**
 * NV2A vertex programs, executed on the CPU.
 *
 * The pushbuffer executor rasterises in software, so a title that transforms
 * its vertices with a vertex program (everything that is not pre-transformed
 * screen-space geometry) needs the program run before there is anything to
 * rasterise. This is that: the transform program and constant memory as the
 * title loads them through NV097 methods, and an interpreter for the MAC/ILU
 * instruction pairs.
 *
 * Instruction decoding follows xemu's vsh.c field table (dwords 1..3 of each
 * 128-bit slot; dword 0 is unused), which is the reference the D3D8 layer's
 * HLSL translator was also derived from.
 */
#ifndef XBOXRECOMP_NV2A_VP_H
#define XBOXRECOMP_NV2A_VP_H

#include <stdint.h>

#define NV2A_VP_SLOTS      136   /* instruction slots */
#define NV2A_VP_CONSTANTS  192   /* float4 constant registers */
#define NV2A_VP_INPUTS     16
#define NV2A_VP_TEMPS      12    /* R0..R11; R12 aliases oPos */

/* Output registers, as FLD_OUT_ADDRESS names them. */
enum {
    NV2A_VP_OPOS = 0, NV2A_VP_OD0 = 3, NV2A_VP_OD1 = 4, NV2A_VP_OFOG = 5,
    NV2A_VP_OPTS = 6, NV2A_VP_OB0 = 7, NV2A_VP_OB1 = 8,
    NV2A_VP_OT0 = 9, NV2A_VP_OT1 = 10, NV2A_VP_OT2 = 11, NV2A_VP_OT3 = 12,
    NV2A_VP_OUTPUTS = 13
};

/* The GPU-side state a title programs: executed with nv2a_vp_run. */
typedef struct {
    uint32_t program[NV2A_VP_SLOTS][4];
    float    constants[NV2A_VP_CONSTANTS][4];
    uint32_t program_load;     /* slot the next SET_TRANSFORM_PROGRAM fills */
    uint32_t constant_load;    /* register the next SET_TRANSFORM_CONSTANT fills */
    uint32_t program_start;    /* first slot to execute */
    uint32_t execution_mode;   /* SET_TRANSFORM_EXECUTION_MODE; bits 1:0 = 2 is program */
    uint32_t program_writes;   /* diagnostics */
    /* Bumped whenever the program or the constants change, so a consumer
     * that caches either (the D3D11 backend) knows when to look again. */
    uint32_t program_gen, constants_gen;
} Nv2aVpState;

/* Method handlers. Each returns 1 if `method` was a vertex-program method. */
int nv2a_vp_method(Nv2aVpState *vp, uint32_t method, uint32_t param);

/* Is the transform stage running a program (as opposed to fixed function)? */
int nv2a_vp_active(const Nv2aVpState *vp);

/* Run the program for one vertex. `in` is v0..v15; `out` receives the output
 * registers (indexed by NV2A_VP_O*). Returns the number of instructions
 * executed, 0 if the program was empty or ran off the end without FINAL. */
int nv2a_vp_run(Nv2aVpState *vp, const float in[NV2A_VP_INPUTS][4],
                float out[NV2A_VP_OUTPUTS][4]);

#endif /* XBOXRECOMP_NV2A_VP_H */
