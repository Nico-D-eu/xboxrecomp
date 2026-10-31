cmake_minimum_required(VERSION 3.20)

# Xbox Kernel Replacement Layer
# Replaces 147 Xbox kernel imports with Win32 equivalents.
# Memory layout, file I/O, threading, synchronization, crypto, HAL.

set(KERNEL_SOURCES
    kernel_path.c
    kernel_pool.c
    kernel_rtl.c
    kernel_memory.c
    kernel_file.c
    kernel_missing.c
    kernel_thread.c
    kernel_sync.c
    kernel_hal.c
    kernel_guest_cpu.c
    kernel_xbox.c
    kernel_ob.c
    kernel_io.c
    kernel_crypto.c
    kernel_thunks.c
    kernel_bridge.c
    xbox_memory_layout.c
    guest_vmem.c
    icall_feedback.c
    recomp_trace.c
    nv2a_pb_scan.c
    nv2a_pb_exec.c
    nv2a_pb_exec_upstream.c
    nv2a_vp.c
    nv2a_combiner.c
    nv2a_hlsl.c
    nv2a_d3d11.c
    nv2a_gl.c               # the executor's OpenGL back end (POSIX; empty on Windows)
    nv2a_flip_hold.c
    nv2a_vsh_interp.c
)

add_library(xbox_kernel STATIC ${KERNEL_SOURCES})

# See src/platform/CMakeLists.txt: resolve src/ relative to this file, not via
# CMAKE_SOURCE_DIR, which is the consuming project's root under add_subdirectory().
get_filename_component(XBOXRECOMP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

target_include_directories(xbox_kernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XBOXRECOMP_SRC_DIR}
)

if(WIN32)
    # platform is an INTERFACE target on Windows: it contributes the src/
    # include path that "platform/xbox_winnt.h" resolves against.
    target_link_libraries(xbox_kernel PUBLIC platform kernel32 bcrypt d3d11 d3dcompiler dxguid)
else()
    # Linux: platform layer (POSIX host primitives) + OpenSSL for crypto,
    # and SDL2 + epoxy for the executor's OpenGL back end (nv2a_gl.c).
    find_package(OpenSSL REQUIRED)
    find_package(SDL2 REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EPOXY REQUIRED IMPORTED_TARGET epoxy)
    target_link_libraries(xbox_kernel PUBLIC platform OpenSSL::Crypto)
    target_link_libraries(xbox_kernel PRIVATE SDL2::SDL2 PkgConfig::EPOXY)
endif()

if(MSVC)
    target_compile_options(xbox_kernel PRIVATE /W4 /WX- /O2 /GS- /Gy)
    target_compile_definitions(xbox_kernel PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(xbox_kernel PRIVATE -Wall -Wextra -Wno-unused-parameter -O2)
endif()

# The trapped-MMIO dispatch (xbox_VehMmioFault, xbox_PosixMmioFault) routes
# to the APU and the USB OHCI models. Link-only: both already call back into
# the kernel, and CMake repeats static libraries in such a cycle.
target_link_libraries(xbox_kernel PRIVATE xbox_apu xbox_usb)

target_compile_definitions(xbox_kernel PRIVATE $<$<CONFIG:Debug>:_DEBUG>)
