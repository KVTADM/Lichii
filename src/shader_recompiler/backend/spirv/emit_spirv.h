// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {

constexpr u32 NUM_TEXTURE_SCALING_WORDS = 4;
constexpr u32 NUM_IMAGE_SCALING_WORDS = 2;
constexpr u32 NUM_TEXTURE_AND_IMAGE_SCALING_WORDS =
    NUM_TEXTURE_SCALING_WORDS + NUM_IMAGE_SCALING_WORDS;

struct RescalingLayout {
    alignas(16) std::array<u32, NUM_TEXTURE_SCALING_WORDS> rescaling_textures;
    alignas(16) std::array<u32, NUM_IMAGE_SCALING_WORDS> rescaling_images;
    u32 down_factor;
};
struct RenderAreaLayout {
    std::array<f32, 4> render_area;
};
constexpr u32 RESCALING_LAYOUT_WORDS_OFFSET = offsetof(RescalingLayout, rescaling_textures);
constexpr u32 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET = offsetof(RescalingLayout, down_factor);
constexpr u32 RENDERAREA_LAYOUT_OFFSET = offsetof(RenderAreaLayout, render_area);

// Mesh fallback buffer (devices without VK_EXT_mesh_shader), one region per draw, in words: a
// VkDrawIndirectCommand whose vertex count the mesh workgroups bump, the vertex capacity, then
// the vertices of the triangle list (one word per stage buffer vertex line).
constexpr u32 MESH_FALLBACK_CAPACITY_WORD = 4;
constexpr u32 MESH_FALLBACK_DEBUG_WORD = 5; ///< nonzero: the vertex shader collapses every vertex
constexpr u32 MESH_FALLBACK_HEADER_WORDS = 16;

[[nodiscard]] std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program, Bindings& bindings);

/// Vertex shader drawing the triangles a mesh program emitted in the mesh fallback: it reads
/// vertex gl_VertexIndex from the fallback buffer (set 1, binding 0) into the outputs the mesh
/// program would have written.
[[nodiscard]] std::vector<u32> EmitMeshFallbackVertex(const Profile& profile,
                                                      const MeshInfo& mesh);
[[nodiscard]] inline std::vector<u32> EmitSPIRV(const Profile& profile, IR::Program& program) {
    Bindings binding;
    return EmitSPIRV(profile, {}, program, binding);
}

} // namespace Shader::Backend::SPIRV
