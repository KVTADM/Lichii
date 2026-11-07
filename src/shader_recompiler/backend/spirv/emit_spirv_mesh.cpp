// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Switch 2 (NVN) task and mesh programs. They talk to the hardware through the stage buffer
// (ISBE): a task program writes the number of mesh workgroups at byte 4 of its output region and
// its payload after it; each mesh workgroup reads that region (its own index at byte 0) and
// writes its vertices and primitives to an attribute region in groups of 32, one 32-word line per
// attribute, plus the primitive count and 8-bit vertex indices to a map region. Here the task's
// output region is the task payload, the mesh output regions live in workgroup memory, and the
// epilogue (reached by every invocation, see structured_control_flow.cpp) hands them to Vulkan.

#include <algorithm>
#include <span>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void Barrier(EmitContext& ctx) {
    ctx.OpControlBarrier(ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::WorkgroupMemory)));
}

Id LocalInvocation(EmitContext& ctx) {
    return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id), 0U);
}

Id Clamped(EmitContext& ctx, Id index, u32 count) {
    return ctx.OpUMin(ctx.U32[1], index, ctx.Const(count - 1));
}

Id PayloadWord(EmitContext& ctx, Id index) {
    return ctx.OpAccessChain(ctx.mesh_payload_u32, ctx.mesh_payload,
                             Clamped(ctx, index, ctx.mesh_info.payload_words));
}

Id SharedWord(EmitContext& ctx, Id array, Id index) {
    return ctx.OpAccessChain(ctx.mesh_shared_u32, array, index);
}

// Runs body under `if (condition)`.
template <typename Func>
void If(EmitContext& ctx, Id condition, Func&& body) {
    const Id then_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, then_label, merge_label);
    ctx.AddLabel(then_label);
    body();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

// Copies the lines of element `index` (a vertex or a primitive) from the attribute region.
void CopyLines(EmitContext& ctx, Id index, u32 base, u32 num_lines,
               const std::vector<EmitContext::MeshLineTarget>& targets) {
    // Word of line 0: base + (index / 32) * num_lines * 32 + index % 32.
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], index, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1],
                              ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              ctx.OpIAdd(ctx.U32[1], slot, ctx.Const(base)))};
    for (u32 line = 0; line < num_lines; ++line) {
        const EmitContext::MeshLineTarget& target{targets[line]};
        if (!Sirit::ValidId(target.variable)) {
            continue;
        }
        const Id word{ctx.OpLoad(
            ctx.U32[1],
            SharedWord(ctx, ctx.mesh_attributes, ctx.OpIAdd(ctx.U32[1], first, ctx.Const(line * 32))))};
        const Id value{target.is_float ? ctx.OpBitcast(ctx.F32[1], word) : word};
        const Id pointer_type{target.is_float ? ctx.output_f32 : ctx.output_u32};
        const Id pointer{target.element == ~0U
                             ? ctx.OpAccessChain(pointer_type, target.variable, index)
                             : ctx.OpAccessChain(pointer_type, target.variable, index,
                                                 ctx.Const(target.element))};
        ctx.OpStore(pointer, value);
    }
}

// OpenGL-style clip space: z = (z + w) / 2, from the stage buffer's position lines.
void ConvertDepth(EmitContext& ctx, Id vertex) {
    const auto& lines{ctx.mesh_info.vertex_lines};
    const auto z_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionZ))};
    const auto w_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionW))};
    if (z_line == lines.end() || w_line == lines.end()) {
        return;
    }
    const u32 num_lines{static_cast<u32>(lines.size())};
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], vertex, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], vertex, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              slot)};
    const auto read = [&](auto line) {
        const u32 offset{static_cast<u32>(line - lines.begin()) * 32};
        return ctx.OpBitcast(ctx.F32[1],
                             ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_attributes,
                                                               ctx.OpIAdd(ctx.U32[1], first,
                                                                          ctx.Const(offset)))));
    };
    const Id z{ctx.OpFMul(ctx.F32[1], ctx.OpFAdd(ctx.F32[1], read(z_line), read(w_line)),
                          ctx.Const(0.5f))};
    ctx.OpStore(ctx.OpAccessChain(ctx.output_f32, ctx.output_position, vertex, ctx.Const(2U)), z);
}

void TaskEpilogue(EmitContext& ctx) {
    Barrier(ctx);
    const Id count{ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, ctx.Const(1U)))};
    const Id one{ctx.Const(1U)};
    // Marker for OpEmitMeshTasksEXT count, 1, 1, payload (PatchMeshInstructions).
    ctx.OpStore(count, one);
    ctx.OpStore(one, ctx.mesh_payload);
    ctx.mesh_tasks_marker = {count, one, one, ctx.mesh_payload};
}

void MeshEpilogue(EmitContext& ctx) {
    const MeshInfo& mesh{ctx.mesh_info};
    Barrier(ctx);
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.mesh_primitive_count),
                              ctx.Const(mesh.max_primitives))};
    const Id max_vertices{ctx.Const(mesh.max_vertices)};
    // Marker for OpSetMeshOutputsEXT max_vertices, count (PatchMeshInstructions).
    ctx.OpStore(max_vertices, count);
    ctx.mesh_outputs_marker = {max_vertices, count};

    const Id invocation{LocalInvocation(ctx)};
    const u32 threads{ctx.mesh_threads};
    const bool convert_depth{ctx.runtime_info.convert_depth_mode && !ctx.profile.support_native_ndc &&
                             Sirit::ValidId(ctx.output_position)};
    const u32 vertex_lines{static_cast<u32>(mesh.vertex_lines.size())};
    // Points need a size per vertex: the fixed state one, unless the program writes its own.
    const bool points{mesh.PrimitiveVertices() == 1};
    const Id point_size{ctx.Const(ctx.runtime_info.fixed_state_point_size.value_or(1.0f))};
    for (u32 first = 0; first < mesh.max_vertices; first += threads) {
        const Id vertex{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
        const auto copy = [&] {
            if (points) {
                ctx.OpStore(ctx.OpAccessChain(ctx.output_f32, ctx.mesh_point_size, vertex),
                            point_size);
            }
            CopyLines(ctx, vertex, 0, vertex_lines, ctx.mesh_vertex_targets);
            if (convert_depth) {
                ConvertDepth(ctx, vertex);
            }
        };
        if (first + threads > mesh.max_vertices) {
            If(ctx, ctx.OpULessThan(ctx.U1, vertex, max_vertices), copy);
        } else {
            copy();
        }
    }
    const u32 primitive_lines{static_cast<u32>(mesh.primitive_lines.size())};
    const u32 primitive_vertices{mesh.PrimitiveVertices()};
    for (u32 first = 0; first < mesh.max_primitives; first += threads) {
        const Id primitive{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
        If(ctx, ctx.OpULessThan(ctx.U1, primitive, count), [&] {
            // Vertex indices: map bytes 4 + primitive_vertices * primitive.
            const Id byte{ctx.OpIAdd(
                ctx.U32[1], ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(primitive_vertices)),
                ctx.Const(4U))};
            std::array<Id, 3> indices;
            for (u32 element = 0; element < primitive_vertices; ++element) {
                const Id address{ctx.OpIAdd(ctx.U32[1], byte, ctx.Const(element))};
                indices[element] = ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_map, address));
            }
            const Id index_type{ctx.U32[primitive_vertices]};
            const Id output_indices{ctx.TypePointer(spv::StorageClass::Output, index_type)};
            ctx.OpStore(ctx.OpAccessChain(output_indices, ctx.mesh_primitive_indices, primitive),
                        primitive_vertices == 1
                            ? indices[0]
                            : ctx.OpCompositeConstruct(
                                  index_type,
                                  std::span<const Id>(indices.data(), primitive_vertices)));
            CopyLines(ctx, primitive, ctx.mesh_primitive_base, primitive_lines,
                      ctx.mesh_primitive_targets);
        });
    }
}
Id FallbackWord(EmitContext& ctx, Id index) {
    return ctx.OpAccessChain(ctx.mesh_fallback_u32, ctx.mesh_fallback_buffer, ctx.u32_zero_value,
                             index);
}

// Mesh fallback (no VK_EXT_mesh_shader): the program runs as a compute shader and appends its
// triangles, three vertices each, to the fallback buffer, one word per vertex line, for
// EmitMeshFallbackVertex to draw.
void MeshFallbackEpilogue(EmitContext& ctx) {
    const MeshInfo& mesh{ctx.mesh_info};
    Barrier(ctx);
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.mesh_primitive_count),
                              ctx.Const(mesh.max_primitives))};
    const u32 primitive_vertices{mesh.PrimitiveVertices()};
    const Id invocation{LocalInvocation(ctx)};
    // One invocation reserves the workgroup's vertices.
    If(ctx, ctx.OpIEqual(ctx.U1, invocation, ctx.u32_zero_value), [&] {
        const Id total{ctx.OpIMul(ctx.U32[1], count, ctx.Const(primitive_vertices))};
        const Id base{ctx.OpAtomicIAdd(ctx.U32[1], FallbackWord(ctx, ctx.u32_zero_value),
                                       ctx.Const(static_cast<u32>(spv::Scope::Device)),
                                       ctx.u32_zero_value, total)};
        ctx.OpStore(ctx.mesh_fallback_base, base);
    });
    Barrier(ctx);
    const Id base{ctx.OpLoad(ctx.U32[1], ctx.mesh_fallback_base)};
    const Id capacity{
        ctx.OpLoad(ctx.U32[1], FallbackWord(ctx, ctx.Const(MESH_FALLBACK_CAPACITY_WORD)))};

    const auto& lines{mesh.vertex_lines};
    const u32 num_lines{static_cast<u32>(lines.size())};
    const auto z_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionZ))};
    const auto w_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionW))};
    const bool convert_depth{ctx.runtime_info.convert_depth_mode && !ctx.profile.support_native_ndc &&
                             z_line != lines.end() && w_line != lines.end()};
    const u32 z_index{static_cast<u32>(z_line - lines.begin())};
    const u32 w_index{static_cast<u32>(w_line - lines.begin())};
    const Id last_vertex{ctx.Const(mesh.max_vertices > 0 ? mesh.max_vertices - 1 : 0U)};
    const u32 threads{ctx.mesh_threads};
    for (u32 first = 0; first < mesh.max_primitives; first += threads) {
        const Id primitive{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
        If(ctx, ctx.OpULessThan(ctx.U1, primitive, count), [&] {
            // Vertex indices: map bytes 4 + primitive_vertices * primitive.
            const Id byte{ctx.OpIAdd(
                ctx.U32[1], ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(primitive_vertices)),
                ctx.Const(4U))};
            for (u32 element = 0; element < primitive_vertices; ++element) {
                const Id output{ctx.OpIAdd(
                    ctx.U32[1], base,
                    ctx.OpIAdd(ctx.U32[1],
                               ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(primitive_vertices)),
                               ctx.Const(element)))};
                If(ctx, ctx.OpULessThan(ctx.U1, output, capacity), [&] {
                    const Id address{ctx.OpIAdd(ctx.U32[1], byte, ctx.Const(element))};
                    const Id vertex{ctx.OpUMin(
                        ctx.U32[1], ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_map, address)),
                        last_vertex)};
                    // Word of line 0 in the attribute region (see CopyLines).
                    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], vertex, ctx.Const(5U))};
                    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], vertex, ctx.Const(31U))};
                    const Id source{ctx.OpIAdd(
                        ctx.U32[1], ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)), slot)};
                    const Id destination{ctx.OpIAdd(
                        ctx.U32[1], ctx.Const(MESH_FALLBACK_HEADER_WORDS),
                        ctx.OpIMul(ctx.U32[1], output, ctx.Const(num_lines)))};
                    const auto read = [&](u32 line) {
                        return ctx.OpLoad(ctx.U32[1],
                                          SharedWord(ctx, ctx.mesh_attributes,
                                                     ctx.OpIAdd(ctx.U32[1], source,
                                                                ctx.Const(line * 32))));
                    };
                    for (u32 line = 0; line < num_lines; ++line) {
                        Id word{read(line)};
                        if (convert_depth && line == z_index) {
                            // OpenGL-style clip space: z = (z + w) / 2.
                            const Id z{ctx.OpBitcast(ctx.F32[1], word)};
                            const Id w{ctx.OpBitcast(ctx.F32[1], read(w_index))};
                            word = ctx.OpBitcast(
                                ctx.U32[1], ctx.OpFMul(ctx.F32[1], ctx.OpFAdd(ctx.F32[1], z, w),
                                                       ctx.Const(0.5f)));
                        }
                        ctx.OpStore(FallbackWord(ctx, ctx.OpIAdd(ctx.U32[1], destination,
                                                                 ctx.Const(line))),
                                    word);
                    }
                });
            }
        });
    }
}
} // Anonymous namespace

Id EmitIsbeRead(EmitContext& ctx, Id address) {
    const Id workgroup{
        ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), 0U)};
    if (ctx.stage == Stage::Task) {
        return workgroup; // the input region holds the workgroup index (VERTEX_ID)
    }
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    const Id word{ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, index))};
    return ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, address, ctx.u32_zero_value), workgroup,
                        word);
}

void EmitIsbeWriteAttribute(EmitContext& ctx, Id address, Id value) {
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    if (ctx.stage == Stage::Task) {
        ctx.OpStore(PayloadWord(ctx, index), value);
        return;
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_attributes, Clamped(ctx, index, ctx.mesh_attribute_words)),
                value);
}

void EmitIsbeWriteMapU8(EmitContext& ctx, Id address, Id value) {
    if (ctx.stage != Stage::Mesh) {
        throw NotImplementedException("Stage buffer map write outside mesh programs");
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_map, Clamped(ctx, address, ctx.mesh_map_bytes)),
                ctx.OpBitwiseAnd(ctx.U32[1], value, ctx.Const(0xffU)));
}

void EmitIsbeWriteMapU32(EmitContext& ctx, const IR::Value& address, Id value) {
    // The primitive count: the map is written from its end, so its first word is at byte 3.
    if (ctx.stage != Stage::Mesh || !address.IsImmediate() || address.U32() != 3) {
        throw NotImplementedException("32-bit stage buffer map write");
    }
    ctx.OpStore(ctx.mesh_primitive_count, value);
}

void EmitMeshPrologue(EmitContext& ctx) {
    if (ctx.stage != Stage::Mesh) {
        return;
    }
    // Every invocation stores the zero: no branch here, the entry block's label is a phi
    // predecessor for later blocks.
    ctx.OpStore(ctx.mesh_primitive_count, ctx.u32_zero_value);
    if (ctx.runtime_info.mesh_fallback) {
        // No task program fills the input region: zero it, so that nothing the program reads
        // from it (loop bounds) is garbage. Clamped stores, still without a branch.
        const Id invocation{LocalInvocation(ctx)};
        const u32 words{ctx.mesh_info.payload_words};
        for (u32 first = 0; first < words; first += ctx.mesh_threads) {
            const Id index{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
            ctx.OpStore(PayloadWord(ctx, index), ctx.u32_zero_value);
        }
    }
    Barrier(ctx);
}

void EmitMeshEpilogue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task) {
        TaskEpilogue(ctx);
    } else if (ctx.stage == Stage::Mesh) {
        if (ctx.runtime_info.mesh_fallback) {
            MeshFallbackEpilogue(ctx);
        } else {
            MeshEpilogue(ctx);
        }
    }
}

std::vector<u32> EmitMeshFallbackVertex(const Profile& profile, const MeshInfo& mesh) {
    Sirit::Module module{profile.supported_spirv};
    module.AddCapability(spv::Capability::Shader);
    module.AddExtension("SPV_KHR_storage_buffer_storage_class");
    const Id void_id{module.TypeVoid()};
    const Id bool_id{module.TypeBool()};
    const Id u32_id{module.TypeInt(32, false)};
    const Id s32_id{module.TypeInt(32, true)};
    const Id f32_id{module.TypeFloat(32)};
    const Id vec4_id{module.TypeVector(f32_id, 4)};
    const auto u32_const = [&](u32 value) { return module.Constant(u32_id, value); };
    const Id f32_zero{module.Constant(f32_id, 0.0f)};
    const Id f32_one{module.Constant(f32_id, 1.0f)};

    // The fallback buffer, read only: vertex stages may not store without
    // vertexPipelineStoresAndAtomics.
    const Id array_type{module.TypeRuntimeArray(u32_id)};
    module.Decorate(array_type, spv::Decoration::ArrayStride, 4U);
    const Id struct_type{module.TypeStruct(array_type)};
    module.Decorate(struct_type, spv::Decoration::Block);
    module.MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
    module.MemberDecorate(struct_type, 0, spv::Decoration::NonWritable);
    const Id buffer{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::StorageBuffer, struct_type),
        spv::StorageClass::StorageBuffer)};
    module.Decorate(buffer, spv::Decoration::Binding, 0U);
    module.Decorate(buffer, spv::Decoration::DescriptorSet, 1U);
    const Id buffer_u32{module.TypePointer(spv::StorageClass::StorageBuffer, u32_id)};

    std::vector<Id> interfaces;
    const Id vertex_index{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::Input, s32_id), spv::StorageClass::Input)};
    module.Decorate(vertex_index, spv::Decoration::BuiltIn, spv::BuiltIn::VertexIndex);
    interfaces.push_back(vertex_index);
    const Id output_vec4{module.TypePointer(spv::StorageClass::Output, vec4_id)};
    const Id position{module.AddGlobalVariable(output_vec4, spv::StorageClass::Output)};
    module.Decorate(position, spv::Decoration::BuiltIn, spv::BuiltIn::Position);
    interfaces.push_back(position);
    std::array<Id, IR::NUM_GENERICS> generics{};
    for (const u32 line : mesh.vertex_lines) {
        const IR::Attribute attribute{line};
        if (!IR::IsGeneric(attribute)) {
            continue;
        }
        const u32 index{IR::GenericAttributeIndex(attribute)};
        if (!Sirit::ValidId(generics[index])) {
            generics[index] = module.AddGlobalVariable(output_vec4, spv::StorageClass::Output);
            module.Decorate(generics[index], spv::Decoration::Location, index);
            interfaces.push_back(generics[index]);
        }
    }
    if (profile.supported_spirv >= 0x00010400) {
        interfaces.push_back(buffer);
    }

    const Id main{module.OpFunction(void_id, spv::FunctionControlMask::MaskNone,
                                    module.TypeFunction(void_id))};
    module.AddLabel(module.OpLabel());
    const Id index{module.OpBitcast(u32_id, module.OpLoad(s32_id, vertex_index))};
    const Id capacity{module.OpLoad(
        u32_id, module.OpAccessChain(buffer_u32, buffer, u32_const(0),
                                     u32_const(MESH_FALLBACK_CAPACITY_WORD)))};
    const Id debug{module.OpLoad(
        u32_id, module.OpAccessChain(buffer_u32, buffer, u32_const(0),
                                     u32_const(MESH_FALLBACK_DEBUG_WORD)))};
    // Vertices past the capacity (the workgroups ran out of room) collapse to the origin.
    // The vertex count the workgroups reserved (the draw may cover the whole capacity).
    const Id count{module.OpLoad(
        u32_id, module.OpAccessChain(buffer_u32, buffer, u32_const(0), u32_const(0)))};
    const Id in_range{module.OpLogicalAnd(
        bool_id,
        module.OpLogicalAnd(bool_id, module.OpULessThan(bool_id, index, capacity),
                            module.OpULessThan(bool_id, index, count)),
        module.OpIEqual(bool_id, debug, u32_const(0)))};
    const Id safe_index{module.OpSelect(u32_id, in_range, index, u32_const(0))};
    const u32 num_lines{static_cast<u32>(mesh.vertex_lines.size())};
    const Id first{module.OpIAdd(u32_id, u32_const(MESH_FALLBACK_HEADER_WORDS),
                                 module.OpIMul(u32_id, safe_index, u32_const(num_lines)))};
    // Each output is stored once, whole. A vertex out of range goes outside the clip volume
    // (x = 2w, w = 1): a position with w = 0 hangs the RX 5700 XT's rasterizer.
    const Id f32_two{module.Constant(f32_id, 2.0f)};
    std::array<std::array<Id, 4>, IR::NUM_GENERICS> generic_values{};
    std::array<Id, 4> position_values{f32_zero, f32_zero, f32_zero, f32_one};
    for (auto& values : generic_values) {
        values = {f32_zero, f32_zero, f32_zero, f32_one};
    }
    for (u32 line = 0; line < num_lines; ++line) {
        const IR::Attribute attribute{mesh.vertex_lines[line]};
        Id* target{};
        if (IR::IsGeneric(attribute)) {
            target = &generic_values[IR::GenericAttributeIndex(attribute)]
                                    [IR::GenericAttributeElement(attribute)];
        } else if (attribute >= IR::Attribute::PositionX && attribute <= IR::Attribute::PositionW) {
            target = &position_values[static_cast<u32>(attribute) -
                                      static_cast<u32>(IR::Attribute::PositionX)];
        } else {
            continue;
        }
        const Id word{module.OpLoad(
            u32_id, module.OpAccessChain(buffer_u32, buffer, u32_const(0),
                                         module.OpIAdd(u32_id, first, u32_const(line))))};
        *target = module.OpBitcast(f32_id, word);
    }
    const Id position_value{module.OpCompositeConstruct(vec4_id, position_values[0],
                                                        position_values[1], position_values[2],
                                                        position_values[3])};
    const Id outside{module.ConstantComposite(vec4_id, f32_two, f32_two, f32_zero, f32_one)};
    module.OpStore(position, module.OpSelect(vec4_id, in_range, position_value, outside));
    for (u32 generic = 0; generic < IR::NUM_GENERICS; ++generic) {
        if (Sirit::ValidId(generics[generic])) {
            const auto& values{generic_values[generic]};
            module.OpStore(generics[generic], module.OpCompositeConstruct(vec4_id, values[0],
                                                                        values[1], values[2],
                                                                        values[3]));
        }
    }
    module.OpReturn();
    module.OpFunctionEnd();
    module.AddEntryPoint(spv::ExecutionModel::Vertex, main, "main",
                         std::span<const Id>(interfaces.data(), interfaces.size()));
    return module.Assemble();
}

} // namespace Shader::Backend::SPIRV
