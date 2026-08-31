#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::span<const DType> layer_dtypes,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (!layer_dtypes.empty() && layer_dtypes.size() < layers) {
        throw std::invalid_argument("Paged KV per-layer dtype table is shorter than the layer count");
    }
    // Per-layer resolution: BF16 entries inherit the global dtype. Accepted
    // per-layer storages are the quantized codecs with their native group.
    const auto layer_dtype = [&](std::uint32_t layer) {
        const DType override_dtype = layer_dtypes.empty() ? DType::BF16 : layer_dtypes[layer];
        const DType selected = override_dtype == DType::BF16 ? dtype : override_dtype;
        if (selected != DType::BF16 && selected != DType::I8 && selected != DType::NVFP4 &&
            selected != DType::FP8_E4M3FN) {
            throw std::invalid_argument("Paged KV per-layer dtype is invalid");
        }
        return selected;
    };
    const auto layer_quant_group = [&](DType selected) {
        return selected == DType::I8
                   ? kKvInt8QuantGroup
                   : (selected == DType::BF16
                          ? 0
                          : (selected == DType::FP8_E4M3FN ? kKvFp8QuantGroup
                                                           : kNvfp4KvQuantGroup));
    };
    (void)quant_group;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        if (selected != DType::BF16) {
            const std::int32_t group = layer_quant_group(selected);
            if (head_dim % group != 0) {
                throw std::invalid_argument("Paged KV per-layer quantization is invalid");
            }
        }
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * 4ULL);
    std::array<DType, 16> stored{};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        stored[layer]        = selected;
        const std::int32_t group = layer_quant_group(selected);
        if (selected == DType::BF16) {
            geometry.planes.push_back({DType::BF16, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::BF16, head_dim, kv_heads, 256});
        } else if (selected == DType::I8) {
            geometry.planes.push_back({DType::I8, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::I8, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else if (selected == DType::FP8_E4M3FN) {
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim, kv_heads, 256});
            // FP8 per-group scales are FP16 in the production codecs; the
            // attention kernels require FP16 scale planes.
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else {
            // NVFP4 tier: K keeps E2M1 packed codes with per-16 E4M3FN scales;
            // V stores ISO3 sign-magnitude nibbles in the same plane geometry
            // (semantic split via v_dtype, no extra payload).
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
        .layer_dtypes = stored,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    const std::span<const DType> layer_dtypes(spec.layer_kv_dtypes.data(),
                                              spec.full_attention_layers);
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                layer_dtypes, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   {}, spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), dtype_(layout.dtype), quant_group_(layout.quant_group),
      layer_dtypes_(layout.layer_dtypes) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::I8
                                    ? kKvInt8QuantGroup
                                    : (layer_dtypes_[layer] == DType::FP8_E4M3FN
                                           ? kKvFp8QuantGroup
                                           : 0)),
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = dtype_ == DType::I8 || dtype_ == DType::FP8_E4M3FN;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::I8
                                    ? kKvInt8QuantGroup
                                    : (layer_dtypes_[layer] == DType::FP8_E4M3FN
                                           ? kKvFp8QuantGroup
                                           : 0)),
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
