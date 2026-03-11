#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace scalar_quantizer {

int amx_bf16_ip_a_rows(
        const uint16_t* a,
        const uint16_t* b,
        size_t d,
        size_t a_rows,
        float* out);

bool amx_bf16_prepare_thread(size_t a_rows = 16);

} // namespace scalar_quantizer
} // namespace faiss