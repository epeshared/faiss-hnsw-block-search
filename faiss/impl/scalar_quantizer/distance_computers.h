/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/simd_levels.h>

#if defined(FAISS_OPT_AMX)
#include <faiss/impl/scalar_quantizer/amx_bf16_ip.h>
#endif

#if defined(__AVX512BF16__)
#include <immintrin.h>
#endif

namespace faiss {

namespace scalar_quantizer {

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, SIMDLevel SL>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, SIMDLevel SL>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#if defined(FAISS_OPT_AMX) && defined(__AMX_TILE__) && defined(__AMX_BF16__)

template <SIMDLevel SL>
struct DCBF16IPAmx : SQDistanceComputer {
    using Sim = SimilarityIP<SL>;

    QuantizerBF16<SL> quant;
    std::vector<uint16_t> qbf16;

    DCBF16IPAmx(size_t d, const std::vector<float>& trained)
            : quant(d, trained), qbf16(d) {}

    void set_query(const float* x) final {
        q = x;
        for (size_t i = 0; i < quant.d; ++i) {
            qbf16[i] = encode_bf16(x[i]);
        }
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16_fallback(
            const uint16_t* a,
            const uint16_t* b) const {
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 32 <= quant.d; i += 32) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + i));
            const __m512i vb = _mm512_loadu_si512((const void*)(b + i));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)va, (__m512bh)vb);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; i < quant.d; ++i) {
            sum += decode_bf16(a[i]) * decode_bf16(b[i]);
        }
        return sum;
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16(
            const uint16_t* a,
            const uint16_t* b) const {
        float out = 0.0f;
        if (amx_bf16_ip_a_rows(a, b, quant.d, 1, &out) == 0) {
            return out;
        }
        return compute_code_ip_bf16_fallback(a, b);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const auto* code1 = reinterpret_cast<const uint16_t*>(codes + i * code_size);
        const auto* code2 = reinterpret_cast<const uint16_t*>(codes + j * code_size);
        return compute_code_ip_bf16(code1, code2);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_ip_bf16(qbf16.data(), reinterpret_cast<const uint16_t*>(code));
    }

    void distances_batch(size_t n, const idx_t* idx, float* dis) override {
        if (n == 0) {
            return;
        }

        if (n < 8) {
            for (size_t i = 0; i < n; ++i) {
                dis[i] = this->operator()(idx[i]);
            }
            return;
        }

        thread_local std::vector<uint16_t> packed;
        packed.resize(n * quant.d);
        for (size_t i = 0; i < n; ++i) {
            const auto* code = reinterpret_cast<const uint16_t*>(
                    codes + idx[i] * code_size);
            std::memcpy(
                    packed.data() + i * quant.d,
                    code,
                    quant.d * sizeof(uint16_t));
        }

        if (amx_bf16_ip_a_rows(packed.data(), qbf16.data(), quant.d, n, dis) ==
            0) {
            return;
        }

        for (size_t i = 0; i < n; ++i) {
            const auto* code = reinterpret_cast<const uint16_t*>(
                    codes + idx[i] * code_size);
            dis[i] = compute_code_ip_bf16_fallback(qbf16.data(), code);
        }
    }
};

#endif

/*******************************************************************
 * Selection function
 *******************************************************************/

template <SIMDLevel SL>
SQDistanceComputer* sq_select_distance_computer(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer
} // namespace faiss
