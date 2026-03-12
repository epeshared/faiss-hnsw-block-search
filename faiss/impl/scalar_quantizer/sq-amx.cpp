#include <faiss/impl/scalar_quantizer/amx_bf16_ip.h>

#include <faiss/utils/bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

namespace faiss {
namespace scalar_quantizer {

#if defined(__linux__)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILECFG
#define XFEATURE_XTILECFG 17
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

#if defined(__linux__) && defined(__AMX_TILE__) && defined(__AMX_BF16__)
static inline bool amx_enable_this_thread_once() {
    static thread_local int t_state = 0;
    if (t_state != 0) {
        return t_state > 0;
    }

    long st_cfg = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILECFG);
    long st_data = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    t_state = (st_cfg == 0 && st_data == 0) ? 1 : -1;
    return t_state > 0;
}

static inline void amx_set_cfg_for_rows(int rows) {
    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prev_rows = -1;

    if (rows == prev_rows) {
        return;
    }

    constexpr int kBlock = 32;
    const int a_rows = rows;
    const int n = 1;
    const int a_colsb = kBlock * 2;
    const int b_colsb = n * 4;
    const int b_rows = kBlock / 2;
    const int c_colsb = n * 4;
    const int c_rows = a_rows;

    std::memset(cfg, 0, sizeof(cfg));
    cfg[0] = 1;

    cfg[16 + 2 * 0] = (unsigned char)a_colsb;
    cfg[48 + 0] = (unsigned char)a_rows;
    cfg[16 + 2 * 1] = (unsigned char)b_colsb;
    cfg[48 + 1] = (unsigned char)b_rows;
    cfg[16 + 2 * 2] = (unsigned char)c_colsb;
    cfg[48 + 2] = (unsigned char)c_rows;

    _tile_loadconfig(cfg);
    prev_rows = rows;
}

static inline int amx_ip_rows_16x1_bf16(
        const uint16_t* a,
        const uint16_t* b,
        size_t d,
        size_t a_rows,
        float* out) {
    if (a_rows == 0 || a_rows > 16) {
        return -1;
    }
    if (!amx_enable_this_thread_once()) {
        return -1;
    }

    const size_t block_count = d / 32;
    const size_t tail_count = d % 32;
    if (block_count == 0) {
        return -1;
    }

    amx_set_cfg_for_rows((int)a_rows);
    _tile_zero(2);

    const int a_stride = (int)(d * sizeof(uint16_t));
    const int b_stride = 4;
    for (size_t blk = 0; blk < block_count; ++blk) {
        const uint16_t* a_blk = a + blk * 32;
        const uint16_t* b_blk = b + blk * 32;
        _tile_loadd(0, (const void*)a_blk, a_stride);
        _tile_loadd(1, (const void*)b_blk, b_stride);
        _tile_dpbf16ps(2, 0, 1);
    }

    _tile_stored(2, (void*)out, 4);

    if (tail_count != 0) {
        const size_t base = block_count * 32;
        for (size_t r = 0; r < a_rows; ++r) {
            const uint16_t* a_tail = a + r * d + base;
            const uint16_t* b_tail = b + base;
            float sum = 0.0f;
            for (size_t i = 0; i < tail_count; ++i) {
                sum += decode_bf16(a_tail[i]) * decode_bf16(b_tail[i]);
            }
            out[r] += sum;
        }
    }

    return 0;
}
#endif

int amx_bf16_ip_a_rows(
        const uint16_t* a,
        const uint16_t* b,
        size_t d,
        size_t a_rows,
        float* out) {
#if !defined(FAISS_OPT_AMX) || !defined(__AMX_TILE__) || !defined(__AMX_BF16__)
    (void)a;
    (void)b;
    (void)d;
    (void)a_rows;
    (void)out;
    return -1;
#else
    if (a_rows == 0) {
        return 0;
    }
    if (a == nullptr || b == nullptr || out == nullptr) {
        return -1;
    }

    constexpr size_t kMaxRows = 16;
    size_t done = 0;
    while (done < a_rows) {
        const size_t cur = std::min(kMaxRows, a_rows - done);
#if defined(__linux__)
        if (amx_ip_rows_16x1_bf16(a + done * d, b, d, cur, out + done) != 0) {
            return -1;
        }
#else
        (void)cur;
        return -1;
#endif
        done += cur;
    }
    return 0;
#endif
}

bool amx_bf16_prepare_thread(size_t a_rows) {
#if !defined(FAISS_OPT_AMX) || !defined(__AMX_TILE__) || !defined(__AMX_BF16__)
    (void)a_rows;
    return false;
#else
#if defined(__linux__)
    if (a_rows == 0) {
        a_rows = 1;
    }
    a_rows = std::min<size_t>(a_rows, 16);
    if (!amx_enable_this_thread_once()) {
        return false;
    }
    amx_set_cfg_for_rows((int)a_rows);
    return true;
#else
    (void)a_rows;
    return false;
#endif
#endif
}

} // namespace scalar_quantizer
} // namespace faiss