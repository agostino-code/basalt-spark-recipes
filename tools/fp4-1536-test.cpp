// fp4_1536_test - the FP4 prompt-expert product at Hy4's 1536/1536 geometry, one expert and a few tokens.
#include "basalt/prefill/moe_fp4.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fp4 = basalt::prefill::fp4;

namespace {
constexpr int N = 1536, F = 1536;
constexpr size_t GU_ROW = (size_t) N / 64 * 36, D_ROW = (size_t) F / 64 * 36;
constexpr size_t UP_OFF = GU_ROW * F, DOWN_OFF = 2 * UP_OFF, TAIL_OFF = DOWN_OFF + D_ROW * N, BYTES = TAIL_OFF + 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};
std::vector<uint8_t> make_blob(std::mt19937& rng, float tail[3]) {
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<uint8_t> b(BYTES);
    std::vector<float> row(N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ex(-10.0f, 3.0f), ts(0.004f, 0.03f);
    auto fill = [&](uint8_t* dst, int rows, int cols, size_t row_bytes) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 16) {
                const float m = std::exp2(ex(rng));
                for (int j = 0; j < 16; ++j) row[(size_t) (c + j)] = nd(rng) * m;
            }
            tr->from_float_ref(row.data(), dst + (size_t) r * row_bytes, cols);
        }
    };
    fill(b.data(), 2 * F, N, GU_ROW);
    fill(b.data() + DOWN_OFF, N, F, D_ROW);
    for (int i = 0; i < 3; ++i) tail[i] = ts(rng);
    const float t4[4] = {tail[0], tail[1], tail[2], 0.0f};
    std::memcpy(b.data() + TAIL_OFF, t4, 16);
    return b;
}
}  // namespace

int main() {
    int dev = 0;
    ck(cudaGetDeviceCount(&dev), "device count");
    if (dev <= 0) return 77;
    std::mt19937 rng(7);
    float tail[3];
    const std::vector<uint8_t> host = make_blob(rng, tail);
    const std::vector<uint8_t> host2 = make_blob(rng, tail);
    Dev blob(2 * BYTES);
    ck(cudaMemcpy(blob.p, host.data(), BYTES, cudaMemcpyHostToDevice), "blob");
    ck(cudaMemcpy((uint8_t*) blob.p + BYTES, host2.data(), BYTES, cudaMemcpyHostToDevice), "blob2");
    constexpr int T = 16;
    std::vector<float> x((size_t) T * N), h((size_t) T * F), out((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);
    Dev dx((size_t) T * N * 4), dh((size_t) T * F * 4), dout((size_t) T * N * 4);
    Dev ids(T * 4), bounds(3 * 4), tile(3 * 4);
    std::vector<int32_t> id(T), bd{0, 8, T};
    for (int i = 0; i < T; ++i) id[i] = i;
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(ids.p, id.data(), id.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(bounds.p, bd.data(), bd.size() * 4, cudaMemcpyHostToDevice), "bounds");
    fp4::plan(bounds.as<int32_t>(), 2, tile.as<int32_t>(), nullptr);
    // one-term token planes, like the production Qwen group
    Dev a4x(fp4::token_bytes(T, N)), gsx(T * 4);
    fp4::quantize_tokens(dx.as<float>(), T, N, N, a4x.p, gsx.as<float>(), nullptr, false);
    ck(cudaDeviceSynchronize(), "quantize_tokens");
    fp4::Product gu;
    gu.blob[0] = blob.as<uint8_t>(); gu.blob[1] = blob.as<uint8_t>() + BYTES; gu.w_off = 0; gu.w_rows = 2 * F; gu.w_cols = N; gu.n = 2;
    gu.a4 = a4x.p; gu.gs = gsx.as<float>(); gu.ids = ids.as<int32_t>(); gu.tokens = T;
    gu.bounds = bounds.as<int32_t>(); gu.swiglu = true; gu.tail_off = TAIL_OFF;
    gu.dst = dh.as<float>(); gu.ld_dst = F; gu.max_rows = 8;
    fp4::run(gu, nullptr);
    ck(cudaDeviceSynchronize(), "gate/up");
    // H one-term, then the down product
    Dev a4h(fp4::a4_bytes(T, F, 2)), gsh(fp4::scale_count(T, 2) * 4);
    fp4::quantize(dh.as<float>(), nullptr, bounds.as<int32_t>(), tile.as<int32_t>(), 2, a4h.p, gsh.as<float>(), F, F, T,
                  nullptr, false);
    ck(cudaDeviceSynchronize(), "quantize H");
    fp4::Product dn;
    dn.blob[0] = blob.as<uint8_t>(); dn.blob[1] = blob.as<uint8_t>() + BYTES; dn.w_off = DOWN_OFF; dn.w_rows = N; dn.w_cols = F; dn.n = 2;
    dn.a4 = a4h.p; dn.gs = gsh.as<float>(); dn.bounds = bounds.as<int32_t>(); dn.tile_off = tile.as<int32_t>();
    dn.dst = dout.as<float>(); dn.ld_dst = N; dn.max_rows = 8;
    fp4::run(dn, nullptr);
    ck(cudaDeviceSynchronize(), "down");
    ck(cudaMemcpy(out.data(), dout.p, out.size() * 4, cudaMemcpyDeviceToHost), "out");
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v);
    std::printf("fp4 1536: OK, finite=%d, out[0]=%g\n", (int) finite, out[0]);
    return finite ? 0 : 1;
}
