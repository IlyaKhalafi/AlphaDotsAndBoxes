#include "_kernels_core.hpp"

#include <cmath>

// Select SIMD instructions at runtime; binaries retain a portable baseline.
#if defined(__GNUC__) && !defined(__clang__) && defined(__GLIBC__) && defined(__x86_64__)
#define ALPHABOXES_TARGETS __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define ALPHABOXES_TARGETS
#endif

#if defined(_MSC_VER)
#define ALPHABOXES_RESTRICT __restrict
#else
#define ALPHABOXES_RESTRICT __restrict__
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ALPHABOXES_INLINE inline __attribute__((always_inline))
#else
#define ALPHABOXES_INLINE __forceinline
#endif

namespace alphaboxes {
namespace {

// Keep the contiguous float32 reduction order used by NumPy: eight lanes,
// pairwise combination, and recursive splitting above 128 elements.
// https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/loops_utils.h.src
template <bool square>
ALPHABOXES_INLINE float pairwise_block(const float* values, std::size_t count) noexcept {
    const auto read = [](float value) { return square ? value * value : value; };
    if (count < 8) {
        float sum = -0.0f;
        for (std::size_t i = 0; i < count; ++i) sum += read(values[i]);
        return sum;
    }
    float lanes[8];
    for (std::size_t lane = 0; lane < 8; ++lane) lanes[lane] = read(values[lane]);
    const std::size_t aligned = count & ~std::size_t(7);
    for (std::size_t i = 8; i < aligned; i += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane) lanes[lane] += read(values[i + lane]);
    }
    float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (std::size_t i = aligned; i < count; ++i) sum += read(values[i]);
    return sum;
}

template <bool square>
float pairwise_sum(const float* values, std::size_t count) noexcept {
    if (count <= 128) return pairwise_block<square>(values, count);
    const std::size_t half = (count / 2) & ~std::size_t(7);
    return pairwise_sum<square>(values, half) + pairwise_sum<square>(values + half, count - half);
}

}  // namespace

ALPHABOXES_TARGETS
void activation_prepare(float* ALPHABOXES_RESTRICT values,
                        const float* ALPHABOXES_RESTRICT bias,
                        float* ALPHABOXES_RESTRICT denominator,
                        std::size_t nodes, std::size_t width) noexcept {
    for (std::size_t row = 0; row < nodes; ++row) {
        for (std::size_t channel = 0; channel < width; ++channel) {
            const std::size_t index = row * width + channel;
            values[index] += bias[channel];
            float exponent = -values[index];
            if (exponent < -80.0f) exponent = -80.0f;
            if (exponent > 80.0f) exponent = 80.0f;
            denominator[index] = exponent;
        }
    }
}

ALPHABOXES_TARGETS
void activation_finish(float* ALPHABOXES_RESTRICT values,
                       float* ALPHABOXES_RESTRICT denominator, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        denominator[i] += 1.0f;
        values[i] /= denominator[i];
    }
}

ALPHABOXES_TARGETS
void grid_aggregate(const float* ALPHABOXES_RESTRICT hidden,
                    float* ALPHABOXES_RESTRICT output, int rows, int cols,
                    std::size_t width, float epsilon) noexcept {
    const std::size_t horizontal = std::size_t(rows + 1) * cols;
    const std::size_t edges = horizontal + std::size_t(rows) * (cols + 1);
    const std::size_t nodes = edges + std::size_t(rows) * cols;
    const float self_scale = 1.0f + epsilon;
    for (std::size_t node = 0; node < nodes; ++node) {
        std::size_t neighbors[4] = {nodes, nodes, nodes, nodes};
        if (node < horizontal) {
            const std::size_t row = node / cols, col = node % cols;
            if (row > 0) neighbors[0] = edges + (row - 1) * cols + col;
            if (row < std::size_t(rows)) {
                neighbors[row > 0 ? 1 : 0] = edges + row * cols + col;
            }
        } else if (node < edges) {
            const std::size_t index = node - horizontal;
            const std::size_t row = index / (cols + 1), col = index % (cols + 1);
            if (col > 0) neighbors[0] = edges + row * cols + col - 1;
            if (col < std::size_t(cols)) {
                neighbors[col > 0 ? 1 : 0] = edges + row * cols + col;
            }
        } else {
            const std::size_t box = node - edges, row = box / cols, col = box % cols;
            neighbors[0] = row * cols + col;
            neighbors[1] = (row + 1) * cols + col;
            neighbors[2] = horizontal + row * (cols + 1) + col;
            neighbors[3] = neighbors[2] + 1;
        }
        const float* own = hidden + node * width;
        float* result = output + node * width;
        const float* first = hidden + neighbors[0] * width;
        if (node < edges) {
            if (neighbors[1] == nodes) {
                for (std::size_t channel = 0; channel < width; ++channel) {
                    const float message = first[channel] + 0.0f;
                    result[channel] = self_scale * own[channel] + message;
                }
            } else {
                const float* second = hidden + neighbors[1] * width;
                for (std::size_t channel = 0; channel < width; ++channel) {
                    const float message = (first[channel] + second[channel]) + 0.0f;
                    result[channel] = self_scale * own[channel] + message;
                }
            }
        } else {
            const float* second = hidden + neighbors[1] * width;
            const float* third = hidden + neighbors[2] * width;
            const float* fourth = hidden + neighbors[3] * width;
            for (std::size_t channel = 0; channel < width; ++channel) {
                float message = first[channel] + second[channel];
                message += third[channel];
                message += fourth[channel];
                result[channel] = self_scale * own[channel] + message;
            }
        }
    }
}

ALPHABOXES_TARGETS
void residual_norm(float* ALPHABOXES_RESTRICT output,
                   const float* ALPHABOXES_RESTRICT bias,
                   const float* ALPHABOXES_RESTRICT residual,
                   const float* ALPHABOXES_RESTRICT scale,
                   const float* ALPHABOXES_RESTRICT shift,
                   const float* ALPHABOXES_RESTRICT mask,
                   std::size_t nodes, std::size_t width) noexcept {
    for (std::size_t node = 0; node < nodes; ++node) {
        float* row = output + node * width;
        const float* skip = residual + node * width;
        for (std::size_t channel = 0; channel < width; ++channel) {
            row[channel] += bias[channel];
            row[channel] += skip[channel];
        }
        const float sum = width <= 128 ? pairwise_block<false>(row, width)
                                      : pairwise_sum<false>(row, width);
        const float mean = (0.0f + sum) / float(width);
        for (std::size_t channel = 0; channel < width; ++channel) row[channel] -= mean;
        const float variance = (width <= 128 ? pairwise_block<true>(row, width)
                                            : pairwise_sum<true>(row, width)) / float(width) + 1e-5f;
        const float denominator = std::sqrt(variance);
        for (std::size_t channel = 0; channel < width; ++channel) {
            row[channel] /= denominator;
            row[channel] *= scale[channel];
            row[channel] += shift[channel];
            if (mask[node] != 1.0f) row[channel] *= mask[node];
        }
    }
}

}  // namespace alphaboxes
