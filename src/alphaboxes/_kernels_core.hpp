#pragma once

#include <cstddef>

namespace alphaboxes {

void activation_prepare(float* values, const float* bias, float* denominator,
                        std::size_t nodes, std::size_t width) noexcept;
void activation_finish(float* values, float* denominator, std::size_t count) noexcept;

void grid_aggregate(const float* hidden, float* output, int rows, int cols,
                    std::size_t width, float epsilon) noexcept;

void residual_norm(float* output, const float* bias, const float* residual,
                   const float* scale, const float* shift, const float* mask,
                   std::size_t nodes, std::size_t width) noexcept;

}  // namespace alphaboxes
