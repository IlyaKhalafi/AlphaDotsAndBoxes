"""Validated buffer bindings for optional C++ float32 inference kernels."""

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

API_VERSION = 1

cdef extern from "_kernels_core.hpp" namespace "alphaboxes":
    void cpp_prepare "alphaboxes::activation_prepare"(
        float*, const float*, float*, size_t, size_t
    ) noexcept nogil
    void cpp_finish "alphaboxes::activation_finish"(float*, float*, size_t) noexcept nogil
    void cpp_grid "alphaboxes::grid_aggregate"(
        const float*, float*, int, int, size_t, float
    ) noexcept nogil
    void cpp_norm "alphaboxes::residual_norm"(
        float*, const float*, const float*, const float*, const float*,
        const float*, size_t, size_t
    ) noexcept nogil


cdef bint overlap(uintptr_t left, size_t left_size, uintptr_t right, size_t right_size) noexcept:
    return left < right + right_size and right < left + left_size


def activation_prepare(float[:, ::1] values, const float[::1] bias,
                       float[:, ::1] denominator):
    cdef Py_ssize_t nodes = values.shape[0], width = values.shape[1]
    if nodes < 1 or width < 1 or bias.shape[0] != width:
        raise ValueError("Activation parameter width differs or input is empty.")
    if denominator.shape[0] != nodes or denominator.shape[1] != width:
        raise ValueError("Activation denominator shape differs.")
    if values.strides[0] != width * <Py_ssize_t>sizeof(float) or denominator.strides[0] != width * <Py_ssize_t>sizeof(float):
        raise ValueError("Activation arrays must be C-contiguous.")
    cdef uintptr_t source = <uintptr_t>&values[0, 0], destination = <uintptr_t>&denominator[0, 0]
    cdef size_t length = nodes * width * sizeof(float), bias_length = width * sizeof(float)
    if (overlap(source, length, destination, length)
        or overlap(source, length, <uintptr_t>&bias[0], bias_length)
        or overlap(destination, length, <uintptr_t>&bias[0], bias_length)):
        raise ValueError("Activation arrays must not overlap.")
    with nogil:
        cpp_prepare(&values[0, 0], &bias[0], &denominator[0, 0], nodes, width)


def activation_finish(float[:, ::1] values, float[:, ::1] denominator):
    cdef Py_ssize_t nodes = values.shape[0], width = values.shape[1]
    if nodes < 1 or width < 1 or denominator.shape[0] != nodes or denominator.shape[1] != width:
        raise ValueError("Activation shapes differ or input is empty.")
    if values.strides[0] != width * <Py_ssize_t>sizeof(float) or denominator.strides[0] != width * <Py_ssize_t>sizeof(float):
        raise ValueError("Activation arrays must be C-contiguous.")
    cdef size_t length = nodes * width * sizeof(float)
    if overlap(<uintptr_t>&values[0, 0], length, <uintptr_t>&denominator[0, 0], length):
        raise ValueError("Activation arrays must not overlap.")
    with nogil:
        cpp_finish(&values[0, 0], &denominator[0, 0], nodes * width)


def aggregate(const float[:, ::1] hidden, float[:, ::1] output,
              int rows, int cols, float epsilon):
    cdef Py_ssize_t nodes = hidden.shape[0], width = hidden.shape[1]
    if rows < 1 or cols < 1 or nodes != 3 * <Py_ssize_t>rows * cols + rows + cols or width < 1:
        raise ValueError("Grid dimensions do not match the hidden array.")
    if output.shape[0] != nodes or output.shape[1] != width:
        raise ValueError("Aggregate output shape differs from hidden array.")
    if hidden.strides[0] != width * <Py_ssize_t>sizeof(float) or output.strides[0] != width * <Py_ssize_t>sizeof(float):
        raise ValueError("Aggregate arrays must be C-contiguous.")
    cdef uintptr_t source = <uintptr_t>&hidden[0, 0]
    cdef uintptr_t destination = <uintptr_t>&output[0, 0]
    cdef size_t length = nodes * width * sizeof(float)
    if overlap(source, length, destination, length):
        raise ValueError("Aggregate output must not overlap its input.")
    with nogil:
        cpp_grid(&hidden[0, 0], &output[0, 0], rows, cols, width, epsilon)


def normalize(float[:, ::1] output, const float[::1] bias,
              const float[:, ::1] residual, const float[::1] scale,
              const float[::1] shift, const float[::1] mask):
    cdef Py_ssize_t nodes = output.shape[0], width = output.shape[1]
    if nodes < 1 or width < 1:
        raise ValueError("Normalization requires a nonempty array.")
    if residual.shape[0] != nodes or residual.shape[1] != width or mask.shape[0] != nodes:
        raise ValueError("Normalization node shapes differ.")
    if bias.shape[0] != width or scale.shape[0] != width or shift.shape[0] != width:
        raise ValueError("Normalization parameter widths differ.")
    if output.strides[0] != width * <Py_ssize_t>sizeof(float) or residual.strides[0] != width * <Py_ssize_t>sizeof(float):
        raise ValueError("Normalization arrays must be C-contiguous.")
    cdef uintptr_t destination = <uintptr_t>&output[0, 0]
    cdef size_t length = nodes * width * sizeof(float), parameter_length = width * sizeof(float)
    if (overlap(destination, length, <uintptr_t>&residual[0, 0], length)
        or overlap(destination, length, <uintptr_t>&bias[0], parameter_length)
        or overlap(destination, length, <uintptr_t>&scale[0], parameter_length)
        or overlap(destination, length, <uintptr_t>&shift[0], parameter_length)
        or overlap(destination, length, <uintptr_t>&mask[0], nodes * sizeof(float))):
        raise ValueError("Normalization output must not overlap its inputs.")
    with nogil:
        cpp_norm(&output[0, 0], &bias[0], &residual[0, 0], &scale[0], &shift[0],
                 &mask[0], nodes, width)
