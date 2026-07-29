"""Fused SIMD evaluator for muparser-compatible bytecode."""

from std.algorithm import parallelize
from std.math import (
    abs,
    acos,
    asin,
    atan,
    atan2,
    cos,
    cosh,
    exp,
    floor,
    log,
    log10,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime MAX_STACK = 64
comptime PARALLEL_WORK = 5_000_000
comptime FAST_PARALLEL_ELEMENTS = 4_000_000


@always_inline
def load_variable[width: Int](
    addresses: I64Ptr, strides: I64Ptr, variable: Int, index: Int
) -> SIMD[DType.float64, width]:
    var source = F64Ptr(unsafe_from_address=Int(addresses[variable]))
    if strides[variable] == 0:
        return SIMD[DType.float64, width](source[0])
    return source.load[width=width](index)


@always_inline
def store_variable[width: Int](
    addresses: I64Ptr,
    strides: I64Ptr,
    variable: Int,
    index: Int,
    value: SIMD[DType.float64, width],
):
    var destination = F64Ptr(unsafe_from_address=Int(addresses[variable]))
    if strides[variable] == 0:
        destination[0] = value[width - 1]
    else:
        destination.store(index, value)


@always_inline
def execute_multiply_add[width: Int](
    addresses: I64Ptr,
    strides: I64Ptr,
    lhs_variable: Int,
    rhs_variable: Int,
    add_variable: Int,
    index: Int,
) -> SIMD[DType.float64, width]:
    return (
        load_variable[width](
            addresses, strides, lhs_variable, index
        )
        * load_variable[width](
            addresses, strides, rhs_variable, index
        )
        + load_variable[width](
            addresses, strides, add_variable, index
        )
    )


@always_inline
def execute_conditional_multiply_divide[width: Int](
    constants: F64Ptr,
    addresses: I64Ptr,
    strides: I64Ptr,
    a_variable: Int,
    b_variable: Int,
    c_variable: Int,
    condition_constant: Int,
    index: Int,
) -> SIMD[DType.float64, width]:
    var a = load_variable[width](
        addresses, strides, a_variable, index
    )
    var b = load_variable[width](
        addresses, strides, b_variable, index
    )
    var c = load_variable[width](
        addresses, strides, c_variable, index
    )
    return c.gt(constants[condition_constant]).select(
        a * b + c, a / b - c
    )


# muparser: include/muParserTemplateMagic.h MathImpl
# muparser: src/muParserBase.cpp ParserBase::ParseCmdCodeBulk
@always_inline
def execute_chunk[width: Int](
    code: I64Ptr,
    code_count: Int,
    constants: F64Ptr,
    addresses: I64Ptr,
    strides: I64Ptr,
    index: Int,
) -> SIMD[DType.float64, width]:
    var stack = InlineArray[SIMD[DType.float64, width], MAX_STACK](
        uninitialized=True
    )
    var sp = 0
    for pc in range(code_count):
        var op = Int(code[pc * 3])
        var argument = Int(code[pc * 3 + 1])
        var argc = Int(code[pc * 3 + 2])
        if op == 20:
            stack[sp] = load_variable[width](
                addresses, strides, argument, index
            )
            sp += 1
        elif op == 21:
            stack[sp] = SIMD[DType.float64, width](constants[argument])
            sp += 1
        elif op == 22:
            var value = load_variable[width](
                addresses, strides, argument, index
            )
            stack[sp] = value * value
            sp += 1
        elif op == 23:
            var value = load_variable[width](
                addresses, strides, argument, index
            )
            stack[sp] = value * value * value
            sp += 1
        elif op == 24:
            var value = load_variable[width](
                addresses, strides, argument, index
            )
            stack[sp] = value * value * value * value
            sp += 1
        elif op == 25:
            var value = load_variable[width](
                addresses, strides, argument, index
            )
            stack[sp] = (
                value * constants[argc] + constants[argc + 1]
            )
            sp += 1
        elif op == 0:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].le(stack[sp]).select(1.0, 0.0)
        elif op == 1:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].ge(stack[sp]).select(1.0, 0.0)
        elif op == 2:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].eq(stack[sp]).select(0.0, 1.0)
        elif op == 3:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].eq(stack[sp]).select(1.0, 0.0)
        elif op == 4:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].lt(stack[sp]).select(1.0, 0.0)
        elif op == 5:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].gt(stack[sp]).select(1.0, 0.0)
        elif op == 6:
            sp -= 1
            stack[sp - 1] += stack[sp]
        elif op == 7:
            sp -= 1
            stack[sp - 1] -= stack[sp]
        elif op == 8:
            sp -= 1
            stack[sp - 1] *= stack[sp]
        elif op == 9:
            sp -= 1
            stack[sp - 1] /= stack[sp]
        elif op == 10:
            sp -= 1
            stack[sp - 1] = pow(stack[sp - 1], stack[sp])
        elif op == 11:
            sp -= 1
            var either_zero = stack[sp - 1].eq(0.0) | stack[sp].eq(0.0)
            stack[sp - 1] = either_zero.select(0.0, 1.0)
        elif op == 12:
            sp -= 1
            var both_zero = stack[sp - 1].eq(0.0) & stack[sp].eq(0.0)
            stack[sp - 1] = both_zero.select(0.0, 1.0)
        elif op == 13:
            sp -= 1
            stack[sp - 1] = stack[sp]
            store_variable[width](
                addresses, strides, argument, index, stack[sp - 1]
            )
        elif op == 30:
            stack[sp - 1] = -stack[sp - 1]
        elif op == 31:
            pass
        elif op == 40:
            stack[sp - 1] = sin(stack[sp - 1])
        elif op == 41:
            stack[sp - 1] = cos(stack[sp - 1])
        elif op == 42:
            stack[sp - 1] = tan(stack[sp - 1])
        elif op == 43:
            stack[sp - 1] = 1.0 / tan(stack[sp - 1])
        elif op == 44:
            stack[sp - 1] = asin(stack[sp - 1])
        elif op == 45:
            stack[sp - 1] = acos(stack[sp - 1])
        elif op == 46:
            stack[sp - 1] = atan(stack[sp - 1])
        elif op == 47:
            sp -= 1
            stack[sp - 1] = atan2(stack[sp - 1], stack[sp])
        elif op == 48:
            stack[sp - 1] = sinh(stack[sp - 1])
        elif op == 49:
            stack[sp - 1] = cosh(stack[sp - 1])
        elif op == 50:
            stack[sp - 1] = tanh(stack[sp - 1])
        elif op == 51:
            stack[sp - 1] = 1.0 / tanh(stack[sp - 1])
        elif op == 52:
            var value = stack[sp - 1]
            stack[sp - 1] = log(value + sqrt(value * value + 1.0))
        elif op == 53:
            var value = stack[sp - 1]
            stack[sp - 1] = log(value + sqrt(value * value - 1.0))
        elif op == 54:
            var value = stack[sp - 1]
            stack[sp - 1] = 0.5 * log((1.0 + value) / (1.0 - value))
        elif op == 55:
            stack[sp - 1] = log(stack[sp - 1]) / log(2.0)
        elif op == 56:
            stack[sp - 1] = log10(stack[sp - 1])
        elif op == 57:
            stack[sp - 1] = log(stack[sp - 1])
        elif op == 58:
            stack[sp - 1] = exp(stack[sp - 1])
        elif op == 59:
            stack[sp - 1] = sqrt(stack[sp - 1])
        elif op == 60:
            var value = stack[sp - 1]
            stack[sp - 1] = value.lt(0.0).select(
                -1.0, value.gt(0.0).select(1.0, 0.0)
            )
        elif op == 61:
            stack[sp - 1] = floor(stack[sp - 1] + 0.5)
        elif op == 62:
            stack[sp - 1] = abs(stack[sp - 1])
        elif op == 63:
            sp -= argc - 1
            var total = SIMD[DType.float64, width](0.0)
            for i in range(argc):
                total += stack[sp - 1 + i]
            stack[sp - 1] = total
        elif op == 64:
            sp -= argc - 1
            var total = SIMD[DType.float64, width](0.0)
            for i in range(argc):
                total += stack[sp - 1 + i]
            stack[sp - 1] = total / Float64(argc)
        elif op == 65:
            sp -= argc - 1
            var result = stack[sp - 1]
            for i in range(1, argc):
                var candidate = stack[sp - 1 + i]
                result = candidate.lt(result).select(candidate, result)
            stack[sp - 1] = result
        elif op == 66:
            sp -= argc - 1
            var result = stack[sp - 1]
            for i in range(1, argc):
                var candidate = stack[sp - 1 + i]
                result = candidate.gt(result).select(candidate, result)
            stack[sp - 1] = result
        elif op == 70:
            sp -= 2
            stack[sp - 1] = stack[sp - 1].eq(0.0).select(
                stack[sp + 1], stack[sp]
            )
        elif op == 71:
            sp -= 1
            stack[sp - 1] = stack[sp]
    return stack[0]

@export("mmup_evaluate_f64")
def mmup_evaluate_f64(
    code_addr: Int,
    code_count: Int,
    constants_addr: Int,
    addresses_addr: Int,
    strides_addr: Int,
    destination_addr: Int,
    n: Int,
    requested_workers: Int,
) abi("C") -> Int:
    if code_count < 1 or code_count > 256 or n < 0:
        return 1
    if n == 0:
        return 0
    var code = I64Ptr(unsafe_from_address=code_addr)
    var constants = F64Ptr(unsafe_from_address=constants_addr)
    var addresses = I64Ptr(unsafe_from_address=addresses_addr)
    var strides = I64Ptr(unsafe_from_address=strides_addr)
    var destination = F64Ptr(unsafe_from_address=destination_addr)
    var fast_path = 0
    var fast_a = 0
    var fast_b = 0
    var fast_c = 0
    var fast_constant = 0
    if (
        code_count == 5
        and code[0] == 20
        and code[3] == 20
        and code[6] == 8
        and code[9] == 20
        and code[12] == 6
    ):
        fast_path = 1
        fast_a = Int(code[1])
        fast_b = Int(code[4])
        fast_c = Int(code[10])
    elif (
        code_count == 14
        and code[0] == 20
        and code[3] == 21
        and code[6] == 5
        and code[9] == 20
        and code[12] == 20
        and code[15] == 8
        and code[18] == 20
        and code[21] == 6
        and code[24] == 20
        and code[27] == 20
        and code[30] == 9
        and code[33] == 20
        and code[36] == 7
        and code[39] == 70
        and code[10] == code[25]
        and code[13] == code[28]
        and code[1] == code[19]
        and code[1] == code[34]
    ):
        fast_path = 2
        fast_c = Int(code[1])
        fast_constant = Int(code[4])
        fast_a = Int(code[10])
        fast_b = Int(code[13])
    var workers = min(requested_workers, num_physical_cores())
    var work_per_element = code_count
    for pc in range(code_count):
        var op = Int(code[pc * 3])
        var argc = Int(code[pc * 3 + 2])
        if op == 10 or (op >= 40 and op <= 58):
            work_per_element += 16
        elif op == 9 or op == 59:
            work_per_element += 4
        elif op >= 63 and op <= 66:
            work_per_element += argc
    if (
        (fast_path > 0 and n < FAST_PARALLEL_ELEMENTS)
        or (
            fast_path == 0
            and n < PARALLEL_WORK // max(work_per_element, 1)
        )
    ):
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + W <= end:
            if fast_path == 1:
                destination.store(
                    i,
                    execute_multiply_add[W](
                        addresses,
                        strides,
                        fast_a,
                        fast_b,
                        fast_c,
                        i,
                    ),
                )
            elif fast_path == 2:
                destination.store(
                    i,
                    execute_conditional_multiply_divide[W](
                        constants,
                        addresses,
                        strides,
                        fast_a,
                        fast_b,
                        fast_c,
                        fast_constant,
                        i,
                    ),
                )
            else:
                destination.store(
                    i,
                    execute_chunk[W](
                        code, code_count, constants, addresses, strides, i
                    ),
                )
            i += W
        while i < end:
            if fast_path == 1:
                destination[i] = execute_multiply_add[1](
                    addresses,
                    strides,
                    fast_a,
                    fast_b,
                    fast_c,
                    i,
                )[0]
            elif fast_path == 2:
                destination[i] = execute_conditional_multiply_divide[1](
                    constants,
                    addresses,
                    strides,
                    fast_a,
                    fast_b,
                    fast_c,
                    fast_constant,
                    i,
                )[0]
            else:
                destination[i] = execute_chunk[1](
                    code, code_count, constants, addresses, strides, i
                )[0]
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    return 0
