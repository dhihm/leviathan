import Foundation
import Metal

/// Errors thrown by the router residency helpers. All cases are `Equatable`
/// so tests can assert on the exact category.
public enum RouterResidencyError: Error, Equatable {
    case divisorNotAllowed(Int)
    case nonPositiveThreadExecutionWidth(Int)
    case nonPositiveMaxThreads(Int)
    case nonPositiveRows(Int)
    case nonPositiveThreadCount(Int)
    case negativeElementCount(Int)
    case byteSpanOverflow
    case gridThreadsOverflow
    case negativeScaleOffset(Int)
    case misalignedScaleOffset(Int)
    case pipelineNotFound(String)
}

/// Kernel + provenance for moving ONLY the second per-element router scale
/// multiply (`h[i] *= Float(routerScale[i])`) from the CPU onto the GPU.
///
/// The original decode path performs two ordered float rounds per element:
///   1. `g4_scale`:         `h[i] *= Float(1 / sqrt(Double(hidden)))`  (GPU)
///   2. CPU element loop:   `h[i] *= Float(routerScale[i])`           (CPU)
/// This kernel replaces round 2 only. Round 1 stays in its own separate,
/// ordered dispatch on the original pipeline so the two factors are never
/// fused or reassociated — each element keeps exactly two float multiply
/// rounds, preserving the original accumulation semantics.
public enum RouterResidencyKernel {

    /// Name of the Metal compute kernel inside `source`.
    public static let kernelName = "g4_router_scale"

    /// Name of the resident-expert output-reset kernel inside `source`:
    /// a bit-preserving uint32 zero (+0.0 bits, no arithmetic) exactly
    /// matching the CPU `memset` pattern it replaces.
    public static let zeroKernelName = "g4_router_resident_zero"

    /// Bump whenever `source` changes. The runtime must bind
    /// kernelName/version/sourceSha256/compile-option choice explicitly
    /// before integration. v2 adds g4_router_resident_zero. v3 computes
    /// the second-scale multiply bit-exactly from raw IEEE-754 bits when a
    /// subnormal operand/result, underflow, or signed zero is possible,
    /// because M1Max hardware multiplies flush subnormal outputs to zero
    /// even under precise compile options; normal-range products keep the
    /// ordinary hardware multiply.
    public static let version: Int = 3

    public static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    // Source version 3: subnormal results now round directly from the
    // exact 48-bit product with a single round-to-nearest ties-to-even,
    // fixing a double-rounding divergence from the CPU multiply.
    // Second router scale multiply only (x[i] *= s[i], in place).
    // The first factor is applied by the original g4_scale kernel in a
    // separate, ordered dispatch; the two multiplies are never fused or
    // reassociated.
    //
    // M1Max GPUs flush subnormal results of the hardware float multiply
    // to zero even with precise compile options, so any case that can
    // produce or consume a subnormal (or a signed zero) is computed from
    // the raw IEEE-754 bits: a 24x24 -> ulong significand product,
    // exponent derivation, and round-to-nearest ties-to-even, reassembled
    // via as_type<float>(uint) so no floating arithmetic ever touches a
    // reconstructed subnormal. The routing decision is made from the
    // exponent bits BEFORE the multiply so FTZ cannot lose information.
    // Everything the hardware multiply already answers exactly — inf/NaN
    // propagation, and operand pairs whose derived biased result exponent
    // is > 0 with both exponent fields nonzero (result at or above the
    // min-normal binade, never subnormal after rounding) — keeps the
    // ordinary hardware multiply.
    static inline uint g4_mul_bits(uint ab, uint sb) {
        const int ea = int((ab >> 23) & 0xFFu);
        const int es = int((sb >> 23) & 0xFFu);
        const int en = ea + es - 127;  // biased exponent of the leading result bits
        const bool special = (ab & 0x7F800000u) == 0x7F800000u ||
                             (sb & 0x7F800000u) == 0x7F800000u;
        if (special || (en > 0 && ea != 0 && es != 0)) {
            return as_type<uint>(as_type<float>(ab) * as_type<float>(sb));
        }
        const uint sign = (ab ^ sb) & 0x80000000u;
        const uint ma = ab & 0x7FFFFFu;
        const uint ms = sb & 0x7FFFFFu;
        if ((ea == 0 && ma == 0) || (es == 0 && ms == 0)) {
            return sign;  // signed zero: exact xor of the operand signs
        }
        const int wa = (ea == 0) ? -149 : ea - 150;
        const int ws = (es == 0) ? -149 : es - 150;
        const ulong p = ulong(ea == 0 ? ma : (ma | 0x800000u)) *
                        ulong(es == 0 ? ms : (ms | 0x800000u));
        const int bitlen = 64 - clz(p);
        int e = wa + ws + bitlen - 1;  // unbiased floor-log2 of the exact product
        if (e < -126) {
            // Subnormal / underflow result: round DIRECTLY from the exact
            // 48-bit product to m in units of 2^-149. Rounding p to a 24-bit
            // significand first and then to subnormal precision would round
            // twice and can differ from a single round-to-nearest
            // ties-to-even, so this branch runs before any 24-bit rounding.
            // p is at most 48 bits, so sh >= 64 means the product is below
            // half the smallest subnormal and rounds to the signed zero.
            // m == 0x800000 lands exactly on the min-normal encoding.
            const int q = wa + ws + 149;
            if (q >= 0) {
                return sign | uint(p << q);  // safe: e < -126 keeps m <= 23 bits
            }
            const int sh = -q;
            if (sh >= 64) { return sign; }
            uint m = uint(p >> sh);
            const ulong rem = p & ((1ul << sh) - 1ul);
            const ulong midpoint = 1ul << (sh - 1);
            if (rem > midpoint || (rem == midpoint && (m & 1u) != 0)) { m += 1u; }
            return sign | m;
        }
        uint sig;
        if (bitlen <= 24) {
            sig = uint(p) << (24 - bitlen);  // exact, no rounding bits
        } else {
            const uint sh = uint(bitlen - 24);
            const ulong rem = p & ((1ul << sh) - 1ul);
            sig = uint(p >> sh);
            const ulong midpoint = 1ul << (sh - 1);
            if (rem > midpoint || (rem == midpoint && (sig & 1u) != 0)) { sig += 1u; }
        }
        if (sig == 0x1000000u) { sig = 0x800000u; e += 1; }  // carry into next binade
        if (e > 127) { return sign | 0x7F800000u; }          // overflow to inf
        return sign | (uint(e + 127) << 23) | (sig & 0x7FFFFFu);  // normal result
    }

    kernel void g4_router_scale(constant const uint& n [[buffer(0)]],
                                device float* x [[buffer(1)]],
                                device const uchar* wbase [[buffer(2)]],
                                constant const ulong& scaleOffset [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        device const float* s =
            reinterpret_cast<device const float*>(wbase + scaleOffset);
        x[gid] = as_type<float>(g4_mul_bits(as_type<uint>(x[gid]),
                                            as_type<uint>(s[gid])));
    }

    // Bit-preserving resident-expert output reset (y[gid] = 0u): writes
    // the exact +0.0 bit pattern of the original CPU memset with no
    // arithmetic, so it may replace that memset inside an ordered GPU
    // stream without changing any observable result.
    kernel void g4_router_resident_zero(constant const uint& n [[buffer(0)]],
                                        device uint* y [[buffer(1)]],
                                        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        y[gid] = 0u;
    }
    """

    /// SHA-256 of `source` as lowercase hex, for explicit provenance binding.
    public static var sourceSha256: String {
        SHA256.hex(SHA256.hash(Array(source.utf8)))
    }

    /// Launch geometry divisors the runtime may request. `0` means keep
    /// the original single-group geometry (the derivation returns nil).
    public static let allowedLaunchDivisors: [Int] = [0, 1, 2, 4, 8]

    /// Compile options that disable fast-math transformations. On SDKs
    /// exposing `mathMode`, `.safe` prevents transformations that affect
    /// results (per Apple's docs this is not a promise about all GPU
    /// denormal behavior — that must be tested on actual hardware);
    /// otherwise fall back to `fastMathEnabled = false`.
    public static func makeCompileOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        return options
    }

    /// Compile `source` with precise options and build the pipeline state
    /// for the named kernel (defaults to the second-scale kernel).
    public static func makePipeline(name: String = kernelName,
                                    device: MTLDevice) throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(source: source, options: makeCompileOptions())
        guard let function = library.makeFunction(name: name) else {
            throw RouterResidencyError.pipelineNotFound(name)
        }
        return try device.makeComputePipelineState(function: function)
    }
}

/// Pure, checked geometry helpers for the scalar router F32 matvec
/// (`g4_f32_matvec`, one thread per output row). No GPU-name heuristics:
/// every limit comes from the actual pipeline/device capabilities.
public enum RouterResidencyGeometry {

    /// Threads per threadgroup for the scalar router matvec.
    ///
    /// divisor == 0 means "original geometry" and returns nil so the caller
    /// dispatches with its existing single-group shape. For divisor > 0 the
    /// thread count is `threadExecutionWidth / divisor`, bounded by the
    /// pipeline's `maxTotalThreadsPerThreadgroup` and by `outputRows`
    /// (never launch more threads than rows). All inputs must be positive.
    public static func scalarRouterThreads(divisor: Int,
                                           threadExecutionWidth: Int,
                                           maxTotalThreadsPerThreadgroup: Int,
                                           outputRows: Int) throws -> Int? {
        guard RouterResidencyKernel.allowedLaunchDivisors.contains(divisor) else {
            throw RouterResidencyError.divisorNotAllowed(divisor)
        }
        guard threadExecutionWidth > 0 else {
            throw RouterResidencyError.nonPositiveThreadExecutionWidth(threadExecutionWidth)
        }
        guard maxTotalThreadsPerThreadgroup > 0 else {
            throw RouterResidencyError.nonPositiveMaxThreads(maxTotalThreadsPerThreadgroup)
        }
        guard outputRows > 0 else {
            throw RouterResidencyError.nonPositiveRows(outputRows)
        }
        if divisor == 0 { return nil }
        let threads = threadExecutionWidth / divisor
        guard threads > 0 else {
            throw RouterResidencyError.nonPositiveThreadCount(threads)
        }
        return min(threads, maxTotalThreadsPerThreadgroup, outputRows)
    }

    /// Overflow-safe ceil(rows / threadsPerGroup). Never computes
    /// `rows + threadsPerGroup - 1`, which can itself overflow.
    public static func groupCount(rows: Int, threadsPerGroup: Int) throws -> Int {
        guard threadsPerGroup > 0 else {
            throw RouterResidencyError.nonPositiveThreadCount(threadsPerGroup)
        }
        guard rows > 0 else {
            throw RouterResidencyError.nonPositiveRows(rows)
        }
        let q = rows / threadsPerGroup
        return rows % threadsPerGroup == 0 ? q : q + 1
    }

    /// Total threads for a padded grid (`groups * threadsPerGroup`). The
    /// extra tail threads are guarded by the kernels' `gid >= n` check.
    public static func gridThreads(rows: Int, threadsPerGroup: Int) throws -> Int {
        let groups = try groupCount(rows: rows, threadsPerGroup: threadsPerGroup)
        let (product, overflow) = groups.multipliedReportingOverflow(by: threadsPerGroup)
        guard !overflow else { throw RouterResidencyError.gridThreadsOverflow }
        return product
    }

    /// Byte span of `elementCount` floats; throws instead of overflowing.
    public static func byteSpan(elementCount: Int) throws -> Int {
        guard elementCount >= 0 else {
            throw RouterResidencyError.negativeElementCount(elementCount)
        }
        let (product, overflow) =
            elementCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        guard !overflow else { throw RouterResidencyError.byteSpanOverflow }
        return product
    }

    /// Validate a scale tensor byte offset inside the weight base buffer.
    /// Must be non-negative and float-aligned; returns the offset on success.
    public static func checkedScaleOffset(_ offset: Int) throws -> Int {
        guard offset >= 0 else { throw RouterResidencyError.negativeScaleOffset(offset) }
        guard offset % MemoryLayout<Float>.alignment == 0 else {
            throw RouterResidencyError.misalignedScaleOffset(offset)
        }
        return offset
    }
}
