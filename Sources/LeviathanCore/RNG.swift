/// Deterministic SplitMix64 pseudo-random generator. Used to synthesize
/// reproducible float32 weights and inputs for the synthetic network, so
/// artifacts and test vectors are byte-identical across runs and machines.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform float in [-1, 1).
    public mutating func nextUniform() -> Float {
        let u = next() >> 11  // 53 bits of entropy is plenty for float32
        return Float(Double(u) / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }
}
