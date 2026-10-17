/// Hard resource limits enforced during validation and execution.
/// These keep tensor allocation bounded on the target machine and make
/// rejection of oversized graphs explicit rather than accidental.
public enum Limits {
    // Tensors are fixed rank 2: row-major [batch, features] float32.
    // There is no rank knob to bound.
    /// Maximum number of elements in any single tensor.
    public static let maxTensorElements = 1_048_576  // 2^20 float32 = 4 MiB
    /// Maximum value of any single dimension.
    public static let maxDim = 4096
    /// Maximum number of ops in a graph.
    public static let maxOps = 64
    /// Maximum batch size accepted at run time.
    public static let maxBatch = 256
    /// Maximum total bytes of the weights blob in an artifact.
    public static let maxWeightBytes = 16 * 1024 * 1024
    /// Maximum total bytes of a whole artifact file (weights + header + digest).
    public static let maxArtifactBytes = maxWeightBytes + 1_048_576
    /// Maximum bytes accepted for a source graph JSON file.
    public static let maxGraphJSONBytes = 1_048_576
}
