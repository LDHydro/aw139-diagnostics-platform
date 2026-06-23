import Foundation

/// A deterministic, dependency-free embedder for the no-models **demo/smoke
/// test** path. It needs no model files and no internet: it hashes word tokens
/// into a fixed-size bag-of-words vector (FNV-1a → bucket) and L2-normalizes.
///
/// The SAME algorithm is implemented in tools/build_demo_corpus.py, so a corpus
/// built there lives in the same vector space as queries embedded here, and
/// cosine retrieval returns meaningful results — enough to exercise the full UI
/// and retrieval pipeline before wiring up real MLX models.
///
/// This is NOT a semantic embedder; it only matches on shared keywords. Use real
/// MLX embeddings (MLXEmbedder) for production retrieval.
struct HashingEmbedder: Embedder {

    let dimension: Int

    init(dimension: Int = 384) {
        self.dimension = dimension
    }

    func load(progress: @escaping (Double) -> Void) async throws { progress(1) }

    func embed(_ text: String) async throws -> [Float] {
        Self.vector(for: text, dimension: dimension)
    }

    // MARK: - Shared algorithm (must match build_demo_corpus.py)

    static func vector(for text: String, dimension: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dimension)
        for token in tokens(text) {
            let h = fnv1a32(token)
            v[Int(h % UInt32(dimension))] += 1
        }
        // L2 normalize
        var norm: Float = 0
        for x in v { norm += x * x }
        norm = norm.squareRoot()
        if norm > 0 {
            for i in v.indices { v[i] /= norm }
        }
        return v
    }

    /// Lowercase, split on non-alphanumerics, keep tokens of length >= 2.
    static func tokens(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !($0.isLetter || $0.isNumber) })
            .map(String.init)
            .filter { $0.count >= 2 }
    }

    /// FNV-1a 32-bit over the token's UTF-8 bytes.
    static func fnv1a32(_ s: String) -> UInt32 {
        var hash: UInt32 = 2166136261
        for byte in s.utf8 {
            hash = (hash ^ UInt32(byte)) &* 16777619
        }
        return hash
    }
}
