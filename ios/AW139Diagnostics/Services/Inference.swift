import Foundation

/// Produces an embedding vector for a query string (on-device).
protocol Embedder {
    /// Loads model weights into memory. `progress` is 0...1 while downloading/loading.
    func load(progress: @escaping (Double) -> Void) async throws
    func embed(_ text: String) async throws -> [Float]
}

/// Generates a maintenance answer from a system + user prompt (on-device LLM).
protocol TextGenerator {
    func load(progress: @escaping (Double) -> Void) async throws
    /// Streams generated tokens; `onToken` is called with the cumulative text.
    func generate(system: String,
                  user: String,
                  onToken: @escaping (String) -> Void) async throws -> String
}

enum InferenceError: LocalizedError {
    case modelNotFound(String)
    case notLoaded
    case dimensionMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let where_):
            return "Model files not found at \(where_). Transfer the model into the app's Documents/models folder."
        case .notLoaded:
            return "The model has not been loaded yet."
        case .dimensionMismatch(let expected, let got):
            return "Embedding dimension mismatch: corpus uses \(expected)-dim vectors but the query embedder produced \(got)-dim. Re-build the corpus with the same embedding model."
        }
    }
}

/// A deterministic, dependency-free fallback used when no on-device model has
/// been loaded yet. It lets the full retrieval pipeline and UI run (and is handy
/// for development), but it does NOT perform real semantic generation —
/// it surfaces the retrieved documentation so the screen is never empty.
struct StubGenerator: TextGenerator {
    func load(progress: @escaping (Double) -> Void) async throws { progress(1) }

    func generate(system: String,
                  user: String,
                  onToken: @escaping (String) -> Void) async throws -> String {
        let message = """
        ⚠️ No on-device LLM loaded.

        Retrieval ran successfully and the most relevant maintenance \
        documentation is shown below, but generating a synthesized answer \
        requires an MLX language model.

        Load a model from Settings (transfer it into Documents/models/llm) to \
        get full Level-D specialist answers.

        ── Retrieved documentation context ──
        \(user)
        """
        onToken(message)
        return message
    }
}
