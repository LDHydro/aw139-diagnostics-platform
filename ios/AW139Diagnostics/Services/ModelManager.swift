import Foundation

/// Owns the on-device models and tracks their load state for the UI.
///
/// Two modes:
/// - **Demo mode** (no models): uses `HashingEmbedder` for retrieval and the
///   `StubGenerator` for output. Needs no model files — for smoke-testing the UI
///   and retrieval with the bundled synthetic corpus.
/// - **Real mode**: MLX embedder + MLX LLM loaded from Documents/models.
@MainActor
final class ModelManager: ObservableObject {

    enum LoadState: Equatable {
        case idle
        case loading(Double)   // 0...1
        case ready
        case failed(String)
    }

    @Published var llmState: LoadState = .idle
    @Published var embedderState: LoadState = .idle

    /// When true, run fully with no model files (hashing retrieval + stub answer).
    @Published var demoMode = false {
        didSet {
            // Reset load state so the UI reflects the new mode.
            embedderState = demoMode ? .ready : .idle
            llmState = demoMode ? .ready : .idle
        }
    }

    /// Optional Hugging Face fallback IDs, used only if no file-transferred model
    /// is present and the device has internet for a one-time download.
    /// Set to nil to enforce strictly offline operation.
    var llmFallbackHubID: String? = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    private let mlxEmbedder: Embedder = MLXEmbedder()
    private let hashingEmbedder: Embedder = HashingEmbedder()
    private var mlxGenerator: TextGenerator?
    private let stub = StubGenerator()

    /// The embedder used by the pipeline for the current mode.
    var embedder: Embedder { demoMode ? hashingEmbedder : mlxEmbedder }

    /// The generator used by the pipeline: the real MLX model once loaded,
    /// otherwise the stub (also used in demo mode).
    var activeGenerator: TextGenerator { demoMode ? stub : (mlxGenerator ?? stub) }

    var llmReady: Bool { if case .ready = llmState { return true } else { return false } }
    var embedderReady: Bool { if case .ready = embedderState { return true } else { return false } }

    func loadEmbedder() async {
        if demoMode { embedderState = .ready; return }
        embedderState = .loading(0)
        do {
            try await mlxEmbedder.load { [weak self] p in self?.embedderState = .loading(p) }
            embedderState = .ready
        } catch {
            embedderState = .failed(error.localizedDescription)
        }
    }

    func loadLLM() async {
        if demoMode { llmState = .ready; return }
        llmState = .loading(0)
        let gen = MLXGenerator(fallbackHubID: llmFallbackHubID)
        do {
            try await gen.load { [weak self] p in self?.llmState = .loading(p) }
            mlxGenerator = gen
            llmState = .ready
        } catch {
            mlxGenerator = nil
            llmState = .failed(error.localizedDescription)
        }
    }

    /// Load everything needed for a fully offline run.
    func loadAll() async {
        AppPaths.ensureDirectories()
        await loadEmbedder()
        await loadLLM()
    }
}
