import Foundation
import MLX
import MLXEmbedders

/// On-device text embeddings via MLX.
///
/// IMPORTANT: the corpus (corpus.json) must be embedded with the SAME model.
/// tools/build_offline_corpus.py uses BAAI/bge-small-en-v1.5 by default, which
/// matches `ModelConfiguration.bge_small` here. If you change one side, change
/// both — otherwise query and document vectors live in different spaces and
/// retrieval quality collapses.
///
/// The model is read from `Documents/models/embedder` if present, otherwise the
/// preset (downloaded once from Hugging Face) is used.
///
/// NOTE ON MLX VERSIONS: the `loadModelContainer` / `perform` encode block below
/// follows the MLXEmbedders example API. If signatures differ in your resolved
/// package version, this is the one place to adjust.
actor MLXEmbedder: Embedder {

    private var container: ModelContainer?
    private let preset: ModelConfiguration

    init(preset: ModelConfiguration = .bge_small) {
        self.preset = preset
    }

    func load(progress: @escaping (Double) -> Void) async throws {
        if container != nil { return }
        let configuration: ModelConfiguration
        if AppPaths.modelPresent(at: AppPaths.embedderModelDirectory) {
            configuration = ModelConfiguration(directory: AppPaths.embedderModelDirectory)
        } else {
            configuration = preset
        }
        container = try await MLXEmbedders.loadModelContainer(configuration: configuration)
        await MainActor.run { progress(1) }
    }

    func embed(_ text: String) async throws -> [Float] {
        guard let container else { throw InferenceError.notLoaded }

        let vector: [Float] = await container.perform { (model, tokenizer, pooling) in
            var tokens = tokenizer.encode(text: text, addSpecialTokens: true)
            // bge models are trained at 512 tokens; truncate longer queries.
            if tokens.count > 512 { tokens = Array(tokens.prefix(512)) }

            let ids = MLXArray(tokens).reshaped([1, tokens.count])
            let tokenTypes = MLXArray.zeros(like: ids)
            let mask = MLXArray.ones(like: ids)

            let output = pooling(
                model(ids, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                normalize: true,
                applyLayerNorm: true
            )
            output.eval()
            return output.asArray(Float.self)
        }
        return vector
    }
}
