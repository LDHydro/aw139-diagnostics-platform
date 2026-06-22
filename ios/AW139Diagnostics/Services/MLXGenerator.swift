import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// On-device LLM generation via Apple's MLX framework.
///
/// Loads a quantized instruction-tuned model (e.g. Llama-3.2-3B-Instruct-4bit
/// or Qwen2.5-3B-Instruct-4bit) entirely on the iPad's GPU. The model is read
/// from `Documents/models/llm` (file-transferred); if that directory is empty
/// the optional `fallbackHubID` is downloaded once from Hugging Face.
///
/// NOTE ON MLX VERSIONS: the `loadContainer` / `perform` / `generate` call sites
/// below target the mlx-swift-examples `main` API as of mid-2025. The MLX Swift
/// API still evolves; if the project fails to compile after resolving packages,
/// these three call sites are the only places likely to need a small signature
/// tweak to match the installed version.
actor MLXGenerator: TextGenerator {

    private var container: ModelContainer?
    private let fallbackHubID: String?
    private let maxTokens: Int

    init(fallbackHubID: String? = nil, maxTokens: Int = 2500) {
        self.fallbackHubID = fallbackHubID
        self.maxTokens = maxTokens
    }

    func load(progress: @escaping (Double) -> Void) async throws {
        if container != nil { return }

        // Limit MLX's GPU cache so a 3B model coexists with the corpus in RAM.
        MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)

        let configuration: ModelConfiguration
        if AppPaths.modelPresent(at: AppPaths.llmModelDirectory) {
            configuration = ModelConfiguration(directory: AppPaths.llmModelDirectory)
        } else if let hub = fallbackHubID {
            configuration = ModelConfiguration(id: hub)
        } else {
            throw InferenceError.modelNotFound(AppPaths.llmModelDirectory.path)
        }

        container = try await LLMModelFactory.shared.loadContainer(configuration: configuration) { p in
            Task { @MainActor in progress(p.fractionCompleted) }
        }
    }

    func generate(system: String,
                  user: String,
                  onToken: @escaping (String) -> Void) async throws -> String {
        guard let container else { throw InferenceError.notLoaded }

        let cappedTokens = maxTokens
        let result = try await container.perform { context -> GenerateResult in
            let input = try await context.processor.prepare(
                input: UserInput(messages: [
                    ["role": "system", "content": system],
                    ["role": "user", "content": user]
                ])
            )
            let params = GenerateParameters(temperature: 0.0)
            return try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context
            ) { tokens in
                let text = context.tokenizer.decode(tokens: tokens)
                Task { @MainActor in onToken(text) }
                return tokens.count >= cappedTokens ? .stop : .more
            }
        }
        return result.output
    }
}
