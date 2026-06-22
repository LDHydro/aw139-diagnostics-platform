import Foundation
import SwiftUI

@MainActor
final class DiagnosticViewModel: ObservableObject {

    // Owned shared state, also injected into the environment by the App so
    // nested views observe their @Published changes directly.
    let corpus = CorpusStore()
    let models = ModelManager()

    @Published var request = DiagnosticRequest()
    @Published var streamingAnswer = ""
    @Published var result: DiagnosticResult?
    @Published var isRunning = false
    @Published var errorMessage: String?

    var canRun: Bool {
        !request.problemDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && corpus.isLoaded
            && !isRunning
    }

    func run() async {
        guard canRun else { return }
        isRunning = true
        errorMessage = nil
        streamingAnswer = ""
        result = nil

        // Lazily load models on first run.
        if !models.embedderReady { await models.loadEmbedder() }
        if !models.llmReady { await models.loadLLM() }

        let engine = DiagnosticEngine(
            corpus: corpus,
            embedder: models.embedder,
            generator: models.activeGenerator
        )

        do {
            let res = try await engine.run(request) { [weak self] text in
                self?.streamingAnswer = text
            }
            self.result = res
        } catch {
            self.errorMessage = error.localizedDescription
        }
        isRunning = false
    }

    func reset() {
        result = nil
        streamingAnswer = ""
        errorMessage = nil
    }
}
