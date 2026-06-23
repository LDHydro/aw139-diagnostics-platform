import SwiftUI

/// Shows the offline-asset status and how to load/update content by file transfer.
struct SetupView: View {
    @EnvironmentObject private var corpus: CorpusStore
    @EnvironmentObject private var models: ModelManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Toggle("Demo mode (no models)", isOn: $models.demoMode)
                } header: {
                    Text("Smoke Test")
                } footer: {
                    Text("Runs the full UI + retrieval pipeline with no model files, using a built-in hashing embedder. Pair it with the synthetic demo corpus (tools/build_demo_corpus.py). Answers are not AI-generated in this mode.")
                }

                Section("Document Corpus") {
                    statusRow(title: "corpus.json",
                              ok: corpus.isLoaded,
                              detail: corpus.isLoaded
                                ? "\(corpus.count) documents (\(corpus.embeddingDimension)-dim)"
                                : (corpus.loadError ?? "Not loaded"))
                    Button("Reload corpus") { corpus.load() }
                }

                Section("Embedding Model") {
                    modelStatusRow(state: models.embedderState,
                                   present: AppPaths.modelPresent(at: AppPaths.embedderModelDirectory))
                    Button("Load embedding model") { Task { await models.loadEmbedder() } }
                        .disabled(isLoading(models.embedderState))
                }

                Section("Language Model (LLM)") {
                    modelStatusRow(state: models.llmState,
                                   present: AppPaths.modelPresent(at: AppPaths.llmModelDirectory))
                    Button("Load language model") { Task { await models.loadLLM() } }
                        .disabled(isLoading(models.llmState))
                }

                Section("How to load / update content (offline)") {
                    instructions
                }

                Section {
                    Text("Documents path:\n\(AppPaths.documents.path)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("Setup")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var instructions: some View {
        VStack(alignment: .leading, spacing: 10) {
            stepText("1.", "Connect the iPad to a Mac. In Finder, select the iPad → Files tab → “AW139 Diagnostics”.")
            stepText("2.", "Drag corpus.json into the app's folder. Create a “models” folder containing “llm” and “embedder” subfolders with the MLX model files (config.json, *.safetensors, tokenizer files).")
            stepText("3.", "Reopen the app and tap the Load buttons above. No rebuild needed — this is how you push content updates.")
            Text("Tip: the embedding model used to build corpus.json MUST match the embedding model loaded here (default: BAAI/bge-small-en-v1.5). See tools/build_offline_corpus.py.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.top, 4)
        }
    }

    private func stepText(_ n: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(n).fontWeight(.bold)
            Text(text)
        }
        .font(.subheadline)
    }

    private func statusRow(title: String, ok: Bool, detail: String) -> some View {
        HStack {
            Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundStyle(ok ? .green : .orange)
            VStack(alignment: .leading) {
                Text(title)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func modelStatusRow(state: ModelManager.LoadState, present: Bool) -> some View {
        switch state {
        case .idle:
            return AnyView(statusRow(title: "Status",
                                     ok: false,
                                     detail: present ? "Files present — tap Load" : "Not loaded (will try Hugging Face fallback)"))
        case .loading(let p):
            return AnyView(HStack { ProgressView(value: p); Text("\(Int(p * 100))%") })
        case .ready:
            return AnyView(statusRow(title: "Status", ok: true, detail: "Loaded and ready"))
        case .failed(let msg):
            return AnyView(statusRow(title: "Failed", ok: false, detail: msg))
        }
    }

    private func isLoading(_ state: ModelManager.LoadState) -> Bool {
        if case .loading = state { return true }
        return false
    }
}
