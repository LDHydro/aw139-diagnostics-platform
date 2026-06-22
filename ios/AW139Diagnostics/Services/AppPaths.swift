import Foundation

/// Locations for the file-transferred offline assets.
///
/// Everything the app needs to run offline lives in the app's Documents
/// directory, which is exposed over Finder file sharing / the Files app
/// (see Info.plist `UIFileSharingEnabled`). To update content you simply
/// drop new files in here — no rebuild required.
///
///   Documents/
///     corpus.json                     <- offline document corpus (+ embeddings)
///     models/
///       llm/                          <- MLX LLM weights (config.json, *.safetensors, tokenizer)
///       embedder/                     <- MLX embedding model weights
enum AppPaths {

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static var corpusFile: URL {
        documents.appendingPathComponent("corpus.json")
    }

    static var modelsDirectory: URL {
        documents.appendingPathComponent("models", isDirectory: true)
    }

    static var llmModelDirectory: URL {
        modelsDirectory.appendingPathComponent("llm", isDirectory: true)
    }

    static var embedderModelDirectory: URL {
        modelsDirectory.appendingPathComponent("embedder", isDirectory: true)
    }

    static func ensureDirectories() {
        let fm = FileManager.default
        for dir in [modelsDirectory, llmModelDirectory, embedderModelDirectory] {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    /// A model directory is considered "present" if it contains a config.json.
    static func modelPresent(at dir: URL) -> Bool {
        exists(dir.appendingPathComponent("config.json"))
    }
}
