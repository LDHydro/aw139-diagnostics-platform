import SwiftUI

struct ResultView: View {
    @ObservedObject var vm: DiagnosticViewModel
    @EnvironmentObject private var models: ModelManager

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                if case .loading(let p) = models.llmState {
                    loadingBanner("Loading language model…", progress: p)
                }
                if case .loading(let p) = models.embedderState {
                    loadingBanner("Loading embedding model…", progress: p)
                }

                if let error = vm.errorMessage {
                    Label(error, systemImage: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                }

                if let result = vm.result {
                    queryKindBadge(result.kind)
                }

                if !vm.streamingAnswer.isEmpty {
                    Text(vm.streamingAnswer)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if vm.isRunning {
                    HStack { ProgressView(); Text("Retrieving documentation…") }
                        .foregroundStyle(.secondary)
                } else if vm.result == nil && vm.errorMessage == nil {
                    ContentUnavailableView(
                        "No diagnostic yet",
                        systemImage: "wrench.and.screwdriver",
                        description: Text("Fill in the form and tap Run Diagnostic.")
                    )
                    .padding(.top, 60)
                }

                if let result = vm.result, !result.retrieved.isEmpty {
                    sourcesSection(result.retrieved)
                }
            }
            .padding()
        }
        .navigationTitle("Diagnosis")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func loadingBanner(_ title: String, progress: Double) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.medium))
            ProgressView(value: progress)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func queryKindBadge(_ kind: QueryKind) -> some View {
        let (label, color): (String, Color) = {
            switch kind {
            case .faultCode: return ("Fault Isolation", .red)
            case .calibration: return ("Calibration / Adjustment", .orange)
            case .procedure: return ("Procedure / Task", .blue)
            case .electrical: return ("Electrical Diagnostic", .purple)
            case .general: return ("General Maintenance", .gray)
            }
        }()
        return Text(label)
            .font(.caption.weight(.bold))
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private func sourcesSection(_ docs: [RetrievedDocument]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider().padding(.vertical, 4)
            Text("Source Documents")
                .font(.headline)
            ForEach(docs) { doc in
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(doc.dmcCode.isEmpty ? doc.ataIdentifier : doc.dmcCode)
                            .font(.system(.caption, design: .monospaced).weight(.semibold))
                        Spacer()
                        Text(String(format: "%.0f%%", doc.similarity * 100))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Text(doc.document.text.prefix(160) + "…")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }
}
