import SwiftUI

/// Re-creates the web DiagnosticForm: aircraft model, serial number, ATA code /
/// training material, task type, and the free-text problem description.
struct DiagnosticFormView: View {
    @ObservedObject var vm: DiagnosticViewModel
    @EnvironmentObject private var corpus: CorpusStore

    var body: some View {
        Form {
            if !corpus.isLoaded {
                Section {
                    Label(corpus.loadError ?? "Corpus not loaded.",
                          systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                } header: {
                    Text("Offline corpus")
                }
            } else {
                Section {
                    Label("\(corpus.count) documents indexed (\(corpus.embeddingDimension)-dim)",
                          systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                }
            }

            Section("Aircraft") {
                TextField("Aircraft model", text: $vm.request.aircraftModel)
                TextField("Serial number (e.g., 41287)", text: $vm.request.serialNumber)
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
            }

            Section("System") {
                Picker("ATA / Training material", selection: $vm.request.ata) {
                    Text("Select…").tag("")
                    ForEach(ATACatalog.trainingMaterials) { e in
                        Text(e.system).tag(e.code)
                    }
                    ForEach(ATACatalog.ataCodes) { e in
                        Text(e.label).tag(e.code)
                    }
                }

                Picker("Task type", selection: $vm.request.taskType) {
                    ForEach(TaskType.allCases) { t in
                        Text(t.label).tag(t)
                    }
                }
                .pickerStyle(.segmented)
            }

            Section("Problem description") {
                TextEditor(text: $vm.request.problemDescription)
                    .frame(minHeight: 140)
                    .overlay(alignment: .topLeading) {
                        if vm.request.problemDescription.isEmpty {
                            Text("Describe the fault, CAS message, or task…")
                                .foregroundStyle(.secondary)
                                .padding(.top, 8)
                                .allowsHitTesting(false)
                        }
                    }
            }

            Section {
                Button(action: { Task { await vm.run() } }) {
                    HStack {
                        if vm.isRunning {
                            ProgressView().padding(.trailing, 4)
                        }
                        Text(vm.isRunning ? "Analyzing…" : "Run Diagnostic")
                            .frame(maxWidth: .infinity)
                            .fontWeight(.semibold)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!vm.canRun)
            }

            Section {
                Text("All analysis runs on-device. Always verify against the applicable AMM/IETP. This tool does not replace approved maintenance data.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
