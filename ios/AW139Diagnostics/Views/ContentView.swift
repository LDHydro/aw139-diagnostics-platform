import SwiftUI

/// iPad split layout: diagnostic form on the left, result on the right.
struct ContentView: View {
    @EnvironmentObject private var vm: DiagnosticViewModel
    @EnvironmentObject private var corpus: CorpusStore
    @State private var showSetup = false

    var body: some View {
        NavigationSplitView {
            DiagnosticFormView(vm: vm)
                .navigationTitle("AW139 Diagnostics")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button { showSetup = true } label: {
                            Image(systemName: "gearshape")
                        }
                        .accessibilityLabel("Setup")
                    }
                }
        } detail: {
            ResultView(vm: vm)
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(isPresented: $showSetup) {
            SetupView()
        }
        .onAppear {
            corpus.loadIfNeeded()
            if !corpus.isLoaded { showSetup = true }
        }
    }
}
