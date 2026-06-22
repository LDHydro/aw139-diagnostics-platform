import SwiftUI

@main
struct AW139DiagnosticsApp: App {
    @StateObject private var vm = DiagnosticViewModel()

    init() {
        AppPaths.ensureDirectories()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(vm)
                .environmentObject(vm.corpus)
                .environmentObject(vm.models)
        }
    }
}
