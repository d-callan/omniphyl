import SwiftUI

@main
struct HyPhyStudioApp: App {
    public init() {
        if CommandLine.arguments.contains("--verify") || CommandLine.arguments.contains("--test") {
            VerificationHarness.run()
            exit(0)
        }
    }

    var body: some Scene {
        DocumentGroup(newDocument: HyPhyProjectDocument()) { file in
            MainView(document: file.$document)
        }
        .commands {
            SidebarCommands()
            CommandGroup(replacing: .appInfo) {
                Button("About HyPhy Studio") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            NSApplication.AboutPanelOptionKey.applicationName: "HyPhy Studio",
                            NSApplication.AboutPanelOptionKey.version: "3.0.0",
                            NSApplication.AboutPanelOptionKey.credits: NSAttributedString(string: "HyPhy Development Team • Sergei L. Kosakovsky Pond")
                        ]
                    )
                }
            }
            CommandMenu("Analysis") {
                Button("Run Selected Method...") {
                    // Triggers run sheet
                }
                .keyboardShortcut("r", modifiers: .command)

                Divider()

                Button("Test All Branches") {
                    // Set all branches tested
                }
                Button("Clear Branch Selection") {
                    // Clear branch selection
                }
            }
        }
    }
}
