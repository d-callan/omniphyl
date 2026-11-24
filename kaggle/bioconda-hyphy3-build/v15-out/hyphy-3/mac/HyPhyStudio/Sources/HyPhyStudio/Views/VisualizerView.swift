import SwiftUI
import WebKit

public final class VisualizerCoordinator: NSObject, WKScriptMessageHandler {
    public var onSiteSelected: ((Int) -> Void)?
    public var onBranchSelected: ((String) -> Void)?
    private var isReady = false
    private var pendingPayload: String?
    private weak var webView: WKWebView?

    func attach(webView: WKWebView) {
        self.webView = webView
    }

    public func loadResultJSON(_ jsonString: String) {
        if isReady {
            let b64 = Data(jsonString.utf8).base64EncodedString()
            let script = "if (window.loadHyPhyResult) { window.loadHyPhyResult(JSON.parse(atob('\(b64)'))); }"
            webView?.evaluateJavaScript(script)
        } else {
            pendingPayload = jsonString
        }
    }

    public func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "hyphyBridge",
              let dict = message.body as? [String: Any],
              let action = dict["action"] as? String else { return }

        switch action {
        case "viewerReady":
            isReady = true
            if let pending = pendingPayload {
                loadResultJSON(pending)
                pendingPayload = nil
            }
        case "siteSelected":
            if let site = dict["site"] as? Int {
                onSiteSelected?(site)
            }
        case "branchSelected":
            if let branch = dict["branch"] as? String {
                onBranchSelected?(branch)
            }
        default:
            break
        }
    }
}

public struct VisualizerWebView: NSViewRepresentable {
    public let resultJSON: String
    public var onSiteSelect: ((Int) -> Void)?
    public var onBranchSelect: ((String) -> Void)?

    public init(
        resultJSON: String,
        onSiteSelect: ((Int) -> Void)? = nil,
        onBranchSelect: ((String) -> Void)? = nil
    ) {
        self.resultJSON = resultJSON
        self.onSiteSelect = onSiteSelect
        self.onBranchSelect = onBranchSelect
    }

    public func makeCoordinator() -> VisualizerCoordinator {
        let coord = VisualizerCoordinator()
        coord.onSiteSelected = onSiteSelect
        coord.onBranchSelected = onBranchSelect
        return coord
    }

    public func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let contentController = WKUserContentController()
        contentController.add(context.coordinator, name: "hyphyBridge")
        config.userContentController = contentController

        let webView = WKWebView(frame: .zero, configuration: config)
        context.coordinator.attach(webView: webView)

        // Load viewer_shell.html from Bundle or relative resources
        if let shellURL = Bundle.module.url(forResource: "viewer_shell", withExtension: "html") {
            webView.loadFileURL(shellURL, allowingReadAccessTo: shellURL.deletingLastPathComponent())
        }

        return webView
    }

    public func updateNSView(_ nsView: WKWebView, context: Context) {
        context.coordinator.onSiteSelected = onSiteSelect
        context.coordinator.onBranchSelected = onBranchSelect
        if !resultJSON.isEmpty {
            context.coordinator.loadResultJSON(resultJSON)
        }
    }
}
