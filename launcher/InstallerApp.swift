// Install OpenSuno.app: choose the Studio, the render node or both, press Continue, and everything is
// put where it belongs (~/OpenSuno, /Applications, launchd). The work itself is installer.py, run with the
// Python runtime this app unpacks first, so nothing needs to be installed on the Mac beforehand.
import AppKit

@main
final class Installer: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let studio = NSButton(checkboxWithTitle: "Studio — the interface, opens in your browser", target: nil, action: nil)
    let node = NSButton(checkboxWithTitle: "Render node — renders songs on this Mac's GPU (menu bar app)", target: nil, action: nil)
    let analysis = NSButton(checkboxWithTitle: "Cover analysis — turn recordings into covers (about 2 GB more)", target: nil, action: nil)
    let login = NSButton(checkboxWithTitle: "Start the render node when I log in", target: nil, action: nil)
    let hint = NSTextField(wrappingLabelWithString: "Everything is installed into ~/OpenSuno and Applications. Python packages are downloaded during installation, so an internet connection is needed.")
    let button = NSButton(title: "Continue", target: nil, action: nil)
    let status = NSTextField(labelWithString: "")
    let detail = NSTextField(labelWithString: "")
    let spinner = NSProgressIndicator()
    var errors = Data()

    static func main() {
        let app = NSApplication.shared
        let delegate = Installer()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 520, height: 300), styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Install OpenSuno"
        window.center()
        window.isReleasedWhenClosed = false
        let content = window.contentView!
        let title = NSTextField(labelWithString: "What should this Mac do?")
        title.font = .boldSystemFont(ofSize: 15)
        title.frame = NSRect(x: 24, y: 256, width: 470, height: 22)
        content.addSubview(title)
        studio.frame = NSRect(x: 24, y: 218, width: 470, height: 20)
        node.frame = NSRect(x: 24, y: 190, width: 470, height: 20)
        analysis.frame = NSRect(x: 44, y: 164, width: 450, height: 20)
        login.frame = NSRect(x: 44, y: 140, width: 450, height: 20)
        studio.state = .on
        node.state = .on
        analysis.state = .on
        login.state = .on
        for box in [studio, node, analysis, login] {
            box.target = self
            box.action = #selector(choiceChanged)
            content.addSubview(box)
        }
        hint.frame = NSRect(x: 24, y: 84, width: 470, height: 44)
        hint.font = .systemFont(ofSize: 11)
        hint.textColor = .secondaryLabelColor
        content.addSubview(hint)
        button.frame = NSRect(x: 396, y: 20, width: 100, height: 32)
        button.bezelStyle = .rounded
        button.keyEquivalent = "\r"
        button.target = self
        button.action = #selector(start)
        content.addSubview(button)
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.frame = NSRect(x: 24, y: 26, width: 18, height: 18)
        spinner.isHidden = true
        content.addSubview(spinner)
        status.frame = NSRect(x: 50, y: 40, width: 340, height: 18)
        status.font = .systemFont(ofSize: 12)
        status.lineBreakMode = .byTruncatingTail
        content.addSubview(status)
        detail.frame = NSRect(x: 50, y: 22, width: 340, height: 16)
        detail.font = .systemFont(ofSize: 10)
        detail.textColor = .secondaryLabelColor
        detail.lineBreakMode = .byTruncatingTail
        content.addSubview(detail)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func choiceChanged() {
        let nodeChosen = node.state == .on
        analysis.isEnabled = nodeChosen
        login.isEnabled = nodeChosen
        button.isEnabled = nodeChosen || studio.state == .on
    }

    @objc func start() {
        for box in [studio, node, analysis, login] { box.isEnabled = false }
        button.isEnabled = false
        spinner.isHidden = false
        spinner.startAnimation(nil)
        status.stringValue = "Preparing…"
        var flags: [String] = []
        if studio.state == .on { flags.append("--studio") }
        if node.state == .on {
            flags.append("--node")
            if analysis.state == .on { flags.append("--analysis") }
            if login.state == .on { flags.append("--login") }
        }
        DispatchQueue.global().async { self.install(flags) }
    }

    func install(_ flags: [String]) {
        let resources = Bundle.main.resourceURL!
        let home = ProcessInfo.processInfo.environment["OPENSUNO_HOME"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("OpenSuno")
        let scratch = home.appendingPathComponent(".installer-runtime")
        do {
            try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: scratch.path) { try FileManager.default.removeItem(at: scratch) }
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        } catch {
            fail("Could not prepare \(home.path): \(error.localizedDescription)")
            return
        }
        // /usr/bin/tar is always present; the unpacked Python then runs the real installer.
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-xzf", resources.appendingPathComponent("payload/runtime/python312.tar.gz").path, "-C", scratch.path]
        do {
            try tar.run()
            tar.waitUntilExit()
        } catch {
            fail("Could not unpack the Python runtime: \(error.localizedDescription)")
            return
        }
        guard tar.terminationStatus == 0 else { fail("Could not unpack the Python runtime."); return }
        let process = Process()
        process.executableURL = scratch.appendingPathComponent("python/bin/python3")
        process.arguments = ["-u", resources.appendingPathComponent("installer.py").path] + flags
        process.currentDirectoryURL = home
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        environment["PYTHONWARNINGS"] = "ignore"
        process.environment = environment
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        out.fileHandleForReading.readabilityHandler = { handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            let lines = text.split(separator: "\n").map(String.init).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            guard !lines.isEmpty else { return }
            DispatchQueue.main.async {
                for line in lines {
                    // Headline steps end with an ellipsis; anything else is detail under them.
                    if line.hasSuffix("…") { self.status.stringValue = line; self.detail.stringValue = "" }
                    else { self.detail.stringValue = line }
                }
            }
        }
        err.fileHandleForReading.readabilityHandler = { handle in self.errors.append(handle.availableData) }
        do {
            try process.run()
        } catch {
            fail("The installer could not start: \(error.localizedDescription)")
            return
        }
        process.waitUntilExit()
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        errors.append(err.fileHandleForReading.readDataToEndOfFile())
        try? FileManager.default.removeItem(at: scratch)
        if process.terminationStatus == 0 {
            DispatchQueue.main.async { self.finish(flags) }
        } else {
            let message = String(decoding: errors, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            fail(message.isEmpty ? "The installation did not finish." : message)
        }
    }

    func finish(_ flags: [String]) {
        spinner.stopAnimation(nil)
        spinner.isHidden = true
        status.stringValue = "Installed."
        var parts: [String] = []
        if flags.contains("--studio") { parts.append("The Studio is starting and opens in your browser.") }
        if flags.contains("--node") { parts.append("The render node runs from the waveform icon in the menu bar. A Studio on another Mac pairs with the address and token found in that menu.") }
        detail.stringValue = ""
        let alert = NSAlert()
        alert.messageText = "OpenSuno is installed"
        alert.informativeText = parts.joined(separator: "\n\n")
        alert.addButton(withTitle: "Done")
        alert.runModal()
        NSApp.terminate(nil)
    }

    func fail(_ message: String) {
        DispatchQueue.main.async {
            self.spinner.stopAnimation(nil)
            self.spinner.isHidden = true
            self.status.stringValue = "Installation failed."
            let alert = NSAlert()
            alert.alertStyle = .critical
            alert.messageText = "OpenSuno could not be installed"
            alert.informativeText = message
            alert.addButton(withTitle: "Quit")
            alert.runModal()
            NSApp.terminate(nil)
        }
    }
}
