// OpenSuno.app: a small window while the bundled bootstrap installs or starts the Studio, then it quits.
// The Studio itself keeps running in the background (launchd) and lives in the browser.
import AppKit

@main
final class Launcher: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let status = NSTextField(labelWithString: "Starting OpenSuno…")
    let spinner = NSProgressIndicator()
    var errors = Data()

    static func main() {
        let app = NSApplication.shared
        let delegate = Launcher()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 120), styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "OpenSuno"
        window.center()
        window.isReleasedWhenClosed = false
        let content = window.contentView!
        spinner.style = .spinning
        spinner.controlSize = .regular
        spinner.frame = NSRect(x: 24, y: 44, width: 32, height: 32)
        spinner.startAnimation(nil)
        status.frame = NSRect(x: 68, y: 50, width: 310, height: 20)
        status.font = .systemFont(ofSize: 13)
        status.lineBreakMode = .byTruncatingTail
        content.addSubview(spinner)
        content.addSubview(status)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.global().async { self.bootstrap() }
    }

    func bootstrap() {
        let resources = Bundle.main.resourceURL!
        let process = Process()
        // The self-contained app carries its runtime; the app placed by "Install OpenSuno" uses the installed one.
        let bundled = resources.appendingPathComponent("payload/.venv/bin/python")
        let home = ProcessInfo.processInfo.environment["OPENSUNO_HOME"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("OpenSuno")
        let installed = home.appendingPathComponent(".venv/bin/python")
        process.executableURL = FileManager.default.fileExists(atPath: bundled.path) ? bundled : installed
        process.arguments = ["-u", resources.appendingPathComponent("bootstrap.py").path]
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        process.environment = environment
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        out.fileHandleForReading.readabilityHandler = { handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            guard let line = text.split(separator: "\n").last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) else { return }
            DispatchQueue.main.async { self.status.stringValue = String(line) }
        }
        err.fileHandleForReading.readabilityHandler = { handle in self.errors.append(handle.availableData) }
        do {
            try process.run()
        } catch {
            fail("The launcher inside the app could not start: \(error.localizedDescription)")
            return
        }
        process.waitUntilExit()
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        errors.append(err.fileHandleForReading.readDataToEndOfFile())
        if process.terminationStatus == 0 {
            DispatchQueue.main.async { NSApp.terminate(nil) }
        } else {
            let message = String(decoding: errors, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            fail(message.isEmpty ? "OpenSuno did not start. See ~/Library/Logs/OpenSuno/server.log or run Install OpenSuno again." : message)
        }
    }

    func fail(_ message: String) {
        DispatchQueue.main.async {
            self.spinner.stopAnimation(nil)
            let alert = NSAlert()
            alert.alertStyle = .critical
            alert.messageText = "OpenSuno could not start"
            alert.informativeText = message
            alert.addButton(withTitle: "Quit")
            alert.runModal()
            NSApp.terminate(nil)
        }
    }
}
