// OpenSuno Render Node menu bar app.
// Supervises the render node through launchd, shows what it is doing (stage, speed, queue),
// starts/stops it, and can register both itself and the node to start at login.
// Built by launcher/build_installer.py; no dependencies beyond AppKit.
import AppKit
import ServiceManagement

let label = "local.opensuno.rendernode"
let home = FileManager.default.homeDirectoryForCurrentUser
let support = home.appendingPathComponent("Library/Application Support/OpenSuno")
let logs = home.appendingPathComponent("Library/Logs/OpenSuno")
let agents = home.appendingPathComponent("Library/LaunchAgents")
let configURL = support.appendingPathComponent("render-node.json")
let plistURL = agents.appendingPathComponent(label + ".plist")
let logURL = logs.appendingPathComponent("render-node.log")
let root = (Bundle.main.object(forInfoDictionaryKey: "OpenSunoRoot") as? String) ?? ""

struct Config: Codable {
    var token = ""
    var port = 7863
    var launchAtLogin = false
    var idleMinutes = 30

    init() {}
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        token = try values.decodeIfPresent(String.self, forKey: .token) ?? ""
        port = try values.decodeIfPresent(Int.self, forKey: .port) ?? 7863
        launchAtLogin = try values.decodeIfPresent(Bool.self, forKey: .launchAtLogin) ?? false
        idleMinutes = try values.decodeIfPresent(Int.self, forKey: .idleMinutes) ?? 30
    }

    static func load() -> Config {
        var config = Config()
        if let data = try? Data(contentsOf: configURL), let saved = try? JSONDecoder().decode(Config.self, from: data) {
            config = saved
        }
        if config.token.isEmpty {
            config.token = Config.randomToken()
            config.save()
        }
        return config
    }

    func save() {
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(self) {
            try? data.write(to: configURL)
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
        }
    }

    static func randomToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 24)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return Data(bytes).base64EncodedString().replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
    }
}

@discardableResult
func launchctl(_ arguments: String...) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do { try process.run() } catch { return (1, error.localizedDescription) }
    process.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func lanAddress() -> String? {
    var list: UnsafeMutablePointer<ifaddrs>? = nil
    guard getifaddrs(&list) == 0, let first = list else { return nil }
    defer { freeifaddrs(list) }
    var fallback: String? = nil
    for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
        let flags = Int32(pointer.pointee.ifa_flags)
        guard let address = pointer.pointee.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
              flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        guard getnameinfo(address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
        let name = String(cString: host)
        let interface = String(cString: pointer.pointee.ifa_name)
        if interface == "en0" { return name }
        if fallback == nil { fallback = name }
    }
    return fallback
}

final class NodeService {
    var config = Config.load()
    let domain = "gui/\(getuid())"
    var service: String { "\(domain)/\(label)" }

    func writePlist() {
        try? FileManager.default.createDirectory(at: agents, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let plist: [String: Any] = [
            "Label": label,
            "ProgramArguments": [root + "/.venv/bin/python", "-u", root + "/app/render_node.py"],
            "WorkingDirectory": root,
            "EnvironmentVariables": [
                "PATH": root + "/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONUNBUFFERED": "1",
                "OPENSUNO_NODE_TOKEN": config.token,
                "OPENSUNO_NODE_PORT": String(config.port),
                "OPENSUNO_NODE_IDLE_MINUTES": String(config.idleMinutes),
            ],
            "RunAtLoad": config.launchAtLogin,
            "KeepAlive": ["SuccessfulExit": false],
            "ProcessType": "Interactive",
            "StandardOutPath": logURL.path,
            "StandardErrorPath": logURL.path,
        ]
        if let data = try? PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0) {
            try? data.write(to: plistURL)
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: plistURL.path)
        }
    }

    var loaded: Bool { launchctl("print", service).status == 0 }
    var running: Bool {
        let result = launchctl("print", service)
        return result.status == 0 && result.output.range(of: #"\bpid = \d+"#, options: .regularExpression) != nil
    }

    func start() -> String? {
        writePlist()
        if !loaded {
            let result = launchctl("bootstrap", domain, plistURL.path)
            if result.status != 0 { return "Could not start the render node: " + result.output.trimmingCharacters(in: .whitespacesAndNewlines) }
        }
        let result = launchctl("kickstart", service)
        return result.status == 0 ? nil : "Could not start the render node: " + result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func stop() {
        launchctl("bootout", service)
    }

    /// Apply a changed token, port or login setting: the plist is rewritten and a running node restarted.
    func apply() -> String? {
        let wasRunning = running
        if loaded { stop() }
        writePlist()
        if wasRunning { return start() }
        if config.launchAtLogin { launchctl("bootstrap", domain, plistURL.path) }
        return nil
    }
}

/// The status block at the top of the menu: readable, never highlighted, clearly not a command.
final class InfoView: NSView {
    let headline = NSTextField(labelWithString: "")
    let lines = (0..<3).map { _ in NSTextField(labelWithString: "") }
    let stack = NSStackView()
    static let width: CGFloat = 340

    init() {
        super.init(frame: NSRect(x: 0, y: 0, width: InfoView.width, height: 10))
        headline.font = .systemFont(ofSize: 13, weight: .semibold)
        headline.textColor = .labelColor
        for line in lines {
            line.font = .systemFont(ofSize: 12)
            line.textColor = .secondaryLabelColor
            line.lineBreakMode = .byWordWrapping
            line.maximumNumberOfLines = 2
            line.preferredMaxLayoutWidth = InfoView.width - 28
        }
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 3
        stack.edgeInsets = NSEdgeInsets(top: 6, left: 14, bottom: 6, right: 14)
        stack.addArrangedSubview(headline)
        lines.forEach(stack.addArrangedSubview)
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor),
            stack.topAnchor.constraint(equalTo: topAnchor),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }

    func show(_ title: String, _ details: [String]) {
        headline.stringValue = title
        for (line, text) in zip(lines, details + Array(repeating: "", count: max(0, lines.count - details.count))) {
            line.stringValue = text
            line.isHidden = text.isEmpty
        }
        layoutSubtreeIfNeeded()
        frame.size = NSSize(width: InfoView.width, height: stack.fittingSize.height)
    }
}

final class MenuApp: NSObject, NSApplicationDelegate {
    let node = NodeService()
    var item: NSStatusItem!
    var health: [String: Any]? = nil
    var timer: Timer?
    var pollInFlight = false

    let info = InfoView()
    let infoItem = NSMenuItem()
    let addressLine = NSMenuItem()
    let startStop = NSMenuItem()
    let loginToggle = NSMenuItem()
    let restartItem = NSMenuItem()

    func applicationDidFinishLaunching(_ notification: Notification) {
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.imagePosition = .imageLeading
        let menu = NSMenu()
        infoItem.view = info
        infoItem.isEnabled = false
        menu.addItem(infoItem)
        menu.addItem(.separator())
        addressLine.target = self; addressLine.action = #selector(copyAddress)
        menu.addItem(addressLine)
        menu.addItem(.separator())
        startStop.target = self; startStop.action = #selector(toggleNode)
        menu.addItem(startStop)
        restartItem.title = "Restart node"; restartItem.target = self; restartItem.action = #selector(restartNode)
        menu.addItem(restartItem)
        loginToggle.title = "Start at login"; loginToggle.target = self; loginToggle.action = #selector(toggleLogin)
        menu.addItem(loginToggle)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Copy token", action: #selector(copyToken), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Change token…", action: #selector(changeToken), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Change port…", action: #selector(changePort), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Open log", action: #selector(openLog), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Open Studio", action: #selector(openStudio), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit", action: #selector(quit), keyEquivalent: "q").target = self
        item.menu = menu
        // Opening the app means "run the render node"; quitting it stops the node again.
        if let error = node.start() { alert("Render node", error) }
        // The installer can ask for "start at login" before this app ever ran; the login item is registered here.
        if #available(macOS 13.0, *), node.config.launchAtLogin, SMAppService.mainApp.status != .enabled {
            try? SMAppService.mainApp.register()
        }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.poll() }
        poll()
    }

    // MARK: polling

    func poll() {
        guard !pollInFlight else { return }
        pollInFlight = true
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(node.config.port)/v1/health")!, timeoutInterval: 2)
        request.setValue("Bearer " + node.config.token, forHTTPHeaderField: "Authorization")
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            var parsed: [String: Any]? = nil
            if let data = data, (response as? HTTPURLResponse)?.statusCode == 200,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any], json["ok"] as? Bool == true {
                parsed = json
            }
            DispatchQueue.main.async {
                self?.health = parsed
                self?.pollInFlight = false
                self?.refresh()
            }
        }.resume()
    }

    func stageText(_ progress: [String: Any]) -> String {
        let stage = progress["stage"] as? String ?? ""
        switch stage {
        case "loading": return progress["loading_detail"] as? String ?? "Loading model"
        case "planning", "arranging": return "Writing the score"
        case "music_tokens": return "Composing"
        case "synthesis":
            let steps = progress["steps"] as? Int ?? 0, total = progress["step_total"] as? Int ?? 0
            return total > 0 ? "Rendering audio · step \(steps)/\(total)" : "Rendering audio"
        case "decoding_audio": return "Decoding audio"
        case "complete": return "Finishing"
        default: return stage.hasPrefix("analysis") ? "Analyzing recording" : (stage.isEmpty ? "Working" : stage.capitalized)
        }
    }

    func refresh() {
        let button = item.button!
        let symbol: String
        var title = ""
        var headline: String
        var details: [String] = []
        if let health = health {
            let busy = health["busy"] as? Bool ?? false
            let gpu = (health["gpu"] as? [String: Any])?["name"] as? String ?? "GPU"
            let models = (health["models"] as? [[String: Any]] ?? []).filter { $0["ready"] as? Bool == true }.compactMap { $0["label"] as? String }
            let queue = (health["queue"] as? [Any])?.count ?? 0
            let completed = health["completed"] as? Int ?? 0
            let loaded = health["models_loaded"] as? Bool ?? false
            if busy {
                symbol = "waveform.badge.mic"
                let progress = health["progress"] as? [String: Any] ?? [:]
                let rate = progress["tokens_per_second"] as? Double ?? 0
                let tokens = progress["tokens"] as? Int ?? 0
                let elapsed = progress["elapsed"] as? Double ?? 0
                let name = health["active_title"] as? String ?? "song"
                headline = "Rendering “\(name)”"
                details.append(stageText(progress) + (tokens > 0 ? " · \(tokens) tokens" : ""))
                details.append((rate > 0 ? String(format: "%.1f tokens/s", rate) : "Speed —") + (elapsed > 0 ? String(format: " · %d:%02d", Int(elapsed) / 60, Int(elapsed) % 60) : ""))
                title = rate > 0 ? String(format: "%.1f tok/s", rate) : "Rendering"
            } else {
                symbol = "waveform"
                headline = "Render node running · " + gpu
                details.append(models.isEmpty ? "No models installed yet · download them from the Studio's Models panel" : "Models ready: " + models.joined(separator: ", "))
                details.append(loaded ? "Model in memory, ready to start" : "Idle" + (completed > 0 ? " · \(completed) job\(completed == 1 ? "" : "s") done" : ""))
            }
            if busy || queue > 0 { details.append(queue > 0 ? "\(queue) job\(queue == 1 ? "" : "s") waiting" : "Queue empty") }
            startStop.title = "Stop node"
            restartItem.isHidden = false
        } else {
            let starting = node.running
            symbol = "waveform.slash"
            headline = starting ? "Render node starting…" : "Render node stopped"
            details.append(starting ? "Waiting for it to answer" : "Songs sent here will wait until it starts")
            startStop.title = starting ? "Stop node" : "Start node"
            restartItem.isHidden = !starting
            title = starting ? "" : "Off"
        }
        info.show(headline, details)
        let address = lanAddress().map { "http://\($0):\(node.config.port)" } ?? "http://127.0.0.1:\(node.config.port)"
        addressLine.title = "Copy address  \(address)"
        loginToggle.state = node.config.launchAtLogin ? .on : .off
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "OpenSuno Render Node")
        image?.isTemplate = true
        button.image = image
        button.title = title.isEmpty ? "" : " " + title
        button.toolTip = headline
    }

    // MARK: actions

    func alert(_ message: String, _ text: String = "") {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = text
        alert.runModal()
    }

    @objc func toggleNode() {
        if health != nil || node.running {
            node.stop()
            health = nil
        } else if let error = node.start() {
            alert("Render node", error)
        }
        refresh()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.poll() }
    }

    @objc func restartNode() {
        node.stop()
        health = nil
        if let error = node.start() { alert("Render node", error) }
        refresh()
    }

    @objc func toggleLogin() {
        node.config.launchAtLogin.toggle()
        node.config.save()
        if #available(macOS 13.0, *) {
            do {
                if node.config.launchAtLogin { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            } catch {
                alert("Start at login", "The menu bar app could not be registered: \(error.localizedDescription). The render node itself will still start at login.")
            }
        }
        if let error = node.apply() { alert("Render node", error) }
        refresh()
    }

    @objc func copyAddress() {
        let address = lanAddress().map { "http://\($0):\(node.config.port)" } ?? "http://127.0.0.1:\(node.config.port)"
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(address, forType: .string)
    }

    @objc func copyToken() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(node.config.token, forType: .string)
    }

    func ask(_ message: String, _ text: String, value: String, secure: Bool = false) -> String? {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = text
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 320, height: 24))
        field.stringValue = value
        alert.accessoryView = field
        alert.window.initialFirstResponder = field
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertFirstButtonReturn ? field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines) : nil
    }

    @objc func changeToken() {
        guard let value = ask("Render node token", "Studios connect with this token. Leave empty to generate a new one.", value: node.config.token) else { return }
        node.config.token = value.isEmpty ? Config.randomToken() : value
        node.config.save()
        if let error = node.apply() { alert("Render node", error) }
        health = nil
        refresh()
    }

    @objc func changePort() {
        guard let value = ask("Render node port", "Studios connect to this port.", value: String(node.config.port)) else { return }
        guard let port = Int(value), (1024...65535).contains(port) else { alert("Render node port", "Enter a number between 1024 and 65535."); return }
        node.config.port = port
        node.config.save()
        if let error = node.apply() { alert("Render node", error) }
        health = nil
        refresh()
    }

    @objc func openLog() {
        if !FileManager.default.fileExists(atPath: logURL.path) { alert("Render node log", "The node has not written a log yet.") ; return }
        NSWorkspace.shared.open(logURL)
    }

    @objc func openStudio() {
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:7862/")!)
    }

    @objc func quit() {
        NSApp.terminate(nil)
    }

    /// Quitting the app (menu, Cmd+Q or logout) takes the node down with it; a queued song is left for next time.
    func applicationWillTerminate(_ notification: Notification) {
        timer?.invalidate()
        node.stop()
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = MenuApp()
app.delegate = delegate
app.run()
