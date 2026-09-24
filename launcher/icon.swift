import AppKit
let size = NSSize(width: 1024, height: 1024)
let image = NSImage(size: size)
image.lockFocus()
let tile = NSBezierPath(roundedRect: NSRect(x: 64, y: 64, width: 896, height: 896), xRadius: 200, yRadius: 200)
NSColor(calibratedWhite: 0.075, alpha: 1).setFill()
tile.fill()
let circle = NSBezierPath(ovalIn: NSRect(x: 192, y: 192, width: 640, height: 640))
let gradient = NSGradient(starting: NSColor(red: 1, green: 0.47, blue: 0.15, alpha: 1), ending: NSColor(red: 0.98, green: 0.16, blue: 0.57, alpha: 1))!
gradient.draw(in: circle, angle: 35)
let style: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 440, weight: .semibold), .foregroundColor: NSColor.white]
let note = "♫" as NSString
let measured = note.size(withAttributes: style)
note.draw(at: NSPoint(x: (1024 - measured.width) / 2, y: (1024 - measured.height) / 2 + 15), withAttributes: style)
image.unlockFocus()
let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
