import UIKit

/// A deliberately small, local-only dictation keyboard. iOS forbids a
/// keyboard extension from using the microphone, so the containing app owns
/// capture and local inference. This extension only reads append-only text
/// from the App Group and inserts it through the documented text proxy. Apple
/// gates App Group access behind the user's Full Access switch; this target
/// contains no network client and never transmits the shared text.
final class KeyboardViewController: UIInputViewController {
    private let statusLabel = UILabel()
    private let previewLabel = UILabel()
    private let pasteButton = UIButton(type: .system)
    private var pollTimer: Timer?
    private var sessionID = ""
    private var insertedCharacters = 0
    private var latest = DictationSnapshot.empty

    override func viewDidLoad() {
        super.viewDidLoad()
        hasDictationKey = true
        view.backgroundColor = UIColor(red: 0.035, green: 0.055, blue: 0.050, alpha: 1)
        buildInterface()
        poll()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) {
            [weak self] _ in self?.poll()
        }
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func buildInterface() {
        statusLabel.font = .monospacedSystemFont(ofSize: 13, weight: .bold)
        statusLabel.textColor = UIColor(red: 0.34, green: 0.94, blue: 0.66, alpha: 1)
        statusLabel.textAlignment = .center

        previewLabel.font = .systemFont(ofSize: 13)
        previewLabel.textColor = UIColor(white: 0.82, alpha: 1)
        previewLabel.textAlignment = .center
        previewLabel.numberOfLines = 2

        pasteButton.setTitle("Paste last transcript", for: .normal)
        pasteButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
        pasteButton.addTarget(self, action: #selector(pasteLast), for: .touchUpInside)
        style(pasteButton, accent: true)

        let globe = UIButton(type: .system)
        globe.setTitle("🌐", for: .normal)
        globe.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
        style(globe, accent: false)
        globe.addTarget(
            self, action: #selector(handleInputModeList(from:with:)), for: .allTouchEvents)
        let space = key("space", action: #selector(insertSpace))
        let delete = key("⌫", action: #selector(deleteOneCharacter))
        let enter = key("return", action: #selector(insertReturn))
        let keys = UIStackView(arrangedSubviews: [globe, space, delete, enter])
        keys.axis = .horizontal
        keys.spacing = 8
        keys.distribution = .fillEqually

        let stack = UIStackView(arrangedSubviews: [statusLabel, previewLabel, pasteButton, keys])
        stack.axis = .vertical
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            stack.topAnchor.constraint(equalTo: view.topAnchor, constant: 12),
            stack.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -12),
            view.heightAnchor.constraint(greaterThanOrEqualToConstant: 210),
            pasteButton.heightAnchor.constraint(equalToConstant: 42),
            keys.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    private func key(_ title: String, action: Selector) -> UIButton {
        let button = UIButton(type: .system)
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
        button.addTarget(self, action: action, for: .touchUpInside)
        style(button, accent: false)
        return button
    }

    private func style(_ button: UIButton, accent: Bool) {
        button.setTitleColor(accent ? .black : .white, for: .normal)
        button.backgroundColor = accent
            ? UIColor(red: 0.34, green: 0.94, blue: 0.66, alpha: 1)
            : UIColor(white: 0.16, alpha: 1)
        button.layer.cornerRadius = 9
    }

    private func poll() {
        let snapshot = DictationBridge.read()
        latest = snapshot
        if snapshot.sessionID != sessionID {
            sessionID = snapshot.sessionID
            insertedCharacters = 0
        }

        switch snapshot.state {
        case .listening:
            statusLabel.text = "● LISTENING LOCALLY"
            pasteButton.setTitle("Listening in FrankenWhisper…", for: .normal)
            pasteButton.isEnabled = false
            insertNewText(from: snapshot)
        case .finishing:
            statusLabel.text = "FINISHING ON DEVICE…"
            pasteButton.setTitle("Finishing transcription…", for: .normal)
            pasteButton.isEnabled = false
            insertNewText(from: snapshot)
        case .failed:
            statusLabel.text = "DICTATION NEEDS ATTENTION"
            previewLabel.text = snapshot.message ?? "Open FrankenWhisper to retry."
            pasteButton.setTitle("Paste recovered text", for: .normal)
            pasteButton.isEnabled = !snapshot.text.isEmpty
        case .idle:
            statusLabel.text = "START LIVE DICTATION IN FRANKENWHISPER"
            pasteButton.setTitle("Paste last transcript", for: .normal)
            pasteButton.isEnabled = !snapshot.text.isEmpty
        }
        if snapshot.state != .failed {
            previewLabel.text = snapshot.text.isEmpty
                ? "Full Access enables only the local app handoff. Nothing is transmitted."
                : String(snapshot.text.suffix(180))
        }
    }

    private func insertNewText(from snapshot: DictationSnapshot) {
        let count = snapshot.text.count
        guard count > insertedCharacters else { return }
        let start = snapshot.text.index(snapshot.text.startIndex, offsetBy: insertedCharacters)
        textDocumentProxy.insertText(String(snapshot.text[start...]))
        insertedCharacters = count
    }

    @objc private func pasteLast() {
        guard !latest.text.isEmpty else { return }
        textDocumentProxy.insertText(latest.text)
        insertedCharacters = latest.text.count
    }

    @objc private func insertSpace() { textDocumentProxy.insertText(" ") }
    @objc private func deleteOneCharacter() { textDocumentProxy.deleteBackward() }
    @objc private func insertReturn() { textDocumentProxy.insertText("\n") }
}
