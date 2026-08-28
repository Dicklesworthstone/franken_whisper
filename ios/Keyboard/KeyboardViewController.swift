import SwiftUI
import UIKit

private struct DictationStartLink: View {
    let title: String
    let onTap: () -> Void

    var body: some View {
        Link(destination: URL(string: "frankenwhisper://dictate")!) {
            Text(title)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(.black)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(red: 0.34, green: 0.94, blue: 0.66))
                .clipShape(RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .simultaneousGesture(TapGesture().onEnded { onTap() })
    }
}

/// A system-wide, local-only dictation keyboard. iOS never gives keyboard
/// extensions microphone access, so a user tap makes a visible transition to
/// the containing app, which owns capture and local inference. On iOS 26.4+
/// the user then swipes back to the original app while this extension reads
/// append-only text from the App Group and inserts it at the cursor.
final class KeyboardViewController: UIInputViewController {
    private let statusLabel = UILabel()
    private let previewLabel = UILabel()
    private let micButton = UIButton(type: .system)
    private let micContainer = UIView()
    private var micLinkHost: UIHostingController<DictationStartLink>?
    private var letterButtons: [UIButton] = []
    private var pollTimer: Timer?
    private var sessionID = ""
    private var insertedCharacters = 0
    private var latest = DictationSnapshot.empty
    private var isShifted = true
    private var observedActiveSession = false

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

    override func textDidChange(_ textInput: (any UITextInput)?) {
        super.textDidChange(textInput)
        guard !isShifted else { return }
        let before = textDocumentProxy.documentContextBeforeInput ?? ""
        if before.isEmpty || before.hasSuffix("\n") || before.hasSuffix(". ")
            || before.hasSuffix("! ") || before.hasSuffix("? ")
        {
            isShifted = true
            refreshLetterTitles()
        }
    }

    private func buildInterface() {
        statusLabel.font = .monospacedSystemFont(ofSize: 11, weight: .bold)
        statusLabel.textColor = accentColor
        statusLabel.numberOfLines = 2

        micButton.setTitle("🎙  Start", for: .normal)
        micButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .bold)
        micButton.addTarget(self, action: #selector(toggleDictation), for: .touchUpInside)
        style(micButton, accent: true)
        micButton.translatesAutoresizingMaskIntoConstraints = false
        micContainer.addSubview(micButton)

        let linkHost = UIHostingController(
            rootView: DictationStartLink(title: "🎙  Start", onTap: { [weak self] in
                self?.dictationLinkTapped()
            }))
        linkHost.view.backgroundColor = .clear
        linkHost.view.translatesAutoresizingMaskIntoConstraints = false
        addChild(linkHost)
        micContainer.addSubview(linkHost.view)
        linkHost.didMove(toParent: self)
        micLinkHost = linkHost

        NSLayoutConstraint.activate([
            micContainer.widthAnchor.constraint(equalToConstant: 104),
            micButton.leadingAnchor.constraint(equalTo: micContainer.leadingAnchor),
            micButton.trailingAnchor.constraint(equalTo: micContainer.trailingAnchor),
            micButton.topAnchor.constraint(equalTo: micContainer.topAnchor),
            micButton.bottomAnchor.constraint(equalTo: micContainer.bottomAnchor),
            linkHost.view.leadingAnchor.constraint(equalTo: micContainer.leadingAnchor),
            linkHost.view.trailingAnchor.constraint(equalTo: micContainer.trailingAnchor),
            linkHost.view.topAnchor.constraint(equalTo: micContainer.topAnchor),
            linkHost.view.bottomAnchor.constraint(equalTo: micContainer.bottomAnchor),
        ])

        let statusRow = UIStackView(arrangedSubviews: [statusLabel, micContainer])
        statusRow.axis = .horizontal
        statusRow.spacing = 8
        statusRow.alignment = .center

        previewLabel.font = .systemFont(ofSize: 11)
        previewLabel.textColor = UIColor(white: 0.78, alpha: 1)
        previewLabel.numberOfLines = 1
        previewLabel.textAlignment = .center

        let row1 = letterRow("qwertyuiop")
        let row2 = letterRow("asdfghjkl")

        let shift = key("⇧", action: #selector(toggleShift))
        let row3Letters = Array("zxcvbnm").map { letterKey(String($0)) }
        let delete = key("⌫", action: #selector(deleteOneCharacter))
        let row3 = UIStackView(arrangedSubviews: [shift] + row3Letters + [delete])
        configureKeyRow(row3)

        let globe = key("🌐", action: #selector(showInputModeList))
        let comma = key(",", action: #selector(insertComma))
        let space = key("space", action: #selector(insertSpace))
        let period = key(".", action: #selector(insertPeriod))
        let enter = key("return", action: #selector(insertReturn))
        let bottom = UIStackView(arrangedSubviews: [globe, comma, space, period, enter])
        configureKeyRow(bottom)
        globe.widthAnchor.constraint(equalToConstant: 44).isActive = true
        comma.widthAnchor.constraint(equalToConstant: 36).isActive = true
        period.widthAnchor.constraint(equalToConstant: 36).isActive = true
        enter.widthAnchor.constraint(equalToConstant: 68).isActive = true

        let stack = UIStackView(arrangedSubviews: [statusRow, previewLabel, row1, row2, row3, bottom])
        stack.axis = .vertical
        stack.spacing = 7
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6),
            stack.topAnchor.constraint(equalTo: view.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -8),
            view.heightAnchor.constraint(greaterThanOrEqualToConstant: 300),
            statusRow.heightAnchor.constraint(equalToConstant: 40),
            row1.heightAnchor.constraint(equalToConstant: 42),
            row2.heightAnchor.constraint(equalToConstant: 42),
            row3.heightAnchor.constraint(equalToConstant: 42),
            bottom.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    private var accentColor: UIColor {
        UIColor(red: 0.34, green: 0.94, blue: 0.66, alpha: 1)
    }

    private func letterRow(_ letters: String) -> UIStackView {
        let row = UIStackView(arrangedSubviews: letters.map { letterKey(String($0)) })
        configureKeyRow(row)
        return row
    }

    private func configureKeyRow(_ row: UIStackView) {
        row.axis = .horizontal
        row.spacing = 4
        row.distribution = .fillEqually
    }

    private func letterKey(_ letter: String) -> UIButton {
        let button = key(letter, action: #selector(insertLetter(_:)))
        button.accessibilityIdentifier = letter
        letterButtons.append(button)
        return button
    }

    private func key(_ title: String, action: Selector) -> UIButton {
        let button = UIButton(type: .system)
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 16, weight: .medium)
        button.addTarget(self, action: action, for: .touchUpInside)
        style(button, accent: false)
        return button
    }

    private func style(_ button: UIButton, accent: Bool) {
        button.setTitleColor(accent ? .black : .white, for: .normal)
        button.backgroundColor = accent ? accentColor : UIColor(white: 0.16, alpha: 1)
        button.layer.cornerRadius = 7
    }

    private func poll() {
        let snapshot = DictationBridge.read()
        latest = snapshot
        if snapshot.sessionID != sessionID {
            sessionID = snapshot.sessionID
            insertedCharacters = 0
            observedActiveSession = false
        }

        switch snapshot.state {
        case .listening:
            observedActiveSession = true
            statusLabel.text = "● LISTENING LOCALLY"
            showActionButton(title: "■  Finish", enabled: true)
            insertNewText(from: snapshot)
        case .finishing:
            observedActiveSession = true
            statusLabel.text = "FINISHING ON DEVICE…"
            showActionButton(title: "Finishing…", enabled: false)
            insertNewText(from: snapshot)
        case .failed:
            statusLabel.text = "DICTATION NEEDS ATTENTION"
            showStartLink(title: "🎙  Retry")
        case .idle:
            if observedActiveSession { insertNewText(from: snapshot) }
            statusLabel.text = hasFullAccess
                ? "PRIVATE · ON-DEVICE WHISPER"
                : "FULL ACCESS NEEDED FOR LOCAL HANDOFF"
            showStartLink(title: "🎙  Start")
        }

        if snapshot.state == .failed {
            previewLabel.text = snapshot.message ?? "Open FrankenWhisper to retry."
        } else if snapshot.text.isEmpty {
            previewLabel.text = hasFullAccess
                ? "Tap Start, then swipe back when FrankenWhisper is listening."
                : "Typing still works. Full Access only enables the on-device app handoff."
        } else {
            previewLabel.text = String(snapshot.text.suffix(120))
        }
    }

    private func insertNewText(from snapshot: DictationSnapshot) {
        let count = snapshot.text.count
        guard count > insertedCharacters else { return }
        let start = snapshot.text.index(snapshot.text.startIndex, offsetBy: insertedCharacters)
        textDocumentProxy.insertText(String(snapshot.text[start...]))
        insertedCharacters = count
    }

    private func showStartLink(title: String) {
        guard hasFullAccess else {
            showActionButton(title: title, enabled: true)
            return
        }
        micLinkHost?.rootView = DictationStartLink(title: title, onTap: { [weak self] in
            self?.dictationLinkTapped()
        })
        micLinkHost?.view.isHidden = false
        micButton.isHidden = true
    }

    private func showActionButton(title: String, enabled: Bool) {
        micLinkHost?.view.isHidden = true
        micButton.isHidden = false
        micButton.setTitle(title, for: .normal)
        micButton.isEnabled = enabled
    }

    private func dictationLinkTapped() {
        statusLabel.text = "OPENING FRANKENWHISPER…"
        previewLabel.text = "Wait for the listening screen, then swipe back to this app."
    }

    @objc private func toggleDictation() {
        switch latest.state {
        case .listening:
            DictationBridge.writeCommand(.stop)
            statusLabel.text = "FINISHING ON DEVICE…"
            micButton.isEnabled = false
        case .finishing:
            break
        case .idle, .failed:
            guard hasFullAccess else {
                statusLabel.text = "ENABLE FULL ACCESS IN KEYBOARD SETTINGS"
                previewLabel.text = "Typing remains local; Full Access enables only the local app handoff."
                return
            }
            // Full-access starts are rendered as a SwiftUI Link. Keeping this
            // branch inert prevents a future refactor from reintroducing the
            // programmatic open that iOS 26 rejects for keyboard extensions.
            showStartLink(title: latest.state == .failed ? "🎙  Retry" : "🎙  Start")
        }
    }

    @objc private func insertLetter(_ sender: UIButton) {
        guard let letter = sender.accessibilityIdentifier else { return }
        textDocumentProxy.insertText(isShifted ? letter.uppercased() : letter)
        if isShifted {
            isShifted = false
            refreshLetterTitles()
        }
    }

    @objc private func toggleShift() {
        isShifted.toggle()
        refreshLetterTitles()
    }

    private func refreshLetterTitles() {
        for button in letterButtons {
            guard let letter = button.accessibilityIdentifier else { continue }
            button.setTitle(isShifted ? letter.uppercased() : letter, for: .normal)
        }
    }

    @objc private func showInputModeList(_ sender: UIButton, event: UIEvent) {
        handleInputModeList(from: sender, with: event)
    }

    @objc private func insertComma() { textDocumentProxy.insertText(",") }
    @objc private func insertPeriod() { textDocumentProxy.insertText(".") }
    @objc private func insertSpace() { textDocumentProxy.insertText(" ") }
    @objc private func deleteOneCharacter() { textDocumentProxy.deleteBackward() }
    @objc private func insertReturn() { textDocumentProxy.insertText("\n") }
}
