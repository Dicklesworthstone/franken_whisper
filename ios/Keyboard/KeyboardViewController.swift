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

/// Smoothly interpolates the real frequency-band energy published by the
/// containing app. No synthetic/random animation is used: silence is flat.
private final class LiveSpectrumView: UIView {
    private var target = [Float](repeating: 0, count: 14)
    private var displayed = [Float](repeating: 0, count: 14)
    private var targetLevel: Float = 0
    private var displayedLevel: Float = 0
    private var displayLink: CADisplayLink?

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false
        isAccessibilityElement = true
        accessibilityLabel = "Live microphone spectrum"
        let link = CADisplayLink(target: self, selector: #selector(animateFrame))
        link.preferredFramesPerSecond = 30
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    deinit { displayLink?.invalidate() }

    func update(bands: [Float], level: Float, active: Bool) {
        if active, bands.count == target.count {
            target = bands.map { min(1, max(0, $0)) }
            targetLevel = min(1, max(0, level))
            accessibilityValue = "\(Int(targetLevel * 100)) percent"
        } else {
            target = [Float](repeating: 0, count: target.count)
            targetLevel = 0
            accessibilityValue = "silent"
        }
    }

    @objc private func animateFrame() {
        displayedLevel += (targetLevel - displayedLevel) * 0.2
        for index in displayed.indices {
            displayed[index] += (target[index] - displayed[index]) * 0.24
        }
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext(), !displayed.isEmpty else { return }
        let accent = UIColor(red: 0.34, green: 0.94, blue: 0.66, alpha: 1)
        context.setFillColor(UIColor(white: 0.02, alpha: 0.65).cgColor)
        UIBezierPath(roundedRect: rect, cornerRadius: 7).fill()

        let gap: CGFloat = 2
        let horizontalPadding: CGFloat = 6
        let usableWidth = max(1, rect.width - horizontalPadding * 2)
        let barWidth = max(2, (usableWidth - gap * CGFloat(displayed.count - 1)) / CGFloat(displayed.count))
        let centerY = rect.midY
        for (index, energy) in displayed.enumerated() {
            let minimum: CGFloat = displayedLevel > 0.01 ? 2 : 1
            let height = minimum + CGFloat(energy) * max(2, rect.height - 8)
            let barRect = CGRect(
                x: horizontalPadding + CGFloat(index) * (barWidth + gap),
                y: centerY - height / 2,
                width: barWidth,
                height: height)
            context.setFillColor(accent.withAlphaComponent(0.38 + CGFloat(displayedLevel) * 0.62).cgColor)
            UIBezierPath(roundedRect: barRect, cornerRadius: barWidth / 2).fill()
        }
    }
}

/// A system-wide, local-only dictation keyboard. iOS never gives keyboard
/// extensions microphone access, so the first tap makes a visible transition
/// to the containing app, which owns capture and local inference. That app
/// keeps a time-bounded local session armed in the background; subsequent
/// Start/Finish taps stay in this keyboard and cross only the App Group.
final class KeyboardViewController: UIInputViewController {
    private let statusLabel = UILabel()
    private let spectrumView = LiveSpectrumView()
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
        statusLabel.numberOfLines = 1
        statusLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        spectrumView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            spectrumView.widthAnchor.constraint(equalToConstant: 112),
            spectrumView.heightAnchor.constraint(equalToConstant: 32),
        ])

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

        let statusRow = UIStackView(arrangedSubviews: [spectrumView, statusLabel, micContainer])
        statusRow.axis = .horizontal
        statusRow.spacing = 8
        statusRow.alignment = .center

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

        let stack = UIStackView(arrangedSubviews: [statusRow, row1, row2, row3, bottom])
        stack.axis = .vertical
        stack.spacing = 7
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6),
            stack.topAnchor.constraint(equalTo: view.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -8),
            view.heightAnchor.constraint(greaterThanOrEqualToConstant: 282),
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

        if let expiresAt = snapshot.expiresAt,
           expiresAt <= Date().timeIntervalSince1970,
           snapshot.state == .armed || snapshot.state == .listening || snapshot.state == .finishing
        {
            // The host can be terminated before publishing its final idle
            // snapshot. Do not present that stale App Group value as a live
            // microphone service forever.
            statusLabel.text = "SESSION EXPIRED"
            showStartLink(title: "🎙  Start")
            spectrumView.update(bands: [], level: 0, active: false)
            return
        }

        switch snapshot.state {
        case .armed:
            if observedActiveSession { insertNewText(from: snapshot) }
            statusLabel.text = "READY · \(minutesRemaining(snapshot))m"
            showActionButton(title: "🎙  Start", enabled: true)
        case .listening:
            observedActiveSession = true
            statusLabel.text = "● LISTENING"
            showActionButton(title: "■  Finish", enabled: true)
            insertNewText(from: snapshot)
        case .finishing:
            observedActiveSession = true
            statusLabel.text = "PROCESSING…"
            showActionButton(title: "Finishing…", enabled: false)
            insertNewText(from: snapshot)
        case .failed:
            statusLabel.text = "NEEDS APP"
            showStartLink(title: "🎙  Retry")
        case .idle:
            if observedActiveSession { insertNewText(from: snapshot) }
            statusLabel.text = hasFullAccess ? "ON-DEVICE" : "FULL ACCESS"
            showStartLink(title: "🎙  Start")
        }
        spectrumView.update(
            bands: snapshot.spectrum ?? [],
            level: snapshot.level ?? 0,
            active: snapshot.state == .listening)
    }

    private func minutesRemaining(_ snapshot: DictationSnapshot) -> Int {
        guard let expiresAt = snapshot.expiresAt else { return 60 }
        return max(0, Int(ceil((expiresAt - Date().timeIntervalSince1970) / 60)))
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
        statusLabel.text = "OPENING APP…"
    }

    @objc private func toggleDictation() {
        switch latest.state {
        case .armed:
            DictationBridge.writeCommand(.start)
            statusLabel.text = "STARTING…"
            micButton.isEnabled = false
        case .listening:
            DictationBridge.writeCommand(.stop)
            statusLabel.text = "PROCESSING…"
            micButton.isEnabled = false
        case .finishing:
            break
        case .idle, .failed:
            guard hasFullAccess else {
                statusLabel.text = "FULL ACCESS NEEDED"
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
