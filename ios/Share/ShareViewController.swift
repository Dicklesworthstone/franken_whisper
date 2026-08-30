import UIKit
import UniformTypeIdentifiers

final class ShareViewController: UIViewController {
    private let statusLabel = UILabel()
    private let openButton = UIButton(type: .system)
    private var stagedURL: URL?
    private var cancelled = false

    deinit {
        // Covers a sheet dismissed by the system rather than our Cancel
        // button. A successful handoff clears `stagedURL` first, so only
        // abandoned private copies are reclaimed here.
        if let stagedURL {
            FrankenWhisperSharedStore.discardStagedMedia(at: stagedURL)
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        configureView()
        stageSharedMedia()
    }

    private func configureView() {
        view.backgroundColor = UIColor(red: 0.002, green: 0.025, blue: 0.025, alpha: 1)
        let mark = UIImageView(image: UIImage(systemName: "waveform.badge.mic"))
        mark.tintColor = UIColor(red: 0.25, green: 0.82, blue: 0.96, alpha: 1)
        mark.preferredSymbolConfiguration = UIImage.SymbolConfiguration(pointSize: 30, weight: .bold)

        let title = UILabel()
        title.text = "SPEECH OBSERVATORY"
        title.textColor = .white
        title.font = .monospacedSystemFont(ofSize: 17, weight: .black)

        statusLabel.text = "Securing the media locally…"
        statusLabel.textColor = UIColor.white.withAlphaComponent(0.66)
        statusLabel.font = .preferredFont(forTextStyle: .subheadline)
        statusLabel.numberOfLines = 0
        statusLabel.textAlignment = .center

        var configuration = UIButton.Configuration.filled()
        configuration.title = "Open FrankenWhisper"
        configuration.image = UIImage(systemName: "waveform")
        configuration.imagePadding = 8
        configuration.baseBackgroundColor = UIColor(red: 0.04, green: 0.55, blue: 0.44, alpha: 1)
        configuration.cornerStyle = .capsule
        openButton.configuration = configuration
        openButton.isEnabled = false
        openButton.addTarget(self, action: #selector(openObservatory), for: .touchUpInside)

        let cancel = UIButton(type: .system)
        cancel.setTitle("Cancel", for: .normal)
        cancel.tintColor = UIColor.white.withAlphaComponent(0.62)
        cancel.addTarget(self, action: #selector(cancelShare), for: .touchUpInside)

        let stack = UIStackView(arrangedSubviews: [mark, title, statusLabel, openButton, cancel])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
            stack.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 360),
            openButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
        ])
    }

    private func stageSharedMedia() {
        let providers = (extensionContext?.inputItems as? [NSExtensionItem])?
            .compactMap(\.attachments).flatMap { $0 } ?? []
        let supported = [UTType.audio, .movie, .mpeg4Audio]
        guard let pair = providers.lazy.compactMap({ provider -> (NSItemProvider, UTType)? in
            supported.first(where: { provider.hasItemConformingToTypeIdentifier($0.identifier) })
                .map { (provider, $0) }
        }).first else {
            showFailure("Share one audio or video file to transcribe it.")
            return
        }

        pair.0.loadFileRepresentation(forTypeIdentifier: pair.1.identifier) { [weak self] url, _ in
            guard let url else {
                Task { @MainActor in self?.showFailure("The shared media could not be opened.") }
                return
            }
            do {
                let staged = try FrankenWhisperSharedStore.stageMedia(
                    from: url,
                    preferredExtension: pair.1.preferredFilenameExtension
                )
                Task { @MainActor [weak self] in
                    guard let self, !self.cancelled else {
                        FrankenWhisperSharedStore.discardStagedMedia(at: staged)
                        return
                    }
                    self.stagedURL = staged
                    self.statusLabel.text = "Media secured locally. Ready for private transcription."
                    self.openButton.isEnabled = true
                }
            } catch {
                Task { @MainActor in self?.showFailure("Could not stage that file: \(error.localizedDescription)") }
            }
        }
    }

    private func showFailure(_ message: String) {
        statusLabel.text = message
        statusLabel.textColor = UIColor(red: 0.97, green: 0.44, blue: 0.44, alpha: 1)
    }

    @objc private func openObservatory() {
        guard let url = URL(string: "frankenwhisper://new") else { return }
        openButton.isEnabled = false
        extensionContext?.open(url) { [weak self] opened in
            Task { @MainActor in
                guard let self else { return }
                if opened {
                    self.stagedURL = nil
                    self.extensionContext?.completeRequest(returningItems: nil)
                } else {
                    self.showFailure("FrankenWhisper could not be opened. Try again or cancel this share.")
                    self.openButton.isEnabled = true
                }
            }
        }
    }

    @objc private func cancelShare() {
        cancelled = true
        if let stagedURL {
            FrankenWhisperSharedStore.discardStagedMedia(at: stagedURL)
            self.stagedURL = nil
        }
        extensionContext?.cancelRequest(withError: CocoaError(.userCancelled))
    }
}
