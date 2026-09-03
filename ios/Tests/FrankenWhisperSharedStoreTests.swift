import XCTest
import UIKit

final class FrankenWhisperSharedStoreTests: XCTestCase {
    func testLiveDictationSurfacesRemainReadableInBothAppearances() throws {
        let darkTraits = UITraitCollection(userInterfaceStyle: .dark)
        let lightTraits = UITraitCollection(userInterfaceStyle: .light)
        let background = UIColor(Lab.background)
        let panel = UIColor(Lab.panelStrong)
        let text = UIColor(Lab.textPrimary)

        let darkBackground = try rgba(background.resolvedColor(with: darkTraits))
        let lightBackground = try rgba(background.resolvedColor(with: lightTraits))
        let darkPanel = try rgba(panel.resolvedColor(with: darkTraits))
        let lightPanel = try rgba(panel.resolvedColor(with: lightTraits))
        let darkText = try rgba(text.resolvedColor(with: darkTraits))
        let lightText = try rgba(text.resolvedColor(with: lightTraits))

        XCTAssertGreaterThan(contrastRatio(darkText, darkBackground), 7)
        XCTAssertGreaterThan(contrastRatio(lightText, lightBackground), 7)
        XCTAssertGreaterThan(contrastRatio(darkText, darkPanel), 7)
        XCTAssertGreaterThan(contrastRatio(lightText, lightPanel), 7)
        XCTAssertNotEqual(darkBackground, lightBackground)
        XCTAssertNotEqual(darkPanel, lightPanel)
    }

    func testOnlyGeneratedStagedMediaNamesAreAccepted() {
        let generated = "\(UUID().uuidString).m4a"
        XCTAssertTrue(FrankenWhisperSharedStore.isValidStagedMediaName(generated))

        for malformed in [
            ".",
            "..",
            "../recording.m4a",
            "/tmp/recording.m4a",
            "recording.m4a",
            "\(UUID().uuidString)",
            "\(UUID().uuidString).m4a/extra",
            "\(UUID().uuidString).thisextensionistoolong"
        ] {
            XCTAssertFalse(
                FrankenWhisperSharedStore.isValidStagedMediaName(malformed),
                "unexpectedly accepted \(malformed)"
            )
        }
    }

    private func rgba(_ color: UIColor) throws -> [CGFloat] {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            throw XCTSkip("Theme color could not be resolved in the active color space")
        }
        return [red, green, blue, alpha]
    }

    private func relativeLuminance(_ rgba: [CGFloat]) -> CGFloat {
        func linear(_ component: CGFloat) -> CGFloat {
            component <= 0.04045
                ? component / 12.92
                : pow((component + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * linear(rgba[0]) + 0.7152 * linear(rgba[1]) + 0.0722 * linear(rgba[2])
    }

    private func contrastRatio(_ first: [CGFloat], _ second: [CGFloat]) -> CGFloat {
        let firstLuminance = relativeLuminance(first)
        let secondLuminance = relativeLuminance(second)
        return (max(firstLuminance, secondLuminance) + 0.05)
            / (min(firstLuminance, secondLuminance) + 0.05)
    }
}
