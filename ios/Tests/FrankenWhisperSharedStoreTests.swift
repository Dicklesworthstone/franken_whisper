import XCTest

final class FrankenWhisperSharedStoreTests: XCTestCase {
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
}
