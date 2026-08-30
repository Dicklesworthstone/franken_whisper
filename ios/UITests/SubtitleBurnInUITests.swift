import XCTest

final class SubtitleBurnInUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testImportedPhotosVideoReachesRealSubtitleExporter() throws {
        guard ProcessInfo.processInfo.environment["FW_SUBTITLE_E2E_SAMPLE_PHOTOS"] == "1" else {
            throw XCTSkip(
                "Ultra-stress E2E is opt-in: boot an isolated simulator, add the operator "
                    + "sample with simctl addmedia, then set FW_SUBTITLE_E2E_SAMPLE_PHOTOS=1."
            )
        }
        let app = XCUIApplication()
        app.launchEnvironment["FW_SUBTITLE_E2E_CAPTURE"] = "1"
        app.launch()

        let videoPicker = app.buttons["fw.videoPicker"]
        XCTAssertTrue(
            videoPicker.waitForExistence(timeout: 180),
            "The real engine never became ready. UI hierarchy:\n\(app.debugDescription)"
        )
        XCTAssertTrue(
            videoPicker.waitUntilEnabled(timeout: 180),
            "The real Photos video picker stayed disabled after model hydration. "
                + "UI hierarchy:\n\(app.debugDescription)"
        )
        videoPicker.tap()

        // PHPicker is a privacy-preserving remote view hosted inside the app's
        // process, not the Photos app. Query the real visible media grid by its
        // system accessibility label so this still exercises PhotosPicker and
        // PickedVideo.loadTransferable exactly as a user tap does.
        let firstVideo = app.images.matching(
            NSPredicate(format: "label BEGINSWITH %@", "Video,")
        ).firstMatch
        XCTAssertTrue(
            firstVideo.waitForExistence(timeout: 30),
            "The system Photos picker did not expose the imported sample. "
                + "App hierarchy:\n\(app.debugDescription)"
        )
        // PHPicker's first grid cell can report an origin of -0.0 points,
        // which makes XCUIElement.tap() reject it as not hittable even though
        // its visible center is well inside the picker. Tap that real element's
        // center coordinate; this still drives the same system picker action.
        firstVideo.coordinate(
            withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)
        ).tap()

        let addButton = app.buttons.matching(
            NSPredicate(format: "label IN %@", ["Add", "Done"])
        ).firstMatch
        if addButton.waitForExistence(timeout: 2), addButton.isEnabled {
            addButton.tap()
        }

        let transcribe = app.buttons["fw.transcribe"]
        XCTAssertTrue(
            transcribe.waitForExistence(timeout: 60),
            "The picked movie never returned through the app's media-import path"
        )
        XCTAssertTrue(
            transcribe.waitUntilEnabled(timeout: 60),
            "The picked movie never finished AVFoundation audio extraction"
        )
        transcribe.tap()

        let studio = app.buttons["fw.subtitleStudio"]
        XCTAssertTrue(
            studio.waitForExistence(timeout: 300),
            "The real FFI transcription did not produce decoder-aligned subtitle controls. "
                + "UI hierarchy:\n\(app.debugDescription)"
        )
        XCTAssertTrue(studio.isEnabled, "The native run returned no DTW word alignment")
        studio.tap()

        let burn = app.buttons["fw.burnSubtitles"]
        XCTAssertTrue(burn.waitForExistence(timeout: 30), "Subtitle Studio did not open")
        burn.tap()

        let status = app.descendants(matching: .any)["fw.subtitleExportStatus"]
        XCTAssertTrue(
            status.waitForExistence(timeout: 300),
            "The production AVFoundation burn-in exporter did not finish"
        )
        XCTAssertTrue(
            status.label.contains("ready"),
            "The production exporter reported failure: \(status.label)"
        )
    }
}

private extension XCUIElement {
    func waitUntilEnabled(timeout: TimeInterval) -> Bool {
        let predicate = NSPredicate(format: "enabled == true")
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: self)
        return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
    }
}
