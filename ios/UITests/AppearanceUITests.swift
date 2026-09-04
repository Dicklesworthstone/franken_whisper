import XCTest

final class FrankenWhisperAppearanceUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testAppearanceTogglePersistsLightModeAcrossLaunches() throws {
        let app = XCUIApplication()
        app.launch()

        let toggle = app.buttons["appearance-toggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 12))
        XCTAssertTrue(
            ["Switch to light mode", "Switch to dark mode"].contains(toggle.label),
            "Appearance control exposed an unexpected state: \(toggle.label)"
        )

        if toggle.label == "Switch to dark mode" {
            toggle.tap()
            XCTAssertEqual(toggle.label, "Switch to light mode")
        }

        toggle.tap()
        XCTAssertEqual(toggle.label, "Switch to dark mode")
        keepScreenshot(of: app, named: "Remembered light appearance")

        app.terminate()
        app.launch()

        let relaunchedToggle = app.buttons["appearance-toggle"]
        XCTAssertTrue(relaunchedToggle.waitForExistence(timeout: 12))
        XCTAssertEqual(relaunchedToggle.label, "Switch to dark mode")
    }

    func testTranscriptHistoryIsReadableInDarkAndLightAppearances() throws {
        let app = XCUIApplication()
        app.launch()

        let appearance = app.buttons["appearance-toggle"]
        XCTAssertTrue(appearance.waitForExistence(timeout: 12))
        if appearance.label == "Switch to dark mode" {
            appearance.tap()
        }
        XCTAssertEqual(appearance.label, "Switch to light mode")

        let history = app.buttons["transcript-history-button"]
        XCTAssertTrue(history.waitForExistence(timeout: 5))
        XCTAssertEqual(history.value as? String, "Empty")
        history.tap()

        let library = app.descendants(matching: .any)["transcript-history-library"]
        XCTAssertTrue(library.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["No recent transcripts"].exists)
        XCTAssertTrue(
            app.staticTexts["Finished batch transcriptions appear here automatically."].exists
        )
        keepScreenshot(of: app, named: "Transcript history dark appearance")
        app.buttons["Done"].tap()

        appearance.tap()
        XCTAssertEqual(appearance.label, "Switch to dark mode")
        history.tap()
        XCTAssertTrue(library.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["No recent transcripts"].exists)
        keepScreenshot(of: app, named: "Transcript history light appearance")
        app.buttons["Done"].tap()

        appearance.tap()
        XCTAssertEqual(appearance.label, "Switch to light mode")
    }

    private func keepScreenshot(of app: XCUIApplication, named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
