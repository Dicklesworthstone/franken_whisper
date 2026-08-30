import XCTest

// The unit-test target compiles ModelStore without the native Engine bridge.
// Keep this minimal error surface aligned with the one operation ModelStore
// needs so response validation can run without loading model code.
enum EngineError: Error {
    case invalid(String)
}

final class ModelStoreTests: XCTestCase {
    func testEngineLifecycleFenceRejectsDelayedOlderUnload() {
        var fence = EngineLifecycleFence()

        XCTAssertTrue(fence.accept(8))
        XCTAssertTrue(fence.accept(9))
        XCTAssertFalse(fence.accept(8))
        XCTAssertEqual(fence.latestToken, 9)
    }

    func testValidPartialResponseAcceptsExactReturnedRange() throws {
        let response = try httpResponse(
            status: 206,
            headers: ["Content-Range": "bytes 100-149/1000"])

        let length = try ModelStore.validatedResponseLength(
            response: response,
            dataCount: 50,
            requestedRange: 100...199,
            fileBytes: 1_000,
            label: "fixture")

        XCTAssertEqual(length, 50)
    }

    func testZeroLengthPartialResponseIsRejectedInsteadOfSpinning() throws {
        let response = try httpResponse(
            status: 206,
            headers: ["Content-Range": "bytes 100-149/1000"])

        XCTAssertThrowsError(
            try ModelStore.validatedResponseLength(
                response: response,
                dataCount: 0,
                requestedRange: 100...199,
                fileBytes: 1_000,
                label: "fixture"))
    }

    func testWrongContentRangeStartIsRejected() throws {
        let response = try httpResponse(
            status: 206,
            headers: ["Content-Range": "bytes 0-49/1000"])

        XCTAssertThrowsError(
            try ModelStore.validatedResponseLength(
                response: response,
                dataCount: 50,
                requestedRange: 100...199,
                fileBytes: 1_000,
                label: "fixture"))
    }

    func testFullResponseMustContainTheWholeFile() throws {
        let response = try httpResponse(status: 200)
        XCTAssertEqual(
            try ModelStore.validatedResponseLength(
                response: response,
                dataCount: 1_000,
                requestedRange: 0...199,
                fileBytes: 1_000,
                label: "fixture"),
            1_000)
        XCTAssertThrowsError(
            try ModelStore.validatedResponseLength(
                response: response,
                dataCount: 200,
                requestedRange: 0...199,
                fileBytes: 1_000,
                label: "fixture"))
    }

    private func httpResponse(
        status: Int,
        headers: [String: String] = [:]
    ) throws -> HTTPURLResponse {
        try XCTUnwrap(
            HTTPURLResponse(
                url: URL(string: "https://example.invalid/model.bin")!,
                statusCode: status,
                httpVersion: "HTTP/1.1",
                headerFields: headers))
    }
}
