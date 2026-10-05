// Tiny standalone Metal event/ticket hardware smoke for the router overlap
// runtime primitives. It drives the ACTUAL public RouterOverlapTicketLedger
// with a local MTLCommandBuffer adapter (the runtime adapter is internal).
// No model, GGUF, hashing, runner or large allocation: three 1 KiB shared
// buffers and two integer-write kernels.
//
// Protocol: prints BENCHMARK_READY, flushes, and requires the exact stdin
// line GO before creating any Metal object or dispatching anything.
// Prints TIMED_WINDOW_END only when bounded cleanup observed a terminal
// status (completed or error) for every committed command. Otherwise it
// prints TIMED_WINDOW_NOT_ENDED, still writes and prints the report, and
// exits 3, which dominates every other exit code. CPU exit never implies GPU
// completion. The optional --report file is created exclusively after the
// END or NOT_ENDED line. Exit codes: 0 pass, 1 assertions failed with
// cleanup observed, 2 no GO, 3 timed window not ended (GPU work may be
// outstanding), 64 usage, 70 encode, 73 report exists/create, 74 report write.
//
// Real: one retained MTLSharedEvent with explicit encodeWaitForEvent /
// encodeSignalEvent between R and S command buffers, commits and drains
// through the shared ledger, and a CPU signaledValue raise releasing an
// already committed waiter on a second command queue. Every CPU wait is
// bounded by status polling BEFORE the ledger's waitUntilCompleted runs.
// Simulated (labelled): one round's router handle REPORTS failure from the
// host over a command that succeeded on the GPU, exercising only the ledger
// cleanup choreography. NOT tested: genuine GPU command failure and whether
// Metal signals an event from a failed command buffer. No GPU busy-time or
// performance claims; the adapter reports 0 GPU microseconds.

import Foundation
import Metal
import LeviathanCore

struct SmokeError: Error, CustomStringConvertible {
    let description: String
}

final class LocalCommandHandle: RouterOverlapCommandHandle {
    let buffer: MTLCommandBuffer
    let simulatedHostFailure: Bool

    init(_ buffer: MTLCommandBuffer, simulatedHostFailure: Bool) {
        self.buffer = buffer
        self.simulatedHostFailure = simulatedHostFailure
    }

    func commitHandle() { buffer.commit() }
    func waitHandle() { buffer.waitUntilCompleted() }
    var handleSucceeded: Bool {
        !simulatedHostFailure && buffer.status == .completed && buffer.error == nil
    }
    /// Not measured: this smoke makes no GPU busy-time claim.
    var handleIntervalMicros: Double { 0 }
    var handleFailureDescription: String {
        simulatedHostFailure
            ? "SIMULATED host-reported failure over a succeeded command (not a GPU failure)"
            : "status \(buffer.status.rawValue), \(String(describing: buffer.error))"
    }
}

struct RoundRecord: Codable {
    let round: Int
    let simulatedHostFailure: Bool
    let routerSignal: UInt64
    let sharedSignal: UInt64
    let signalsMonotonic: Bool
    let routerDataOK: Bool
    let sharedDataOK: Bool
    /// Ledger state only; the GPU may already have finished S.
    let sharedPendingInLedgerAfterRouterDrain: Bool
    /// Raw MTLCommandBufferStatus of S observed at the R drain (informational).
    let sharedCommandStatusObservedAtRouterDrain: Int
    let eventValueAfterRound: UInt64
}

struct CpuRaiseRecord: Codable {
    let queue: String
    let waitValue: UInt64
    let eventValueBeforeRaise: UInt64
    /// Informational: S was not complete during a 20 ms observation.
    let waiterNotCompleteBeforeRaise: Bool
    let releasedWithinBound: Bool
    let finalCommandStatus: Int
    let dataOK: Bool
}

struct SmokeReport: Codable {
    let schema: String
    let status: String
    let device: String
    let deviceRegistryID: UInt64
    let rounds: [RoundRecord]
    let ledgerAsyncCommitted: Int
    let ledgerDrained: Int
    let ledgerFailedDrains: Int
    let ledgerLastCommittedSignal: UInt64
    let ledgerConsistencyRejection: String?
    let unblockCalls: [UInt64]
    let eventInitialValue: UInt64
    let eventFinalValue: UInt64
    let cpuRaise: CpuRaiseRecord?
    let boundedCleanupObservedCompletion: Bool
    let thrown: String?
    let failures: [String]
    let realPrimitives: [String]
    let simulated: [String]
    let notTested: [String]
    let gpuBusyTimeClaimed: Bool
    let performanceClaims: Bool
}

struct Context {
    let queue: MTLCommandQueue
    let fill: MTLComputePipelineState
    let derive: MTLComputePipelineState
    let bufR: MTLBuffer
    let bufS: MTLBuffer
    let event: MTLSharedEvent
    let ledger: RouterOverlapTicketLedger<LocalCommandHandle>
}

final class Smoke {
    static let count = 256
    static let boundSeconds = 5.0
    static let roundCount = 4
    static let simulatedRound = 3
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void smoke_fill(device uint* out [[buffer(0)]],
                           constant uint& base [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        out[gid] = base + gid;
    }

    kernel void smoke_derive(device const uint* src [[buffer(0)]],
                             device uint* out [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        out[gid] = src[gid] ^ 0xA5A5A5A5u;
    }
    """

    var failures: [String] = []
    var committed: [MTLCommandBuffer] = []
    var rounds: [RoundRecord] = []
    var unblockCalls: [UInt64] = []
    var cpuRaise: CpuRaiseRecord?
    var event: MTLSharedEvent?
    var ledger: RouterOverlapTicketLedger<LocalCommandHandle>?
    var deviceName = ""
    var registryID: UInt64 = 0
    var eventInitial: UInt64 = 0
    var cpuWaitValue: UInt64 = 0

    func fail(_ message: String) { failures.append(message) }

    /// Bounded CPU polling; never waits forever.
    static func waitBounded(_ cmds: [MTLCommandBuffer], seconds: Double) -> Bool {
        let deadline = DispatchTime.now().uptimeNanoseconds + UInt64(seconds * 1e9)
        while true {
            if cmds.allSatisfy({ $0.status == .completed || $0.status == .error }) { return true }
            if DispatchTime.now().uptimeNanoseconds >= deadline { return false }
            usleep(200)
        }
    }

    static func verify(_ b: MTLBuffer, _ expected: (UInt32) -> UInt32) -> Bool {
        let p = b.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<count where p[i] != expected(UInt32(i)) { return false }
        return true
    }

    func unblock(_ value: UInt64, _ event: MTLSharedEvent) {
        unblockCalls.append(value)
        if event.signaledValue < value { event.signaledValue = value }
    }

    func makeCommand(_ queue: MTLCommandQueue, waitFor value: UInt64,
                     event: MTLSharedEvent) throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let cmd = queue.makeCommandBuffer() else {
            throw SmokeError(description: "command buffer allocation failed")
        }
        if value > 0 { cmd.encodeWaitForEvent(event, value: value) }
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw SmokeError(description: "compute encoder allocation failed")
        }
        return (cmd, enc)
    }

    func dispatch(_ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState) {
        let width = max(1, min(Self.count, pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(
            MTLSize(width: (Self.count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    func commitTicket(_ c: Context, _ cmd: MTLCommandBuffer, _ enc: MTLComputeCommandEncoder,
                      kind: RouterOverlapTicketKind, simulated: Bool) throws -> UInt64 {
        var ended = false
        do {
            let handle = LocalCommandHandle(cmd, simulatedHostFailure: simulated)
            let signal = try c.ledger.commit(handle, kind: kind) { value in
                enc.endEncoding()
                ended = true
                cmd.encodeSignalEvent(c.event, value: value)
            }
            committed.append(cmd)
            return signal
        } catch {
            if !ended { enc.endEncoding() }
            throw error
        }
    }

    func run() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SmokeError(description: "no Metal device")
        }
        deviceName = device.name
        registryID = device.registryID
        guard let queue = device.makeCommandQueue(),
              let queue2 = device.makeCommandQueue(),
              let event = device.makeSharedEvent() else {
            throw SmokeError(description: "queue/event allocation failed")
        }
        self.event = event
        eventInitial = event.signaledValue
        let library = try device.makeLibrary(source: Self.source, options: nil)
        guard let fillFn = library.makeFunction(name: "smoke_fill"),
              let deriveFn = library.makeFunction(name: "smoke_derive") else {
            throw SmokeError(description: "smoke kernels missing")
        }
        let fill = try device.makeComputePipelineState(function: fillFn)
        let derive = try device.makeComputePipelineState(function: deriveFn)
        let bytes = Self.count * MemoryLayout<UInt32>.stride
        guard let bufR = device.makeBuffer(length: bytes, options: .storageModeShared),
              let bufS = device.makeBuffer(length: bytes, options: .storageModeShared),
              let bufU = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw SmokeError(description: "buffer allocation failed")
        }
        let ledger = RouterOverlapTicketLedger<LocalCommandHandle>(
            lastCommittedSignal: event.signaledValue)
        self.ledger = ledger
        let c = Context(queue: queue, fill: fill, derive: derive, bufR: bufR, bufS: bufS,
                        event: event, ledger: ledger)
        for round in 0..<Self.roundCount {
            try runRound(round, simulated: round == Self.simulatedRound, c)
        }
        let expectedTickets = 2 * Self.roundCount
        if ledger.asyncCommitted != expectedTickets || ledger.drained != expectedTickets
            || ledger.failedDrains != 1 {
            fail("ledger counters: committed=\(ledger.asyncCommitted) drained=\(ledger.drained) failedDrains=\(ledger.failedDrains); expected \(expectedTickets)/\(expectedTickets)/1")
        }
        // Last: the CPU-raised waiter is outside the ledger, so no ledger
        // commit may follow it on this event.
        try cpuRaiseTest(queue2, fill: fill, buffer: bufU, c)
    }

    func runRound(_ round: Int, simulated: Bool, _ c: Context) throws {
        let n = Self.count
        memset(c.bufR.contents(), 0, n * 4)
        memset(c.bufS.contents(), 0, n * 4)
        var base = UInt32(0x0100_0000 + round * 0x1_0000)
        var count = UInt32(n)
        let before = c.ledger.lastCommittedSignal

        let (rCmd, rEnc) = try makeCommand(c.queue, waitFor: before, event: c.event)
        rEnc.setComputePipelineState(c.fill)
        rEnc.setBuffer(c.bufR, offset: 0, index: 0)
        rEnc.setBytes(&base, length: 4, index: 1)
        rEnc.setBytes(&count, length: 4, index: 2)
        dispatch(rEnc, c.fill)
        let rSignal = try commitTicket(c, rCmd, rEnc, kind: .router, simulated: simulated)

        let (sCmd, sEnc) = try makeCommand(c.queue, waitFor: rSignal, event: c.event)
        sEnc.setComputePipelineState(c.derive)
        sEnc.setBuffer(c.bufR, offset: 0, index: 0)
        sEnc.setBuffer(c.bufS, offset: 0, index: 1)
        sEnc.setBytes(&count, length: 4, index: 2)
        dispatch(sEnc, c.derive)
        let sSignal = try commitTicket(c, sCmd, sEnc, kind: .shared, simulated: false)

        let monotonic = rSignal == before + 1 && sSignal == rSignal + 1
        if !monotonic { fail("round \(round): signals not monotonic") }

        // Bounded completion BEFORE the ledger's unbounded waits. A failure
        // report extends the drain to S, so the simulated round bounds both.
        let bounded = simulated ? [rCmd, sCmd] : [rCmd]
        guard Self.waitBounded(bounded, seconds: Self.boundSeconds) else {
            throw SmokeError(description: "round \(round): bounded wait expired before R drain")
        }
        let sStatusAtDrain = Int(sCmd.status.rawValue)
        let rOutcome = c.ledger.drain(through: rSignal) { self.unblock($0, c.event) }
        let sPending = c.ledger.pending.count == 1
            && c.ledger.pending.first?.kind == .shared
            && c.ledger.pending.first?.signal == sSignal

        if simulated {
            if unblockCalls != [sSignal, sSignal] {
                fail("round \(round): expected two unblock calls through \(sSignal), got \(unblockCalls)")
            }
            let simulatedReported = rOutcome.firstFailure?.contains("SIMULATED") ?? false
            if !simulatedReported || rOutcome.drained != 2 || c.ledger.hasPending {
                fail("round \(round): simulated failure did not drain every committed ticket")
            }
            if rOutcome.unblockedThrough != sSignal {
                fail("round \(round): unblock was not raised through the highest committed signal")
            }
            if !c.ledger.isPoisoned {
                fail("round \(round): simulated host-reported failure did not poison the ledger")
            }
        } else {
            if !unblockCalls.isEmpty {
                fail("round \(round): unblock called during a non-failing round: \(unblockCalls)")
            }
            if rOutcome.drained != 1 || rOutcome.firstFailure != nil || !sPending {
                fail("round \(round): R drain did not leave exactly S pending in the ledger")
            }
        }
        let rBase = base
        let rDataOK = Self.verify(c.bufR) { rBase &+ $0 }
        if !rDataOK { fail("round \(round): R data mismatch") }

        if !simulated {
            guard Self.waitBounded([sCmd], seconds: Self.boundSeconds) else {
                throw SmokeError(description: "round \(round): bounded wait expired before S drain")
            }
            let sOutcome = c.ledger.drain(through: nil) { self.unblock($0, c.event) }
            if sOutcome.drained != 1 || sOutcome.firstFailure != nil || c.ledger.hasPending {
                fail("round \(round): S drain mismatch")
            }
        }
        let sDataOK = Self.verify(c.bufS) { (rBase &+ $0) ^ 0xA5A5_A5A5 }
        if !sDataOK { fail("round \(round): S data mismatch") }
        let eventAfter = c.event.signaledValue
        if simulated {
            // CPU-raised by the simulated-failure unblock: lower bound only.
            if eventAfter < sSignal { fail("round \(round): event below the S signal after the simulated drain") }
        } else if eventAfter != sSignal {
            // No CPU raise reaches S here, so equality is the GPU-signal proof.
            fail("round \(round): event \(eventAfter) != S signal \(sSignal) after S drained")
        }

        rounds.append(RoundRecord(
            round: round, simulatedHostFailure: simulated,
            routerSignal: rSignal, sharedSignal: sSignal, signalsMonotonic: monotonic,
            routerDataOK: rDataOK, sharedDataOK: sDataOK,
            sharedPendingInLedgerAfterRouterDrain: sPending,
            sharedCommandStatusObservedAtRouterDrain: sStatusAtDrain,
            eventValueAfterRound: eventAfter))
    }

    func cpuRaiseTest(_ queue2: MTLCommandQueue, fill: MTLComputePipelineState,
                      buffer: MTLBuffer, _ c: Context) throws {
        let (waitValue, overflow) = c.ledger.lastCommittedSignal.addingReportingOverflow(1)
        let before = c.event.signaledValue
        guard !overflow, before < waitValue else {
            fail("cpu raise: event already at or beyond the wait value; test skipped")
            return
        }
        memset(buffer.contents(), 0, Self.count * 4)
        var base: UInt32 = 0x0C0D_0000
        var count = UInt32(Self.count)
        let (cmd, enc) = try makeCommand(queue2, waitFor: waitValue, event: c.event)
        enc.setComputePipelineState(fill)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBytes(&base, length: 4, index: 1)
        enc.setBytes(&count, length: 4, index: 2)
        dispatch(enc, fill)
        enc.endEncoding()
        cpuWaitValue = waitValue
        cmd.commit()
        committed.append(cmd)
        let notComplete = !Self.waitBounded([cmd], seconds: 0.02)
        c.event.signaledValue = waitValue
        let released = Self.waitBounded([cmd], seconds: Self.boundSeconds)
        var dataOK = false
        if !released {
            fail("cpu raise: committed waiter not released within the bound; supervisor owns containment")
        } else if cmd.status != .completed || cmd.error != nil {
            fail("cpu raise: released waiter did not complete successfully")
        } else {
            dataOK = Self.verify(buffer) { 0x0C0D_0000 &+ $0 }
            if !dataOK { fail("cpu raise: data mismatch") }
        }
        cpuRaise = CpuRaiseRecord(
            queue: "second MTLCommandQueue, same retained MTLSharedEvent",
            waitValue: waitValue, eventValueBeforeRaise: before,
            waiterNotCompleteBeforeRaise: notComplete, releasedWithinBound: released,
            finalCommandStatus: Int(cmd.status.rawValue), dataOK: dataOK)
    }

    /// Raises the event through every committed or awaited value, then
    /// polls every committed command boundedly.
    func cleanup() -> Bool {
        guard let event else { return true }
        let target = max(ledger?.lastCommittedSignal ?? 0, cpuWaitValue)
        if event.signaledValue < target { event.signaledValue = target }
        return Self.waitBounded(committed, seconds: Self.boundSeconds)
    }

    func report(thrown: String?, cleanupOK: Bool) -> SmokeReport {
        SmokeReport(
            schema: "router-overlap-event-smoke-v1",
            status: failures.isEmpty ? "pass" : "fail",
            device: deviceName, deviceRegistryID: registryID,
            rounds: rounds,
            ledgerAsyncCommitted: ledger?.asyncCommitted ?? 0,
            ledgerDrained: ledger?.drained ?? 0,
            ledgerFailedDrains: ledger?.failedDrains ?? 0,
            ledgerLastCommittedSignal: ledger?.lastCommittedSignal ?? 0,
            ledgerConsistencyRejection: ledger?.consistencyRejection,
            unblockCalls: unblockCalls,
            eventInitialValue: eventInitial,
            eventFinalValue: event?.signaledValue ?? 0,
            cpuRaise: cpuRaise,
            boundedCleanupObservedCompletion: cleanupOK,
            thrown: thrown,
            failures: failures,
            realPrimitives: [
                "one retained MTLSharedEvent with encodeWaitForEvent before each compute encoder and encodeSignalEvent per ticket",
                "public LeviathanCore RouterOverlapTicketLedger commit/drain with a local MTLCommandBuffer adapter",
                "bounded status polling before every ledger waitUntilCompleted",
                "CPU signaledValue raise releasing an already committed waiter on a second command queue",
            ],
            simulated: [
                "round \(Self.simulatedRound): the router handle REPORTS failure from the host over a command that SUCCEEDED on the GPU; only ledger cleanup choreography (drain extension to S, poison, two unblock calls both through the highest committed signal) is exercised",
            ],
            notTested: [
                "genuine GPU command-buffer failure",
                "whether Metal signals an event from a failed command buffer",
                "ledger command-buffer interval accounting (adapter reports 0 microseconds)",
                "runner integration and model execution",
                "whether a later GPU signal of a lower value lowers signaledValue after a CPU raise (never encountered by this tool)",
                "GPU busy time or performance",
            ],
            gpuBusyTimeClaimed: false,
            performanceClaims: false)
    }
}

// MARK: - entry

var reportPath: String?
var argIndex = 1
let arguments = CommandLine.arguments
while argIndex < arguments.count {
    if arguments[argIndex] == "--report", argIndex + 1 < arguments.count, reportPath == nil {
        reportPath = arguments[argIndex + 1]
        argIndex += 2
    } else {
        FileHandle.standardError.write(Data("usage: router-overlap-event-smoke [--report <new-file>]\nexit: 0 pass, 1 fail (cleanup observed), 2 no GO, 3 timed window not ended (GPU work may be outstanding; dominant), 64 usage, 70 encode, 73 report exists/create, 74 report write\n".utf8))
        exit(64)
    }
}
if let reportPath, FileManager.default.fileExists(atPath: reportPath) {
    FileHandle.standardError.write(Data("report path already exists; refusing\n".utf8))
    exit(73)
}

print("BENCHMARK_READY")
fflush(nil)
guard let goLine = readLine(strippingNewline: true), goLine == "GO" else {
    FileHandle.standardError.write(Data("exact GO not received; no GPU work issued\n".utf8))
    exit(2)
}

let smoke = Smoke()
var thrown: String?
do {
    try smoke.run()
} catch {
    thrown = "\(error)"
    smoke.fail("thrown: \(error)")
}
if let rejection = smoke.ledger?.consistencyRejection {
    smoke.fail("ledger consistency: \(rejection)")
}
let cleanupOK = smoke.cleanup()
if cleanupOK {
    print("TIMED_WINDOW_END")
} else {
    smoke.fail("bounded cleanup did not observe a terminal status for every committed command; GPU work may still be outstanding; supervisor owns containment")
    let nonterminal = smoke.committed.filter { $0.status != .completed && $0.status != .error }.count
    print("TIMED_WINDOW_NOT_ENDED committed=\(smoke.committed.count) nonterminal=\(nonterminal)")
}
fflush(nil)

let finalReport = smoke.report(thrown: thrown, cleanupOK: cleanupOK)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
var exitCode: Int32 = cleanupOK ? (finalReport.status == "pass" ? 0 : 1) : 3
do {
    let data = try encoder.encode(finalReport)
    if let reportPath {
        let fd = open(reportPath, O_WRONLY | O_CREAT | O_EXCL, 0o644)
        if fd < 0 {
            FileHandle.standardError.write(Data("exclusive report creation failed\n".utf8))
            exitCode = 73
        } else {
            let written = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
            if written != data.count || close(fd) != 0 { exitCode = 74 }
        }
    }
    print(String(decoding: data, as: UTF8.self))
} catch {
    FileHandle.standardError.write(Data("report encoding failed\n".utf8))
    exitCode = 70
}
print("ROUTER_OVERLAP_EVENT_SMOKE status=\(finalReport.status)")
fflush(nil)
if !cleanupOK { exitCode = 3 }
exit(exitCode)
