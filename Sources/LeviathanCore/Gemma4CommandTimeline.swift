import Foundation
import Metal

/// Errors from the opt-in command-buffer timeline diagnostics.
public enum Gemma4CommandTimelineError: Error, Equatable, Sendable {
    /// Recorder capacity outside 1...Gemma4CommandTimeline.maxCapacity.
    case invalidCapacity(Int)
    /// Device-probe command-buffer count outside 1...64.
    case invalidProbeCount(Int)
}

/// Opt-in Metal command-buffer timeline diagnostics: pure event/summary
/// types, a bounded recorder, and a validating analysis.
///
/// Clock basis: host timestamps are `mach_absolute_time()` converted to
/// seconds with `mach_timebase_info`, the conversion QuartzCore documents
/// for `CACurrentMediaTime()` (CABase.h). MTLCommandBuffer.h documents
/// `GPUStartTime` / `GPUEndTime` as host time in seconds (0 until the
/// buffer started / completed). That both share one basis is CHECKED per
/// event, not assumed: an event is used only if its GPU interval lies
/// inside its own host interval [before commit, after wait].
///
/// Interpretation limits: this is a single-run diagnostic trace, not a
/// performance or gain benchmark. A GPU-end-to-next-commit gap is host
/// wall time between two command buffers: it contains wait-return
/// latency, scheduling, CPU work and encoding, and may include page
/// faults or contention from anything else on the machine. It is NOT a
/// measurement of CPU busy time, and nothing here attributes time to
/// another process.
public enum Gemma4CommandTimeline {

    /// Hard upper bound on recorded events per recorder.
    public static let maxCapacity = 8192

    /// Stored in every trace so serialized traces carry their clock basis.
    public static let clockBasis = "mach_absolute_time seconds via mach_timebase_info (CACurrentMediaTime basis)"

    /// Seconds per mach tick; NaN if the timebase is unavailable, which
    /// makes every host timestamp non-finite and every event rejected.
    private static let secondsPerTick: Double = {
        var info = mach_timebase_info_data_t()
        guard mach_timebase_info(&info) == KERN_SUCCESS, info.numer != 0, info.denom != 0 else {
            return .nan
        }
        return Double(info.numer) / Double(info.denom) / 1e9
    }()

    /// Current host time in seconds on the CACurrentMediaTime basis.
    public static func hostSeconds() -> Double {
        Double(mach_absolute_time()) * secondsPerTick
    }

    /// The commit/wait site that produced an event.
    public enum Site: String, Codable, Equatable, Sendable {
        /// Gemma4Runner.flushOpenSegment (one deferred segment of kernels).
        case deferredFlush
        /// Gemma4Runner.dispatch per-kernel submission.
        case directDispatch
        /// Gemma4Runner.dispatchPipeline per-kernel submission.
        case directDispatchPipeline
        /// Standalone deviceProbe blit (never a runner command buffer).
        case deviceProbe
    }

    /// One committed-and-waited command buffer. Host times are seconds on
    /// `clockBasis`; gpuStart/gpuEnd are the RAW MTLCommandBuffer
    /// gpuStartTime/gpuEndTime values, unvalidated.
    public struct Event: Codable, Equatable, Sendable {
        /// 1-based, assigned by the recorder to every recorded attempt
        /// (dropped events consume sequence numbers too).
        public let sequence: Int
        public let site: Site
        /// Compute dispatches encoded into this buffer; nil when unknown.
        public let kernelCount: Int?
        /// Host seconds spent encoding; nil when unknown. Direct sites:
        /// from just before makeCommandBuffer to just before commit.
        /// Deferred segments: the sum of the per-kernel encode calls plus
        /// endEncoding; CPU work between kernels is NOT included.
        public let encodeSeconds: Double?
        /// Host time immediately before commit().
        public let hostBeforeCommit: Double
        /// Host time immediately after commit() returned.
        public let hostAfterCommit: Double
        /// Host time immediately after waitUntilCompleted() returned.
        public let hostAfterWait: Double
        public let gpuStart: Double
        public let gpuEnd: Double
        /// status == .completed and error == nil when recorded.
        public let completed: Bool
        public let statusRaw: Int

        public init(sequence: Int, site: Site, kernelCount: Int?, encodeSeconds: Double?,
                    hostBeforeCommit: Double, hostAfterCommit: Double, hostAfterWait: Double,
                    gpuStart: Double, gpuEnd: Double, completed: Bool, statusRaw: Int) {
            self.sequence = sequence
            self.site = site
            self.kernelCount = kernelCount
            self.encodeSeconds = encodeSeconds
            self.hostBeforeCommit = hostBeforeCommit
            self.hostAfterCommit = hostAfterCommit
            self.hostAfterWait = hostAfterWait
            self.gpuStart = gpuStart
            self.gpuEnd = gpuEnd
            self.completed = completed
            self.statusRaw = statusRaw
        }
    }

    /// A bounded snapshot of recorded events.
    public struct Trace: Codable, Equatable, Sendable {
        public let clockBasis: String
        public let capacity: Int
        /// Events not stored because the recorder was full (always the
        /// tail of the run; their sequence numbers were still consumed).
        public let droppedCount: Int
        public let events: [Event]

        public var truncated: Bool { droppedCount > 0 }

        public init(clockBasis: String = Gemma4CommandTimeline.clockBasis, capacity: Int,
                    droppedCount: Int, events: [Event]) {
            self.clockBasis = clockBasis
            self.capacity = capacity
            self.droppedCount = droppedCount
            self.events = events
        }
    }

    /// Bounded event storage: keeps the first `capacity` events and counts
    /// the rest as dropped. Storage grows only when events are recorded.
    public struct Recorder: Equatable, Sendable {
        public let capacity: Int
        public private(set) var events: [Event] = []
        public private(set) var droppedCount = 0
        public private(set) var lastSequence = 0

        /// Recorder with the maximum capacity.
        public init() {
            capacity = Gemma4CommandTimeline.maxCapacity
        }

        public init(capacity: Int) throws {
            guard capacity >= 1, capacity <= Gemma4CommandTimeline.maxCapacity else {
                throw Gemma4CommandTimelineError.invalidCapacity(capacity)
            }
            self.capacity = capacity
        }

        public mutating func record(site: Site, kernelCount: Int?, encodeSeconds: Double?,
                                    hostBeforeCommit: Double, hostAfterCommit: Double,
                                    hostAfterWait: Double, gpuStart: Double, gpuEnd: Double,
                                    completed: Bool, statusRaw: Int) {
            lastSequence += 1
            guard events.count < capacity else {
                droppedCount += 1
                return
            }
            events.append(Event(sequence: lastSequence, site: site, kernelCount: kernelCount,
                                encodeSeconds: encodeSeconds,
                                hostBeforeCommit: hostBeforeCommit,
                                hostAfterCommit: hostAfterCommit, hostAfterWait: hostAfterWait,
                                gpuStart: gpuStart, gpuEnd: gpuEnd,
                                completed: completed, statusRaw: statusRaw))
        }

        public func trace() -> Trace {
            Trace(capacity: capacity, droppedCount: droppedCount, events: events)
        }
    }

    /// Why an event was excluded from every derived quantity.
    public enum RejectionReason: String, Codable, Equatable, Sendable {
        case sequenceNotIncreasing
        case notCompleted
        case nonFinite
        case nonPositiveTimestamp
        case hostOrderReversed
        case negativeEncode
        case gpuReversed
        case gpuOutsideHostBounds
        case overlapsPrevious
    }

    public struct Rejection: Codable, Equatable, Sendable {
        public let sequence: Int
        public let reason: RejectionReason

        public init(sequence: Int, reason: RejectionReason) {
            self.sequence = sequence
            self.reason = reason
        }
    }

    /// Host wall time from one buffer's GPU end to the next buffer's
    /// commit, split at the first buffer's wait return. Derived only for
    /// two valid events with adjacent sequence numbers.
    public struct Gap: Codable, Equatable, Sendable {
        public let fromSequence: Int
        public let toSequence: Int
        public let gpuEndToNextCommitSeconds: Double
        public let gpuEndToWaitReturnSeconds: Double
        public let waitReturnToNextCommitSeconds: Double

        public init(fromSequence: Int, toSequence: Int, gpuEndToNextCommitSeconds: Double,
                    gpuEndToWaitReturnSeconds: Double, waitReturnToNextCommitSeconds: Double) {
            self.fromSequence = fromSequence
            self.toSequence = toSequence
            self.gpuEndToNextCommitSeconds = gpuEndToNextCommitSeconds
            self.gpuEndToWaitReturnSeconds = gpuEndToWaitReturnSeconds
            self.waitReturnToNextCommitSeconds = waitReturnToNextCommitSeconds
        }
    }

    /// Derived totals over VALID events only (seconds).
    public struct Summary: Codable, Equatable, Sendable {
        public let eventCount: Int
        public let validEventCount: Int
        public let rejections: [Rejection]
        public let droppedCount: Int
        public let truncated: Bool
        /// Sequence numbers absent from the trace before the last event.
        public let missingSequenceCount: Int
        public let encodeSeconds: Double
        public let encodeUnknownCount: Int
        public let commitCallSeconds: Double
        public let waitCallSeconds: Double
        public let gpuSeconds: Double
        public let commitToGpuStartSeconds: Double
        public let gpuEndToWaitReturnSeconds: Double
        public let gaps: [Gap]
        public let gapTotalSeconds: Double
        public let gapMaxSeconds: Double?
        /// Consecutive valid events not adjacent in sequence (a missing or
        /// rejected event lies between them); no gap is derived for them.
        public let gapPairsSkipped: Int
        /// First valid before-commit to last valid after-wait.
        public let hostSpanSeconds: Double?
    }

    static func rejection(for e: Event, lastSequence: Int,
                          previousValid: Event?) -> RejectionReason? {
        guard e.sequence > lastSequence else { return .sequenceNotIncreasing }
        guard e.completed else { return .notCompleted }
        let stamps = [e.hostBeforeCommit, e.hostAfterCommit, e.hostAfterWait, e.gpuStart, e.gpuEnd]
        guard stamps.allSatisfy({ $0.isFinite }), e.encodeSeconds?.isFinite ?? true else {
            return .nonFinite
        }
        guard stamps.allSatisfy({ $0 > 0 }) else { return .nonPositiveTimestamp }
        guard e.hostBeforeCommit <= e.hostAfterCommit,
              e.hostAfterCommit <= e.hostAfterWait else { return .hostOrderReversed }
        if let encode = e.encodeSeconds, encode < 0 { return .negativeEncode }
        guard e.gpuStart <= e.gpuEnd else { return .gpuReversed }
        guard e.gpuStart >= e.hostBeforeCommit, e.gpuEnd <= e.hostAfterWait else {
            return .gpuOutsideHostBounds
        }
        if let p = previousValid,
           e.hostBeforeCommit < p.hostAfterWait || e.gpuStart < p.gpuEnd {
            return .overlapsPrevious
        }
        return nil
    }

    /// Validates every event in order and derives totals and gaps from the
    /// valid ones only. Invalid values are reported, never clamped.
    public static func analyze(_ trace: Trace) -> Summary {
        var lastSequence = 0
        var missing = 0
        var previousValid: Event?
        var rejections: [Rejection] = []
        var gaps: [Gap] = []
        var gapPairsSkipped = 0
        var validCount = 0
        var encode = 0.0
        var encodeUnknown = 0
        var commitCall = 0.0
        var waitCall = 0.0
        var gpu = 0.0
        var commitToGpuStart = 0.0
        var gpuEndToWaitReturn = 0.0
        var firstCommit: Double?
        var lastWait: Double?
        for e in trace.events {
            if e.sequence > lastSequence {
                missing += e.sequence - lastSequence - 1
            }
            if let reason = rejection(for: e, lastSequence: lastSequence,
                                      previousValid: previousValid) {
                rejections.append(Rejection(sequence: e.sequence, reason: reason))
                lastSequence = max(lastSequence, e.sequence)
                continue
            }
            lastSequence = e.sequence
            validCount += 1
            if let s = e.encodeSeconds { encode += s } else { encodeUnknown += 1 }
            commitCall += e.hostAfterCommit - e.hostBeforeCommit
            waitCall += e.hostAfterWait - e.hostAfterCommit
            gpu += e.gpuEnd - e.gpuStart
            commitToGpuStart += e.gpuStart - e.hostBeforeCommit
            gpuEndToWaitReturn += e.hostAfterWait - e.gpuEnd
            if let p = previousValid {
                if e.sequence - p.sequence == 1 {
                    gaps.append(Gap(fromSequence: p.sequence, toSequence: e.sequence,
                                    gpuEndToNextCommitSeconds: e.hostBeforeCommit - p.gpuEnd,
                                    gpuEndToWaitReturnSeconds: p.hostAfterWait - p.gpuEnd,
                                    waitReturnToNextCommitSeconds: e.hostBeforeCommit - p.hostAfterWait))
                } else {
                    gapPairsSkipped += 1
                }
            }
            if firstCommit == nil { firstCommit = e.hostBeforeCommit }
            lastWait = e.hostAfterWait
            previousValid = e
        }
        return Summary(
            eventCount: trace.events.count,
            validEventCount: validCount,
            rejections: rejections,
            droppedCount: trace.droppedCount,
            truncated: trace.truncated,
            missingSequenceCount: missing,
            encodeSeconds: encode,
            encodeUnknownCount: encodeUnknown,
            commitCallSeconds: commitCall,
            waitCallSeconds: waitCall,
            gpuSeconds: gpu,
            commitToGpuStartSeconds: commitToGpuStart,
            gpuEndToWaitReturnSeconds: gpuEndToWaitReturn,
            gaps: gaps,
            gapTotalSeconds: gaps.reduce(0.0) { $0 + $1.gpuEndToNextCommitSeconds },
            gapMaxSeconds: gaps.map(\.gpuEndToNextCommitSeconds).max(),
            gapPairsSkipped: gapPairsSkipped,
            hostSpanSeconds: firstCommit.flatMap { first in lastWait.map { $0 - first } })
    }

    /// Standalone clock-basis probe: commits `commandBuffers` tiny blit
    /// fills one at a time on a fresh device queue and records them exactly
    /// as the runner sites do. Used to validate GPU timestamps against host
    /// bounds on real hardware; it never touches a runner.
    public static func deviceProbe(commandBuffers: Int = 3) throws -> Trace {
        guard commandBuffers >= 1, commandBuffers <= 64 else {
            throw Gemma4CommandTimelineError.invalidProbeCount(commandBuffers)
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let scratch = device.makeBuffer(length: 1 << 16, options: .storageModeShared) else {
            throw Gemma4Error.metalUnavailable
        }
        var recorder = try Recorder(capacity: commandBuffers)
        for i in 0..<commandBuffers {
            let encodeStart = hostSeconds()
            guard let cmd = queue.makeCommandBuffer(),
                  let blit = cmd.makeBlitCommandEncoder() else {
                throw Gemma4Error.metalUnavailable
            }
            blit.fill(buffer: scratch, range: 0..<scratch.length,
                      value: UInt8(truncatingIfNeeded: i))
            blit.endEncoding()
            let beforeCommit = hostSeconds()
            cmd.commit()
            let afterCommit = hostSeconds()
            cmd.waitUntilCompleted()
            let afterWait = hostSeconds()
            recorder.record(site: .deviceProbe, kernelCount: nil,
                            encodeSeconds: beforeCommit - encodeStart,
                            hostBeforeCommit: beforeCommit, hostAfterCommit: afterCommit,
                            hostAfterWait: afterWait,
                            gpuStart: cmd.gpuStartTime, gpuEnd: cmd.gpuEndTime,
                            completed: cmd.status == .completed && cmd.error == nil,
                            statusRaw: Int(cmd.status.rawValue))
        }
        return recorder.trace()
    }
}
