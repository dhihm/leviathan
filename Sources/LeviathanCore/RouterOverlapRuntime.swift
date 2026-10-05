// Default-off, prepared-SSA-only DIAGNOSTIC runtime for the router/shared
// FFN overlap schedule proven by RouterOverlapPlan.
//
// Per FFN window the executor follows the plan permutation
//   i0, i6, i7, i8 | commit R | i1 ... i5 | commit S | i9 (wait R) | i10, i11
// with ONLY the two remapped h versions (router_norm / router_scale
// outputs) bound to a dedicated routerIn buffer. R and S are committed
// asynchronously as tickets on the runner's own queue. Every deferred
// segment opened through beginSegment while the scope is active encodes an
// explicit wait on the last committed signal of one dedicated
// MTLSharedEvent BEFORE its compute encoder, and each ticket signals the
// next value, so ticket ordering never relies on implicit queue behavior.
// Direct synchronous dispatches and flushes do not wait on the event; they
// drain every committed ticket first. A beginSegment call with an encoder
// already open takes the ordinary flush-then-open path. router_topk drains tickets in commit
// order up to R only, then copies the completed router scores without
// flushing S; the CPU FP64 routing code is unchanged.
//
// The decisions about ordering, accounting and read legality live in pure
// shared types (ticket ledger, scope predicate, binding check, step
// derivation, alias guard, token expectation). The runner drives them with
// MTLCommandBuffer handles; the pure tests drive the same types with fake
// handles. No kernel math, MSL, candidate, plan schema or digest changes.
//
// Failure handling: the first failed ticket POISONS the ledger and state.
// Every committed ticket is still drained before any buffer reuse, but no
// further overlap operation (token preflight, segment open, commit, score
// read, synchronous submission) runs in that scope; the scope stays active
// so raw/control execution remains refused until exit. The next scope entry
// replaces a poisoned, fully drained state (new event, ledger and routerIn).
//
// NOT established by this source (requires later small GPU tests): whether
// Metal signals an event from a failed command buffer; whether the CPU
// unblock below (raising signaledValue) releases already-committed waiters;
// and whether a later, lower GPU encodeSignalEvent can lower signaledValue
// after a CPU raise (a review hypothesis: the SDK header only declares
// signaledValue readwrite). Poisoning and the post-drain re-raise are
// precautions, not proofs.

import Foundation
import Metal

public enum RouterOverlapRuntimeError: Error, Equatable, CustomStringConvertible {
    case rejected(String)

    public var description: String {
        switch self {
        case .rejected(let m): return "router overlap runtime rejected: \(m)"
        }
    }
}

// MARK: - Ticket ledger (shared by the runner and the pure tests)

/// The command-buffer operations the ledger needs. The runner adapts
/// MTLCommandBuffer; tests adapt fakes. No completion handlers.
public protocol RouterOverlapCommandHandle: AnyObject {
    func commitHandle()
    func waitHandle()
    var handleSucceeded: Bool { get }
    /// Command-buffer interval in microseconds (gpuEndTime - gpuStartTime).
    /// May include an encoded event wait; never busy time or utilization.
    var handleIntervalMicros: Double { get }
    var handleFailureDescription: String { get }
}

public enum RouterOverlapTicketKind: String, Sendable {
    /// Segment ending at router_proj: writes the router score buffer.
    case router
    /// Segment ending at shared_post1: never writes the router score buffer.
    case shared
}

public struct RouterOverlapTicket<Handle: RouterOverlapCommandHandle> {
    public let handle: Handle
    public let kind: RouterOverlapTicketKind
    public let signal: UInt64
}

public struct RouterOverlapDrainOutcome: Equatable, Sendable {
    public var drained = 0
    public var succeeded = 0
    public var intervalMicros = 0.0
    public var firstFailure: String?
    public var unblockedThrough: UInt64?
}

/// Single-threaded ledger of committed, not yet drained tickets. The
/// signal counter starts at the event's current value, only increases and
/// is never reset while the event is retained; overflow fails BEFORE any
/// encode or commit. Each successfully completed ticket's command-buffer
/// interval (gpuEndTime - gpuStartTime) is accounted exactly once; it may
/// include the ticket's encoded event wait and is NEVER busy time or
/// utilization. The first failed ticket POISONS the ledger: every later
/// commit is refused before encode, while draining already committed
/// tickets keeps working so cleanup can finish.
public final class RouterOverlapTicketLedger<Handle: RouterOverlapCommandHandle> {
    public private(set) var lastCommittedSignal: UInt64
    public private(set) var pending: [RouterOverlapTicket<Handle>] = []
    public private(set) var asyncCommitted = 0
    public private(set) var drained = 0
    public private(set) var failedDrains = 0
    public private(set) var intervalMicrosAccounted = 0.0
    public private(set) var lastSucceededRouterSignal: UInt64 = 0
    /// First failure observed by any drain; non-nil means poisoned.
    public private(set) var firstFailure: String?

    public var isPoisoned: Bool { firstFailure != nil }

    public init(lastCommittedSignal: UInt64 = 0) {
        self.lastCommittedSignal = lastCommittedSignal
    }

    public var hasPending: Bool { !pending.isEmpty }

    /// True while any pending ticket may still write the router scores.
    public var pendingRouterWrites: Bool { pending.contains { $0.kind == .router } }

    /// Reserves the next signal value, lets the caller end encoding and
    /// encode the signal, commits the handle exactly once and records it.
    @discardableResult
    public func commit(_ handle: Handle, kind: RouterOverlapTicketKind,
                       encodeSignal: (UInt64) -> Void) throws -> UInt64 {
        if let failure = firstFailure {
            throw RouterOverlapRuntimeError.rejected(
                "ledger poisoned by a failed ticket (" + failure + "); nothing committed")
        }
        let (next, overflow) = lastCommittedSignal.addingReportingOverflow(1)
        guard !overflow else {
            throw RouterOverlapRuntimeError.rejected(
                "event signal sequence exhausted at \(lastCommittedSignal); nothing committed")
        }
        guard !pending.contains(where: { $0.handle === handle }) else {
            throw RouterOverlapRuntimeError.rejected("command handle is already pending")
        }
        encodeSignal(next)
        handle.commitHandle()
        lastCommittedSignal = next
        asyncCommitted += 1
        pending.append(RouterOverlapTicket(handle: handle, kind: kind, signal: next))
        return next
    }

    /// Waits for pending tickets in commit order up to `target` (nil: all).
    /// A failed ticket poisons the ledger, extends the drain to EVERY
    /// committed ticket, calls `unblock` with the highest committed signal
    /// before any later wait, and once more after the last wait.
    public func drain(through target: UInt64?,
                      unblock: (UInt64) -> Void) -> RouterOverlapDrainOutcome {
        var outcome = RouterOverlapDrainOutcome()
        var limit = target
        while let ticket = pending.first {
            if let bound = limit, ticket.signal > bound { break }
            pending.removeFirst()
            ticket.handle.waitHandle()
            drained += 1
            outcome.drained += 1
            if ticket.handle.handleSucceeded {
                outcome.succeeded += 1
                let micros = ticket.handle.handleIntervalMicros
                if micros.isFinite, micros >= 0 {
                    intervalMicrosAccounted += micros
                    outcome.intervalMicros += micros
                }
                if ticket.kind == .router { lastSucceededRouterSignal = ticket.signal }
            } else {
                failedDrains += 1
                if firstFailure == nil {
                    firstFailure = "\(ticket.kind.rawValue) ticket \(ticket.signal): "
                        + ticket.handle.handleFailureDescription
                }
                if outcome.firstFailure == nil {
                    outcome.firstFailure = "\(ticket.kind.rawValue) ticket \(ticket.signal): "
                        + ticket.handle.handleFailureDescription
                }
                limit = nil
                if outcome.unblockedThrough == nil {
                    outcome.unblockedThrough = lastCommittedSignal
                    unblock(lastCommittedSignal)
                }
            }
        }
        // Re-raise after the LAST wait of a failing drain: tickets completing
        // after the first raise executed their own lower encodeSignalEvent
        // values. Whether that can lower signaledValue is a review
        // hypothesis, not established; the re-raise is cheap and the
        // poisoned ledger refuses every further commit regardless.
        if outcome.firstFailure != nil { unblock(lastCommittedSignal) }
        return outcome
    }

    /// nil when the counters are mutually consistent.
    public var consistencyRejection: String? {
        guard drained + pending.count == asyncCommitted else {
            return "drained \(drained) + pending \(pending.count) != committed \(asyncCommitted)"
        }
        guard failedDrains <= drained else { return "failed drains exceed drains" }
        guard intervalMicrosAccounted.isFinite, intervalMicrosAccounted >= 0 else {
            return "accounted command-buffer interval is not finite and nonnegative"
        }
        guard (failedDrains == 0) == (firstFailure == nil) else {
            return "failure record disagrees with the failed drain count"
        }
        var previous: UInt64 = 0
        for ticket in pending {
            guard ticket.signal > previous, ticket.signal <= lastCommittedSignal else {
                return "pending signals are not strictly increasing"
            }
            previous = ticket.signal
        }
        guard lastSucceededRouterSignal <= lastCommittedSignal else {
            return "router success beyond the committed sequence"
        }
        return nil
    }
}

// MARK: - Router score read guard

public enum RouterOverlapReadGuard {
    /// nil when the router scores may be copied without a flush: the
    /// requested buffer IS the router score buffer, the router ticket that
    /// wrote it completed successfully, no pending ticket writes it, and
    /// the checked byte count fits. The ticket `kind` is asserted by the
    /// caller; that a shared ticket never writes the router scores is
    /// justified only by the plan's physical replay, not checked here.
    public static func routerReadRejection(requested: ObjectIdentifier,
                                           router: ObjectIdentifier,
                                           routerSignal: UInt64,
                                           lastSucceededRouterSignal: UInt64,
                                           pendingRouterWrites: Bool,
                                           count: Int, bufferLength: Int) -> String? {
        guard requested == router else {
            return "foreign buffer: only the router score buffer is readable here"
        }
        guard routerSignal > 0, lastSucceededRouterSignal == routerSignal else {
            return "router ticket \(routerSignal) has not completed successfully"
        }
        guard !pendingRouterWrites else {
            return "a pending ticket may still write the router scores"
        }
        guard count > 0 else { return "score count must be positive" }
        let (bytes, overflow) = count.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard !overflow, bytes <= bufferLength else {
            return "score read exceeds the router buffer"
        }
        return nil
    }
}

// MARK: - Scope predicate

public enum RouterOverlapScopePhase: Sendable {
    /// withRouterOverlapDiagnosticScope entry (nesting refused).
    case entry
    /// Whole-token preflight before any GPU work.
    case token
}

/// Plain values the scope predicate decides on; the runner fills it from
/// its actual state.
public struct RouterOverlapScopeSnapshot: Equatable, Sendable {
    public var scopeActive: Bool
    public var eventAndScratchReady: Bool
    public var openEncoderPresent: Bool
    public var pendingTickets: Int
    public var expertCacheEnabled: Bool
    public var residentWeightsMapped: Bool
    public var moeBatch: Int?
    public var moeRowsPerGroup: Int?
    /// Entry: the installed routed source is the recorded E4_R32 emission.
    /// Token: MoE artifact digest and Q4 width unchanged since entry.
    public var installIdentityVerified: Bool
    public var q4LaunchWidth: Int?
    public var rmsGeneratedInstalled: Bool
    /// Every per-call axis flag is at its default outside execution.
    public var persistentAxisFlagsClean: Bool
    public var layerProbeInstalled: Bool
    public var moeInputProbeInstalled: Bool
    public var traceLayers: Bool
    public var breakdownInstrumentation: Bool
    public var commandTimelineEnabled: Bool
    public var attentionDiagnosticActive: Bool

    public init(scopeActive: Bool, eventAndScratchReady: Bool, openEncoderPresent: Bool,
                pendingTickets: Int, expertCacheEnabled: Bool, residentWeightsMapped: Bool,
                moeBatch: Int?, moeRowsPerGroup: Int?, installIdentityVerified: Bool,
                q4LaunchWidth: Int?, rmsGeneratedInstalled: Bool,
                persistentAxisFlagsClean: Bool, layerProbeInstalled: Bool,
                moeInputProbeInstalled: Bool, traceLayers: Bool,
                breakdownInstrumentation: Bool, commandTimelineEnabled: Bool,
                attentionDiagnosticActive: Bool) {
        self.scopeActive = scopeActive
        self.eventAndScratchReady = eventAndScratchReady
        self.openEncoderPresent = openEncoderPresent
        self.pendingTickets = pendingTickets
        self.expertCacheEnabled = expertCacheEnabled
        self.residentWeightsMapped = residentWeightsMapped
        self.moeBatch = moeBatch
        self.moeRowsPerGroup = moeRowsPerGroup
        self.installIdentityVerified = installIdentityVerified
        self.q4LaunchWidth = q4LaunchWidth
        self.rmsGeneratedInstalled = rmsGeneratedInstalled
        self.persistentAxisFlagsClean = persistentAxisFlagsClean
        self.layerProbeInstalled = layerProbeInstalled
        self.moeInputProbeInstalled = moeInputProbeInstalled
        self.traceLayers = traceLayers
        self.breakdownInstrumentation = breakdownInstrumentation
        self.commandTimelineEnabled = commandTimelineEnabled
        self.attentionDiagnosticActive = attentionDiagnosticActive
    }
}

public enum RouterOverlapScope {
    public static let requiredMoeBatch = 4
    public static let requiredMoeRowsPerGroup = 32
    /// The one Q4 launch width (besides the default nil) the scope accepts.
    public static let permittedQ4LaunchWidth = 32

    /// nil when the snapshot qualifies for `phase`.
    public static func rejection(_ s: RouterOverlapScopeSnapshot,
                                 phase: RouterOverlapScopePhase) -> String? {
        switch phase {
        case .entry:
            if s.scopeActive { return "nested router overlap scope refused" }
        case .token:
            if !s.scopeActive { return "no active router overlap scope" }
            if !s.eventAndScratchReady { return "dedicated event/routerIn not allocated" }
        }
        if s.openEncoderPresent { return "a deferred encoder is open" }
        if s.pendingTickets != 0 { return "\(s.pendingTickets) tickets still pending" }
        if s.expertCacheEnabled { return "expert-cache (SSD) mode is not supported" }
        if !s.residentWeightsMapped { return "resident weight mapping missing" }
        guard s.moeBatch == requiredMoeBatch, s.moeRowsPerGroup == requiredMoeRowsPerGroup else {
            return "installed routed lowering must be v1 E4_R32"
        }
        if !s.installIdentityVerified { return "installed lowering identity not verified" }
        if let width = s.q4LaunchWidth, width != permittedQ4LaunchWidth {
            return "Q4 launch width \(width) is not the standard \(permittedQ4LaunchWidth)"
        }
        if s.rmsGeneratedInstalled { return "generated RMS kernel would pre-empt statistics reuse" }
        if !s.persistentAxisFlagsClean { return "runner axis flags are not at their defaults" }
        if s.layerProbeInstalled { return "layerProbe must be nil" }
        if s.moeInputProbeInstalled { return "moeInputProbe must be nil" }
        if s.traceLayers { return "traceLayers must be off" }
        if s.breakdownInstrumentation { return "breakdown instrumentation must be off" }
        if s.commandTimelineEnabled { return "command timeline must be off" }
        if s.attentionDiagnosticActive { return "attention diagnostic must not be installed" }
        return nil
    }

    /// What a caller is about to execute while a router overlap state may
    /// exist.
    public enum ExecutionKind: Equatable, Sendable {
        /// A diagnostic preparation token (requires a clean active scope).
        case diagnosticPrepared
        /// Anything refused inside an active scope (raw/control execution,
        /// direct forward, engineIdentity, forge), poisoned or not.
        case outsideScopeOnly(String)
    }

    /// nil when the execution may proceed. A poisoned scope stays active,
    /// so outside-only work is still refused until scope exit; outside the
    /// scope a retained poisoned state never blocks control execution.
    public static func executionRejection(_ kind: ExecutionKind, scopeActive: Bool,
                                          poisoned: Bool) -> String? {
        switch kind {
        case .diagnosticPrepared:
            guard scopeActive else {
                return "a diagnostic preparation executes only inside the router overlap scope"
            }
            guard !poisoned else {
                return "router overlap scope is poisoned by a failed ticket; no further overlap execution"
            }
            return nil
        case .outsideScopeOnly(let what):
            guard scopeActive else { return nil }
            return what + " refused inside the router overlap diagnostic scope"
                + (poisoned ? " (scope poisoned)" : "")
        }
    }

    public enum StateDisposition: Equatable, Sendable {
        case allocate
        case reuse
        /// Poisoned and fully drained: replace event, ledger and routerIn.
        case reallocate
        case refuse
    }

    /// Scope-entry decision for the retained runtime state.
    public static func stateDisposition(hasState: Bool, poisoned: Bool,
                                        pendingTickets: Int) -> StateDisposition {
        guard hasState else { return .allocate }
        guard pendingTickets == 0 else { return .refuse }
        return poisoned ? .reallocate : .reuse
    }

    /// nil when the candidate is a legal, valid, full modern v1 candidate
    /// with router GPU residency (any allowed launch divisor).
    public static func candidateRejection(_ c: GemmaForgeCandidate) -> String? {
        guard GemmaForge.candidates().contains(c) else {
            return "candidate is not in the legal set"
        }
        do { try GemmaForge.validateCandidate(c) } catch {
            return "candidate invalid: \(error)"
        }
        let required = [c.segmentBatching, c.vectorizedLoads, c.gpuCacheAppend,
                        c.parallelAttention, c.rmsnormCooperative, c.matvecCooperativeQ6,
                        c.rmsStatisticsReuse, c.routerGPUResidency]
        let forbidden = [c.top5Diagnostic, c.matvecTiledQ4, c.matvecTiledQ6,
                         c.rmsnormTiled, c.matvecTiledF32]
        guard !required.contains(false), !forbidden.contains(true) else {
            return "candidate is not the full modern v1 flag set"
        }
        return nil
    }
}

// MARK: - Dedicated scratch alias guard

public struct RouterOverlapBufferExtent: Equatable, Sendable {
    public let name: String
    public let identity: ObjectIdentifier
    /// CPU address of the shared-storage contents.
    public let base: UInt
    public let length: Int

    public init(name: String, identity: ObjectIdentifier, base: UInt, length: Int) {
        self.name = name
        self.identity = identity
        self.base = base
        self.length = length
    }
}

public enum RouterOverlapAliasGuard {
    /// nil when the dedicated buffer is large enough, is no other buffer
    /// object, and its byte range overlaps no other buffer's range.
    public static func rejection(dedicated: RouterOverlapBufferExtent, requiredBytes: Int,
                                 others: [RouterOverlapBufferExtent]) -> String? {
        guard requiredBytes > 0, dedicated.length >= requiredBytes else {
            return "\(dedicated.name) holds \(dedicated.length) bytes, needs \(requiredBytes)"
        }
        guard let dedicatedEnd = end(dedicated) else {
            return "\(dedicated.name) address range overflows"
        }
        for other in others {
            if other.identity == dedicated.identity {
                return "\(dedicated.name) is the same object as \(other.name)"
            }
            guard let otherEnd = end(other) else {
                return "\(other.name) address range is invalid"
            }
            if other.length > 0, dedicated.base < otherEnd, other.base < dedicatedEnd {
                return "\(dedicated.name) overlaps \(other.name)"
            }
        }
        return nil
    }

    private static func end(_ e: RouterOverlapBufferExtent) -> UInt? {
        guard e.length >= 0 else { return nil }
        let (sum, overflow) = e.base.addingReportingOverflow(UInt(e.length))
        return overflow ? nil : sum
    }
}

// MARK: - Prepare-time plan binding

/// Values the prepare-time binding check decides on.
public struct RouterOverlapBindingFacts: Equatable, Sendable {
    public var planAnalysis: Int
    public var digestVerified: Bool
    public var canonicalScheduleMatches: Bool
    public var scheduleDigestMatches: Bool
    public var statisticsDigestMatches: Bool
    public var statisticsAnalysisMatches: Bool
    public var planEpsilonBits: UInt32
    public var statisticsEpsilonBits: UInt32
    public var runnerEpsilonBits: UInt32
    public var configMatches: Bool
    public var layerCount: Int
    public var windowCount: Int
    public var instructionCount: Int
    public var permutationCount: Int
    public var remappedCount: Int
    public var uniqueRemappedCount: Int
    public var remappedMatchesWindows: Bool
    public var controlRejectedIndexMatches: Bool
    public var statisticsGroups: Int
    public var statisticsReuses: Int

    public init(planAnalysis: Int, digestVerified: Bool, canonicalScheduleMatches: Bool,
                scheduleDigestMatches: Bool, statisticsDigestMatches: Bool,
                statisticsAnalysisMatches: Bool, planEpsilonBits: UInt32,
                statisticsEpsilonBits: UInt32, runnerEpsilonBits: UInt32,
                configMatches: Bool, layerCount: Int, windowCount: Int,
                instructionCount: Int, permutationCount: Int, remappedCount: Int,
                uniqueRemappedCount: Int, remappedMatchesWindows: Bool,
                controlRejectedIndexMatches: Bool, statisticsGroups: Int,
                statisticsReuses: Int) {
        self.planAnalysis = planAnalysis
        self.digestVerified = digestVerified
        self.canonicalScheduleMatches = canonicalScheduleMatches
        self.scheduleDigestMatches = scheduleDigestMatches
        self.statisticsDigestMatches = statisticsDigestMatches
        self.statisticsAnalysisMatches = statisticsAnalysisMatches
        self.planEpsilonBits = planEpsilonBits
        self.statisticsEpsilonBits = statisticsEpsilonBits
        self.runnerEpsilonBits = runnerEpsilonBits
        self.configMatches = configMatches
        self.layerCount = layerCount
        self.windowCount = windowCount
        self.instructionCount = instructionCount
        self.permutationCount = permutationCount
        self.remappedCount = remappedCount
        self.uniqueRemappedCount = uniqueRemappedCount
        self.remappedMatchesWindows = remappedMatchesWindows
        self.controlRejectedIndexMatches = controlRejectedIndexMatches
        self.statisticsGroups = statisticsGroups
        self.statisticsReuses = statisticsReuses
    }
}

public enum RouterOverlapBindingCheck {
    /// nil when the plan binds the prepared schedule, statistics, epsilon
    /// and config exactly. Old plans are never retagged: a different
    /// analysis version is a rejection.
    public static func rejection(_ f: RouterOverlapBindingFacts) -> String? {
        guard f.planAnalysis == RouterOverlapPlan.analysisVersion else {
            return "plan analysis \(f.planAnalysis) != \(RouterOverlapPlan.analysisVersion)"
        }
        guard f.digestVerified else { return "plan digest does not verify" }
        guard f.canonicalScheduleMatches else { return "plan schedule is not the prepared SSA" }
        guard f.scheduleDigestMatches else { return "plan schedule digest mismatch" }
        guard f.statisticsDigestMatches else { return "statistics plan digest mismatch" }
        guard f.statisticsAnalysisMatches else { return "statistics analysis version mismatch" }
        guard f.planEpsilonBits == f.statisticsEpsilonBits,
              f.planEpsilonBits == f.runnerEpsilonBits else {
            return "epsilon bits differ between plan, statistics and runner"
        }
        guard f.configMatches else { return "graph config mismatch" }
        guard f.layerCount > 0, f.windowCount == f.layerCount else {
            return "window count \(f.windowCount) != layer count \(f.layerCount)"
        }
        guard f.instructionCount > 0, f.permutationCount == f.instructionCount else {
            return "permutation length \(f.permutationCount) != \(f.instructionCount)"
        }
        let (twice, overflow) = f.windowCount.multipliedReportingOverflow(by: 2)
        guard !overflow, f.remappedCount == twice,
              f.uniqueRemappedCount == f.remappedCount, f.remappedMatchesWindows else {
            return "remapped value set is not exactly two values per window"
        }
        guard f.controlRejectedIndexMatches else {
            return "control replay rejection index is not the first shared gate"
        }
        guard f.statisticsGroups == f.windowCount, f.statisticsReuses == twice else {
            return "statistics shape is not one group and two reuses per window"
        }
        return nil
    }
}

/// Runtime action after (or, for awaitRouter, at) a canonical instruction.
public enum RouterOverlapStep: Equatable, Sendable {
    case plain
    case commitRouter(layer: Int)
    case commitShared(layer: Int)
    case awaitRouter(layer: Int)
}

/// Immutable prepare-time binding of one analyzed plan to one prepared SSA:
/// the permutation, a dense per-instruction step table and the remapped
/// value set, all derived ONCE. Nothing is hashed or re-analyzed per token.
public struct RouterOverlapPreparedBinding: Sendable {
    public let plan: RouterOverlapPlan
    public let permutation: [Int]
    public let steps: [RouterOverlapStep]
    public let remappedValueIDs: Set<String>
    public let windowCount: Int
    public let statisticsGroups: Int
    public let statisticsReuses: Int

    /// Checked accessor; out-of-range indices are plain.
    public func step(at index: Int) -> RouterOverlapStep {
        guard index >= 0, index < steps.count else { return .plain }
        return steps[index]
    }

    /// Dense step table: commitRouter at i8, commitShared at i5 and
    /// awaitRouter at i9 of every window, after checking that each window
    /// is 12 contiguous in-range indices executed as
    /// i0, i6, i7, i8, i1 ... i5, i9, i10, i11 and that no position outside
    /// a window is permuted.
    public static func deriveSteps(windows: [(layer: Int, canonicalIndices: [Int])],
                                   permutation: [Int],
                                   instructionCount n: Int) throws -> [RouterOverlapStep] {
        func reject(_ m: String) -> RouterOverlapRuntimeError { .rejected("steps: " + m) }
        guard n > 0, permutation.count == n,
              permutation.allSatisfy({ $0 >= 0 && $0 < n }),
              Set(permutation).count == n else {
            throw reject("order is not a permutation of \(n) instructions")
        }
        guard !windows.isEmpty else { throw reject("no windows") }
        var steps = [RouterOverlapStep](repeating: .plain, count: n)
        var covered = [Bool](repeating: false, count: n)
        var layers = Set<Int>()
        for w in windows {
            let idx = w.canonicalIndices
            guard layers.insert(w.layer).inserted else {
                throw reject("duplicate layer \(w.layer)")
            }
            guard idx.count == 12, let i0 = idx.first, i0 >= 0, i0 <= n - 12,
                  idx == Array(i0..<(i0 + 12)) else {
                throw reject("layer \(w.layer) window is not 12 contiguous in-range indices")
            }
            for j in idx {
                guard !covered[j] else { throw reject("windows overlap at \(j)") }
                covered[j] = true
            }
            let expected = [idx[0], idx[6], idx[7], idx[8], idx[1], idx[2],
                            idx[3], idx[4], idx[5], idx[9], idx[10], idx[11]]
            guard Array(permutation[i0..<(i0 + 12)]) == expected else {
                throw reject("layer \(w.layer) order is not i0, i6, i7, i8, i1..i5, i9, i10, i11")
            }
            steps[idx[8]] = .commitRouter(layer: w.layer)
            steps[idx[5]] = .commitShared(layer: w.layer)
            steps[idx[9]] = .awaitRouter(layer: w.layer)
        }
        for i in 0..<n where !covered[i] && permutation[i] != i {
            throw reject("position \(i) outside every window is permuted")
        }
        return steps
    }

    /// Prepare-time only: verifies the plan digest, computes the schedule
    /// and statistics digests once, checks every binding fact and derives
    /// the step table.
    static func make(plan: RouterOverlapPlan, ssa: GemmaSSAIR,
                     statisticsPlan: RMSStatisticsPlan,
                     config: GemmaGraphIR.GraphConfig,
                     runnerEpsilon: Float) throws -> RouterOverlapPreparedBinding {
        var digestVerified = true
        do { try RouterOverlapPlan.verifyDigest(plan) } catch { digestVerified = false }
        let scheduleDigest = try? GemmaSSA.digest(ssa)
        let statisticsDigest = try? RouterOverlapPlan.digestOfStatisticsPlan(statisticsPlan)
        let remapped = Set(plan.remappedValueIDs)
        let controlMatches = plan.windows.first.map {
            $0.canonicalIndices.count > 1
                && plan.controlRejectedIndex == $0.canonicalIndices[1]
        } ?? false
        let facts = RouterOverlapBindingFacts(
            planAnalysis: plan.analysis, digestVerified: digestVerified,
            canonicalScheduleMatches: plan.canonicalSchedule == ssa,
            scheduleDigestMatches: scheduleDigest == plan.canonicalScheduleDigest,
            statisticsDigestMatches: statisticsDigest == plan.statisticsPlanDigest,
            statisticsAnalysisMatches: plan.statisticsAnalysis == statisticsPlan.analysis,
            planEpsilonBits: plan.epsilonBits,
            statisticsEpsilonBits: statisticsPlan.epsilonBits,
            runnerEpsilonBits: runnerEpsilon.bitPattern,
            configMatches: plan.config == config,
            layerCount: config.layerCount, windowCount: plan.windows.count,
            instructionCount: ssa.instructions.count,
            permutationCount: plan.permutation.count,
            remappedCount: plan.remappedValueIDs.count,
            uniqueRemappedCount: remapped.count,
            remappedMatchesWindows: plan.windows.flatMap(\.remappedValueIDs)
                == plan.remappedValueIDs,
            controlRejectedIndexMatches: controlMatches,
            statisticsGroups: statisticsPlan.groupCount,
            statisticsReuses: statisticsPlan.reuseCount)
        if let reason = RouterOverlapBindingCheck.rejection(facts) {
            throw GemmaForgeError.invalidPlan("router overlap binding rejected: " + reason)
        }
        let steps = try deriveSteps(
            windows: plan.windows.map { (layer: $0.layer, canonicalIndices: $0.canonicalIndices) },
            permutation: plan.permutation, instructionCount: ssa.instructions.count)
        return RouterOverlapPreparedBinding(
            plan: plan, permutation: plan.permutation, steps: steps,
            remappedValueIDs: remapped, windowCount: plan.windows.count,
            statisticsGroups: statisticsPlan.groupCount,
            statisticsReuses: statisticsPlan.reuseCount)
    }
}

// MARK: - Per-token counter expectation

public struct RouterOverlapCounters: Equatable, Sendable {
    public var applied: Int
    public var asyncCommitted: Int
    public var drained: Int
    public var failedDrains: Int
    public var pending: Int
    public var statsComputed: Int
    public var statsReused: Int
    public var statsFallback: Int
    public var routerScaleFallback: Int
    public var residentZeroFallback: Int
    public var routerLaunchFallback: Int
    public var moeFallback: Int

    public init(applied: Int, asyncCommitted: Int, drained: Int, failedDrains: Int,
                pending: Int, statsComputed: Int, statsReused: Int, statsFallback: Int,
                routerScaleFallback: Int, residentZeroFallback: Int,
                routerLaunchFallback: Int, moeFallback: Int) {
        self.applied = applied
        self.asyncCommitted = asyncCommitted
        self.drained = drained
        self.failedDrains = failedDrains
        self.pending = pending
        self.statsComputed = statsComputed
        self.statsReused = statsReused
        self.statsFallback = statsFallback
        self.routerScaleFallback = routerScaleFallback
        self.residentZeroFallback = residentZeroFallback
        self.routerLaunchFallback = routerLaunchFallback
        self.moeFallback = moeFallback
    }
}

public enum RouterOverlapTokenExpectation {
    /// nil when one overlap token applied every window exactly once: two
    /// async commits and two drains per window, nothing failed or left
    /// pending, exact statistics compute/reuse counts, and zero fallbacks.
    public static func rejection(before b: RouterOverlapCounters,
                                 after a: RouterOverlapCounters,
                                 windows: Int, statisticsGroups: Int,
                                 statisticsReuses: Int) -> String? {
        let (twice, overflow) = windows.multipliedReportingOverflow(by: 2)
        guard windows > 0, !overflow else { return "window count invalid" }
        let checks: [(String, Int, Int)] = [
            ("applied windows", a.applied &- b.applied, windows),
            ("async commits", a.asyncCommitted &- b.asyncCommitted, twice),
            ("drained tickets", a.drained &- b.drained, twice),
            ("failed drains", a.failedDrains &- b.failedDrains, 0),
            ("pending tickets at token end", a.pending, 0),
            ("statistics computes", a.statsComputed &- b.statsComputed, statisticsGroups),
            ("statistics reuses", a.statsReused &- b.statsReused, statisticsReuses),
            ("statistics fallbacks", a.statsFallback &- b.statsFallback, 0),
            ("router scale fallbacks", a.routerScaleFallback &- b.routerScaleFallback, 0),
            ("resident zero fallbacks", a.residentZeroFallback &- b.residentZeroFallback, 0),
            ("router launch fallbacks", a.routerLaunchFallback &- b.routerLaunchFallback, 0),
            ("MoE resident fallbacks", a.moeFallback &- b.moeFallback, 0),
        ]
        for (name, got, want) in checks where got != want {
            return "\(name): \(got) != \(want)"
        }
        return nil
    }
}

// MARK: - Metal adapter and runner state

final class RouterOverlapMetalHandle: RouterOverlapCommandHandle {
    let buffer: MTLCommandBuffer

    init(_ buffer: MTLCommandBuffer) { self.buffer = buffer }

    func commitHandle() { buffer.commit() }
    func waitHandle() { buffer.waitUntilCompleted() }
    var handleSucceeded: Bool { buffer.status == .completed && buffer.error == nil }
    /// Command-buffer interval; may include the encoded event wait. Never
    /// busy time or utilization.
    var handleIntervalMicros: Double { (buffer.gpuEndTime - buffer.gpuStartTime) * 1e6 }
    var handleFailureDescription: String {
        "status \(buffer.status.rawValue), \(String(describing: buffer.error))"
    }
}

/// Runner-owned overlap state: one dedicated event, one dedicated routerIn
/// buffer (hidden * 4 bytes) and the ticket ledger, allocated together at
/// the first scope entry and retained while clean; the signal counter
/// continues with the event and is never reset. A poisoned, fully drained
/// state is replaced as a whole at the next scope entry (the old event is
/// never reused).
final class RouterOverlapRuntimeState {
    let event: MTLSharedEvent
    let routerIn: MTLBuffer
    let ledger: RouterOverlapTicketLedger<RouterOverlapMetalHandle>
    var active = false
    var pendingRouterSignal: UInt64?
    var appliedWindows = 0
    var fallbackEvents = 0
    var entryMoeDigest: String?
    var entryQ4Width: Int?
    var lastDrainFailure: String?

    /// A failed ticket poisons the state until the next scope entry
    /// replaces it.
    var poisoned: Bool { ledger.isPoisoned }
    /// Synchronous submissions (flush, direct dispatch) must run the overlap
    /// guard first.
    var needsSynchronousGuard: Bool { ledger.hasPending || (active && poisoned) }

    init(event: MTLSharedEvent, routerIn: MTLBuffer) {
        self.event = event
        self.routerIn = routerIn
        self.ledger = RouterOverlapTicketLedger(lastCommittedSignal: event.signaledValue)
    }

    /// Raises (never lowers) the event so waiters behind a failed ticket
    /// can complete; their results are discarded by the failing run.
    func unblock(_ value: UInt64) {
        if event.signaledValue < value { event.signaledValue = value }
    }
}

// MARK: - Runner integration

extension Gemma4Runner {

    public var routerOverlapScopeActive: Bool { routerOverlapState?.active ?? false }
    /// Windows whose router scores were read from a completed R ticket.
    public var routerOverlapAppliedCount: Int { routerOverlapState?.appliedWindows ?? 0 }
    /// Refused tokens/windows inside the scope (qualified runs: 0).
    public var routerOverlapFallbackCount: Int { routerOverlapState?.fallbackEvents ?? 0 }
    public var routerOverlapAsyncCommittedCount: Int { routerOverlapState?.ledger.asyncCommitted ?? 0 }
    public var routerOverlapDrainedCount: Int { routerOverlapState?.ledger.drained ?? 0 }
    public var routerOverlapPendingCount: Int { routerOverlapState?.ledger.pending.count ?? 0 }
    public var routerOverlapPoisoned: Bool { routerOverlapState?.poisoned ?? false }
    /// Diagnostic only: summed command-buffer intervals of successfully
    /// completed async overlap tickets. May include event waits; never busy
    /// time or utilization. Not part of totalGpuMicros.
    public var routerOverlapCommandBufferIntervalMicros: Double { routerOverlapIntervalMicros }

    /// DIAGNOSTIC ONLY. Runs `body` with the router overlap runtime armed
    /// for diagnostic preparations (GemmaPreparedExecution
    /// .prepareRouterOverlapDiagnostic). Entry is refused (before any GPU
    /// work) when nested, with an open encoder or pending tickets, in
    /// expert-cache mode, without the installed v1 E4_R32 routed lowering,
    /// with a generated RMS kernel, non-default axis flags, any probe,
    /// trace, breakdown, timeline or attention diagnostic. On every exit
    /// the open encoder is discarded without submit and all committed
    /// tickets are drained; a body error is preserved. Never a serving
    /// default, never persisted, never production-selectable.
    public func withRouterOverlapDiagnosticScope<T>(_ body: () throws -> T) throws -> T {
        let entry = routerOverlapScopeSnapshot(phase: .entry, state: routerOverlapState)
        if let reason = RouterOverlapScope.rejection(entry, phase: .entry) {
            throw Gemma4Error.badTensorGeometry("router overlap scope refused: " + reason)
        }
        let state = try routerOverlapEnsureState()
        if let reason = routerOverlapAliasRejection(state) {
            throw Gemma4Error.badTensorGeometry("router overlap scope refused: " + reason)
        }
        state.entryMoeDigest = moeGenerated?.artifactDigest
        state.entryQ4Width = q4LaunchWidth
        state.pendingRouterSignal = nil
        state.lastDrainFailure = nil
        state.active = true
        let result: T
        do {
            result = try body()
        } catch {
            _ = routerOverlapCloseScope(state)
            throw error
        }
        if let problem = routerOverlapCloseScope(state) {
            throw Gemma4Error.badTensorGeometry("router overlap scope exit: " + problem)
        }
        guard moeGenerated?.artifactDigest == state.entryMoeDigest,
              q4LaunchWidth == state.entryQ4Width else {
            throw Gemma4Error.badTensorGeometry(
                "router overlap scope exit: install state was displaced during the scope")
        }
        return result
    }

    func routerOverlapRefuseInScope(_ what: String) throws {
        if let reason = RouterOverlapScope.executionRejection(
            .outsideScopeOnly(what), scopeActive: routerOverlapScopeActive,
            poisoned: routerOverlapState?.poisoned ?? false) {
            throw Gemma4Error.badTensorGeometry(reason)
        }
    }

    /// Lazily allocates the dedicated event and routerIn buffer together;
    /// any failure throws before GPU work and stores nothing.
    func routerOverlapEnsureState() throws -> RouterOverlapRuntimeState {
        switch RouterOverlapScope.stateDisposition(
            hasState: routerOverlapState != nil,
            poisoned: routerOverlapState?.poisoned ?? false,
            pendingTickets: routerOverlapState?.ledger.pending.count ?? 0) {
        case .reuse:
            if let state = routerOverlapState { return state }
        case .refuse:
            throw Gemma4Error.badTensorGeometry(
                "router overlap: retained state still has pending tickets")
        case .allocate, .reallocate:
            break
        }
        let device = q4RunnerDevice
        guard let event = device.makeSharedEvent() else {
            throw Gemma4Error.badTensorGeometry("router overlap: event allocation failed")
        }
        let (bytes, overflow) = hidden.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard !overflow, bytes > 0,
              let routerIn = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw Gemma4Error.badTensorGeometry("router overlap: routerIn allocation failed")
        }
        routerIn.label = "leviathan.routerOverlap.routerIn"
        let state = RouterOverlapRuntimeState(event: event, routerIn: routerIn)
        routerOverlapState = state
        return state
    }

    func routerOverlapScopeSnapshot(phase: RouterOverlapScopePhase,
                                    state: RouterOverlapRuntimeState?) -> RouterOverlapScopeSnapshot {
        let installed = moeGenerated
        let identityVerified: Bool
        switch phase {
        case .entry:
            // Entry only (never per token): the installed routed region must
            // be the recorded v1 E4_R32 emission.
            if let installed {
                identityVerified = MoEKernelEmitter.sourcesSha256([installed.kernels.source])
                    == Q4LaunchArtifact.baselineMoeSourceSha256
            } else {
                identityVerified = false
            }
        case .token:
            identityVerified = installed != nil && state != nil
                && installed?.artifactDigest == state?.entryMoeDigest
                && q4LaunchWidth == state?.entryQ4Width
        }
        let axisFlags = [autoSegment, q4TiledKernel, q6TiledKernel, rmsTiledKernel,
                         f32TiledKernel, vectorizedLoads, gpuCacheAppendKernel,
                         parallelAttentionKernel, rmsCooperativeKernel,
                         matvecCooperativeQ6Kernel, rmsStatisticsReuse, routerGPUResidency]
        return RouterOverlapScopeSnapshot(
            scopeActive: state?.active ?? false,
            eventAndScratchReady: state != nil,
            openEncoderPresent: openEnc != nil || openCmd != nil,
            pendingTickets: state?.ledger.pending.count ?? 0,
            expertCacheEnabled: expertCache != nil,
            residentWeightsMapped: expertCache == nil && residentWeightsBuffer != nil,
            moeBatch: installed?.schedule.batch,
            moeRowsPerGroup: installed?.schedule.rowsPerGroup,
            installIdentityVerified: identityVerified,
            q4LaunchWidth: q4LaunchWidth,
            rmsGeneratedInstalled: rmsGeneratedKernel != nil,
            persistentAxisFlagsClean: !axisFlags.contains(true) && routerLaunchDivisor == 0,
            layerProbeInstalled: layerProbe != nil,
            moeInputProbeInstalled: moeInputProbe != nil,
            traceLayers: traceLayers,
            breakdownInstrumentation: breakdownInstrumentation,
            commandTimelineEnabled: commandTimelineEnabled,
            attentionDiagnosticActive: attentionDiagnosticActive)
    }

    /// routerIn must not be, nor overlap, any activation, expert, MoE,
    /// statistics, embedding, frequency, KV-cache or weight buffer.
    func routerOverlapAliasRejection(_ state: RouterOverlapRuntimeState) -> String? {
        func extent(_ name: String, _ b: MTLBuffer) -> RouterOverlapBufferExtent {
            RouterOverlapBufferExtent(name: name, identity: ObjectIdentifier(b),
                                      base: UInt(bitPattern: b.contents()), length: b.length)
        }
        let named: [(String, MTLBuffer?)] = [
            ("x", bufX), ("h", bufH), ("t1", bufT1), ("t2", bufT2), ("t3", bufT3),
            ("kv", bufKV), ("gu", bufGU), ("act", bufAct), ("shared", bufShared),
            ("router", bufRouter), ("embed", bufEmbed), ("logits", bufLogits),
            ("expertGU", bufExpertGU), ("expertAct", bufExpertAct),
            ("expertAcc", bufExpertAcc), ("localFreq", localFreqBuffer),
            ("globalFreq", globalFreqBuffer), ("weights", residentWeightsBuffer),
            ("rmsStatsInverse", rmsStatsInverseBuffer),
            ("moeScratchGU", moeScratch?.gu), ("moeScratchAct", moeScratch?.act),
            ("moeScratchDN", moeScratch?.dn),
        ]
        var others: [RouterOverlapBufferExtent] = []
        others.reserveCapacity(named.count + kCaches.count + vCaches.count)
        for (name, buffer) in named {
            if let buffer { others.append(extent(name, buffer)) }
        }
        for (i, b) in kCaches.enumerated() { others.append(extent("kCache\(i)", b)) }
        for (i, b) in vCaches.enumerated() { others.append(extent("vCache\(i)", b)) }
        let (required, overflow) = hidden.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard !overflow else { return "routerIn size overflows" }
        return RouterOverlapAliasGuard.rejection(dedicated: extent("routerIn", state.routerIn),
                                                 requiredBytes: required, others: others)
    }

    /// Whole-token guards before any GPU work of a diagnostic execution.
    func routerOverlapTokenPreflight(binding: RouterOverlapPreparedBinding) throws {
        if let reason = RouterOverlapScope.executionRejection(
            .diagnosticPrepared, scopeActive: routerOverlapScopeActive,
            poisoned: routerOverlapState?.poisoned ?? false) {
            routerOverlapState?.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry("router overlap token refused: " + reason)
        }
        guard let state = routerOverlapState, state.active else {
            throw Gemma4Error.badTensorGeometry(
                "router overlap: a diagnostic preparation executes only inside the scope")
        }
        let snapshot = routerOverlapScopeSnapshot(phase: .token, state: state)
        if let reason = RouterOverlapScope.rejection(snapshot, phase: .token) {
            state.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry("router overlap token refused: " + reason)
        }
        if let reason = routerOverlapAliasRejection(state) {
            state.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry("router overlap token refused: " + reason)
        }
        guard binding.windowCount == config.layerCount else {
            state.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry("router overlap: plan windows != model layers")
        }
    }

    func routerOverlapRouterIn() throws -> MTLBuffer {
        try routerOverlapLiveState("routerIn").routerIn
    }

    /// The active, unpoisoned state; every overlap operation goes through
    /// this gate.
    func routerOverlapLiveState(_ what: String) throws -> RouterOverlapRuntimeState {
        guard let state = routerOverlapState, state.active else {
            throw Gemma4Error.badTensorGeometry("router overlap: " + what + " outside the scope")
        }
        if let failure = state.ledger.firstFailure {
            throw Gemma4Error.commandBufferFailed(
                "router overlap poisoned, " + what + " refused: " + failure)
        }
        return state
    }

    /// Non-draining segment open: waits on the last signaled value BEFORE
    /// the compute encoder is created.
    func routerOverlapBeginSegment(_ state: RouterOverlapRuntimeState) throws {
        if let failure = state.ledger.firstFailure {
            throw Gemma4Error.commandBufferFailed(
                "router overlap poisoned, segment open refused: " + failure)
        }
        guard let cmd = q4RunnerQueue.makeCommandBuffer() else {
            throw Gemma4Error.metalUnavailable
        }
        let wait = state.ledger.lastCommittedSignal
        if wait > 0 { cmd.encodeWaitForEvent(state.event, value: wait) }
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw Gemma4Error.metalUnavailable
        }
        openCmd = cmd
        openEnc = enc
    }

    /// Commits the open segment asynchronously as one ticket signaling the
    /// next event value; submittedBuffers grows exactly once.
    func routerOverlapCommit(_ kind: RouterOverlapTicketKind) throws {
        let state = try routerOverlapLiveState("commit")
        guard let cmd = openCmd, let enc = openEnc else {
            state.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry(
                "router overlap: no open segment to commit as a \(kind.rawValue) ticket")
        }
        switch kind {
        case .router:
            guard state.pendingRouterSignal == nil else {
                throw Gemma4Error.badTensorGeometry("router overlap: previous router ticket unread")
            }
        case .shared:
            guard state.pendingRouterSignal != nil else {
                throw Gemma4Error.badTensorGeometry("router overlap: shared ticket without router ticket")
            }
        }
        let event = state.event
        let signal = try state.ledger.commit(RouterOverlapMetalHandle(cmd), kind: kind) { value in
            enc.endEncoding()
            cmd.encodeSignalEvent(event, value: value)
        }
        openCmd = nil
        openEnc = nil
        submittedBuffers += 1
        if kind == .router { state.pendingRouterSignal = signal }
    }

    func routerOverlapAfterInstruction(_ step: RouterOverlapStep) throws {
        switch step {
        case .commitRouter: try routerOverlapCommit(.router)
        case .commitShared: try routerOverlapCommit(.shared)
        case .plain, .awaitRouter: return
        }
    }

    /// router_topk under overlap: drains tickets in commit order up to the
    /// router ticket (S stays pending), then copies the completed scores
    /// without a flush after the read guard accepts.
    func routerOverlapRouterScores(buffer: MTLBuffer, count: Int) throws -> [Float] {
        let state = try routerOverlapLiveState("router score read")
        guard let routerSignal = state.pendingRouterSignal else {
            throw Gemma4Error.badTensorGeometry(
                "router overlap: router_topk without a committed router ticket")
        }
        guard openEnc == nil, openCmd == nil else {
            throw Gemma4Error.badTensorGeometry(
                "router overlap: router_topk reached with an uncommitted segment")
        }
        let outcome = state.ledger.drain(through: routerSignal) { state.unblock($0) }
        routerOverlapIntervalMicros += outcome.intervalMicros
        state.pendingRouterSignal = nil
        if let failure = outcome.firstFailure {
            state.lastDrainFailure = failure
            throw Gemma4Error.commandBufferFailed("router overlap ticket: " + failure)
        }
        let router: MTLBuffer = bufRouter
        if let reason = RouterOverlapReadGuard.routerReadRejection(
            requested: ObjectIdentifier(buffer), router: ObjectIdentifier(router),
            routerSignal: routerSignal,
            lastSucceededRouterSignal: state.ledger.lastSucceededRouterSignal,
            pendingRouterWrites: state.ledger.pendingRouterWrites,
            count: count, bufferLength: buffer.length) {
            throw Gemma4Error.badTensorGeometry("router overlap: score read refused: " + reason)
        }
        let scores = [Float](UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
        state.appliedWindows += 1
        return scores
    }

    /// Synchronous-submission guard (flush, direct dispatch): drains every
    /// committed ticket in commit order. On a failure in this drain, or
    /// while an ACTIVE scope is poisoned, discards the open encoder without
    /// submitting it and throws. Outside an active scope a retained
    /// poisoned state never blocks control execution.
    func routerOverlapDrainAll(_ state: RouterOverlapRuntimeState) throws {
        var failedNow: String?
        if state.ledger.hasPending {
            let outcome = state.ledger.drain(through: nil) { state.unblock($0) }
            routerOverlapIntervalMicros += outcome.intervalMicros
            failedNow = outcome.firstFailure
        }
        let failure = failedNow ?? (state.active ? state.ledger.firstFailure : nil)
        guard let failure else { return }
        state.lastDrainFailure = failure
        state.pendingRouterSignal = nil
        discardOpenSegment()
        throw Gemma4Error.commandBufferFailed("router overlap poisoned: " + failure)
    }

    /// Non-throwing drain for cleanup paths; works while poisoned. A failure
    /// is recorded (and poisons the ledger), never thrown, so the caller's
    /// original error is preserved.
    func routerOverlapDrainDiscarding(_ state: RouterOverlapRuntimeState) {
        let outcome = state.ledger.drain(through: nil) { state.unblock($0) }
        routerOverlapIntervalMicros += outcome.intervalMicros
        if let failure = outcome.firstFailure { state.lastDrainFailure = failure }
        state.pendingRouterSignal = nil
    }

    /// Scope exit: no open encoder and no pending ticket survive, even when
    /// poisoned. A poisoned or inconsistent ledger is reported as a problem.
    func routerOverlapCloseScope(_ state: RouterOverlapRuntimeState) -> String? {
        var problem: String?
        if openEnc != nil || openCmd != nil {
            problem = "uncommitted encoder at scope exit (discarded without submit)"
            discardOpenSegment()
        }
        if state.ledger.hasPending { routerOverlapDrainDiscarding(state) }
        if problem == nil, let failure = state.ledger.firstFailure {
            problem = "scope poisoned by a failed router overlap ticket: " + failure
        }
        if problem == nil, let reason = state.ledger.consistencyRejection {
            problem = "ledger inconsistent at scope exit: " + reason
        }
        state.pendingRouterSignal = nil
        state.active = false
        return problem
    }

    func routerOverlapCounterSnapshot() -> RouterOverlapCounters {
        let state = routerOverlapState
        return RouterOverlapCounters(
            applied: state?.appliedWindows ?? 0,
            asyncCommitted: state?.ledger.asyncCommitted ?? 0,
            drained: state?.ledger.drained ?? 0,
            failedDrains: state?.ledger.failedDrains ?? 0,
            pending: state?.ledger.pending.count ?? 0,
            statsComputed: rmsStatsComputedCount,
            statsReused: rmsStatsReusedCount,
            statsFallback: rmsStatsFallbackCount,
            routerScaleFallback: routerScaleFallback,
            residentZeroFallback: residentZeroFallback,
            routerLaunchFallback: routerLaunchFallback,
            moeFallback: moeResidentFallbackCount)
    }

    /// Terminal per-token check after the final logits read.
    func routerOverlapVerifyToken(binding: RouterOverlapPreparedBinding,
                                  since before: RouterOverlapCounters) throws {
        let after = routerOverlapCounterSnapshot()
        if let reason = RouterOverlapTokenExpectation.rejection(
            before: before, after: after, windows: binding.windowCount,
            statisticsGroups: binding.statisticsGroups,
            statisticsReuses: binding.statisticsReuses) {
            routerOverlapState?.fallbackEvents += 1
            throw Gemma4Error.badTensorGeometry("router overlap token expectation failed: " + reason)
        }
        guard openEnc == nil, routerOverlapState?.pendingRouterSignal == nil else {
            throw Gemma4Error.badTensorGeometry("router overlap: token ended with open work")
        }
        if let reason = routerOverlapState?.ledger.consistencyRejection {
            throw Gemma4Error.badTensorGeometry("router overlap ledger inconsistent: " + reason)
        }
    }
}
