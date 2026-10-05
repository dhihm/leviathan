// Phase 54 own-process memory, submission and readback diagnostics.
// Diagnostic only and default off: no artifact, no installation, no winner,
// no activation, no promotion; gainQualified, installable and
// productionTPOTClaim are always false.
//
// Measured on THIS process only:
// - Darwin task_info(mach_task_self_) TASK_EVENTS_INFO (cumulative faults,
//   pageins, copy-on-write faults, context switches) and TASK_VM_INFO
//   (virtual, resident, phys_footprint) at runner, gate, reference and run
//   boundaries, never inside a timed decode call. A failed call is recorded
//   as UNKNOWN with its kernel return code; no value is invented as zero.
// - MTLDevice.currentAllocatedSize as a point sample (allocation capacity,
//   not physical residency).
// - Source-derived K/V capacity from the runner's configured maxContext
//   (the existing 4096 bound) next to the actual allocated cache lengths.
// - The opt-in breakdown instrumentation split into declared nested
//   phases, read through Gemma4Runner.submissionBreakdown() (the original
//   breakdownCounters() tuple is unchanged).
//
// Protocol: the unchanged phase-53 gate runs first in qualification-only
// mode on the same runner (instrumentation off, all three fixtures and the
// guard refusal tests). Only if it passed: inside the Q4 reference width
// 32 the exact incumbent (v1 E4R32, declared modern candidate with par64
// and router launch divisor 1) is installed and prepared, and per fixture
// in fixed order a fresh direct reference is rerun, then the OFF arm is
// profiled once as a declared warmup and once as the measured run, both
// with breakdown instrumentation ON and both fully validated. ON (router
// overlap) is never profiled: its scope refuses instrumentation.

import Foundation
import Metal

public struct OwnProcessDiagnosticsError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

// MARK: - Own-process task_info

public enum OwnProcessDeltaStatus: String, Codable, Sendable {
    case known
    /// One side of the pair was not captured (task_info failed or no device).
    case unknown
    /// A cumulative counter decreased (wrap or reset); no value is reported.
    case nonMonotonic
}

public struct OwnProcessTaskEvents: Codable, Equatable, Sendable {
    public let status: String
    public let faults: Int64?
    public let pageins: Int64?
    public let cowFaults: Int64?
    public let contextSwitches: Int64?

    public init(status: String, faults: Int64?, pageins: Int64?, cowFaults: Int64?,
                contextSwitches: Int64?) {
        self.status = status
        self.faults = faults
        self.pageins = pageins
        self.cowFaults = cowFaults
        self.contextSwitches = contextSwitches
    }
}

public struct OwnProcessVMInfo: Codable, Equatable, Sendable {
    public let status: String
    public let virtualBytes: Int64?
    public let residentBytes: Int64?
    public let physFootprintBytes: Int64?

    public init(status: String, virtualBytes: Int64?, residentBytes: Int64?,
                physFootprintBytes: Int64?) {
        self.status = status
        self.virtualBytes = virtualBytes
        self.residentBytes = residentBytes
        self.physFootprintBytes = physFootprintBytes
    }
}

public struct OwnProcessTaskSnapshot: Codable, Equatable, Sendable {
    public let label: String
    public let uptimeNanos: UInt64
    public let events: OwnProcessTaskEvents
    public let vm: OwnProcessVMInfo
    public let metalStatus: String
    public let metalCurrentAllocatedBytes: Int64?

    public init(label: String, uptimeNanos: UInt64, events: OwnProcessTaskEvents,
                vm: OwnProcessVMInfo, metalStatus: String, metalCurrentAllocatedBytes: Int64?) {
        self.label = label
        self.uptimeNanos = uptimeNanos
        self.events = events
        self.vm = vm
        self.metalStatus = metalStatus
        self.metalCurrentAllocatedBytes = metalCurrentAllocatedBytes
    }
}

public struct OwnProcessCounterDelta: Codable, Equatable, Sendable {
    public let name: String
    public let monotonic: Bool
    public let before: Int64?
    public let after: Int64?
    public let delta: Int64?
    public let status: OwnProcessDeltaStatus
}

public struct OwnProcessSnapshotDelta: Codable, Equatable, Sendable {
    public let from: String
    public let to: String
    public let elapsedNanos: UInt64?
    public let counters: [OwnProcessCounterDelta]
}

public enum OwnProcessInfo {
    /// Native TASK_EVENTS_INFO word count (sizeof / sizeof(natural_t)).
    public static var eventsInfoCount: mach_msg_type_number_t {
        mach_msg_type_number_t(MemoryLayout<task_events_info_data_t>.stride
                               / MemoryLayout<natural_t>.stride)
    }

    /// Native TASK_VM_INFO word count of the struct this binary was built with.
    public static var vmInfoCount: mach_msg_type_number_t {
        mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.stride
                               / MemoryLayout<natural_t>.stride)
    }

    /// Minimum returned word count for phys_footprint to be valid.
    public static var vmInfoCountForPhysFootprint: mach_msg_type_number_t {
        let offset = MemoryLayout<task_vm_info_data_t>.offset(of: \task_vm_info_data_t.phys_footprint)
            ?? MemoryLayout<task_vm_info_data_t>.stride
        let end = offset + MemoryLayout<UInt64>.size
        let word = MemoryLayout<natural_t>.stride
        return mach_msg_type_number_t((end + word - 1) / word)
    }

    public static func taskEvents() -> OwnProcessTaskEvents {
        var info = task_events_info_data_t()
        let full = eventsInfoCount
        var count = full
        let kr = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(full)) {
                task_info(mach_task_self_, task_flavor_t(TASK_EVENTS_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS, count >= full else {
            return OwnProcessTaskEvents(
                status: "UNKNOWN(task_info TASK_EVENTS_INFO kern_return " + String(kr)
                    + ", count " + String(count) + ")",
                faults: nil, pageins: nil, cowFaults: nil, contextSwitches: nil)
        }
        return OwnProcessTaskEvents(status: "ok", faults: Int64(info.faults),
                                    pageins: Int64(info.pageins),
                                    cowFaults: Int64(info.cow_faults),
                                    contextSwitches: Int64(info.csw))
    }

    public static func vmInfo() -> OwnProcessVMInfo {
        var info = task_vm_info_data_t()
        let full = vmInfoCount
        var count = full
        let kr = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(full)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else {
            return OwnProcessVMInfo(
                status: "UNKNOWN(task_info TASK_VM_INFO kern_return " + String(kr) + ")",
                virtualBytes: nil, residentBytes: nil, physFootprintBytes: nil)
        }
        let physKnown = count >= vmInfoCountForPhysFootprint
        return OwnProcessVMInfo(
            status: physKnown ? "ok"
                : "partial(phys_footprint UNKNOWN: returned count " + String(count) + ")",
            virtualBytes: Int64(clamping: info.virtual_size),
            residentBytes: Int64(clamping: info.resident_size),
            physFootprintBytes: physKnown ? Int64(clamping: info.phys_footprint) : nil)
    }

    /// Own-process point sample. `metalAllocatedBytes` nil means no runner
    /// device existed or was sampled: recorded as UNKNOWN, never zero.
    public static func snapshot(label: String, metalAllocatedBytes: Int?) -> OwnProcessTaskSnapshot {
        let events = taskEvents()
        let vm = vmInfo()
        return OwnProcessTaskSnapshot(
            label: label, uptimeNanos: DispatchTime.now().uptimeNanoseconds,
            events: events, vm: vm,
            metalStatus: metalAllocatedBytes == nil ? "UNKNOWN(no runner Metal device sampled)" : "ok",
            metalCurrentAllocatedBytes: metalAllocatedBytes.map { Int64($0) })
    }

    public static func counterDelta(name: String, before: Int64?, after: Int64?,
                                    monotonic: Bool) -> OwnProcessCounterDelta {
        guard let b = before, let a = after else {
            return OwnProcessCounterDelta(name: name, monotonic: monotonic, before: before,
                                          after: after, delta: nil, status: .unknown)
        }
        let (d, overflow) = a.subtractingReportingOverflow(b)
        if overflow {
            return OwnProcessCounterDelta(name: name, monotonic: monotonic, before: before,
                                          after: after, delta: nil, status: .unknown)
        }
        if monotonic && d < 0 {
            return OwnProcessCounterDelta(name: name, monotonic: monotonic, before: before,
                                          after: after, delta: nil, status: .nonMonotonic)
        }
        return OwnProcessCounterDelta(name: name, monotonic: monotonic, before: before,
                                      after: after, delta: d, status: .known)
    }

    /// Fixed-order deltas: cumulative counters first (monotonic), gauges after.
    public static func delta(_ a: OwnProcessTaskSnapshot, _ b: OwnProcessTaskSnapshot) -> OwnProcessSnapshotDelta {
        let counters = [
            counterDelta(name: "faults", before: a.events.faults, after: b.events.faults, monotonic: true),
            counterDelta(name: "pageins", before: a.events.pageins, after: b.events.pageins, monotonic: true),
            counterDelta(name: "cowFaults", before: a.events.cowFaults, after: b.events.cowFaults, monotonic: true),
            counterDelta(name: "contextSwitches", before: a.events.contextSwitches,
                         after: b.events.contextSwitches, monotonic: true),
            counterDelta(name: "virtualBytes", before: a.vm.virtualBytes, after: b.vm.virtualBytes, monotonic: false),
            counterDelta(name: "residentBytes", before: a.vm.residentBytes, after: b.vm.residentBytes, monotonic: false),
            counterDelta(name: "physFootprintBytes", before: a.vm.physFootprintBytes,
                         after: b.vm.physFootprintBytes, monotonic: false),
            counterDelta(name: "metalCurrentAllocatedBytes", before: a.metalCurrentAllocatedBytes,
                         after: b.metalCurrentAllocatedBytes, monotonic: false),
        ]
        let elapsed: UInt64? = b.uptimeNanos >= a.uptimeNanos ? b.uptimeNanos - a.uptimeNanos : nil
        return OwnProcessSnapshotDelta(from: a.label, to: b.label, elapsedNanos: elapsed, counters: counters)
    }
}

// MARK: - Submission / readback breakdown

public enum Gemma4BreakdownSite: String, Codable, Sendable {
    case embed
    case router
    case copy
    case logits
}

/// Value copy of the opt-in breakdown counters. Cumulative since the last
/// breakdownReset(); use minus() for interval deltas. Nesting is declared in
/// `nesting`; nested fields are never to be added to their containers.
public struct Gemma4SubmissionBreakdown: Codable, Equatable, Sendable {
    public var encodeMicros = 0.0
    public var flushMicros = 0.0
    public var flushEndEncodingMicros = 0.0
    public var flushCommitMicros = 0.0
    public var flushWaitMicros = 0.0
    public var flushCommandBufferIntervalMicros = 0.0
    public var flushCount = 0
    public var directSubmitCount = 0
    public var readCopyMicros = 0.0
    public var readCopyCount = 0
    public var writeCopyMicros = 0.0
    public var writeCopyCount = 0
    public var routerSelectionMicros = 0.0
    public var routerSelectionCount = 0
    public var bindingCheckMicros = 0.0
    public var bindingCheckCount = 0
    public var siteEmbedMicros = 0.0
    public var siteEmbedCount = 0
    public var siteRouterMicros = 0.0
    public var siteRouterCount = 0
    public var siteCopyMicros = 0.0
    public var siteCopyCount = 0
    public var siteLogitsMicros = 0.0
    public var siteLogitsCount = 0

    public init() {}

    public static let nesting: [String] = [
        "flushMicros is the CPU wall time of every flushOpenSegment that had an open segment; "
            + "flushEndEncodingMicros, flushCommitMicros and flushWaitMicros are nested inside it and "
            + "are never added to it (the remainder is status checks and accounting).",
        "flushCommandBufferIntervalMicros is the summed gpuEndTime - gpuStartTime of the flushed "
            + "command buffers: interval coverage on the GPU timeline, not CPU time, not busy time, "
            + "and it overlaps flushWaitMicros.",
        "encodeMicros, readCopyMicros, writeCopyMicros, routerSelectionMicros and "
            + "bindingCheckMicros are mutually non-overlapping CPU intervals.",
        "site*Micros are readback-site wall intervals (embed, router score, copy, final logits) "
            + "that CONTAIN the flush and copy time of their site; they overlap flush and copy fields "
            + "and are never added to them.",
        "No field is GPU busy time or utilization; instrumentation itself adds clock reads to "
            + "every measured site.",
    ]

    public func minus(_ o: Gemma4SubmissionBreakdown) -> Gemma4SubmissionBreakdown {
        var d = Gemma4SubmissionBreakdown()
        d.encodeMicros = encodeMicros - o.encodeMicros
        d.flushMicros = flushMicros - o.flushMicros
        d.flushEndEncodingMicros = flushEndEncodingMicros - o.flushEndEncodingMicros
        d.flushCommitMicros = flushCommitMicros - o.flushCommitMicros
        d.flushWaitMicros = flushWaitMicros - o.flushWaitMicros
        d.flushCommandBufferIntervalMicros = flushCommandBufferIntervalMicros - o.flushCommandBufferIntervalMicros
        d.flushCount = flushCount - o.flushCount
        d.directSubmitCount = directSubmitCount - o.directSubmitCount
        d.readCopyMicros = readCopyMicros - o.readCopyMicros
        d.readCopyCount = readCopyCount - o.readCopyCount
        d.writeCopyMicros = writeCopyMicros - o.writeCopyMicros
        d.writeCopyCount = writeCopyCount - o.writeCopyCount
        d.routerSelectionMicros = routerSelectionMicros - o.routerSelectionMicros
        d.routerSelectionCount = routerSelectionCount - o.routerSelectionCount
        d.bindingCheckMicros = bindingCheckMicros - o.bindingCheckMicros
        d.bindingCheckCount = bindingCheckCount - o.bindingCheckCount
        d.siteEmbedMicros = siteEmbedMicros - o.siteEmbedMicros
        d.siteEmbedCount = siteEmbedCount - o.siteEmbedCount
        d.siteRouterMicros = siteRouterMicros - o.siteRouterMicros
        d.siteRouterCount = siteRouterCount - o.siteRouterCount
        d.siteCopyMicros = siteCopyMicros - o.siteCopyMicros
        d.siteCopyCount = siteCopyCount - o.siteCopyCount
        d.siteLogitsMicros = siteLogitsMicros - o.siteLogitsMicros
        d.siteLogitsCount = siteLogitsCount - o.siteLogitsCount
        return d
    }

    /// Nesting and sign consistency of one record (cumulative or delta).
    public func violations(epsilonMicros: Double = 1e-3) -> [String] {
        var f = [String]()
        let times: [(String, Double)] = [
            ("encodeMicros", encodeMicros), ("flushMicros", flushMicros),
            ("flushEndEncodingMicros", flushEndEncodingMicros),
            ("flushCommitMicros", flushCommitMicros), ("flushWaitMicros", flushWaitMicros),
            ("flushCommandBufferIntervalMicros", flushCommandBufferIntervalMicros),
            ("readCopyMicros", readCopyMicros), ("writeCopyMicros", writeCopyMicros),
            ("routerSelectionMicros", routerSelectionMicros),
            ("bindingCheckMicros", bindingCheckMicros), ("siteEmbedMicros", siteEmbedMicros),
            ("siteRouterMicros", siteRouterMicros), ("siteCopyMicros", siteCopyMicros),
            ("siteLogitsMicros", siteLogitsMicros),
        ]
        for (name, v) in times where !(v.isFinite && v >= -epsilonMicros) {
            f.append(name + " is not finite and nonnegative")
        }
        let counts: [(String, Int)] = [
            ("flushCount", flushCount), ("directSubmitCount", directSubmitCount),
            ("readCopyCount", readCopyCount), ("writeCopyCount", writeCopyCount),
            ("routerSelectionCount", routerSelectionCount), ("bindingCheckCount", bindingCheckCount),
            ("siteEmbedCount", siteEmbedCount), ("siteRouterCount", siteRouterCount),
            ("siteCopyCount", siteCopyCount), ("siteLogitsCount", siteLogitsCount),
        ]
        for (name, v) in counts where v < 0 {
            f.append(name + " is negative")
        }
        let nested = flushEndEncodingMicros + flushCommitMicros + flushWaitMicros
        if nested.isFinite, flushMicros.isFinite, nested > flushMicros + epsilonMicros {
            f.append("flush sub-phases (endEncoding + commit + wait) exceed the flush total")
        }
        let counted: [(String, Double, Int)] = [
            ("flush", flushMicros + nested + flushCommandBufferIntervalMicros, flushCount),
            ("readCopy", readCopyMicros, readCopyCount),
            ("writeCopy", writeCopyMicros, writeCopyCount),
            ("routerSelection", routerSelectionMicros, routerSelectionCount),
            ("bindingCheck", bindingCheckMicros, bindingCheckCount),
            ("siteEmbed", siteEmbedMicros, siteEmbedCount),
            ("siteRouter", siteRouterMicros, siteRouterCount),
            ("siteCopy", siteCopyMicros, siteCopyCount),
            ("siteLogits", siteLogitsMicros, siteLogitsCount),
        ]
        for (name, v, n) in counted where n == 0 && abs(v) > epsilonMicros {
            f.append(name + " time recorded without a counted event")
        }
        return f
    }
}

extension Gemma4Runner {
    /// The same monotonic clock as the original breakdown counters.
    static func breakdownClockMicros() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1000
    }

    /// Value copy of every opt-in breakdown counter (original four plus
    /// the phase-54 split). The breakdownCounters() tuple is unchanged.
    public func submissionBreakdown() -> Gemma4SubmissionBreakdown {
        var b = Gemma4SubmissionBreakdown()
        b.encodeMicros = brEncodeMicros
        b.flushMicros = brFlushMicros
        b.flushEndEncodingMicros = brFlushEndEncodingMicros
        b.flushCommitMicros = brFlushCommitMicros
        b.flushWaitMicros = brFlushWaitMicros
        b.flushCommandBufferIntervalMicros = brFlushIntervalMicros
        b.flushCount = brFlushCount
        b.directSubmitCount = brDirectSubmitCount
        b.readCopyMicros = brReadCopyMicros
        b.readCopyCount = brReadCopyCount
        b.writeCopyMicros = brWriteCopyMicros
        b.writeCopyCount = brWriteCopyCount
        b.routerSelectionMicros = brRouterSelectionMicros
        b.routerSelectionCount = brRouterSelectionCount
        b.bindingCheckMicros = brBindingCheckMicros
        b.bindingCheckCount = brBindingCheckCount
        b.siteEmbedMicros = brSiteEmbedMicros
        b.siteEmbedCount = brSiteEmbedCount
        b.siteRouterMicros = brSiteRouterMicros
        b.siteRouterCount = brSiteRouterCount
        b.siteCopyMicros = brSiteCopyMicros
        b.siteCopyCount = brSiteCopyCount
        b.siteLogitsMicros = brSiteLogitsMicros
        b.siteLogitsCount = brSiteLogitsCount
        return b
    }

    func submissionBreakdownResetSplit() {
        brFlushEndEncodingMicros = 0
        brFlushCommitMicros = 0
        brFlushWaitMicros = 0
        brFlushIntervalMicros = 0
        brFlushCount = 0
        brDirectSubmitCount = 0
        brReadCopyCount = 0
        brWriteCopyCount = 0
        brRouterSelectionMicros = 0
        brRouterSelectionCount = 0
        brBindingCheckMicros = 0
        brBindingCheckCount = 0
        brSiteEmbedMicros = 0
        brSiteEmbedCount = 0
        brSiteRouterMicros = 0
        brSiteRouterCount = 0
        brSiteCopyMicros = 0
        brSiteCopyCount = 0
        brSiteLogitsMicros = 0
        brSiteLogitsCount = 0
    }

    /// Site start marker: a clock read only while instrumentation is on.
    func breakdownSiteBegin() -> Double {
        breakdownInstrumentation ? Gemma4Runner.breakdownClockMicros() : 0
    }

    func breakdownSiteEnd(_ site: Gemma4BreakdownSite, _ t0: Double) {
        guard breakdownInstrumentation else { return }
        let d = Gemma4Runner.breakdownClockMicros() - t0
        switch site {
        case .embed:
            brSiteEmbedMicros += d
            brSiteEmbedCount += 1
        case .router:
            brSiteRouterMicros += d
            brSiteRouterCount += 1
        case .copy:
            brSiteCopyMicros += d
            brSiteCopyCount += 1
        case .logits:
            brSiteLogitsMicros += d
            brSiteLogitsCount += 1
        }
    }

    func breakdownRouterSelectionEnd(_ t0: Double) {
        guard breakdownInstrumentation else { return }
        brRouterSelectionMicros += Gemma4Runner.breakdownClockMicros() - t0
        brRouterSelectionCount += 1
    }

    func breakdownBindingCheckEnd(_ t0: Double) {
        guard breakdownInstrumentation else { return }
        brBindingCheckMicros += Gemma4Runner.breakdownClockMicros() - t0
        brBindingCheckCount += 1
    }

    func ownProcessMemoryAccounting(fixtureMaxPositions: Int) -> OwnProcessMemoryAccounting {
        var widths = [Int]()
        for layer in 0..<config.layerCount {
            widths.append(config.isGlobalLayer(layer)
                ? config.globalKVHeads * config.globalHeadDim
                : config.localKVHeads * config.localHeadDim)
        }
        let derived = OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: maxContext)
        let perPosition = OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: 1)
        let fixtureBytes = OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths,
                                                                    positions: fixtureMaxPositions)
        var actual = 0
        for b in kCaches { actual += b.length }
        for b in vCaches { actual += b.length }
        let buffers: [MTLBuffer?] = [bufX, bufH, bufT1, bufT2, bufT3, bufKV, bufGU, bufAct,
                                     bufExpertGU, bufExpertAct, bufExpertAcc, bufShared, bufRouter,
                                     bufEmbed, bufLogits, localFreqBuffer, globalFreqBuffer,
                                     rmsStatsInverseBuffer, moeScratch?.gu, moeScratch?.act,
                                     moeScratch?.dn, routerOverlapState?.routerIn]
        var scratch = 0
        for b in buffers {
            if let b { scratch += b.length }
        }
        return OwnProcessMemoryAccounting(
            ggufFileBytes: ggufFile.byteCount,
            residentWeightsMappingBytes: residentWeightsBuffer?.length,
            kvCapacityPositions: maxContext, kvLayerWidthsFloats: widths,
            kvBytesPerPosition: perPosition, kvDerivedAllocationBytes: derived,
            kvActualAllocationBytes: actual, kvDerivedMatchesActual: derived == actual,
            fixtureMaxPositions: fixtureMaxPositions, fixtureKVBytesUpperBound: fixtureBytes,
            activationAndScratchBytes: scratch,
            metalCurrentAllocatedBytes: metalAllocatedSample(),
            caveats: OwnProcessMemoryAccounting.standardCaveats)
    }
}

// MARK: - Memory accounting

public struct OwnProcessMemoryAccounting: Codable, Equatable, Sendable {
    public let ggufFileBytes: Int
    public let residentWeightsMappingBytes: Int?
    public let kvCapacityPositions: Int
    public let kvLayerWidthsFloats: [Int]
    public let kvBytesPerPosition: Int?
    public let kvDerivedAllocationBytes: Int?
    public let kvActualAllocationBytes: Int
    public let kvDerivedMatchesActual: Bool
    public let fixtureMaxPositions: Int
    public let fixtureKVBytesUpperBound: Int?
    public let activationAndScratchBytes: Int
    public let metalCurrentAllocatedBytes: Int
    public let caveats: [String]

    public static let standardCaveats: [String] = [
        "Allocation capacity (buffer lengths, MTLDevice.currentAllocatedSize) is not physical residency.",
        "The resident GGUF mapping is file-backed and read-only; how many of its pages are resident "
            + "depends on the page cache and is not claimed here.",
        "K/V capacity is derived from the configured maxContext (the existing 4096 bound) and must "
            + "equal the actual cache allocation; the fixture bound is an upper bound on bytes written.",
        "currentAllocatedSize is a point sample; the allocator may retain freed bytes and lazy "
            + "allocations may not have happened yet.",
    ]

    /// K plus V F32 bytes for the given per-layer widths (floats per
    /// position) and positions; nil on invalid input or overflow.
    public static func derivedKVBytes(layerWidths: [Int], positions: Int) -> Int? {
        guard positions >= 0, !layerWidths.isEmpty, layerWidths.allSatisfy({ $0 > 0 }) else {
            return nil
        }
        var total = 0
        for w in layerWidths {
            let (a, o1) = w.multipliedReportingOverflow(by: positions)
            let (b, o2) = a.multipliedReportingOverflow(by: 8)
            let (t, o3) = total.addingReportingOverflow(b)
            if o1 || o2 || o3 { return nil }
            total = t
        }
        return total
    }
}

// MARK: - Reference comparison

public struct OwnProcessWorstElement: Codable, Equatable, Sendable {
    public let step: Int
    public let index: Int
    public let value: Float
    public let reference: Float
    public let absDelta: Double
    public let perElementNormalized: Double
}

/// Full-vocabulary comparison against a fresh direct reference. Records the
/// gate's vector-normalized metric max|d| / max(1, max|ref|), the absolute
/// maximum and the stricter per-element max_i |d_i| / max(1, |ref_i|).
public struct OwnProcessLogitComparison: Codable, Equatable, Sendable {
    public private(set) var comparedSteps = 0
    public private(set) var invalidSteps = 0
    public private(set) var maxAbsDelta = 0.0
    public private(set) var maxVectorNormalized = 0.0
    public private(set) var maxPerElementNormalized = 0.0
    public private(set) var perStepMaxPerElementNormalized: [Double] = []
    public private(set) var worstPerElement: OwnProcessWorstElement? = nil

    public init() {}

    public mutating func add(step: Int, logits: [Float], reference: [Float]) {
        guard !logits.isEmpty, logits.count == reference.count,
              logits.allSatisfy({ $0.isFinite }), reference.allSatisfy({ $0.isFinite }) else {
            invalidSteps += 1
            return
        }
        var maxAbs = 0.0
        var maxRef = 0.0
        var maxElement = 0.0
        var worstIndex = 0
        for i in 0..<logits.count {
            let d = abs(Double(logits[i]) - Double(reference[i]))
            let a = abs(Double(reference[i]))
            if d > maxAbs { maxAbs = d }
            if a > maxRef { maxRef = a }
            let e = d / max(1.0, a)
            if e > maxElement {
                maxElement = e
                worstIndex = i
            }
        }
        if worstPerElement == nil || maxElement > maxPerElementNormalized {
            worstPerElement = OwnProcessWorstElement(
                step: step, index: worstIndex, value: logits[worstIndex],
                reference: reference[worstIndex],
                absDelta: abs(Double(logits[worstIndex]) - Double(reference[worstIndex])),
                perElementNormalized: maxElement)
        }
        comparedSteps += 1
        maxAbsDelta = max(maxAbsDelta, maxAbs)
        maxVectorNormalized = max(maxVectorNormalized, maxAbs / max(1.0, maxRef))
        maxPerElementNormalized = max(maxPerElementNormalized, maxElement)
        perStepMaxPerElementNormalized.append(maxElement)
    }

    /// Both bounds are strict (< contract), like the existing forge and gate.
    public func failures(contract: Double, expectedSteps: Int) -> [String] {
        var f = [String]()
        if invalidSteps > 0 {
            f.append(String(invalidSteps) + " output(s) not comparable (length mismatch, empty or non-finite)")
        }
        if comparedSteps != expectedSteps {
            f.append("compared " + String(comparedSteps) + " of " + String(expectedSteps)
                     + " outputs with the fresh reference")
        }
        if !(maxVectorNormalized < contract) {
            f.append("vector-normalized max diff " + String(maxVectorNormalized) + " not < " + String(contract))
        }
        if !(maxPerElementNormalized < contract) {
            f.append("per-element normalized max diff " + String(maxPerElementNormalized)
                     + " not < " + String(contract))
        }
        return f
    }
}

// MARK: - Readback site expectation

/// Per-forward readback/copy site counts of the OFF executor: one embedding
/// read and write, one router-score read per layer, one read and write per
/// SSA copy, the executor's final logits read and the caller's logits read.
public struct OwnProcessSiteExpectation: Codable, Equatable, Sendable {
    public let embedSites: Int
    public let routerSites: Int
    public let copySites: Int
    public let logitsSites: Int
    public let routerSelections: Int
    public let readCopies: Int
    public let writeCopies: Int

    public static func perForward(layers: Int, copies: Int) -> OwnProcessSiteExpectation {
        OwnProcessSiteExpectation(embedSites: 1, routerSites: layers, copySites: copies,
                                  logitsSites: 1, routerSelections: layers,
                                  readCopies: 3 + layers + copies, writeCopies: 1 + copies)
    }

    /// Exact count checks; binding checks are per executor call.
    public func mismatches(_ b: Gemma4SubmissionBreakdown, forwards: Int, calls: Int,
                           label: String) -> [String] {
        let checks: [(String, Int, Int)] = [
            ("siteEmbedCount", b.siteEmbedCount, embedSites * forwards),
            ("siteRouterCount", b.siteRouterCount, routerSites * forwards),
            ("siteCopyCount", b.siteCopyCount, copySites * forwards),
            ("siteLogitsCount", b.siteLogitsCount, logitsSites * forwards),
            ("routerSelectionCount", b.routerSelectionCount, routerSelections * forwards),
            ("readCopyCount", b.readCopyCount, readCopies * forwards),
            ("writeCopyCount", b.writeCopyCount, writeCopies * forwards),
            ("bindingCheckCount", b.bindingCheckCount, calls),
        ]
        var f = [String]()
        for (name, got, want) in checks where got != want {
            f.append(label + " " + name + " " + String(got) + " != " + String(want))
        }
        return f
    }
}

// MARK: - Records

public struct OwnProcessProfiledRun: Codable, Equatable, Sendable {
    public let fixture: String
    public let warmup: Bool
    public let prefillWallMicros: Double
    public let continuationWallMicros: [Double]
    public let continuationWallCumulativeMicros: Double
    public let continuationTPOTMedianMicros: Double?
    /// Returned command-buffer interval per continuation (OFF: every flushed
    /// buffer of the call). Coverage, not busy time.
    public let continuationCommandBufferIntervalMicros: [Double]
    /// wall - interval per continuation: unattributed host-side time under
    /// instrumentation; not directly measured GPU idle and not recoverable
    /// time by itself.
    public let continuationResidualMicros: [Double]
    public let prefillBreakdown: Gemma4SubmissionBreakdown
    public let continuationBreakdownTotal: Gemma4SubmissionBreakdown
    public let continuationBreakdownPerToken: [Gemma4SubmissionBreakdown]
    public let siteExpectationPerForward: OwnProcessSiteExpectation
    public let totals: RouterOverlapGateTotals
    public let counts: RouterOverlapGatePhaseCounts
    public let totalsMismatchesVsQualifiedOff: [String]
    public let comparison: OwnProcessLogitComparison
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let outputs: Int
    public let allFinite: Bool
    public let processBefore: OwnProcessTaskSnapshot
    public let processAfter: OwnProcessTaskSnapshot
    public let processDelta: OwnProcessSnapshotDelta
    public let preconditionFailures: [String]
    public let error: String?
    public let checksPassed: Bool
    public let failure: String?
}

public enum OwnProcessDiagnosticsStatus: String, Codable, Sendable {
    case profiledChecksPassed
    case profiledChecksFailed
    case qualificationFailed
    case aborted
    case runnerNotRestored
}

public struct OwnProcessDiagnosticsReport: Codable {
    public private(set) var kind = OwnProcessDiagnostics.kind
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionTPOTClaim = false
    public private(set) var notes = OwnProcessDiagnostics.notes
    public private(set) var breakdownNesting = Gemma4SubmissionBreakdown.nesting
    public private(set) var correctnessContract = OwnProcessDiagnostics.correctnessContract
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var memory: OwnProcessMemoryAccounting? = nil
    public var snapshots: [OwnProcessTaskSnapshot] = []
    public var snapshotDeltas: [OwnProcessSnapshotDelta] = []
    public var gate: RouterOverlapGateReport? = nil
    public var runs: [OwnProcessProfiledRun] = []
    public var profileFailures: [String] = []
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: OwnProcessDiagnosticsStatus = .aborted

    public init() {}
}

extension OwnProcessDiagnosticsReport {
    /// Serialization copy with the always-false contract re-asserted; the
    /// embedded gate report gets its own gainQualified normalization.
    func withDiagnosticContract() -> OwnProcessDiagnosticsReport {
        var copy = self
        copy.gainQualified = false
        copy.installable = false
        copy.productionTPOTClaim = false
        copy.gate = copy.gate?.withGainQualifiedContract()
        return copy
    }
}

public struct OwnProcessDiagnosticsOptions: Equatable, Sendable {
    public let gguf: String
    public let region: String
    public let graph: String
    public let fixtures: String
    public let report: String
}

// MARK: - Diagnostic driver

public enum OwnProcessDiagnostics {
    public static let kind = "own-process-memory-submission-readback-diagnostics-off-only-v1"
    /// Supervisor protocol tokens: bound to the existing phase-53 gate
    /// tokens (single source of truth shared with the owned supervisor).
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let correctnessContract = 1e-4
    public static let maxReportBytes = 16 << 20
    public static let usage = "usage: owned-process-diagnostics --gguf M.gguf --region R.json "
        + "--graph G.json --fixtures F.json --report NEW_REPORT.json"

    public static let notes: [String] = [
        "Diagnostic only: no artifact, no installation, no winner, no activation, no promotion; "
            + "gainQualified, installable and productionTPOTClaim are always false.",
        "Instrumentation ON adds clock reads and counter updates at every encode, flush, copy and "
            + "readback site, so profiled wall times are NOT production TPOT and are never compared "
            + "with gate or benchmark timings.",
        "Profiled arm: OFF only (control preparation, declared modern candidate with par64 and "
            + "router launch divisor 1, installed v1 E4R32, Q4 width 32, CPU FP64 routing). The router "
            + "overlap scope refuses breakdown instrumentation, so ON is never profiled.",
        "Per profiled run: 32 frozen IDs, 32 full-vocabulary finite outputs fed by the previous "
            + "actual argmax, exact per-phase counters against the derived OFF expectation, kernel and "
            + "command-buffer totals equal to the qualified OFF run of the same fixture, and every "
            + "output compared with a fresh direct reference rerun under BOTH the gate's "
            + "vector-normalized bound and the stricter per-element bound max_i |d_i| / max(1, |ref_i|), "
            + "each strictly < 1e-4.",
        "Residual = continuation wall - returned command-buffer interval: unattributed host-side "
            + "time under instrumentation; not directly measured GPU idle and not recoverable time by itself.",
        "Process counters come from task_info on this process only (TASK_EVENTS_INFO, TASK_VM_INFO) "
            + "and MTLDevice.currentAllocatedSize, sampled at runner, gate, reference and run boundaries, "
            + "never inside a timed call. A failed call is UNKNOWN and a decreasing cumulative counter is "
            + "nonMonotonic; neither is reported as a number.",
        "Memory accounting separates GGUF file bytes, the resident mapping length, source-derived and "
            + "actual K/V allocation (configured maxContext, the existing 4096 bound) and Metal "
            + "currentAllocatedSize; allocation capacity is not physical residency.",
        "The model SHA-256 is the cached explicit-refresh record; nothing here hashes, re-verifies "
            + "or refreshes the model.",
    ]

    // MARK: Pure protocol helpers

    /// Exactly GO; EOF (nil) and every other line are refused.
    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func reportPathRejection(path: String, exists: Bool,
                                           parentIsDirectory: Bool) -> String? {
        guard path.hasSuffix(".json"), path.count > 5 else {
            return "report path must name a new .json file"
        }
        guard !exists else { return "refusing to overwrite existing " + path }
        guard parentIsDirectory else { return "report directory does not exist" }
        return nil
    }

    public static func parseOptions(_ argv: [String]) throws -> OwnProcessDiagnosticsOptions {
        let keys = ["--gguf", "--region", "--graph", "--fixtures", "--report"]
        var values = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard keys.contains(key), i + 1 < argv.count, values[key] == nil else {
                throw OwnProcessDiagnosticsError(usage)
            }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else {
                throw OwnProcessDiagnosticsError(usage)
            }
            values[key] = value
            i += 2
        }
        guard let gguf = values["--gguf"], let region = values["--region"],
              let graph = values["--graph"], let fixtures = values["--fixtures"],
              let report = values["--report"] else {
            throw OwnProcessDiagnosticsError(usage)
        }
        return OwnProcessDiagnosticsOptions(gguf: gguf, region: region, graph: graph,
                                            fixtures: fixtures, report: report)
    }

    public static func status(restoredPristine: Bool, aborted: Bool,
                              gateStatus: RouterOverlapGateStatus?, runsPassed: [Bool],
                              expectedRuns: Int, profileFailures: [String]) -> OwnProcessDiagnosticsStatus {
        if !restoredPristine { return .runnerNotRestored }
        if aborted { return .aborted }
        guard gateStatus == .qualificationPassed else { return .qualificationFailed }
        if profileFailures.isEmpty, runsPassed.count == expectedRuns, runsPassed.allSatisfy({ $0 }) {
            return .profiledChecksPassed
        }
        return .profiledChecksFailed
    }

    public static func aborted(reason: String, preInitSnapshot: OwnProcessTaskSnapshot) -> OwnProcessDiagnosticsReport {
        var r = OwnProcessDiagnosticsReport()
        r.snapshots = [preInitSnapshot]
        r.abortReason = reason
        r.status = .aborted
        return r
    }

    public static func encodeReport(_ r: OwnProcessDiagnosticsReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: true).encode(r.withDiagnosticContract())
    }

    static func snapshot(_ label: String, _ runner: Gemma4Runner) -> OwnProcessTaskSnapshot {
        OwnProcessInfo.snapshot(label: label, metalAllocatedBytes: runner.metalAllocatedSample())
    }

    // MARK: Entry point

    /// Never throws; every failure is recorded. The caller prints END only
    /// after this returns and RouterOverlapModelGate.quiescent(runner) holds.
    public static func run(region: MoERegionIR, graph: GemmaGraphIR,
                           fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner,
                           modelSha256: String, executableDigest: String,
                           preInitSnapshot: OwnProcessTaskSnapshot,
                           emit: (String) -> Void) -> OwnProcessDiagnosticsReport {
        let proto = RouterOverlapGateProtocol.required
        var report = OwnProcessDiagnosticsReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.snapshots.append(preInitSnapshot)
        report.snapshots.append(snapshot("runner-loaded", runner))
        let maxPositions = fixtures.map { $0.prompt.count + proto.outputsPerRun }.max() ?? 0
        let memory = runner.ownProcessMemoryAccounting(fixtureMaxPositions: maxPositions)
        report.memory = memory
        emit(RouterOverlapModelGate.event("memoryAccounting", memory))
        report.snapshots.append(snapshot("before-gate-qualification", runner))
        let gate = RouterOverlapModelGate.run(
            mode: .qualificationOnly, region: region, graph: graph, fixtures: fixtures,
            runner: runner, modelSha256: modelSha256, executableDigest: executableDigest, emit: emit)
        report.gate = gate
        report.snapshots.append(snapshot("after-gate-qualification", runner))
        switch gate.status {
        case .qualificationPassed:
            do {
                try runner.withQ4DiagnosticReferenceWidth(proto.q4LaunchWidth) { () -> Void in
                    try profile(&report, gate: gate, region: region, graph: graph,
                                fixtures: fixtures, runner: runner, emit: emit)
                }
            } catch {
                report.abortReason = "profile threw: " + String(describing: error)
            }
        case .aborted, .runnerNotRestored:
            report.abortReason = "phase-53 gate ended with status " + gate.status.rawValue
        default:
            emit("phase-53 gate qualification did not pass (" + gate.status.rawValue
                 + "); nothing profiled")
        }
        runner.breakdownInstrumentation = false
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            && RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner)
        report.snapshots.append(snapshot("end", runner))
        report.snapshotDeltas = zip(report.snapshots, report.snapshots.dropFirst()).map {
            OwnProcessInfo.delta($0.0, $0.1)
        }
        report.status = status(restoredPristine: report.restoredPristine,
                               aborted: report.abortReason != nil, gateStatus: gate.status,
                               runsPassed: report.runs.map { $0.checksPassed },
                               expectedRuns: 2 * fixtures.count,
                               profileFailures: report.profileFailures)
        emit("status " + report.status.rawValue + "; restored pristine " + String(report.restoredPristine)
             + "; diagnostic only, gainQualified false, not installable")
        return report
    }

    static func profile(_ report: inout OwnProcessDiagnosticsReport, gate: RouterOverlapGateReport,
                        region: MoERegionIR, graph: GemmaGraphIR,
                        fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner,
                        emit: (String) -> Void) throws {
        let proto = RouterOverlapGateProtocol.required
        defer {
            runner.breakdownInstrumentation = false
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == proto.q4LaunchWidth else {
            throw OwnProcessDiagnosticsError("Q4 reference width 32 is not in effect inside the width scope")
        }
        guard let gateDerived = gate.identity?.derived else {
            report.profileFailures.append("the qualified gate report carries no identity")
            return
        }
        var prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
        let derived = RouterOverlapModelGate.derive(on: prepared.on, runner: runner)
        guard derived == gateDerived, derived.failures.isEmpty else {
            report.profileFailures.append(
                "derived per-forward counts differ from the qualified gate identity or are inconsistent")
            return
        }
        if derived.statisticsGroups > 0 {
            // Allocation only, as in the gate: keeps buffer totals stationary.
            _ = try runner.rmsStatisticsInverse(slots: derived.statisticsGroups)
        }
        let copies = prepared.off.ssa.instructions.filter { $0.op == "copy" }.count
        let sites = OwnProcessSiteExpectation.perForward(layers: derived.layers, copies: copies)
        emit(RouterOverlapModelGate.event("siteExpectationPerForward", sites))
        for fx in fixtures {
            runner.uninstallMoeGeneratedLowering()
            var ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                               candidate: RouterOverlapModelGate.modernCandidate,
                                               derived: derived, off: prepared.off, on: prepared.on)
            report.snapshots.append(snapshot("before-reference " + fx.name, runner))
            let ref = RouterOverlapModelGate.qualify(.reference, ctx, fx, reference: [], off: [],
                                                     offTotals: nil, keep: true)
            report.snapshots.append(snapshot("after-reference " + fx.name, runner))
            guard ref.record.passed, ref.kept.count == proto.outputsPerRun else {
                report.profileFailures.append("fixture " + fx.name + ": fresh reference rerun failed: "
                                              + (ref.record.failure ?? "unknown"))
                continue
            }
            prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
            guard RouterOverlapModelGate.derive(on: prepared.on, runner: runner) == derived else {
                report.profileFailures.append("derived counts changed after re-preparation")
                return
            }
            ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate,
                                           derived: derived, off: prepared.off, on: prepared.on)
            let qualifiedOff = gate.qualification.first { $0.fixture == fx.name }?.off.totals
            for warmup in [true, false] {
                let run = profiledRun(ctx, fx, warmup: warmup, reference: ref.kept,
                                      qualifiedOff: qualifiedOff, sites: sites)
                report.runs.append(run)
                emit(RouterOverlapModelGate.event("profiledRun", run))
            }
        }
    }

    /// One OFF 32-output run with breakdown instrumentation ON. Timers bracket
    /// only executeGraph(prompt) and each executeGraphContinue(previous
    /// argmax); breakdown copies, counter snapshots, comparisons and process
    /// snapshots run between or outside the timers.
    static func profiledRun(_ c: RouterOverlapGateContext, _ fx: RouterOverlapGateFixture,
                            warmup: Bool, reference: [[Float]],
                            qualifiedOff: RouterOverlapGateTotals?,
                            sites: OwnProcessSiteExpectation) -> OwnProcessProfiledRun {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let vocab = r.config.vocabSize
        let tag = fx.name + (warmup ? " warmup" : " measured")
        let pre = RouterOverlapModelGate.preconditions(.off, c, prepared: c.off)
        var g = RouterOverlapGateGreedy()
        var cmp = OwnProcessLogitComparison()
        var prefillWall = -1.0
        var walls = [Double]()
        var intervals = [Double]()
        var perToken = [Gemma4SubmissionBreakdown]()
        var prefillBreakdown = Gemma4SubmissionBreakdown()
        var runBreakdown = Gemma4SubmissionBreakdown()
        let before = snapshot("run-start " + tag, r)
        if pre.isEmpty {
            r.reset()
            r.breakdownReset()
            r.breakdownInstrumentation = true
            let b0 = r.submissionBreakdown()
            g.s0 = RouterOverlapGateCounters(r)
            g.s1 = g.s0
            do {
                let t0 = RouterOverlapModelGate.now()
                let first = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt,
                                                        candidate: c.candidate, prepared: c.off)
                let t1 = RouterOverlapModelGate.now()
                prefillWall = RouterOverlapModelGate.micros(t0, t1)
                g.s1 = RouterOverlapGateCounters(r)
                prefillBreakdown = r.submissionBreakdown().minus(b0)
                g.totals.prefillKernels = first.kernels
                g.totals.prefillBuffers = first.buffers
                g.totals.prefillGpuElapsedMicros = first.gpuMicros
                if RouterOverlapModelGate.fullFinite(first.logits, vocab) {
                    cmp.add(step: 0, logits: first.logits, reference: reference.first ?? [])
                    g.ids.append(first.argmax)
                    g.outputs = 1
                    while g.outputs < proto.outputsPerRun {
                        let bt0 = r.submissionBreakdown()
                        let c0 = RouterOverlapModelGate.now()
                        let step = try GemmaForge.executeGraphContinue(
                            c.graph, runner: r, token: g.ids[g.ids.count - 1],
                            candidate: c.candidate, prepared: c.off)
                        let c1 = RouterOverlapModelGate.now()
                        let bt1 = r.submissionBreakdown()
                        walls.append(RouterOverlapModelGate.micros(c0, c1))
                        intervals.append(step.gpuMicros)
                        perToken.append(bt1.minus(bt0))
                        g.totals.continuationKernels.append(step.kernels)
                        g.totals.continuationBuffers.append(step.buffers)
                        g.totals.continuationGpuElapsedMicros.append(step.gpuMicros)
                        guard RouterOverlapModelGate.fullFinite(step.logits, vocab) else {
                            g.allFinite = false
                            g.error = "output " + String(g.outputs) + " not full-vocabulary finite; run stopped"
                            break
                        }
                        let index = g.outputs
                        cmp.add(step: index, logits: step.logits,
                                reference: index < reference.count ? reference[index] : [])
                        g.ids.append(step.argmax)
                        g.outputs += 1
                    }
                } else {
                    g.allFinite = false
                    g.error = "output 0 not full-vocabulary finite; run stopped"
                }
            } catch {
                g.error = "threw: " + String(describing: error)
            }
            g.s2 = RouterOverlapGateCounters(r)
            runBreakdown = r.submissionBreakdown().minus(b0)
            r.breakdownInstrumentation = false
            r.reset()
        }
        let after = snapshot("run-end " + tag, r)
        let continuationBreakdown = runBreakdown.minus(prefillBreakdown)
        let counts = RouterOverlapModelGate.phaseCounts(g, c.derived, .off, promptCount: fx.prompt.count)
        let relation: [String] = qualifiedOff.map {
            RouterOverlapModelGate.totalsRelation(candidate: g.totals, baseline: $0,
                                                  extraBuffersPerForward: 0,
                                                  promptCount: fx.prompt.count)
        } ?? ["no qualified OFF totals for " + fx.name + " in the gate report"]
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if g.ids != fx.frozenIds { f.append("output IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != proto.outputsPerRun {
                f.append("not 32 full-vocabulary finite outputs")
            }
            if !(prefillWall.isFinite && prefillWall > 0) { f.append("prefill wall invalid") }
            if RouterOverlapModelGate.runTPOT(walls) == nil {
                f.append("continuation walls are not 31 positive values")
            }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            f += relation
            f += cmp.failures(contract: correctnessContract, expectedSteps: proto.outputsPerRun)
            f += prefillBreakdown.violations().map { "prefill breakdown: " + $0 }
            f += continuationBreakdown.violations().map { "continuation breakdown: " + $0 }
            for (i, b) in perToken.enumerated() {
                f += b.violations().map { "token " + String(i) + " breakdown: " + $0 }
            }
            f += sites.mismatches(prefillBreakdown, forwards: fx.prompt.count, calls: 1, label: "prefill")
            f += sites.mismatches(continuationBreakdown, forwards: proto.continuationsPerRun,
                                  calls: proto.continuationsPerRun, label: "continuation")
            let buffersDelta = g.s2["buffers"] - g.s0["buffers"]
            if runBreakdown.directSubmitCount != 0 {
                f.append("OFF executed " + String(runBreakdown.directSubmitCount)
                         + " direct (unsegmented) submissions")
            }
            if runBreakdown.flushCount + runBreakdown.directSubmitCount != buffersDelta {
                f.append("flushes " + String(runBreakdown.flushCount) + " + direct "
                         + String(runBreakdown.directSubmitCount) + " != submitted buffers "
                         + String(buffersDelta))
            }
            for (i, b) in perToken.enumerated() where i < intervals.count {
                let diff = abs(intervals[i] - b.flushCommandBufferIntervalMicros)
                if !(diff <= 1e-3 + 1e-9 * abs(intervals[i])) {
                    f.append("token " + String(i)
                             + ": returned command-buffer interval disagrees with the flushed intervals")
                }
            }
            if !RouterOverlapModelGate.quiescent(r) { f.append("runner not quiescent after the run") }
            if r.breakdownInstrumentation { f.append("breakdown instrumentation left on") }
        }
        let residuals = zip(walls, intervals).map { $0.0 - $0.1 }
        return OwnProcessProfiledRun(
            fixture: fx.name, warmup: warmup, prefillWallMicros: prefillWall,
            continuationWallMicros: walls,
            continuationWallCumulativeMicros: walls.reduce(0, +),
            continuationTPOTMedianMicros: RouterOverlapModelGate.runTPOT(walls),
            continuationCommandBufferIntervalMicros: intervals,
            continuationResidualMicros: residuals,
            prefillBreakdown: prefillBreakdown,
            continuationBreakdownTotal: continuationBreakdown,
            continuationBreakdownPerToken: perToken,
            siteExpectationPerForward: sites, totals: g.totals, counts: counts,
            totalsMismatchesVsQualifiedOff: relation, comparison: cmp, ids: g.ids,
            idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs, allFinite: g.allFinite,
            processBefore: before, processAfter: after,
            processDelta: OwnProcessInfo.delta(before, after),
            preconditionFailures: pre, error: g.error, checksPassed: f.isEmpty,
            failure: f.isEmpty ? nil : f.joined(separator: "; "))
    }
}
