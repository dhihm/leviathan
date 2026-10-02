// Explicit bounded expert-weight cache for the Gemma 4 runtime.
//
// This is the SSD/streaming execution mode: the MoE expert tensors
// (`ffn_gate_up_exps.weight`, `ffn_down_exps.weight` — ~12.8 GB of the
// 14.44 GB checkpoint) are NOT kept GPU-resident through the whole-file
// mapping. Instead the runner compacts every NON-expert tensor into a
// dedicated GPU buffer (~1.6 GB) and this cache holds a bounded set of
// expert slices loaded on demand from the original GGUF with checked
// pread offsets/extents. Prefetch is a SYNCHRONOUS free-slot-only
// initial read-ahead between expert iterations (no async I/O or GPU
// overlap is claimed; it never evicts, so it cannot change results).
//
// Honesty notes (measured claims only):
// - The byte budget is the sum of the two slot pools; pinned and
//   in-flight slots are slices of those same pools, so accounting is
//   bounded by construction, and the peak pinned (union per family)
//   bytes are reported separately.
// - flush-before-reuse: a slot whose encoded GPU work may still be in
//   flight is never overwritten; the cache flushes the runner's open
//   command-buffer segment (completing the GPU work) before reusing it.
// - pread from the GGUF may be served by the OS page cache; no
//   cold-SSD / physical-disk latency claim is made anywhere.

import Foundation
import Metal

/// Reported counters of the bounded expert cache (all separate).
public struct Gemma4ExpertCacheStats: Sendable, Equatable, Codable {
    /// Configured byte budget (the pool allocation ceiling).
    public let budgetBytes: Int
    /// Resident bytes actually allocated as slot pools (<= budget).
    public let residentBytes: Int
    /// gate-up slots × stride + down slots × stride breakdown.
    public let guSlots: Int
    public let dnSlots: Int
    public let guStrideBytes: Int
    public let dnStrideBytes: Int
    /// Bytes of the compacted NON-expert weights buffer (everything
    /// except the two expert tensor families is GPU-resident).
    public let nonExpertResidentBytes: Int
    /// Total expert slice bytes read from the GGUF (may be served warm
    /// from the OS page cache — not a physical-SSD-throughput measure).
    public let sliceBytesRead: Int
    public let misses: Int
    public let hits: Int
    public let evictions: Int
    /// Segment flushes forced because every candidate victim slot had
    /// possibly-in-flight GPU work (flush-before-reuse).
    public let flushesBeforeReuse: Int
    /// Peak bytes simultaneously pinned (union of active + in-flight
    /// slots per family).
    public let peakPinnedBytes: Int
    /// Peak device.currentAllocatedSize sampled at slice loads.
    public let peakDeviceAllocatedBytes: Int
    /// device.currentAllocatedSize sampled when cache-mode construction
    /// completed (a SAMPLE at end of construction, not a startup peak).
    public let initDeviceAllocatedBytes: Int
    /// Synchronous read-ahead: slices copied into FREE slots only
    /// (never evicting). Counts/bytes of prefetch loads.
    public let prefetchSlices: Int
    public let prefetchBytes: Int
    /// Cache hits served from slices installed by prefetch.
    public let prefetchHits: Int
}

/// Pure LRU slot table over `key -> slot` with external pinning
/// (public: a production component whose policy is unit-tested
/// directly).
/// Victim selection returns FREE (never-used) slots before LRU
/// occupied victims and only slots NOT in the pinned set; the caller
/// decides how to unpin (a segment flush clears in-flight pins).
public struct ExpertSlotTable {
    private(set) var slotCount: Int
    private var keyToSlot: [Int: Int] = [:]
    /// Occupied slot indices in use order, least-recently-used first.
    private var order: [Int] = []

    public init(slotCount: Int) {
        precondition(slotCount > 0)
        self.slotCount = slotCount
    }

    /// Returns the slot holding `key` (touching it MRU), or nil on miss.
    public mutating func lookup(_ key: Int) -> Int? {
        guard let s = keyToSlot[key] else { return nil }
        if let i = order.firstIndex(of: s) { order.remove(at: i) }
        order.append(s)
        return s
    }

    /// A slot index holding no key (checked before LRU eviction), or
    /// nil when every slot is occupied.
    public func freeSlot() -> Int? {
        let used = Set(keyToSlot.values)
        return (0..<slotCount).first { !used.contains($0) }
    }

    /// Whether `slot` currently holds a resident key.
    public func isOccupied(_ slot: Int) -> Bool {
        keyToSlot.values.contains(slot)
    }

    /// LRU OCCUPIED slot not in `pinned`, or nil when every occupied
    /// slot is pinned (free slots are `freeSlot`'s business).
    public func victim(pinned: Set<Int>) -> Int? {
        order.first { !pinned.contains($0) }
    }

    /// Removes whatever key occupies `slot` (pre-IO eviction: the slot
    /// stops being addressable BEFORE its bytes are overwritten).
    public mutating func evict(slot: Int) {
        for (k, s) in keyToSlot where s == slot { keyToSlot[k] = nil }
        if let i = order.firstIndex(of: slot) { order.remove(at: i) }
    }

    /// Records `key` as resident in `slot` (post-IO install: only a
    /// fully read slice is ever addressable).
    public mutating func install(key: Int, slot: Int) {
        evict(slot: slot)
        order.append(slot)
        keyToSlot[key] = slot
    }

    public var residentKeys: Int { keyToSlot.count }
}

/// Checked arithmetic helpers shared by the cache geometry and the
/// loader (malformed extreme values are rejected, never trapped).
enum Checked {
    static func add(_ a: Int, _ b: Int) -> Int? {
        let (r, o) = a.addingReportingOverflow(b)
        return o ? nil : r
    }
    static func mul(_ a: Int, _ b: Int) -> Int? {
        let (r, o) = a.multipliedReportingOverflow(by: b)
        return o ? nil : r
    }
}

/// Pure budget arithmetic for the two slot pools (unit-tested without
/// Metal or a GGUF): splits `budgetBytes` into gate-up and down slot
/// counts whose total is within budget, each at least 1 (one expert
/// iteration needs one slot per family). Returns nil when the budget
/// cannot hold one of each slice. All arithmetic is checked.
public enum ExpertCacheGeometry {
    public static func slotCounts(budgetBytes: Int, guStrideBytes: Int,
                                  dnStrideBytes: Int) -> (gu: Int, dn: Int)? {
        guard guStrideBytes > 0, dnStrideBytes > 0, budgetBytes > 0,
              let pair = Checked.add(guStrideBytes, dnStrideBytes),
              budgetBytes >= pair,
              // Cap slot counts far below overflow territory; real
              // strides are megabytes and budgets gigabytes.
              budgetBytes <= Int.max / 2 else { return nil }
        // Proportional split by stride, floored to at least one slot
        // each, then trimmed down until the total fits the budget.
        let share = Double(budgetBytes) * Double(guStrideBytes) / Double(pair)
        guard share.isFinite, share < Double(Int.max) else { return nil }
        var gu = max(1, Int(share / Double(guStrideBytes)))
        var dn = max(1, (budgetBytes - gu * guStrideBytes) / dnStrideBytes)
        while let guBytes = Checked.mul(gu, guStrideBytes),
              let dnBytes = Checked.mul(dn, dnStrideBytes),
              let total = Checked.add(guBytes, dnBytes),
              total > budgetBytes {
            if dn > 1 { dn -= 1 }
            else if gu > 1 { gu -= 1 }
            else { return nil }
        }
        guard let guBytes = Checked.mul(gu, guStrideBytes),
              let dnBytes = Checked.mul(dn, dnStrideBytes),
              let total = Checked.add(guBytes, dnBytes), total <= budgetBytes,
              gu >= 1, dn >= 1 else { return nil }
        return (gu, dn)
    }
}

/// Bounded LRU cache of expert weight slices. Slot pools are two
/// MTLBuffers (gate-up and down families); each slot holds exactly one
/// expert's slice of one layer's tensor, loaded with a checked pread
/// from the original GGUF. Pinning protocol:
/// - `slice(...)` pins the returned slot as ACTIVE for the caller's
///   current expert iteration (at most one slot per family).
/// - `endExpertIteration(segmentPending:)` releases active pins; in
///   deferred-segment mode the slots move to IN-FLIGHT (encoded GPU
///   work may still read them) until the next segment flush.
/// - A victim is never chosen from active or in-flight slots. If all
///   slots of a family are pinned, the caller-supplied `flush` runs
///   first (completing GPU work, clearing in-flight pins) — counted as
///   a flush-before-reuse — and eviction then proceeds.
/// - Load ordering: the victim's mapping is dropped BEFORE the pread
///   (its bytes are dead), and the new key is installed ONLY after the
///   full slice was read; a failed/short read leaves the slot FREE,
///   never holding a partial slice.
public final class Gemma4ExpertCache {
    public enum Family {
        case gateUp
        case down
    }

    public let budgetBytes: Int
    public let guStrideBytes: Int
    public let dnStrideBytes: Int
    let guPool: MTLBuffer
    let dnPool: MTLBuffer
    public let guSlotCount: Int
    public let dnSlotCount: Int
    /// Size of the compacted non-expert weights buffer (reported, not
    /// owned — the runner owns that buffer).
    public var nonExpertResidentBytes: Int = 0

    private let device: MTLDevice
    private var fd: Int32
    private let ggufSize: Int
    private var guTable: ExpertSlotTable
    private var dnTable: ExpertSlotTable
    private var activeGU: Set<Int> = []
    private var activeDN: Set<Int> = []
    private var inflightGU: Set<Int> = []
    private var inflightDN: Set<Int> = []

    private(set) var sliceBytesRead = 0
    private(set) var prefetchSlices = 0
    private(set) var prefetchBytes = 0
    private(set) var prefetchHits = 0
    // Prefetch attribution markers, per family, keyed by SLOT (bounded
    // by pool size): set when prefetch installs a slice, consumed
    // (removed) on the first hit served from that slot, removed on any
    // demand fill of that slot in the SAME family. Hits after an
    // eviction + on-demand reload are NOT prefetch hits. Slot indices
    // are per-family, so a GU eviction never touches DN markers.
    private var prefetchedGUSlots: Set<Int> = []
    private var prefetchedDNSlots: Set<Int> = []
    private(set) var misses = 0
    private(set) var hits = 0
    private(set) var evictions = 0
    private(set) var flushesBeforeReuse = 0
    private var peakPinnedBytes = 0
    private var peakDeviceAllocatedBytes = 0
    private(set) var initDeviceAllocatedBytes = 0

    /// Public for component tests (drives real slot loads against a
    /// synthetic file + Metal device without a GGUF).
    public init(device: MTLDevice, fileURL: URL, ggufSize: Int,
                guStrideBytes: Int, dnStrideBytes: Int, budgetBytes: Int) throws {
        guard let slots = ExpertCacheGeometry.slotCounts(
                budgetBytes: budgetBytes, guStrideBytes: guStrideBytes,
                dnStrideBytes: dnStrideBytes) else {
            throw Gemma4Error.badTensorGeometry(
                "expert cache budget \(budgetBytes) cannot hold one gate-up slice (\(guStrideBytes)) plus one down slice (\(dnStrideBytes))")
        }
        guard let guBytes = Checked.mul(slots.gu, guStrideBytes),
              let dnBytes = Checked.mul(slots.dn, dnStrideBytes),
              guBytes <= device.maxBufferLength, dnBytes <= device.maxBufferLength else {
            throw Gemma4Error.exceedsMaxBufferLength(
                needed: max(guStrideBytes, dnStrideBytes), max: device.maxBufferLength)
        }
        guard let guPool = device.makeBuffer(length: guBytes, options: .storageModeShared),
              let dnPool = device.makeBuffer(length: dnBytes, options: .storageModeShared) else {
            throw Gemma4Error.metalUnavailable
        }
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else { throw Gemma4Error.mmapFailed(errno: errno) }
        self.device = device
        self.fd = fd
        self.ggufSize = ggufSize
        self.budgetBytes = budgetBytes
        self.guStrideBytes = guStrideBytes
        self.dnStrideBytes = dnStrideBytes
        self.guPool = guPool
        self.dnPool = dnPool
        self.guSlotCount = slots.gu
        self.dnSlotCount = slots.dn
        self.guTable = ExpertSlotTable(slotCount: slots.gu)
        self.dnTable = ExpertSlotTable(slotCount: slots.dn)
    }

    deinit { close(fd) }

    var residentBytes: Int {
        Checked.add(Checked.mul(guSlotCount, guStrideBytes) ?? 0,
                    Checked.mul(dnSlotCount, dnStrideBytes) ?? 0) ?? 0
    }

    /// Resolves the resident slice for `key` (hit) or loads it from
    /// `fileOffset` in the GGUF (miss, possibly evicting the family's
    /// LRU unpinned slot). The returned slot is pinned ACTIVE until
    /// `endExpertIteration`. `flush` must complete all outstanding GPU
    /// work reading pool memory (the runner's segment flush).
    public func slice(key: Int, family: Family, fileOffset: Int,
               flush: () throws -> Void) throws -> (buffer: MTLBuffer, offset: Int) {
        let stride = family == .gateUp ? guStrideBytes : dnStrideBytes
        guard fileOffset >= 0, fileOffset < ggufSize,
              let wEnd = Checked.add(fileOffset, stride), wEnd <= ggufSize else {
            throw Gemma4Error.badTensorGeometry(
                "expert slice [\(fileOffset), \(Checked.add(fileOffset, stride).map(String.init) ?? "overflow")) beyond GGUF size \(ggufSize)")
        }
        switch family {
        case .gateUp:
            if let s = guTable.lookup(key) {
                hits += 1
                if prefetchedGUSlots.contains(s) {
                    prefetchHits += 1
                    prefetchedGUSlots.remove(s)   // first consumption only
                }
                activeGU.insert(s)
                trackPeaks()
                return (guPool, Checked.mul(s, guStrideBytes)!)
            }
            misses += 1
            let slot = try acquireSlot(table: &guTable, active: activeGU,
                                       inflight: inflightGU, flush)
            prefetchedGUSlots.remove(slot)   // demand fill replaces content
            let evicted = try loadSlot(table: &guTable, slot: slot, key: key,
                                       pool: guPool, stride: guStrideBytes,
                                       fileOffset: fileOffset)
            if evicted { evictions += 1 }
            activeGU.insert(slot)
            trackPeaks()
            return (guPool, Checked.mul(slot, guStrideBytes)!)
        case .down:
            if let s = dnTable.lookup(key) {
                hits += 1
                if prefetchedDNSlots.contains(s) {
                    prefetchHits += 1
                    prefetchedDNSlots.remove(s)   // first consumption only
                }
                activeDN.insert(s)
                trackPeaks()
                return (dnPool, Checked.mul(s, dnStrideBytes)!)
            }
            misses += 1
            let slot = try acquireSlot(table: &dnTable, active: activeDN,
                                       inflight: inflightDN, flush)
            prefetchedDNSlots.remove(slot)   // demand fill replaces content
            let evicted = try loadSlot(table: &dnTable, slot: slot, key: key,
                                       pool: dnPool, stride: dnStrideBytes,
                                       fileOffset: fileOffset)
            if evicted { evictions += 1 }
            activeDN.insert(slot)
            trackPeaks()
            return (dnPool, Checked.mul(slot, dnStrideBytes)!)
        }
    }

    /// Slot selection on a miss: a FREE slot first (initial fill), then
    /// the family's LRU OCCUPIED slot that is neither active nor
    /// in-flight. If every occupied slot is pinned, flush first
    /// (flush-before-reuse) — the flush completes GPU work and clears
    /// in-flight pins — then retry against active pins only. If even
    /// then no slot is free, the budget cannot run one expert
    /// iteration; fail closed.
    private func acquireSlot(table: inout ExpertSlotTable, active: Set<Int>,
                             inflight: Set<Int>,
                             _ flush: () throws -> Void) throws -> Int {
        if let free = table.freeSlot() { return free }
        if let s = table.victim(pinned: active.union(inflight)) { return s }
        try flush()
        flushesBeforeReuse += 1
        noteSegmentFlushed()
        guard let s = table.victim(pinned: active) else {
            throw Gemma4Error.badTensorGeometry(
                "expert cache exhausted: every slot is actively pinned")
        }
        return s
    }

    /// Load ordering: if the chosen slot was OCCUPIED, its mapping is
    /// dropped (eviction) BEFORE the pread overwrites the bytes; the
    /// new key is installed ONLY after the full read succeeded. A
    /// failed read leaves the slot FREE — no partial slice is ever
    /// addressable. Returns whether a resident slice was evicted.
    private func loadSlot(table: inout ExpertSlotTable, slot: Int, key: Int,
                          pool: MTLBuffer, stride: Int,
                          fileOffset: Int) throws -> Bool {
        let wasOccupied = table.isOccupied(slot)
        if wasOccupied { table.evict(slot: slot) }
        try Gemma4ExpertCache.preadExact(fd: fd,
                                         into: pool.contents() + Checked.mul(slot, stride)!,
                                         count: stride, from: fileOffset)
        table.install(key: key, slot: slot)
        let (read, o) = sliceBytesRead.addingReportingOverflow(stride)
        guard !o else {
            throw Gemma4Error.badTensorGeometry("slice byte counter overflow")
        }
        sliceBytesRead = read
        return wasOccupied
    }

    /// Ends one expert iteration: active pins drop (per-kernel mode —
    /// dispatch already waited for completion) or become in-flight
    /// (deferred-segment mode — cleared at the next segment flush).
    /// Synchronous read-ahead into a FREE slot only: never evicts, so
    /// it cannot change any computation (the slice is byte-identical to
    /// an on-demand load, and no resident slice is disturbed). Blocking
    /// preads performed between expert iterations — NO async I/O/GPU
    /// overlap is claimed. Returns false when the family's pool has no
    /// free slot or the key is already resident (nothing done).
    @discardableResult
    public func prefetch(key: Int, family: Family, fileOffset: Int) throws -> Bool {
        let stride = family == .gateUp ? guStrideBytes : dnStrideBytes
        guard fileOffset >= 0, fileOffset < ggufSize,
              let wEnd = Checked.add(fileOffset, stride), wEnd <= ggufSize else {
            throw Gemma4Error.badTensorGeometry(
                "prefetch slice beyond GGUF size \(ggufSize)")
        }
        switch family {
        case .gateUp:
            if guTable.lookup(key) != nil { return false }
            guard let free = guTable.freeSlot() else { return false }
            try Gemma4ExpertCache.preadExact(
                fd: fd, into: guPool.contents() + Checked.mul(free, guStrideBytes)!,
                count: guStrideBytes, from: fileOffset)
            guTable.install(key: key, slot: free)
            prefetchedGUSlots.insert(free)
        case .down:
            if dnTable.lookup(key) != nil { return false }
            guard let free = dnTable.freeSlot() else { return false }
            try Gemma4ExpertCache.preadExact(
                fd: fd, into: dnPool.contents() + Checked.mul(free, dnStrideBytes)!,
                count: dnStrideBytes, from: fileOffset)
            dnTable.install(key: key, slot: free)
            prefetchedDNSlots.insert(free)
        }
        prefetchSlices += 1
        let (b, o) = prefetchBytes.addingReportingOverflow(stride)
        guard !o else { throw Gemma4Error.badTensorGeometry("prefetch counter overflow") }
        prefetchBytes = b
        let (r, o2) = sliceBytesRead.addingReportingOverflow(stride)
        guard !o2 else { throw Gemma4Error.badTensorGeometry("slice counter overflow") }
        sliceBytesRead = r
        return true
    }

    public func endExpertIteration(segmentPending: Bool) {
        if segmentPending {
            inflightGU.formUnion(activeGU)
            inflightDN.formUnion(activeDN)
        }
        activeGU.removeAll()
        activeDN.removeAll()
    }

    /// All encoded GPU work has completed (runner segment flush or CPU
    /// readback): no slot is in flight any more.
    func noteSegmentFlushed() {
        inflightGU.removeAll()
        inflightDN.removeAll()
    }

    /// Error cleanup: a DISCARDED (never-submitted) segment's encoded
    /// work will never execute, so its pins — active and in-flight —
    /// are dropped wholesale. Called from the runner's
    /// discardOpenSegment; the discarded bytes in the pools are simply
    /// dead until evicted.
    func discardAllPins() {
        activeGU.removeAll()
        activeDN.removeAll()
        inflightGU.removeAll()
        inflightDN.removeAll()
    }

    /// Records the device allocation level at the end of cache-mode
    /// construction (reported as the bounded startup level).
    func noteInitAllocated() {
        initDeviceAllocatedBytes = Int(clamping: device.currentAllocatedSize)
    }

    /// Peak pinned bytes uses the UNION of active and in-flight slots
    /// per family (a slot can be both within one iteration window).
    private func trackPeaks() {
        let pinned = (activeGU.union(inflightGU)).count * guStrideBytes
                  + (activeDN.union(inflightDN)).count * dnStrideBytes
        peakPinnedBytes = max(peakPinnedBytes, pinned)
        peakDeviceAllocatedBytes = max(peakDeviceAllocatedBytes,
                                       Int(clamping: device.currentAllocatedSize))
    }

    /// Full pread loop: EINTR is retried; a short read, EOF, or hard
    /// error aborts the load (no partial slice becomes resident — see
    /// `loadSlot`). Internal static so tests can drive it against a
    /// crafted short file directly.
    public static func preadExact(fd: Int32, into dst: UnsafeMutableRawPointer,
                           count: Int, from offset: Int) throws {
        var done = 0
        while done < count {
            let n = pread(fd, dst + done, count - done, off_t(offset + done))
            if n < 0 {
                if errno == EINTR { continue }
                throw Gemma4Error.mmapFailed(errno: errno)
            }
            guard n > 0 else { throw Gemma4Error.mmapFailed(errno: 0) }   // EOF/short
            done += n
        }
    }

    public func stats() -> Gemma4ExpertCacheStats {
        Gemma4ExpertCacheStats(
            budgetBytes: budgetBytes, residentBytes: residentBytes,
            guSlots: guSlotCount, dnSlots: dnSlotCount,
            guStrideBytes: guStrideBytes, dnStrideBytes: dnStrideBytes,
            nonExpertResidentBytes: nonExpertResidentBytes,
            sliceBytesRead: sliceBytesRead, misses: misses, hits: hits,
            evictions: evictions, flushesBeforeReuse: flushesBeforeReuse,
            peakPinnedBytes: peakPinnedBytes,
            peakDeviceAllocatedBytes: peakDeviceAllocatedBytes,
            initDeviceAllocatedBytes: initDeviceAllocatedBytes,
            prefetchSlices: prefetchSlices, prefetchBytes: prefetchBytes,
            prefetchHits: prefetchHits)
    }
}
