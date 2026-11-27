// swift-tools-version:6.0
import PackageDescription

#if os(Linux)
import Foundation

let corePath = "Sources/LeviathanCore"
let portableSources = ["GGUF.swift", "LlamaCPU.swift"]
let coreDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent(corePath)
let macOnlySources = try FileManager.default.contentsOfDirectory(atPath: coreDirectory.path)
    .filter { $0.hasSuffix(".swift") && !portableSources.contains($0) }
    .sorted()

let package = Package(
    name: "leviathan",
    targets: [
        .target(name: "LeviathanCore", path: corePath,
                exclude: macOnlySources, sources: portableSources),
        .executableTarget(name: "leviathan-cli", dependencies: ["LeviathanCore"],
                          path: "Sources/leviathan-cli", exclude: ["CompositeCLI.swift"]),
    ]
)
#else
let package = Package(
    name: "leviathan",
    platforms: [.macOS(.v13)],
    targets: [
        .target(
            name: "LeviathanCore",
            path: "Sources/LeviathanCore"
        ),
        .executableTarget(
            name: "leviathan-cli",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-cli"
        ),
        .executableTarget(
            name: "leviathan-tests",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-tests"
        ),
        .executableTarget(
            name: "router-overlap-model-gate",
            dependencies: ["LeviathanCore"],
            path: "tools/router_overlap_model_gate"
        ),
        .executableTarget(
            name: "router-overlap-model-gate-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_overlap_model_gate_pure_tests"
        ),
        .executableTarget(
            name: "owned-process-diagnostics",
            dependencies: ["LeviathanCore"],
            path: "tools/owned_process_diagnostics"
        ),
        .executableTarget(
            name: "owned-process-diagnostics-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/owned_process_diagnostics_pure_tests"
        ),
        .executableTarget(
            name: "router-selection-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_selection_pure_tests"
        ),
        .executableTarget(
            name: "router-selection-component",
            dependencies: ["LeviathanCore"],
            path: "tools/router_selection_component"
        ),
        .executableTarget(
            name: "router-route-replay",
            dependencies: ["LeviathanCore"],
            path: "tools/router_route_replay"
        ),
        .executableTarget(
            name: "router-route-replay-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_route_replay_pure_tests"
        ),
        .executableTarget(
            name: "router-device-routing-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_pure_tests"
        ),
        .executableTarget(
            name: "router-device-routing-component",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_component"
        ),
        .executableTarget(
            name: "router-device-routing-gate",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_gate"
        ),
        .executableTarget(
            name: "router-device-routing-gate-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_gate_pure_tests"
        ),
        .executableTarget(
            name: "router-device-routing-benchmark",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_benchmark"
        ),
        .executableTarget(
            name: "router-device-routing-benchmark-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_device_routing_benchmark_pure_tests"
        ),
        .executableTarget(
            name: "kernel-stage-attribution",
            dependencies: ["LeviathanCore"],
            path: "tools/kernel_stage_attribution"
        ),
        .executableTarget(
            name: "kernel-stage-attribution-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/kernel_stage_attribution_pure_tests"
        ),
        .executableTarget(
            name: "q4-weight-stage-component",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_weight_stage_component"
        ),
        .executableTarget(
            name: "q4-weight-stage-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_weight_stage_pure_tests"
        ),
        .executableTarget(
            name: "q4-packed-word-component",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_packed_word_component"
        ),
        .executableTarget(
            name: "q4-packed-word-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_packed_word_pure_tests"
        ),
        .executableTarget(
            name: "q4-register-loop-component",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_register_loop_component"
        ),
        .executableTarget(
            name: "q4-register-loop-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_register_loop_pure_tests"
        ),
        .executableTarget(
            name: "rms-coschedule-component",
            dependencies: ["LeviathanCore"],
            path: "tools/rms_coschedule_component"
        ),
        .executableTarget(
            name: "rms-coschedule-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/rms_coschedule_pure_tests"
        ),
        .executableTarget(
            name: "q4-first-divergence-diagnostic",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_first_divergence_diagnostic"
        ),
        .executableTarget(
            name: "q4-first-divergence-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_first_divergence_pure_tests"
        ),
        .executableTarget(
            name: "q4-prefix-probe-diagnostic",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_prefix_probe_diagnostic"
        ),
        .executableTarget(
            name: "q4-prefix-probe-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/q4_prefix_probe_pure_tests"
        ),
        .executableTarget(
            name: "leviathan-forge-helper",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-forge-helper"
        ),
        .executableTarget(
            name: "leviathan-dag-helper",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-dag-helper"
        ),
        .executableTarget(
            name: "router-overlap-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_overlap_pure_tests"
        ),
        .executableTarget(
            name: "router-overlap-runtime-pure-tests",
            dependencies: ["LeviathanCore"],
            path: "tools/router_overlap_runtime_pure_tests"
        ),
        .executableTarget(
            name: "router-overlap-event-smoke",
            dependencies: ["LeviathanCore"],
            path: "tools/router_overlap_event_smoke"
        ),
    ]
)
#endif
