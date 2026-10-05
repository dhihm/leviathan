// swift-tools-version:6.0
import PackageDescription

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
