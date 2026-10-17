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
            name: "leviathan-forge-helper",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-forge-helper"
        ),
        .executableTarget(
            name: "leviathan-dag-helper",
            dependencies: ["LeviathanCore"],
            path: "Sources/leviathan-dag-helper"
        ),
    ]
)
