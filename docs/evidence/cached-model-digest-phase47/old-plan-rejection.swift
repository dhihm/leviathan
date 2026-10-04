import Foundation
import LeviathanCore
let path = CommandLine.arguments[1]
do {
    _ = try GemmaForge.loadPlan(from: URL(fileURLWithPath: path))
    fatalError("retired actual plan unexpectedly accepted")
} catch {
    print("ACTUAL_RETIRED_PLAN_REJECTED: \(error)")
}
