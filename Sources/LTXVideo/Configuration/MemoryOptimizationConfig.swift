// MemoryOptimizationConfig.swift - Memory optimization for LTX-2 generation
// Copyright 2025

import Foundation
@preconcurrency import MLX

/// Controls how aggressively the pipeline manages GPU memory during generation.
///
/// Higher optimization levels trade speed for lower peak memory usage. Choose
/// a preset based on your machine's available RAM, or use
/// ``recommended(forRAMGB:)`` for automatic selection.
///
/// ## Presets
/// | Preset | Eval Freq | Cache Clear | Unload | VAE Tiling | Best For |
/// |--------|-----------|-------------|--------|------------|----------|
/// | ``disabled`` | 8 | No | No | No | 128+ GB |
/// | ``light`` | 4 | No | Yes | No | 64-96 GB |
/// | ``moderate`` | 2 | Yes | Yes | 8 frames | 32-64 GB |
/// | ``aggressive`` | 1 | Yes | Yes | 6 frames | 16-32 GB |
///
/// ## Example
/// ```swift
/// let pipeline = LTXPipeline(
///     model: .distilled,
///     memoryOptimization: .moderate
/// )
/// ```
public struct MemoryOptimizationConfig: Sendable {
    /// How often to evaluate lazy computation graphs (every N transformer blocks)
    /// Lower values = more frequent eval = lower peak memory but slower
    public var evalFrequency: Int

    /// Whether to call Memory.clearCache() after evaluation
    public var clearCacheOnEval: Bool

    /// Whether to unload each component after use in the pipeline
    /// (e.g., unload text encoder before loading transformer)
    public var unloadAfterUse: Bool

    /// Sleep duration (seconds) after unloading a component, to allow GPU memory reclaim
    public var unloadSleepSeconds: Double

    /// VAE temporal tile size (latent frames per chunk). 0 = disabled (decode all at once).
    /// For long videos, tiling reduces peak VAE memory by ~75%.
    /// Recommended: 8 for videos > 97 frames, 0 for shorter videos.
    public var vaeTemporalTileSize: Int

    /// VAE temporal tile overlap (latent frames). Blended with linear interpolation.
    public var vaeTemporalTileOverlap: Int

    /// Automatically enable VAE temporal tiling when decode activations are estimated
    /// to exceed a memory budget. Manual `vaeTemporalTileSize > 0` takes precedence.
    public var vaeAutoTemporalTiling: Bool

    /// Optional memory budget (GB) used by automatic VAE temporal tiling.
    /// If nil, a budget is derived from current MLX allocations and total system RAM.
    public var vaeDecodeMemoryBudgetGB: Double?

    public init(
        evalFrequency: Int = 4,
        clearCacheOnEval: Bool = false,
        unloadAfterUse: Bool = true,
        unloadSleepSeconds: Double = 0.5,
        vaeTemporalTileSize: Int = 0,
        vaeTemporalTileOverlap: Int = 1,
        vaeAutoTemporalTiling: Bool = true,
        vaeDecodeMemoryBudgetGB: Double? = nil
    ) {
        self.evalFrequency = evalFrequency
        self.clearCacheOnEval = clearCacheOnEval
        self.unloadAfterUse = unloadAfterUse
        self.unloadSleepSeconds = unloadSleepSeconds
        self.vaeTemporalTileSize = vaeTemporalTileSize
        self.vaeTemporalTileOverlap = vaeTemporalTileOverlap
        self.vaeAutoTemporalTiling = vaeAutoTemporalTiling
        self.vaeDecodeMemoryBudgetGB = vaeDecodeMemoryBudgetGB
    }

    // MARK: - Presets

    /// No optimization — keep everything in memory, eval every 8 blocks
    public static let disabled = MemoryOptimizationConfig(
        evalFrequency: 8,
        clearCacheOnEval: false,
        unloadAfterUse: false,
        unloadSleepSeconds: 0,
        vaeTemporalTileSize: 0,
        vaeAutoTemporalTiling: false
    )

    /// Light optimization — eval every 4 blocks, unload after use
    public static let light = MemoryOptimizationConfig(
        evalFrequency: 4,
        clearCacheOnEval: false,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.3,
        vaeTemporalTileSize: 0,
        vaeAutoTemporalTiling: true
    )

    /// Moderate optimization — eval every 2 blocks, clear cache, VAE tiling
    public static let moderate = MemoryOptimizationConfig(
        evalFrequency: 2,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.5,
        vaeTemporalTileSize: 8,
        vaeTemporalTileOverlap: 1,
        vaeAutoTemporalTiling: true
    )

    /// Aggressive optimization — eval every block, clear cache, VAE tiling
    public static let aggressive = MemoryOptimizationConfig(
        evalFrequency: 1,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 1.0,
        vaeTemporalTileSize: 6,
        vaeTemporalTileOverlap: 1,
        vaeAutoTemporalTiling: true
    )

    /// Default preset
    public static let `default` = MemoryOptimizationConfig.light

    /// Auto-select preset based on available system RAM
    public static func recommended(forRAMGB ram: Int) -> MemoryOptimizationConfig {
        switch ram {
        case ...32:
            return .aggressive
        case 33...64:
            return .moderate
        case 65...96:
            return .light
        default:
            return .disabled
        }
    }

    /// Resolve the effective temporal tile size (latent frames) for VAE decode.
    ///
    /// Returns 0 when full decode is estimated to fit the selected budget.
    public func effectiveVAETemporalTileSize(
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        systemRAMGB: Int? = nil,
        currentMLXMemoryBytes: Int? = nil,
        activationBytesPerScalar: Int = 4
    ) -> Int {
        if vaeTemporalTileSize > 0 {
            return vaeTemporalTileSize
        }

        guard vaeAutoTemporalTiling, latentFrames > 0 else {
            return 0
        }

        let budgetBytes = effectiveVAEDecodeBudgetBytes(
            systemRAMGB: systemRAMGB,
            currentMLXMemoryBytes: currentMLXMemoryBytes
        )
        guard budgetBytes > 0 else {
            return 0
        }

        let estimatedFullDecodeBytes = Self.estimatedVAEFullDecodeBytes(
            latentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            activationBytesPerScalar: activationBytesPerScalar
        )
        if estimatedFullDecodeBytes <= budgetBytes {
            LTXDebug.log(
                "VAE auto-tiling decision: budget=\(Self.formatBytesForLog(budgetBytes)), estimate=\(Self.formatBytesForLog(estimatedFullDecodeBytes)), full decode"
            )
            return 0
        }

        let nonTemporalBytes = Self.estimatedVAENonTemporalDecodeBytes(
            latentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            bytesPerScalar: activationBytesPerScalar
        )
        let bytesPerLatentFrame = Self.estimatedVAEPeakActivationBytesPerLatentFrame(
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            bytesPerScalar: activationBytesPerScalar
        )

        let usableTemporalBytes = max(0, budgetBytes - nonTemporalBytes)
        let maxLatentFrames = max(2, Int(usableTemporalBytes / max(1, bytesPerLatentFrame)))
        let tileSize = max(vaeTemporalTileOverlap + 1, maxLatentFrames)
        LTXDebug.log(
            "VAE auto-tiling decision: budget=\(Self.formatBytesForLog(budgetBytes)), estimate=\(Self.formatBytesForLog(estimatedFullDecodeBytes)), tile=\(tileSize), overlap=\(vaeTemporalTileOverlap)"
        )
        return tileSize
    }

    /// Effective decode budget in bytes for automatic VAE tiling.
    ///
    /// When an explicit budget is provided, uses it directly. Otherwise derives a
    /// budget from total RAM after reserving headroom for the OS and subtracting
    /// current MLX allocations already resident in unified memory.
    public func effectiveVAEDecodeBudgetBytes(
        systemRAMGB: Int? = nil,
        currentMLXMemoryBytes: Int? = nil
    ) -> Int64 {
        if let budgetGB = vaeDecodeMemoryBudgetGB {
            return Int64(max(0.0, budgetGB) * 1_073_741_824.0)
        }

        let systemBytes = Int64((systemRAMGB ?? Self.detectedSystemRAMGB()) * 1_073_741_824)
        let mlxBytes = Int64(currentMLXMemoryBytes ?? Self.detectedCurrentMLXMemoryBytes())
        let reservedSystemBytes = max(4 * 1_073_741_824, systemBytes / 4)

        return max(0, systemBytes - reservedSystemBytes - mlxBytes)
    }

    /// Detect total system memory in GB.
    public static func detectedSystemRAMGB() -> Int {
        let bytes = ProcessInfo.processInfo.physicalMemory
        return max(1, Int(bytes / 1_073_741_824))
    }

    /// Detect current MLX memory already resident in unified memory.
    public static func detectedCurrentMLXMemoryBytes() -> Int {
        let snapshot = Memory.snapshot()
        return snapshot.activeMemory + snapshot.cacheMemory
    }

    /// Estimated bytes per latent frame for the largest VAE working activation.
    static func estimatedVAEPeakActivationBytesPerLatentFrame(
        latentHeight: Int,
        latentWidth: Int,
        bytesPerScalar: Int
    ) -> Int64 {
        Int64(512) * 4 * Int64(latentHeight * 4) * Int64(latentWidth * 4) * Int64(bytesPerScalar)
    }

    /// Estimated bytes for the non-temporal portion of decode memory.
    ///
    /// Includes the decoded BCHW tensor, the normalized/transposed output tensor,
    /// and a small extra slack for elementwise post-processing.
    static func estimatedVAENonTemporalDecodeBytes(
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        bytesPerScalar: Int
    ) -> Int64 {
        let pixelFrames = max(1, 8 * (latentFrames - 1) + 1)
        let pixelHeight = latentHeight * 32
        let pixelWidth = latentWidth * 32
        let outputBytes = Int64(pixelFrames) * Int64(pixelHeight) * Int64(pixelWidth) * 3 * Int64(bytesPerScalar)
        let postProcessSlack = max(Int64(256 * 1_048_576), outputBytes / 10)
        return outputBytes * 2 + postProcessSlack
    }

    /// Estimated total bytes required for a full non-tiled VAE decode.
    static func estimatedVAEFullDecodeBytes(
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        activationBytesPerScalar: Int
    ) -> Int64 {
        let activationBytes = estimatedVAEPeakActivationBytesPerLatentFrame(
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            bytesPerScalar: activationBytesPerScalar
        ) * Int64(latentFrames)
        let nonTemporalBytes = estimatedVAENonTemporalDecodeBytes(
            latentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            bytesPerScalar: activationBytesPerScalar
        )
        return activationBytes + nonTemporalBytes
    }

    private static func formatBytesForLog(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        return String(format: "%.1f GiB", gib)
    }
}
