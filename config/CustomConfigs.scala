package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}

// INT8 default Gemmini
// REFV256D128 Saturn RVV
// Default Rocket core
// Without TL Monitors for faster simulation
class GemminiRocketSaturnConfig extends Config(
    new gemmini.DefaultGemminiConfig ++
    new saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.refParams) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new chipyard.config.AbstractConfig)

// INT8 default Gemmini
// REFV256D128 Saturn RVV
// Default Rocket core
// Without TL Monitors for faster simulation
// With FP16 support in the Rocket core for Softmax computation
class GemminiRocketSaturnConfigWithFP16 extends Config(
    new gemmini.DefaultGemminiConfig ++
    new saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.refParams) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new freechips.rocketchip.rocket.WithFP16 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new chipyard.config.AbstractConfig)