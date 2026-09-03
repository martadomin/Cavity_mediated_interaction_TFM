# Shared VMC/DMC core: trial wavefunction, particle moves, energy estimators,
# and the VMC sampler. Included by both julia/vmc/run_vmc.jl and
# julia/dmc/dmc_core.jl, so a fix made here is picked up by both.
#
# Include order follows the physical dependency chain:
#   utils          -> periodic geometry, contact-interaction root finding
#   wavefunctions   -> trial Ψ and its log, built from utils
#   moves           -> particle-move proposals, built from utils
#   energy          -> local energy estimators, built from wavefunctions
#   sampler         -> the VMC Metropolis loop, built from all of the above

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "wavefunctions.jl"))
include(joinpath(@__DIR__, "moves.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "sampler.jl"))
