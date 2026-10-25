"""orbits medium with the cord as a rigid capsule chain (chain_tiers; the FEM setting is unchanged)."""
from .chain_tiers import OrbitsChainTask
from .orbits_medium import OrbitsMediumTask


class OrbitsMediumChainTask(OrbitsChainTask, OrbitsMediumTask):
    variation_id = "orbits_medium_chain"
    tier = "medium"
    asset_resource = "variations/orbits/chain_k0_v3/medium.npz"     # K0, the start (chain_tiers)
    defaults = OrbitsChainTask.chain_defaults(OrbitsMediumTask, 2.6)
    fixed_parameters = (*OrbitsMediumTask.fixed_parameters, "chain")
