"""orbits easy with the cord as a rigid capsule chain (chain_tiers; the FEM setting is unchanged)."""
from .chain_tiers import OrbitsChainTask
from .orbits_easy import OrbitsEasyTask


class OrbitsEasyChainTask(OrbitsChainTask, OrbitsEasyTask):
    variation_id = "orbits_easy_chain"
    tier = "easy"
    asset_resource = "variations/orbits/chain_k0_v3/easy.npz"     # K0, the start (chain_tiers)
    defaults = OrbitsChainTask.chain_defaults(OrbitsEasyTask, 2.14)
    fixed_parameters = (*OrbitsEasyTask.fixed_parameters, "chain")
