"""Runtime composition for the pure-JAX command-line programs.

This module is deliberately orchestration-only.  It builds one controller,
source, reference SDE, and target potential from the Hydra experiment config;
training and checkpoint persistence remain owned by their dedicated modules.
No Torch package is imported anywhere on this path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import yaml
from omegaconf import DictConfig, OmegaConf

from cg_bms_jax.coordinates import (
    AmbientTransform,
    RelativeDomainSpec,
    RelativeSoftWall,
)
from cg_bms_jax.data import GaussianSource, bridge_data_sha256
from cg_bms_jax.model import MLPController, PaiNN, RadialCOMHead, ShapeCOMController
from cg_bms_jax.potential import (
    BMS_AA_CB_INDICES,
    BMS_AA_HA_INDICES,
    Ala2CanonicalSupport,
    AmbientAla2Potential,
    AmbientAllAtomAla2Potential,
    AnalyticMB2DPotential,
    BMSCBImproperRestraint,
    BMSImproperRestraint,
    CartesianBoxDomain,
    CGBGAla2Bundle,
    CGBGAla2PMF,
    CGBGMBPotential,
    MBPMFBundle,
    OpenMMAla2Backend,
    OpenMMAla2Spec,
    SmoothEnergyWindow,
    build_cgbg_mace_energy_fn,
    file_sha256,
)
from cg_bms_jax.process import EDMSDE


def _source_checkout_root() -> Path | None:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "configs").is_dir() and (source_root / "pyproject.toml").is_file():
        return source_root
    cwd = Path.cwd().resolve()
    if (cwd / "configs").is_dir() and (cwd / "pyproject.toml").is_file():
        return cwd
    return None


def repository_root() -> Path:
    """Return the checkout root, or the installed package resource root."""

    checkout = _source_checkout_root()
    if checkout is not None:
        return checkout
    package_root = Path(__file__).resolve().parent
    if (package_root / "configs").is_dir():
        return package_root
    raise FileNotFoundError("Could not locate cg-bms-jax configuration resources")


def config_root() -> Path:
    packaged = Path(__file__).resolve().parent / "configs"
    if packaged.is_dir():
        return packaged
    return repository_root() / "configs"


def compose_config(config_name: str, overrides: list[str] | tuple[str, ...] = ()) -> DictConfig:
    """Compose a repository config without changing the process working directory."""

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(config_root())):
        return compose(config_name=config_name, overrides=list(overrides))


def config_as_dict(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    else:
        value = dict(config)
    if not isinstance(value, dict):
        raise TypeError("The root configuration must be a mapping")
    return value


def canonical_digest(value: Any) -> str:
    """Stable SHA-256 used in runtime/checkpoint compatibility metadata."""

    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    explicit_root = os.environ.get("CG_BMS_PROJECT_ROOT")
    if explicit_root:
        return (Path(explicit_root).expanduser() / path).resolve()
    checkout = _source_checkout_root()
    if checkout is not None:
        return (checkout / path).resolve()
    return (Path.cwd() / path).resolve()


def resolve_project_or_package_path(path: str | Path) -> Path:
    """Resolve a project asset, falling back to a wheel-bundled resource."""

    raw = Path(path).expanduser()
    resolved = resolve_project_path(raw)
    if resolved.exists() or raw.is_absolute():
        return resolved
    packaged = (Path(__file__).resolve().parent / raw).resolve()
    return packaged if packaged.exists() else resolved


def _manifest() -> dict[str, Any]:
    packaged = Path(__file__).resolve().parent / "assets" / "manifest.yaml"
    path = packaged if packaged.is_file() else repository_root() / "assets" / "manifest.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid asset manifest: {path}")
    return value


def _asset_sha(name: str) -> str:
    try:
        return str(_manifest()["files"][name]["sha256"])
    except (KeyError, TypeError) as error:
        raise KeyError(f"Pinned asset {name!r} is absent from assets/manifest.yaml") from error


@dataclass(frozen=True)
class RuntimeSystem:
    """All immutable ingredients shared by training and sampling commands."""

    experiment: Mapping[str, Any]
    controller: nn.Module
    initial_variables: Any
    source: GaussianSource
    sde: EDMSDE
    potential: Any | None
    transform: AmbientTransform | None
    domain: Any | None
    identity: Mapping[str, Any]
    physical_map: Any | None = None

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.experiment["state_shape"])

    @property
    def name(self) -> str:
        return str(self.experiment["name"])

    def apply(self, variables: Any, time: Any, state: Any) -> jax.Array:
        return self.controller.apply(variables, time, state)

    def to_physical(self, state: Any) -> jax.Array:
        value = jnp.asarray(state)
        if self.physical_map is not None:
            return self.physical_map(value)
        return value if self.transform is None else self.transform.to_physical(value)

    def evaluate_target(self, state: Any, *, include_training_wall: bool) -> Any:
        if self.potential is None:
            raise RuntimeError("This RuntimeSystem was built without loading the target potential")
        if isinstance(
            self.potential, (AmbientAla2Potential, AmbientAllAtomAla2Potential)
        ):
            return self.potential.evaluate(state, include_training_wall=include_training_wall)
        if isinstance(self.potential, AnalyticMB2DPotential):
            return self.potential.evaluate(
                state,
                include_training_wall=include_training_wall,
            )
        return self.potential.evaluate(state)

    def domain_metadata(self) -> dict[str, Any]:
        if self.domain is None:
            # reweight.io uses a common schema for molecular and scalar tests.
            return {"box": [], "anchor": 0, "margin": 0.0, "support_mode": "ambient_all"}
        return self.domain.metadata()

    def support_mask(self, state: Any) -> jax.Array:
        physical = self.to_physical(state)
        if self.domain is None:
            axes = tuple(range(1, physical.ndim))
            return jnp.all(jnp.isfinite(physical), axis=axes)
        # The training-wall margin is deliberately not part of formal support.
        return self.domain.support_mask(physical, use_margin=False)


def _controller_variables(controller: nn.Module, key: jax.Array, event_shape: tuple[int, ...]) -> Any:
    params_key, constants_key = jax.random.split(key)
    example = jnp.zeros((1, *event_shape), dtype=jnp.float32)
    return controller.init(
        {"params": params_key, "constants": constants_key},
        jnp.asarray(0.5, dtype=example.dtype),
        example,
    )


def _base_components(experiment: Mapping[str, Any]) -> tuple[GaussianSource, EDMSDE]:
    event_shape = tuple(int(size) for size in experiment["state_shape"])
    source_cfg = experiment["source"]
    sde_cfg = experiment["sde"]
    source = GaussianSource(
        event_shape=event_shape,
        mean=source_cfg["mean"],
        scale=float(source_cfg["sigma"]),
    )
    sde = EDMSDE(
        sigma_min=float(sde_cfg["sigma_min"]),
        sigma_max=float(sde_cfg["sigma_max"]),
        rho=float(sde_cfg["rho"]),
    )
    return source, sde


def _build_mb(
    experiment: Mapping[str, Any],
    *,
    key: jax.Array,
    load_potential: bool,
) -> RuntimeSystem:
    model_cfg = experiment["model"]
    controller = MLPController(
        hidden_features=tuple(int(width) for width in model_cfg["hidden_dims"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
    )
    source, sde = _base_components(experiment)
    variables = _controller_variables(controller, key, source.event_shape)
    potential = None
    pmf_sha = _asset_sha("mb_pmf")
    if load_potential:
        checkpoint = resolve_project_path(experiment["assets"]["pmf"])
        bundle = MBPMFBundle(
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=pmf_sha,
            data_path=str(resolve_project_path(experiment["assets"]["train"])),
            data_sha256=_asset_sha("mb_train"),
            kT=float(experiment["kT"]),
        )
        # The hash is pinned in our versioned manifest; this is the explicit
        # trust boundary required by load_trusted_pickle.
        potential = CGBGMBPotential.from_bundle(bundle, trusted=True)
    manifest = _manifest()
    identity = {
        "experiment_family": "mb_cg1d",
        "event_shape": list(source.event_shape),
        "coordinate_mode": str(experiment["coordinate_mode"]),
        "model_signature": canonical_digest(model_cfg),
        "sde_signature": canonical_digest({key: experiment["sde"][key] for key in ("sigma_min", "sigma_max", "rho")}),
        "pmf_revision": str(manifest["revision"]),
        "pmf_sha256": pmf_sha,
        "training_data_sha256": _asset_sha("mb_train"),
        "coordinate_signature": canonical_digest(
            {
                "event_shape": list(source.event_shape),
                "coordinate_mode": str(experiment["coordinate_mode"]),
                "mapping_name": "mb_cg1d_identity",
                "mapping_indices": [0],
                "standardization_std": 1.0,
                "com_sigma": None,
                "density_mode": "ambient_1d",
            }
        ),
        "species_signature": canonical_digest([0]),
        "density_mode": "ambient_exact",
    }
    return RuntimeSystem(experiment, controller, variables, source, sde, potential, None, None, identity)


def _build_mb2d(
    experiment: Mapping[str, Any],
    *,
    key: jax.Array,
    load_potential: bool,
) -> RuntimeSystem:
    """Build the asset-free analytic two-dimensional Muller--Brown system."""

    model_cfg = experiment["model"]
    controller = MLPController(
        hidden_features=tuple(int(width) for width in model_cfg["hidden_dims"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
    )
    source, sde = _base_components(experiment)
    variables = _controller_variables(controller, key, source.event_shape)

    target_cfg = experiment["target"]
    affine_cfg = experiment["affine"]
    confinement_cfg = target_cfg["confinement"]
    beta = float(target_cfg["beta"])
    kT = float(experiment["kT"])
    if not math.isclose(kT, 1.0 / beta, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "MB2D temperature is specified once: experiment.kT must equal "
            "1 / experiment.target.beta"
        )
    prototype = AnalyticMB2DPotential(
        beta=beta,
        offset=tuple(float(value) for value in affine_cfg["offset"]),
        scale=tuple(float(value) for value in affine_cfg["scale"]),
        physical_box=tuple(
            tuple(float(bound) for bound in axis) for axis in target_cfg["box"]
        ),
        confinement_margin=tuple(
            float(value) for value in confinement_cfg["margin"]
        ),
        confinement_strength=float(confinement_cfg["strength"]),
        extension_width=tuple(
            float(value) for value in confinement_cfg["extension_width"]
        ),
    )
    potential = prototype if load_potential else None
    physical_domain = CartesianBoxDomain(
        lower=prototype.physical_lower,
        upper=prototype.physical_upper,
        margin=prototype.confinement_margin,
    )

    bridge_data_cfg = experiment["bridge_data"]
    if not isinstance(bridge_data_cfg, Mapping):
        raise TypeError("experiment.bridge_data must be a mapping")
    endpoint_data_sha = bridge_data_sha256(bridge_data_cfg)
    endpoint_distribution = str(bridge_data_cfg.get("distribution", ""))
    formal_target_spec = {
        "implementation_abi": "analytic_muller_brown_finite_box_v1",
        "energy": "cg_bg_muller_brown_unbiased",
        "beta": beta,
        "physical_box": [list(axis) for axis in prototype.physical_box],
    }
    formal_target_signature = canonical_digest(formal_target_spec)
    training_target_signature = canonical_digest(
        {
            "formal_target_signature": formal_target_signature,
            "training_confinement": {
                "margin": list(prototype.confinement_margin),
                "strength": prototype.confinement_strength,
            },
            "outside_box_extension": "componentwise_c2_tanh_v1",
            "extension_width": list(prototype.extension_width),
        }
    )
    coordinate_spec = {
        "event_shape": list(source.event_shape),
        "coordinate_mode": str(experiment["coordinate_mode"]),
        "mapping_name": "mb2d_affine",
        "offset": list(prototype.offset),
        "scale": list(prototype.scale),
        "state_box": [
            list(prototype.state_domain.lower),
            list(prototype.state_domain.upper),
        ],
        "physical_box": [list(axis) for axis in prototype.physical_box],
        "density_mode": "ambient_2d",
    }
    topology_signature = canonical_digest(
        {
            "kind": "flat_cartesian_2d",
            "degrees_of_freedom": ["x", "y"],
        }
    )
    identity = {
        "experiment_family": "mb2d_analytic",
        "event_shape": list(source.event_shape),
        "coordinate_mode": str(experiment["coordinate_mode"]),
        "model_signature": canonical_digest(model_cfg),
        "sde_signature": canonical_digest(
            {
                name: experiment["sde"][name]
                for name in ("sigma_min", "sigma_max", "rho")
            }
        ),
        "pmf_revision": "analytic_muller_brown_finite_box_v1",
        # Analytic targets have no external checkpoint.  The formal target
        # digest fills the immutable PMF identity slot used by checkpoints.
        "pmf_sha256": formal_target_signature,
        # This is the configured endpoint-data identity used for checkpoint
        # compatibility. Actual use as an initializer is recorded separately
        # by CheckpointMetadata.warmstart_data_sha256.
        "training_data_sha256": endpoint_data_sha,
        "configured_endpoint_spec_sha256": endpoint_data_sha,
        "endpoint_distribution": endpoint_distribution,
        "synthetic_endpoint_spec_sha256": (
            endpoint_data_sha
            if endpoint_distribution
            == "full_support_diagonal_gaussian_mixture_v1"
            else None
        ),
        "equilibrium_endpoint_spec_sha256": (
            endpoint_data_sha
            if endpoint_distribution == "equilibrium_endpoint_npz_v1"
            else None
        ),
        "coordinate_signature": canonical_digest(coordinate_spec),
        "species_signature": canonical_digest(["x", "y"]),
        "density_mode": "ambient_exact",
        "target_mode": "analytic_mb2d",
        "target_implementation_abi": "analytic_muller_brown_finite_box_v1",
        "target_terms": ["analytic_muller_brown"],
        "training_target_terms": [
            "analytic_muller_brown",
            "finite_box_confinement",
        ],
        "beta": beta,
        "physical_box": [list(axis) for axis in prototype.physical_box],
        "state_box": {
            "lower": list(prototype.state_domain.lower),
            "upper": list(prototype.state_domain.upper),
        },
        "target_signature": formal_target_signature,
        "formal_target_signature": formal_target_signature,
        "training_target_signature": training_target_signature,
        "topology_signature": topology_signature,
        "domain": physical_domain.metadata(),
    }
    return RuntimeSystem(
        experiment,
        controller,
        variables,
        source,
        sde,
        potential,
        None,
        physical_domain,
        identity,
        physical_map=prototype.state_to_physical,
    )


def _build_ala2(
    experiment: Mapping[str, Any],
    *,
    key: jax.Array,
    load_potential: bool,
) -> RuntimeSystem:
    data_path = resolve_project_path(experiment["assets"]["train"])
    checkpoint_path = resolve_project_path(experiment["assets"]["pmf"])
    bundle = CGBGAla2Bundle.from_data_file(
        data_path,
        checkpoint_path,
        temperature_kelvin=float(experiment["temperature_kelvin"]),
    )
    # Assert the locally computed digests against the separately pinned manifest.
    expected_data_sha = _asset_sha("ala2_train")
    expected_pmf_sha = _asset_sha("ala2_pmf")
    if bundle.data_sha256 != expected_data_sha or bundle.checkpoint_sha256 != expected_pmf_sha:
        raise ValueError("Ala2 assets do not match assets/manifest.yaml")

    model_cfg = experiment["model"]
    painn = PaiNN(
        num_features=int(model_cfg["num_features"]),
        num_radial_basis=int(model_cfg["num_radial_basis"]),
        num_layers=int(model_cfg["num_layers"]),
        num_elements=int(model_cfg["num_elements"]),
        r_max=float(model_cfg["r_max_nm"]),
        r_offset=float(model_cfg["r_offset_nm"]),
        time_init_mode=str(model_cfg["time_init_mode"]),
        parity_breaking=bool(model_cfg["parity_breaking"]),
        unique_atom_indices=bool(model_cfg["unique_atom_indices"]),
        conservative=bool(model_cfg["conservative"]),
    )
    com_head = RadialCOMHead(
        hidden_features=tuple(int(width) for width in model_cfg["com_hidden_dims"]),
        time_embedding_dim=32,
    )
    controller = ShapeCOMController(
        shape_model=painn,
        com_head=com_head,
        position_scale=float(bundle.standardization_std_nm),
    )
    source, sde = _base_components(experiment)
    variables = _controller_variables(controller, key, source.event_shape)

    coordinate_cfg = experiment["coordinates"]
    target_cfg = experiment.get("target", {"mode": "pmf_only"})
    if not isinstance(target_cfg, Mapping):
        raise TypeError("experiment.target must be a mapping")
    target_mode = str(target_cfg.get("mode", "pmf_only")).lower()
    target_implementation_abi = str(
        target_cfg.get("implementation_abi", "ala2_pmf_target_v1")
    )
    pmf_lower_bound = None
    pmf_scale = 1.0
    pmf_gate_energy_scale_kj_mol = None
    canonical_support = None
    cb_improper = None

    def build_cb_improper(restraint_cfg: Any) -> BMSCBImproperRestraint:
        if not isinstance(restraint_cfg, Mapping):
            raise TypeError("target.cb_improper must be a mapping when enabled")
        indices = tuple(int(index) for index in restraint_cfg["indices"])
        return BMSCBImproperRestraint(
            indices=indices,
            location=float(restraint_cfg["location_rad"]),
            tolerance=float(restraint_cfg["tolerance_rad"]),
            force_constant_kj_mol=float(restraint_cfg["force_constant_kj_mol"]),
        )

    if target_mode == "pmf_cb_improper":
        cb_improper = build_cb_improper(target_cfg.get("cb_improper"))
    elif target_mode == "pmf_canonical_support":
        supported_canonical_abis = {
            "ala2_canonical_support_v2",
            "ala2_canonical_additive_v3",
        }
        if target_implementation_abi not in supported_canonical_abis:
            raise ValueError(
                "pmf_canonical_support requires "
                "target.implementation_abi=ala2_canonical_support_v2 or "
                "ala2_canonical_additive_v3"
            )
        window_cfg = target_cfg.get("pmf_energy_window")
        support_cfg = target_cfg.get("canonical_support")
        if not isinstance(window_cfg, Mapping):
            raise TypeError("target.pmf_energy_window must be a mapping")
        if not isinstance(support_cfg, Mapping):
            raise TypeError("target.canonical_support must be a mapping")
        if target_implementation_abi == "ala2_canonical_support_v2":
            gate_cfg = target_cfg.get("pmf_topology_gate")
            if not isinstance(gate_cfg, Mapping):
                raise TypeError("target.pmf_topology_gate must be a mapping")
            pmf_gate_energy_scale_kj_mol = float(
                gate_cfg["energy_scale_kj_mol"]
            )
        else:
            required_v3_keys = {
                "pmf_scale",
                "pmf_energy_window",
                "canonical_support",
                "cb_improper",
            }
            missing_v3_keys = sorted(required_v3_keys.difference(target_cfg))
            if missing_v3_keys:
                raise ValueError(
                    "ala2_canonical_additive_v3 requires target fields: "
                    + ", ".join(missing_v3_keys)
                )
            if "pmf_topology_gate" in target_cfg:
                raise ValueError(
                    "ala2_canonical_additive_v3 forbids target.pmf_topology_gate"
                )
            pmf_scale = float(target_cfg["pmf_scale"])
            if not math.isfinite(pmf_scale) or pmf_scale < 0.0:
                raise ValueError(
                    "ala2_canonical_additive_v3 target.pmf_scale must be "
                    "finite and non-negative"
                )
        pmf_lower_bound = SmoothEnergyWindow(
            minimum_kj_mol=float(window_cfg["minimum_kj_mol"]),
            maximum_kj_mol=float(window_cfg["maximum_kj_mol"]),
            lower_softness_kj_mol=float(window_cfg["lower_softness_kj_mol"]),
            upper_softness_kj_mol=float(window_cfg["upper_softness_kj_mol"]),
        )
        box_lengths = tuple(float(value) for value in bundle.box_lengths_nm)
        canonical_support = Ala2CanonicalSupport(
            box_lengths_nm=box_lengths,
            bond_indices=tuple(
                tuple(int(index) for index in pair)
                for pair in support_cfg["bond_indices"]
            ),
            bond_lower_nm=tuple(float(value) for value in support_cfg["bond_lower_nm"]),
            bond_upper_nm=tuple(float(value) for value in support_cfg["bond_upper_nm"]),
            bond_force_constant_kj_mol_nm2=float(
                support_cfg["bond_force_constant_kj_mol_nm2"]
            ),
            angle_indices=tuple(
                tuple(int(index) for index in triple)
                for triple in support_cfg["angle_indices"]
            ),
            angle_lower_rad=tuple(
                float(value) for value in support_cfg["angle_lower_rad"]
            ),
            angle_upper_rad=tuple(
                float(value) for value in support_cfg["angle_upper_rad"]
            ),
            angle_force_constant_kj_mol_rad2=float(
                support_cfg["angle_force_constant_kj_mol_rad2"]
            ),
            repulsion_indices=tuple(
                tuple(int(index) for index in pair)
                for pair in support_cfg["repulsion_indices"]
            ),
            repulsion_min_nm=float(support_cfg["repulsion_min_nm"]),
            repulsion_force_constant_kj_mol_nm2=float(
                support_cfg["repulsion_force_constant_kj_mol_nm2"]
            ),
        )
        if target_implementation_abi == "ala2_canonical_additive_v3":
            cb_improper = build_cb_improper(target_cfg["cb_improper"])
        elif target_cfg.get("cb_improper") is not None:
            cb_improper = build_cb_improper(target_cfg.get("cb_improper"))
    elif target_mode != "pmf_only":
        raise ValueError(
            "Unknown Ala2 target mode "
            f"{target_mode!r}; expected 'pmf_only', 'pmf_cb_improper', "
            "or 'pmf_canonical_support'"
        )
    transform = AmbientTransform(
        n_beads=6,
        physical_std=float(bundle.standardization_std_nm),
        com_std=float(coordinate_cfg["com_sigma"]),
    )
    box = tuple(float(value) for value in bundle.box_lengths_nm)
    margin = float(coordinate_cfg["domain_margin_fraction"]) * min(box)
    domain = RelativeDomainSpec(
        box=box,
        anchor=int(coordinate_cfg["anchor_bead"]),
        margin=margin,
    )
    wall = RelativeSoftWall(domain=domain, strength=float(coordinate_cfg["wall_strength_kT"]))
    potential = None
    if load_potential:
        mace_energy = build_cgbg_mace_energy_fn(bundle, trusted_checkpoint=True)
        pmf = CGBGAla2PMF(bundle, mace_energy)
        potential = AmbientAla2Potential(
            pmf=pmf,
            physical_std_nm=transform.physical_std,
            com_std=transform.resolved_com_std,
            training_wall=wall,
            pmf_lower_bound=pmf_lower_bound,
            pmf_scale=pmf_scale,
            canonical_support=canonical_support,
            pmf_gate_energy_scale_kj_mol=pmf_gate_energy_scale_kj_mol,
            cb_improper=cb_improper,
        )
    manifest = _manifest()
    formal_target_signature = canonical_digest(
        {
            "implementation_abi": target_implementation_abi,
            "target": target_cfg,
            "temperature_kelvin": float(experiment["temperature_kelvin"]),
            "standardization_std_nm": transform.physical_std,
        }
    )
    training_target_signature = canonical_digest(
        {
            "formal_target_signature": formal_target_signature,
            "training_wall": wall.metadata(),
        }
    )
    identity = {
        "experiment_family": "ala2_ambient18_300k",
        "event_shape": list(source.event_shape),
        "coordinate_mode": str(experiment["coordinate_mode"]),
        "model_signature": canonical_digest(model_cfg),
        "sde_signature": canonical_digest({key: experiment["sde"][key] for key in ("sigma_min", "sigma_max", "rho")}),
        "pmf_revision": str(manifest["revision"]),
        "pmf_sha256": expected_pmf_sha,
        "training_data_sha256": expected_data_sha,
        "coordinate_signature": canonical_digest(
            {
                "event_shape": list(source.event_shape),
                "coordinate_mode": str(experiment["coordinate_mode"]),
                "mapping_name": "ala2_core_beta",
                "mapping_indices": [4, 6, 8, 10, 14, 16],
                "species": [int(value) for value in bundle.species],
                "standardization_std": transform.physical_std,
                "com_sigma": transform.resolved_com_std,
                "density_mode": "ambient_18d_aux_com",
            }
        ),
        "species_signature": canonical_digest(
            [int(value) for value in bundle.species]
        ),
        "standardization_std_nm": transform.physical_std,
        "density_mode": "ambient_exact",
        "target_mode": target_mode,
        "target_implementation_abi": target_implementation_abi,
        "pmf_scale": pmf_scale,
        "target_terms": (
            (
                [
                    "pmf_smooth_energy_window",
                    "pmf_scalar_scale",
                ]
                if target_implementation_abi == "ala2_canonical_additive_v3"
                else (
                    ["pmf_smooth_energy_window", "topology_gated_pmf"]
                    if canonical_support is not None
                    else ["pmf"]
                )
            )
            + (
                ["fixed_bonds", "fixed_angles", "nonbonded_repulsion"]
                if canonical_support is not None
                else []
            )
            + (["cb_improper"] if cb_improper is not None else [])
        ),
        "target_signature": formal_target_signature,
        "formal_target_signature": formal_target_signature,
        "training_target_signature": training_target_signature,
        "topology_signature": (
            None
            if canonical_support is None
            else canonical_digest(canonical_support.metadata())
        ),
        "domain": domain.metadata(),
    }
    return RuntimeSystem(experiment, controller, variables, source, sde, potential, transform, domain, identity)


def _build_ala2_all_atom(
    experiment: Mapping[str, Any],
    *,
    key: jax.Array,
    load_potential: bool,
) -> RuntimeSystem:
    """Build the 22-atom full-rank ambient Ala2 target.

    The learned state is 66D.  PaiNN only sees the centred molecular shape,
    while an independent equivariant COM head preserves the three auxiliary
    translational degrees of freedom required for an ordinary Lebesgue density.
    """

    if tuple(int(value) for value in experiment["state_shape"]) != (22, 3):
        raise ValueError("All-atom Ala2 requires state_shape=[22,3]")
    temperature_kelvin = float(experiment["temperature_kelvin"])
    expected_kT = 0.00831446261815324 * temperature_kelvin
    configured_kT = float(experiment["kT"])
    if not math.isclose(configured_kT, expected_kT, rel_tol=1.0e-12):
        raise ValueError(
            "All-atom Ala2 requires kT=R*T in kJ/mol; "
            f"configured {configured_kT}, expected {expected_kT}"
        )
    model_cfg = experiment["model"]
    painn = PaiNN(
        num_features=int(model_cfg["num_features"]),
        num_radial_basis=int(model_cfg["num_radial_basis"]),
        num_layers=int(model_cfg["num_layers"]),
        num_elements=int(model_cfg["num_elements"]),
        r_max=float(model_cfg["r_max_angstrom"]),
        r_offset=float(model_cfg.get("r_offset_angstrom", 0.0)),
        time_init_mode=str(model_cfg["time_init_mode"]),
        parity_breaking=bool(model_cfg["parity_breaking"]),
        unique_atom_indices=bool(model_cfg["unique_atom_indices"]),
        conservative=bool(model_cfg["conservative"]),
    )
    com_head = RadialCOMHead(
        hidden_features=tuple(int(width) for width in model_cfg["com_hidden_dims"]),
        time_embedding_dim=int(model_cfg.get("com_time_embedding_dim", 32)),
    )
    coordinate_cfg = experiment["coordinates"]
    physical_std_angstrom = float(coordinate_cfg["physical_std_angstrom"])
    com_sigma = float(coordinate_cfg["com_sigma"])
    if not math.isclose(
        com_sigma,
        22**-0.5,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "ala2_aa_ambient66 requires coordinates.com_sigma=1/sqrt(22); "
            "otherwise the orthogonal COM target is not N(0,I_3)"
        )
    transform = AmbientTransform(
        n_beads=22,
        physical_std=physical_std_angstrom,
        com_std=com_sigma,
    )
    controller = ShapeCOMController(
        shape_model=painn,
        com_head=com_head,
        position_scale=physical_std_angstrom,
        com_coordinates="orthogonal",
        num_particles=22,
    )
    source, sde = _base_components(experiment)
    source_mean = jnp.asarray(source.mean)
    if not math.isclose(
        source.scale, 1.0, rel_tol=0.0, abs_tol=0.0
    ) or not bool(jnp.all(source_mean == 0.0)):
        raise ValueError(
            "ala2_aa_ambient66 requires the full-rank source N(0,I_66)"
        )
    variables = _controller_variables(controller, key, source.event_shape)

    target_cfg = experiment["target"]
    openmm_cfg = target_cfg["openmm"]
    pdb_path = resolve_project_or_package_path(str(openmm_cfg["pdb"]))
    expected_pdb_sha = str(openmm_cfg["pdb_sha256"]).lower()
    actual_pdb_sha = file_sha256(pdb_path)
    if actual_pdb_sha != expected_pdb_sha:
        raise ValueError(
            "All-atom Ala2 PDB SHA-256 mismatch: "
            f"expected {expected_pdb_sha}, got {actual_pdb_sha}"
        )
    spec = OpenMMAla2Spec(
        pdb_path=str(pdb_path),
        device=str(openmm_cfg.get("device", "cpu")),
        precision=str(openmm_cfg.get("precision", "mixed")),
        device_index=str(openmm_cfg.get("device_index", "0")),
        forcefield=str(openmm_cfg.get("forcefield", "amber99sbildn.xml")),
        implicit_forcefield=str(
            openmm_cfg.get("implicit_forcefield", "amber99_obc.xml")
        ),
        expected_particles=int(openmm_cfg.get("expected_particles", 22)),
        expected_constraints=int(openmm_cfg.get("expected_constraints", 0)),
    )

    restraint_cfg = target_cfg.get("improper_restraints", {})
    if not isinstance(restraint_cfg, Mapping):
        raise TypeError("target.improper_restraints must be a mapping")
    restraints: list[BMSImproperRestraint] = []
    restraint_names: list[str] = []
    for name, default_indices, default_location in (
        ("cb", BMS_AA_CB_INDICES, 0.6154797086703873),
        ("ha", BMS_AA_HA_INDICES, -0.6154797086703873),
    ):
        configured = restraint_cfg.get(name)
        if configured is None:
            continue
        if not isinstance(configured, Mapping):
            raise TypeError(f"target.improper_restraints.{name} must be a mapping")
        if not bool(configured.get("enabled", True)):
            continue
        restraints.append(
            BMSImproperRestraint(
                indices=tuple(
                    int(index)
                    for index in configured.get("indices", default_indices)
                ),
                location=float(configured.get("location_rad", default_location)),
                tolerance=float(
                    configured.get("tolerance_rad", 0.4363323129985824)
                ),
                force_constant_kj_mol=float(
                    configured.get(
                        "force_constant_kj_mol_rad2", 2412.1333030827506
                    )
                ),
            )
        )
        restraint_names.append(name)

    potential = None
    if load_potential:
        potential = AmbientAllAtomAla2Potential(
            OpenMMAla2Backend(spec),
            temperature_kelvin=temperature_kelvin,
            physical_std_angstrom=physical_std_angstrom,
            com_std=com_sigma,
            restraints=restraints,
        )

    target_spec = {
        "implementation_abi": str(
            target_cfg.get(
                "implementation_abi",
                "ala2_aa_openmm_bms_chirality_augmented_com_v1",
            )
        ),
        "temperature_kelvin": temperature_kelvin,
        "openmm": {
            "pdb_sha256": actual_pdb_sha,
            "forcefield": spec.forcefield,
            "implicit_forcefield": spec.implicit_forcefield,
            "expected_particles": spec.expected_particles,
            "expected_constraints": spec.expected_constraints,
        },
        "improper_restraints": [
            {
                "indices": list(restraint.indices),
                "location_rad": restraint.location,
                "tolerance_rad": restraint.tolerance,
                "force_constant_kj_mol_rad2": restraint.force_constant_kj_mol,
            }
            for restraint in restraints
        ],
        "auxiliary_com": {
            "distribution": "cartesian_gaussian",
            "com_sigma": com_sigma,
        },
    }
    target_signature = canonical_digest(target_spec)
    forcefield_profile = (
        f"{spec.forcefield}+{spec.implicit_forcefield or 'vacuum'}"
    )
    openmm_revision = (
        "openmm:"
        + forcefield_profile
        + f":pdb-sha256:{actual_pdb_sha}"
    )
    bridge_data_cfg = experiment.get("bridge_data", {})
    bridge_data_sha = str(
        bridge_data_cfg.get(
            "sha256",
            experiment.get("assets", {}).get("bridge_data_sha256", "unconfigured"),
        )
    )
    atomic_numbers = [
        1,
        6,
        1,
        1,
        6,
        8,
        7,
        1,
        6,
        1,
        6,
        1,
        1,
        1,
        6,
        8,
        7,
        1,
        6,
        1,
        1,
        1,
    ]
    coordinate_spec = {
        "event_shape": [22, 3],
        "coordinate_mode": "ala2_aa_ambient66",
        "atom_order": "official_bms_22",
        "physical_unit": "angstrom",
        "physical_std_angstrom": physical_std_angstrom,
        "com_sigma": com_sigma,
        "density_mode": "ambient_66d_aux_com_exact",
        "shape_dimension": 63,
        "com_dimension": 3,
    }
    identity = {
        "experiment_family": "ala2_aa_ambient66_300k",
        "event_shape": [22, 3],
        "coordinate_mode": str(experiment["coordinate_mode"]),
        "model_signature": canonical_digest(model_cfg),
        "sde_signature": canonical_digest(
            {
                name: experiment["sde"][name]
                for name in ("sigma_min", "sigma_max", "rho")
            }
        ),
        "pmf_revision": openmm_revision,
        "pmf_sha256": target_signature,
        "training_data_sha256": bridge_data_sha,
        "coordinate_signature": canonical_digest(coordinate_spec),
        "species_signature": canonical_digest(atomic_numbers),
        "density_mode": "ambient_66d_aux_com_exact",
        "target_mode": str(target_cfg.get("mode", "openmm_bms_chirality")),
        "target_implementation_abi": target_spec["implementation_abi"],
        "target_terms": [
            openmm_revision,
            *[f"{name}_improper" for name in restraint_names],
            "gaussian_com_auxiliary",
        ],
        "target_signature": target_signature,
        "formal_target_signature": target_signature,
        "training_target_signature": target_signature,
        "topology_signature": canonical_digest(
            {
                "pdb_sha256": actual_pdb_sha,
                "atomic_numbers": atomic_numbers,
                "num_constraints": spec.expected_constraints,
            }
        ),
        "atom_order": "official_bms_22",
        "atom_order_signature": canonical_digest(
            {
                "name": "official_bms_22",
                "atomic_numbers": atomic_numbers,
            }
        ),
        "physical_unit": "angstrom",
        "temperature_kelvin": temperature_kelvin,
        "kT_kj_mol": configured_kT,
        "openmm_forcefield_profile": forcefield_profile,
        "openmm_pdb_sha256": actual_pdb_sha,
        "domain": {
            "box": [],
            "anchor": 0,
            "margin": 0.0,
            "support_mode": "ambient_all",
        },
    }
    return RuntimeSystem(
        experiment,
        controller,
        variables,
        source,
        sde,
        potential,
        transform,
        None,
        identity,
    )


def build_runtime_system(
    config: DictConfig | Mapping[str, Any],
    *,
    key: jax.Array | None = None,
    load_potential: bool = True,
) -> RuntimeSystem:
    """Build a configured scalar, CG, or all-atom BMS system."""

    root = config_as_dict(config)
    experiment = root.get("experiment", root)
    if not isinstance(experiment, Mapping):
        raise TypeError("config.experiment must be a mapping")
    key = jax.random.PRNGKey(0) if key is None else key
    coordinate_mode = str(experiment["coordinate_mode"])
    if coordinate_mode == "identity" and tuple(experiment["state_shape"]) == (1,):
        return _build_mb(experiment, key=key, load_potential=load_potential)
    if coordinate_mode == "mb2d_affine" and tuple(experiment["state_shape"]) == (2,):
        return _build_mb2d(experiment, key=key, load_potential=load_potential)
    if coordinate_mode == "cgbg_ambient18" and tuple(experiment["state_shape"]) == (6, 3):
        return _build_ala2(experiment, key=key, load_potential=load_potential)
    if coordinate_mode == "ala2_aa_ambient66" and tuple(
        experiment["state_shape"]
    ) == (22, 3):
        return _build_ala2_all_atom(
            experiment, key=key, load_potential=load_potential
        )
    raise ValueError(
        f"Unsupported experiment state/coordinate mode: {experiment['state_shape']!r}, {coordinate_mode!r}"
    )


__all__ = [
    "RuntimeSystem",
    "build_runtime_system",
    "canonical_digest",
    "compose_config",
    "config_as_dict",
    "config_root",
    "repository_root",
    "resolve_project_or_package_path",
    "resolve_project_path",
]
