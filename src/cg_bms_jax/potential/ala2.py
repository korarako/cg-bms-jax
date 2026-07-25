"""Pure-JAX adapter for the six-bead CG-BG Ala2 MACE PMF."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.coordinates import (
    RelativeSoftWall,
    com_aux_reduced_energy,
    lift_shape_gradient,
)

from .base import PotentialResult
from .bundle import CGBGAla2Bundle, load_trusted_pickle
from .restraint import BMSCBImproperRestraint
from .support import (
    Ala2CanonicalSupport,
    SmoothEnergyWindow,
    SmoothLowerEnergyBound,
)

Array = jax.Array
SingleEnergyFn = Callable[[Array], Array]


@dataclass(frozen=True)
class CGBGMaceBackend:
    """Exact pure-JAX ForceMatching inference path used by CG-BG.

    ``predict_fractional`` returns energy and the force array emitted by
    ChemTrain for a batch of fractional coordinates.  In the pinned
    periodic-general ABI this returned force follows the physical Cartesian
    convention; the release validation checks it by finite differences.
    """

    predict_fractional: Callable[[Array], tuple[Array, Array]]
    box_lengths_nm: Array

    def evaluate_physical(self, physical_nm: Array) -> tuple[Array, Array]:
        coordinates = jnp.asarray(physical_nm)
        if coordinates.shape[-2:] != (6, 3):
            raise ValueError(f"Expected (...,6,3) coordinates, got {coordinates.shape}")
        leading = coordinates.shape[:-2]
        flat = coordinates.reshape((-1, 6, 3))
        fractional = flat / jnp.asarray(self.box_lengths_nm, dtype=flat.dtype)
        energy, force_physical = self.predict_fractional(fractional)
        # ChemTrain's periodic-general force wrapper returns the force in the
        # physical Cartesian convention even though positions are supplied as
        # fractional coordinates.  The release finite-difference gate checks
        # this pinned ABI explicitly.
        gradient_physical = -jnp.asarray(force_physical)
        return jnp.asarray(energy).reshape(leading), gradient_physical.reshape(coordinates.shape)

    def __call__(self, fractional: Array) -> Array:
        """Compatibility single-energy callable for injected adapter users."""

        value = jnp.asarray(fractional)
        if value.shape != (6, 3):
            raise ValueError(f"Expected one fractional (6,3) conformation, got {value.shape}")
        energy, _force = self.predict_fractional(value[None, ...])
        return jnp.asarray(energy)[0]


def _box_lengths(box: Array) -> Array:
    box = jnp.asarray(box)
    return jnp.diag(box) if box.shape == (3, 3) else box


class CGBGAla2PMF:
    """Physical-coordinate interface around a single-configuration MACE energy."""

    def __init__(
        self,
        bundle: CGBGAla2Bundle,
        energy_fn_fractional: SingleEnergyFn | CGBGMaceBackend,
    ):
        self.bundle = bundle
        self.energy_fn_fractional = energy_fn_fractional
        self._backend = (
            energy_fn_fractional
            if isinstance(energy_fn_fractional, CGBGMaceBackend)
            else None
        )
        self.kT = bundle.kT_kj_mol
        self._box = jnp.asarray(bundle.box_nm)

    def _single_energy(self, physical_nm: Array) -> Array:
        lengths = _box_lengths(self._box)
        # CG-BG's oracle divides Cartesian coordinates by the box length and
        # passes the resulting (possibly negative/out-of-cell) fractional
        # positions directly to periodic_general.  Do not pre-wrap individual
        # beads here: although periodic energies should be equivalent in exact
        # arithmetic, edge construction/order and float32 cutoff behavior must
        # match the published inference path exactly.
        fractional = physical_nm / lengths
        return jnp.asarray(self.energy_fn_fractional(fractional)).reshape(())

    def energy(self, coordinates: Array) -> Array:
        coordinates = jnp.asarray(coordinates)
        if coordinates.shape[-2:] != (6, 3):
            raise ValueError(f"Expected (...,6,3) core-beta coordinates, got {coordinates.shape}")
        if self._backend is not None:
            return self._backend.evaluate_physical(coordinates)[0]
        if coordinates.ndim == 2:
            return self._single_energy(coordinates)
        flat = coordinates.reshape((-1, 6, 3))
        return jax.vmap(self._single_energy)(flat).reshape(coordinates.shape[:-2])

    def energy_and_grad(self, coordinates: Array) -> tuple[Array, Array]:
        coordinates = jnp.asarray(coordinates)
        if self._backend is not None:
            return self._backend.evaluate_physical(coordinates)
        value_grad = jax.value_and_grad(self._single_energy)
        if coordinates.ndim == 2:
            return value_grad(coordinates)
        flat = coordinates.reshape((-1, 6, 3))
        energy, gradient = jax.vmap(value_grad)(flat)
        return energy.reshape(coordinates.shape[:-2]), gradient.reshape(coordinates.shape)

    def evaluate(self, coordinates: Array) -> PotentialResult:
        energy, gradient = self.energy_and_grad(coordinates)
        reduced_energy = energy / self.kT
        reduced_gradient = gradient / self.kT
        reduced_gradient = reduced_gradient - jnp.mean(reduced_gradient, axis=-2, keepdims=True)
        gradient = reduced_gradient * self.kT
        valid = jnp.isfinite(energy) & jnp.all(jnp.isfinite(gradient), axis=(-2, -1))
        return PotentialResult(
            energy=energy,
            gradient=gradient,
            reduced_energy=reduced_energy,
            reduced_gradient=reduced_gradient,
            score=-reduced_gradient,
            valid_mask=valid,
        )


@dataclass(frozen=True)
class AmbientAla2Potential:
    """Ala2 target plus CG-BG's Gaussian auxiliary COM density.

    The dimensional target is the published PMF by default.  Optional PMF
    lower-bound, canonical topology, and CB-improper terms are all explicit
    parts of the formal physical target; they are therefore used consistently
    by forward training, proposal evaluation, and reweighting.  The auxiliary
    COM density and the training-only domain wall remain separate.
    """

    pmf: CGBGAla2PMF
    physical_std_nm: float
    com_std: float | None = None
    training_wall: RelativeSoftWall | None = None
    pmf_lower_bound: SmoothLowerEnergyBound | SmoothEnergyWindow | None = None
    pmf_scale: float = 1.0
    canonical_support: Ala2CanonicalSupport | None = None
    pmf_gate_energy_scale_kj_mol: float | None = None
    cb_improper: BMSCBImproperRestraint | None = None

    def __post_init__(self) -> None:
        pmf_scale = float(self.pmf_scale)
        if not math.isfinite(pmf_scale) or pmf_scale < 0.0:
            raise ValueError("pmf_scale must be finite and non-negative")
        object.__setattr__(self, "pmf_scale", pmf_scale)
        if self.pmf_gate_energy_scale_kj_mol is not None:
            if self.pmf_gate_energy_scale_kj_mol <= 0.0:
                raise ValueError("pmf_gate_energy_scale_kj_mol must be positive")
            if self.canonical_support is None:
                raise ValueError("Topology-gated PMF requires canonical_support")
            if not isinstance(self.pmf_lower_bound, SmoothEnergyWindow):
                raise ValueError("Topology-gated PMF requires a SmoothEnergyWindow")

    @property
    def kT(self) -> float:
        return self.pmf.kT

    def pmf_energy(self, standardized_ambient: Array) -> Array:
        """Return only the published CG-BG PMF component in kJ/mol."""

        physical = jnp.asarray(standardized_ambient) * self.physical_std_nm
        return self.pmf.energy(physical)

    def cb_improper_energy(self, standardized_ambient: Array) -> Array:
        """Return ``U_CB`` in kJ/mol, or exact zeros in PMF-only mode."""

        x = jnp.asarray(standardized_ambient)
        if self.cb_improper is None:
            return jnp.zeros(x.shape[:-2], dtype=x.dtype)
        return self.cb_improper.energy(x * self.physical_std_nm)

    def formal_energy_components(self, standardized_ambient: Array) -> dict[str, Array]:
        """Return every dimensional formal-target component in kJ/mol.

        ``U_pmf`` remains an alias for the raw published PMF for compatibility
        with CG-BG-style analysis.  ``U_pmf_effective`` is the transformed PMF
        before scaling.  ``U_pmf_target_contribution`` is the complete PMF
        term that actually enters ``U_target``.  The legacy topology-gated v2
        path remains available when ``pmf_gate_energy_scale_kj_mol`` is set;
        the additive v3 path leaves it unset.
        """

        x = jnp.asarray(standardized_ambient)
        physical = x * self.physical_std_nm
        raw_pmf = self.pmf.energy(physical)
        effective_pmf = (
            raw_pmf
            if self.pmf_lower_bound is None
            else self.pmf_lower_bound.energy(raw_pmf)
        )
        pmf_scale = jnp.asarray(self.pmf_scale, dtype=raw_pmf.dtype)
        scaled_pmf = pmf_scale * effective_pmf
        zero = jnp.zeros_like(raw_pmf)
        if self.canonical_support is None:
            support_components = {
                "U_bond": zero,
                "U_angle": zero,
                "U_repulsion": zero,
            }
        else:
            support_components = self.canonical_support.energy_components(physical)
        support = (
            support_components["U_bond"]
            + support_components["U_angle"]
            + support_components["U_repulsion"]
        )
        cb = zero if self.cb_improper is None else self.cb_improper.energy(physical)
        if self.pmf_gate_energy_scale_kj_mol is None:
            pmf_target_contribution = scaled_pmf
        else:
            gate_scale = jnp.asarray(
                self.pmf_gate_energy_scale_kj_mol,
                dtype=raw_pmf.dtype,
            )
            gate = jnp.exp(-support / gate_scale)
            assert isinstance(self.pmf_lower_bound, SmoothEnergyWindow)
            anchor = jnp.asarray(
                self.pmf_lower_bound.maximum_kj_mol,
                dtype=raw_pmf.dtype,
            )
            gated_pmf = anchor + gate * (effective_pmf - anchor)
            # Scaling is applied after the legacy v2 gate.  With the default
            # scale of one this is bit-for-bit the historical v2 target.
            pmf_target_contribution = pmf_scale * gated_pmf
        target = pmf_target_contribution + support + cb
        components = {
            "U_pmf": raw_pmf,
            "U_pmf_raw": raw_pmf,
            "U_pmf_effective": effective_pmf,
            "U_pmf_floor_delta": effective_pmf - raw_pmf,
            "U_pmf_transform_delta": effective_pmf - raw_pmf,
            "U_pmf_scaled": scaled_pmf,
            "U_pmf_target_contribution": pmf_target_contribution,
            **support_components,
            "U_support": support,
            "U_cb": cb,
            "U_target": target,
        }
        if self.pmf_gate_energy_scale_kj_mol is not None:
            components.update(
                {
                    "pmf_topology_gate": gate,
                    "U_pmf_gated": gated_pmf,
                }
            )
        return components

    def effective_pmf_energy(self, standardized_ambient: Array) -> Array:
        return self.formal_energy_components(standardized_ambient)["U_pmf_effective"]

    def _energy_grad_components(
        self,
        standardized_ambient: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        x = jnp.asarray(standardized_ambient)
        physical = x * self.physical_std_nm
        raw_pmf, raw_pmf_gradient = self.pmf.energy_and_grad(physical)
        if self.pmf_lower_bound is None:
            effective_pmf = raw_pmf
            pmf_gradient = raw_pmf_gradient
        else:
            effective_pmf = self.pmf_lower_bound.energy(raw_pmf)
            scale = self.pmf_lower_bound.gradient_scale(raw_pmf)
            pmf_gradient = raw_pmf_gradient * scale[..., None, None]
        pmf_scale = jnp.asarray(self.pmf_scale, dtype=raw_pmf.dtype)
        scaled_pmf = pmf_scale * effective_pmf
        scaled_pmf_gradient = pmf_scale * pmf_gradient

        zero = jnp.zeros_like(raw_pmf)
        if self.canonical_support is None:
            support = zero
            support_gradient = jnp.zeros_like(physical)
            support_components = {
                "U_bond": zero,
                "U_angle": zero,
                "U_repulsion": zero,
            }
        else:
            support, support_gradient, support_components = (
                self.canonical_support.energy_and_grad_components(physical)
            )

        if self.pmf_gate_energy_scale_kj_mol is None:
            pmf_target_contribution = scaled_pmf
            pmf_target_gradient = scaled_pmf_gradient
            support_gradient_factor = jnp.ones_like(raw_pmf)
        else:
            gate_scale = jnp.asarray(
                self.pmf_gate_energy_scale_kj_mol,
                dtype=raw_pmf.dtype,
            )
            gate = jnp.exp(-support / gate_scale)
            assert isinstance(self.pmf_lower_bound, SmoothEnergyWindow)
            anchor = jnp.asarray(
                self.pmf_lower_bound.maximum_kj_mol,
                dtype=raw_pmf.dtype,
            )
            delta = effective_pmf - anchor
            gated_pmf = anchor + gate * delta
            gated_pmf_gradient = pmf_gradient * gate[..., None, None]
            pmf_target_contribution = pmf_scale * gated_pmf
            pmf_target_gradient = pmf_scale * gated_pmf_gradient
            # delta <= 0 by construction, so the gate derivative can only
            # strengthen (never reverse) the topology-restoring support force.
            support_gradient_factor = 1.0 - pmf_scale * gate * delta / gate_scale

        if self.cb_improper is None:
            cb = zero
            cb_gradient = jnp.zeros_like(physical)
        else:
            cb, cb_gradient = self.cb_improper.energy_and_grad(physical)

        target = pmf_target_contribution + support + cb
        gradient_physical = (
            pmf_target_gradient
            + support_gradient * support_gradient_factor[..., None, None]
            + cb_gradient
        )
        components = {
            "U_pmf": raw_pmf,
            "U_pmf_raw": raw_pmf,
            "U_pmf_effective": effective_pmf,
            "U_pmf_floor_delta": effective_pmf - raw_pmf,
            "U_pmf_transform_delta": effective_pmf - raw_pmf,
            "U_pmf_scaled": scaled_pmf,
            "U_pmf_target_contribution": pmf_target_contribution,
            **support_components,
            "U_support": support,
            "U_cb": cb,
            "U_target": target,
        }
        if self.pmf_gate_energy_scale_kj_mol is not None:
            components.update(
                {
                    "pmf_topology_gate": gate,
                    "U_pmf_gated": gated_pmf,
                }
            )
        return target, gradient_physical * self.physical_std_nm, components

    def energy(self, standardized_ambient: Array) -> Array:
        return self.formal_energy_components(standardized_ambient)["U_target"]

    def energy_and_grad(self, standardized_ambient: Array) -> tuple[Array, Array]:
        energy, gradient, _components = self._energy_grad_components(
            standardized_ambient
        )
        return energy, gradient

    def evaluate(
        self,
        standardized_ambient: Array,
        *,
        include_training_wall: bool = True,
    ) -> PotentialResult:
        x = jnp.asarray(standardized_ambient)
        energy, grad_dimensional_x, components = self._energy_grad_components(x)
        target_reduced_energy = energy / self.kT
        target_reduced_grad = grad_dimensional_x / self.kT
        reduced_gradient = lift_shape_gradient(
            target_reduced_grad,
            x,
            include_com_aux=True,
            com_std=self.com_std,
        )
        reduced_energy = target_reduced_energy + com_aux_reduced_energy(x, com_std=self.com_std)
        if include_training_wall and self.training_wall is not None:
            physical = x * self.physical_std_nm
            wall_energy = self.training_wall.reduced_energy(physical)
            wall_gradient_x = self.training_wall.reduced_gradient(physical) * self.physical_std_nm
            reduced_energy = reduced_energy + wall_energy
            reduced_gradient = reduced_gradient + wall_gradient_x
        dimensional_gradient = reduced_gradient * self.kT
        valid = jnp.isfinite(reduced_energy) & jnp.all(jnp.isfinite(reduced_gradient), axis=(-2, -1))
        return PotentialResult(
            energy=energy,
            gradient=dimensional_gradient,
            reduced_energy=reduced_energy,
            reduced_gradient=reduced_gradient,
            score=-reduced_gradient,
            valid_mask=valid,
            components=components,
        )

    def formal_reweight_energy(self, standardized_ambient: Array) -> Array:
        """Return the configured physical target, excluding COM and wall terms.

        This is the complete configured physical target.  It intentionally
        excludes only the auxiliary COM density and the training wall.
        """

        return self.energy(standardized_ambient)


def build_cgbg_mace_energy_fn(
    bundle: CGBGAla2Bundle,
    *,
    trusted_checkpoint: bool = False,
    allocation_samples: int | None = None,
) -> CGBGMaceBackend:
    """Restore the exact CG-BG MACE backend without importing Torch.

    ChemTrain/ChemUtils APIs are not stable across revisions.  All dependency
    and ABI errors are isolated here and converted to one actionable failure;
    the rest of the project can use an injected pure-JAX energy function in
    tests and alternative deployments.
    """

    try:
        from chemtrain.data import preprocessing
        from chemtrain.learn import force_matching
        from chemutils.models import mace
        from jax_md import partition, space
        from jax_md_mod import custom_quantity
    except Exception as error:  # pragma: no cover - exercised in production env
        raise RuntimeError(
            "The CG-BG Ala2 MACE backend requires the pinned pure-JAX "
            "ChemTrain and ChemUtils revisions. No Torch module is imported."
        ) from error

    bundle.require_assets()
    params = load_trusted_pickle(
        bundle.checkpoint_path,
        expected_sha256=bundle.checkpoint_sha256,
        trusted=trusted_checkpoint,
    )
    try:  # pragma: no cover - depends on the pinned external ABI
        with np.load(bundle.data_path, allow_pickle=False) as archive:
            raw_allocation_data = {
                "R": np.asarray(archive["R"]),
                "box": np.asarray(archive["box"]),
                "mask": np.asarray(archive["mask"]),
            }
        # Reproduce get_ala_pmf_dataset(..., seed=0) exactly.  In MACE,
        # ``avg_num_neighbors`` is part of the network normalization, so an
        # arbitrary prefix changes the published energy even if it has the
        # same maximum edge capacity.  Production therefore uses the complete
        # shuffled 90% training split.  A prefix is available only as an
        # explicit, approximate ABI diagnostic.
        training, _validation, _test = preprocessing.train_val_test_split(
            raw_allocation_data,
            train_ratio=0.9,
            val_ratio=0.1,
            shuffle=True,
            shuffle_seed=0,
        )
        positions_nm = np.asarray(training["R"])
        boxes = np.asarray(training["box"])
        masks = np.asarray(training["mask"])
        if allocation_samples is not None:
            if int(allocation_samples) <= 0:
                raise ValueError("allocation_samples must be positive or None")
            count = min(int(allocation_samples), positions_nm.shape[0])
            positions_nm = positions_nm[:count]
            boxes = boxes[:count]
            masks = masks[:count]
        box = jnp.asarray(bundle.box_nm)
        lengths = jnp.diag(box)
        dataset = {
            "R": jnp.asarray(positions_nm) / lengths,
            "box": jnp.asarray(boxes),
            "mask": jnp.asarray(masks),
        }
        species = jnp.asarray(bundle.species)
        mask = jnp.asarray(bundle.mask)
        displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)
        cfg = dict(bundle.model_config)
        # The pinned ChemTrain API returns an *allocated NeighborList* (with an
        # ``update`` method), not the ``NeighborListFns`` factory returned by
        # raw jax-md.  Keeping the distinction explicit is important: calling
        # ``allocate`` here would only work against a different ABI.
        neighbor_template, stats = preprocessing.allocate_neighborlist(
            dataset,
            displacement_fn,
            box,
            r_cutoff=float(cfg["r_cutoff"]),
            mask_key="mask",
            box_key="box",
            format=partition.Sparse,
            batch_size=int(cfg.get("batch_size", 256)),
        )
        init_fn, gnn_energy_fn = mace.mace_neighborlist_pp(
            displacement_fn,
            r_cutoff=float(cfg["r_cutoff"]),
            n_species=len(bundle.species),
            max_edges=stats[1],
            per_particle=False,
            avg_num_neighbors=stats[2],
            mode="energy",
            hidden_irreps=cfg["hidden_irreps"],
            max_ell=int(cfg["max_ell"]),
            num_interactions=int(cfg["num_interactions"]),
            correlation=int(cfg["correlation"]),
            readout_mlp_irreps=cfg["readout_irreps"],
            output_irreps=cfg["output_irreps"],
        )
        # Match get_ala_trainer's initialization sequence before replacing the
        # randomly initialized tree with the published checkpoint.  Besides
        # validating the parameter ABI, this fixes any library-side static
        # state established during Haiku/e3nn graph initialization.
        _ = init_fn(
            jax.random.PRNGKey(0),
            dataset["R"][0],
            neighbor_template,
            species=species,
            mask=mask,
        )

        def energy_fn_template(energy_params):
            def energy_fn(position, neighbor, mode=None, **dynamic_kwargs):
                del mode
                if "species" not in dynamic_kwargs:
                    raise KeyError("CG-BG MACE requires species")
                dynamic_kwargs.setdefault(
                    "mask", jnp.ones(position.shape[0], dtype=jnp.bool_)
                )
                return gnn_energy_fn(
                    energy_params,
                    position,
                    neighbor,
                    **dynamic_kwargs,
                )

            return energy_fn

        # Reuse ChemTrain's pure-JAX model composition, but not its trainer or
        # Torch DataLoader.  This preserves the published
        # lax.map(vmap(value_and_grad)) numerical path; a seemingly equivalent
        # direct vmap changes float32 MACE scatter reductions by O(1 kJ/mol).
        feature_fns = {
            "energy_and_force": custom_quantity.energy_force_wrapper(
                energy_fn_template
            )
        }
        quantities = {
            "F": custom_quantity.force_wrapper(None),
            "U": custom_quantity.energy_wrapper(None),
        }
        batched_model = force_matching.init_model(
            neighbor_template,
            quantities,
            feature_extract_fns=feature_fns,
        )

        # Match ChemTrain's single-device shmap_model boundary: the complete
        # observations PyTree and checkpoint parameters cross the JIT boundary
        # as dynamic operands.  Constructing species/box/mask inside the jitted
        # function lets XLA constant-fold them and changes MACE scatter sums.
        predict_model = jax.jit(batched_model)

        # The pinned ChemTrain/MACE GPU lowering produces a non-finite result
        # for lane zero of every inference call, while the same coordinate is
        # finite and agrees with CG-BG when evaluated in lane one.  MACE's
        # float32 scatter reductions are also batch-shape sensitive, so simply
        # prepending a dummy (and changing B to B+1) breaks oracle parity.
        # Keep the real batch size for B>=2 and use two calls instead:
        #
        #   regular = [dummy, real_1, ..., real_B-1]
        #   first   = [dummy, real_0, dummy, ..., dummy]
        #
        # The regular call preserves every published finite lane exactly; the
        # first call recovers real_0 from lane one.  B=1 necessarily uses the
        # minimal two-lane [dummy, real_0] call.  Real invalid inputs are
        # explicitly propagated after inference and cannot be hidden by the
        # sacrificial coordinate.
        sacrificial_fractional = (
            jnp.asarray(bundle.reference_nm, dtype=dataset["R"].dtype) / lengths
        )

        def run_model(batch: Array) -> tuple[Array, Array]:
            batch_count = batch.shape[0]
            observations = {
                "R": batch,
                "U": jnp.zeros((batch_count,), dtype=batch.dtype),
                "F": jnp.zeros_like(batch),
                "box": jnp.broadcast_to(box, (batch_count, 3, 3)),
                "species": jnp.broadcast_to(species, (batch_count, 6)),
                "mask": jnp.broadcast_to(mask, (batch_count, 6)),
            }
            predictions = predict_model(params, observations)
            return (
                jnp.asarray(predictions["U"]).reshape((batch_count,)),
                jnp.asarray(predictions["F"]).reshape(batch.shape),
            )

        def predict_fractional(fractional: Array) -> tuple[Array, Array]:
            fractional = jnp.asarray(fractional)
            if fractional.ndim != 3 or fractional.shape[1:] != (6, 3):
                raise ValueError(
                    f"Expected fractional coordinates (B,6,3), got {fractional.shape}"
                )
            count = fractional.shape[0]
            if count == 0:
                return (
                    jnp.empty((0,), dtype=fractional.dtype),
                    jnp.empty_like(fractional),
                )
            if count == 1:
                recovery = jnp.stack(
                    (sacrificial_fractional, fractional[0]), axis=0
                )
                recovery_energy, recovery_force = run_model(recovery)
                energy = recovery_energy[1:]
                force = recovery_force[1:]
            else:
                regular = fractional.at[0].set(sacrificial_fractional)
                regular_energy, regular_force = run_model(regular)
                recovery = jnp.broadcast_to(
                    sacrificial_fractional, fractional.shape
                ).at[1].set(fractional[0])
                recovery_energy, recovery_force = run_model(recovery)
                energy = jnp.concatenate(
                    (recovery_energy[1:2], regular_energy[1:]), axis=0
                )
                force = jnp.concatenate(
                    (recovery_force[1:2], regular_force[1:]), axis=0
                )
            input_valid = jnp.all(jnp.isfinite(fractional), axis=(-2, -1))
            nan = jnp.asarray(jnp.nan, dtype=fractional.dtype)
            return (
                jnp.where(input_valid, energy, nan),
                jnp.where(input_valid[:, None, None], force, nan),
            )

        backend = CGBGMaceBackend(
            predict_fractional=predict_fractional,
            box_lengths_nm=lengths,
        )
        # Compile once so an incompatible checkpoint/tree fails before a long
        # BMS run starts.  The reference path intentionally includes forces.
        _ = backend.evaluate_physical(jnp.asarray(bundle.reference_nm)[None, ...])
        return backend
    except Exception as error:
        raise RuntimeError(
            "Failed to reconstruct the pinned CG-BG MACE ABI. Verify CG-BG "
            f"{bundle.cgbg_revision}, ChemTrain {bundle.chemtrain_revision}, "
            "ChemUtils, data metadata, and checkpoint digest."
        ) from error
