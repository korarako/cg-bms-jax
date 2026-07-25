from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cg_bms_jax.data import GaussianSource
from cg_bms_jax.experiment import pretrain_bridge, sample_reweight, sample_sde
from cg_bms_jax.runtime import build_runtime_system, compose_config


def test_hydra_composes_smoke_experiment_for_every_entrypoint():
    overrides = {
        "pretrain_bridge": [
            "experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"
        ],
        "train_forward": ["experiment=mb_cg1d_smoke"],
        "train_backward": ["experiment=mb_cg1d_smoke", "forward_checkpoint=/tmp/fwd"],
        "sample_sde": ["experiment=mb_cg1d_smoke", "forward_checkpoint=/tmp/fwd"],
        "sample_reweight": [
            "experiment=mb_cg1d_smoke",
            "forward_checkpoint=/tmp/fwd",
            "backward_checkpoint=/tmp/bwd",
        ],
        "evaluate": ["experiment=mb_cg1d_smoke", "samples=/tmp/samples.npz"],
    }
    for name, values in overrides.items():
        config = compose_config(name, values)
        if name == "pretrain_bridge":
            assert config.experiment.name.endswith("canonical_additive_v3")
            assert list(config.experiment.state_shape) == [6, 3]
            assert config.pretrain.updates == 100000
            assert config.pretrain.batch_size == 64
            assert config.data.random_rotation is True
            assert config.data.add_com_noise is True
        else:
            assert config.experiment.name == "mb_cg1d_smoke"
            assert list(config.experiment.state_shape) == [1]
            assert config.experiment.training.outer_iterations == 2
        if name in {"train_backward", "sample_reweight"}:
            assert config.forward_controller_kind == "forward"


def test_hydra_composes_all_atom_three_arm_smoke_matrix() -> None:
    entries = (
        (
            "train_forward",
            "ala2_allatom_bms_pure_energy_smoke",
            "pure_energy",
        ),
        (
            "pretrain_bridge",
            "ala2_allatom_bms_bridge_only_smoke",
            "bridge_only",
        ),
        (
            "train_forward",
            "ala2_allatom_bms_bridge_then_energy_smoke",
            "bridge_then_energy",
        ),
    )
    for entrypoint, experiment, arm in entries:
        config = compose_config(entrypoint, [f"experiment={experiment}"])
        assert config.experiment.coordinate_mode == "ala2_aa_ambient66"
        assert list(config.experiment.state_shape) == [22, 3]
        assert config.experiment.dimension == 66
        assert config.experiment.workflow.arm == arm
        assert config.experiment.target.mode == "openmm_bms_chirality"
        assert config.experiment.target.openmm.forcefield == "amber99sbildn.xml"
        assert (
            config.experiment.target.openmm.implicit_forcefield
            == "implicit/obc1.xml"
        )


def test_hydra_composes_single_arm_all_atom_bridge_formal() -> None:
    config = compose_config(
        "pretrain_bridge",
        ["experiment=ala2_allatom_bms_bridge_formal"],
    )
    experiment = config.experiment
    assert experiment.coordinate_mode == "ala2_aa_ambient66"
    assert list(experiment.state_shape) == [22, 3]
    assert experiment.model.num_features == 128
    assert experiment.model.num_radial_basis == 64
    assert experiment.model.num_layers == 5
    assert experiment.sde.steps == 50
    assert experiment.training.batch_size == 32
    assert experiment.training.rollout_samples == 2048
    assert experiment.training.buffer_capacity == 2048
    assert experiment.formal_bridge.updates == 100000
    assert experiment.formal_bridge.backward_updates == 100000
    assert experiment.formal_bridge.pf_formal_samples == 20000
    assert experiment.formal_bridge.formal_weight_clip is None
    assert config.pretrain.updates == 100000
    assert config.pretrain.batch_size == 32
    assert config.checkpoint_every == 10000
    evaluation = compose_config(
        "evaluate",
        [
            "experiment=ala2_allatom_bms_bridge_formal",
            "samples=/tmp/proposal.npz",
            "reference_split=validation",
        ],
    )
    assert evaluation.reference_split == "validation"


def test_hydra_composes_single_arm_all_atom_energy_formal() -> None:
    config = compose_config(
        "train_forward",
        ["experiment=ala2_allatom_bms_energy_formal"],
    )
    experiment = config.experiment
    assert experiment.name == "ala2_allatom_bms_energy_formal"
    assert experiment.coordinate_mode == "ala2_aa_ambient66"
    assert list(experiment.state_shape) == [22, 3]
    assert experiment.dimension == 66
    assert experiment.target.mode == "openmm_bms_chirality"
    assert experiment.model.num_features == 128
    assert experiment.model.num_radial_basis == 64
    assert experiment.model.num_layers == 5
    assert experiment.sde.steps == 50
    assert experiment.training.outer_iterations == 1000
    assert experiment.training.rollout_samples == 256
    assert experiment.training.buffer_capacity == 2048
    assert experiment.training.batch_size == 32
    assert experiment.training.gradient_steps == 100
    assert experiment.training.previous_model_interval_outer == 25
    assert experiment.training.damping == 0.0
    assert experiment.training.terminal_score_clip_norm == 100.0
    assert experiment.formal_energy.updates == 100000
    assert experiment.workflow.arm == "pure_energy_formal"


def test_hydra_composes_stabilized_all_atom_energy_formal() -> None:
    config = compose_config(
        "train_forward",
        ["experiment=ala2_allatom_bms_energy_formal_d10"],
    )
    experiment = config.experiment
    assert experiment.name == "ala2_allatom_bms_energy_formal_d10"
    assert experiment.workflow.arm == "pure_energy_formal_d10"
    assert experiment.training.damping == 10.0
    assert experiment.training.outer_iterations == 1000
    assert experiment.training.rollout_samples == 256
    assert experiment.training.buffer_capacity == 2048
    assert experiment.training.batch_size == 32
    assert experiment.training.gradient_steps == 100
    assert experiment.training.previous_model_interval_outer == 25
    assert experiment.model.num_features == 128
    assert experiment.model.num_layers == 5


def test_mb2d_hydra_entries_compose_without_external_assets():
    bridge = compose_config(
        "pretrain_flat_bridge",
        [
            "experiment=mb2d_analytic_smoke",
            "pretrain.updates=8",
            "pretrain.batch_size=32",
            "checkpoint_every=null",
        ],
    )
    validation = compose_config(
        "validate_pf",
        [
            "experiment=mb2d_analytic_smoke",
            "forward_checkpoint=/tmp/fwd",
            "backward_checkpoint=/tmp/bwd",
        ],
    )
    backward = compose_config(
        "train_backward",
        [
            "experiment=mb2d_analytic_smoke",
            "forward_checkpoint=/tmp/fwd",
            "updates=17",
        ],
    )

    assert bridge.experiment.coordinate_mode == "mb2d_affine"
    assert list(bridge.experiment.state_shape) == [2]
    assert bridge.pretrain.updates == 8
    assert bridge.checkpoint_every is None
    assert validation.num_samples == 8
    assert validation.likelihood.solver == "dopri5"
    assert backward.updates == 17


def test_warm_backward_and_pf_require_an_explicit_controller_kind_override():
    backward = compose_config(
        "train_backward",
        [
            "experiment=mb_cg1d_smoke",
            "forward_checkpoint=/tmp/warm",
            "forward_controller_kind=forward_pretrain",
        ],
    )
    likelihood = compose_config(
        "sample_reweight",
        [
            "experiment=mb_cg1d_smoke",
            "forward_checkpoint=/tmp/warm",
            "backward_checkpoint=/tmp/bwd",
            "forward_controller_kind=forward_pretrain",
        ],
    )
    assert backward.forward_controller_kind == "forward_pretrain"
    assert likelihood.forward_controller_kind == "forward_pretrain"


def test_mb_runtime_factory_initializes_a_pure_jax_controller_without_assets():
    config = compose_config("train_forward", ["experiment=mb_cg1d_smoke"])
    system = build_runtime_system(config, key=jax.random.PRNGKey(4), load_potential=False)
    state = jnp.zeros((3, 1), dtype=jnp.float32)
    value = system.apply(system.initial_variables, 0.3, state)
    assert value.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(value)))
    assert system.potential is None
    assert system.identity["density_mode"] == "ambient_exact"


def test_hydra_composes_ala2_128x5_medium_experiment():
    config = compose_config(
        "train_forward",
        ["experiment=ala2_ambient18_300k_128x5_medium"],
    )
    assert config.experiment.name == "ala2_ambient18_300k_128x5_medium"
    assert list(config.experiment.state_shape) == [6, 3]
    assert config.experiment.dimension == 18
    assert config.experiment.model.num_features == 128
    assert config.experiment.model.num_layers == 5
    assert config.experiment.model.num_radial_basis == 64
    assert config.experiment.sde.steps == 100
    assert config.experiment.training.outer_iterations == 20
    assert config.experiment.training.gradient_steps == 500
    assert config.experiment.training.batch_size == 64


def test_sample_reweight_progress_config_and_message():
    config = compose_config(
        "sample_reweight",
        [
            "experiment=mb_cg1d_smoke",
            "forward_checkpoint=/tmp/fwd",
            "backward_checkpoint=/tmp/bwd",
            "progress_every_batches=7",
        ],
    )
    assert config.progress_every_batches == 7
    assert config.target_batch_size is None
    assert config.reuse_pf_stage is True
    message = sample_reweight._progress_message(
        batch_number=2,
        total_batches=4,
        processed_samples=512,
        total_samples=1024,
        ode_steps=37,
        elapsed_seconds=2.0,
    )
    assert message == (
        "[pf-ode] batch=2/4 samples=512/1024 ( 50.00%) "
        "ode_steps=37 elapsed=2.0s rate=256.0 samples/s eta=2.0s"
    )


class _ChunkTrackingTargetSystem:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def evaluate_target(self, state, *, include_training_wall):
        assert not include_training_wall
        self.batch_sizes.append(int(state.shape[0]))
        energy = jnp.sum(state**2, axis=1)
        return SimpleNamespace(
            energy=energy,
            reduced_energy=energy / 2.0,
            valid_mask=jnp.all(jnp.isfinite(state), axis=1),
            components={"U_test": energy},
        )

    def support_mask(self, state):
        return jnp.all(jnp.isfinite(state), axis=1)

    def to_physical(self, state):
        return state * 3.0


def test_target_evaluation_is_chunked_and_concatenated():
    system = _ChunkTrackingTargetSystem()
    state = np.arange(14, dtype=np.float32).reshape(7, 2)
    result = sample_reweight._evaluate_target_in_batches(
        system,
        state,
        batch_size=3,
        progress_every_batches=2,
    )

    assert system.batch_sizes == [3, 3, 1]
    expected_energy = np.sum(state**2, axis=1)
    np.testing.assert_allclose(result.energy, expected_energy)
    np.testing.assert_allclose(result.reduced_energy, expected_energy / 2.0)
    np.testing.assert_allclose(result.physical_coordinates, state * 3.0)
    np.testing.assert_allclose(result.components["U_test"], expected_energy)
    assert result.valid_mask.all()
    assert result.support_mask.all()


def test_pf_stage_round_trip_and_metadata_guard(tmp_path):
    flow_config = sample_reweight.ProbabilityFlowConfig(
        dt0=0.005,
        rtol=1.0e-5,
        atol=1.0e-5,
        max_steps=128,
    )
    metadata = sample_reweight._pf_stage_metadata(
        forward_sha256="1" * 64,
        backward_sha256="2" * 64,
        seed=3,
        num_samples=4,
        batch_size=2,
        flow_config=flow_config,
    )
    state = np.arange(8, dtype=np.float32).reshape(4, 2)
    logq = np.linspace(-3.0, -1.0, 4, dtype=np.float32)
    initial_state = state + 1.0
    initial_logq = logq - 1.0
    step_counts = np.asarray([17, 18], dtype=np.int64)
    path = tmp_path / "proposal.pf_stage.npz"
    sample_reweight._save_pf_stage(
        path,
        state=state,
        logq=logq,
        initial_state=initial_state,
        initial_logq=initial_logq,
        step_counts=step_counts,
        metadata=metadata,
    )

    loaded = sample_reweight._load_pf_stage(
        path,
        expected_metadata=metadata,
        event_shape=(2,),
    )
    for actual, expected in zip(
        loaded,
        (state, logq, initial_state, initial_logq, step_counts),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    invalid_metadata = {**metadata, "seed": 4}
    with np.testing.assert_raises_regex(ValueError, "metadata"):
        sample_reweight._load_pf_stage(
            path,
            expected_metadata=invalid_metadata,
            event_shape=(2,),
        )


class _FakeSystem:
    def __init__(self):
        from cg_bms_jax.process import EDMSDE

        self.source = GaussianSource((1,), mean=0.0, scale=1.0)
        self.sde = EDMSDE(sigma_min=0.01, sigma_max=1.0, rho=7.0)
        self.identity = {"experiment_family": "synthetic"}
        self.transform = None

    def apply(self, variables, time, state):
        del variables, time
        return jnp.zeros_like(state)

    def evaluate_target(self, state, *, include_training_wall):
        assert not include_training_wall
        return SimpleNamespace(
            energy=jnp.sum(state**2, axis=1),
            valid_mask=jnp.all(jnp.isfinite(state), axis=1),
        )

    def support_mask(self, state):
        return jnp.all(jnp.isfinite(state), axis=1)

    def to_physical(self, state):
        return state


def test_sde_sampler_archive_is_proposal_only_and_never_claims_logq(tmp_path, monkeypatch):
    fake_checkpoint = SimpleNamespace(
        variables={},
        digest="0" * 64,
        metadata=SimpleNamespace(to_dict=lambda: {"role": "forward"}),
    )
    monkeypatch.setattr(sample_sde, "build_runtime_system", lambda *args, **kwargs: _FakeSystem())
    monkeypatch.setattr(sample_sde, "load_controller_checkpoint", lambda *args, **kwargs: fake_checkpoint)
    output = tmp_path / "proposal.npz"
    config = OmegaConf.create(
        {
            "seed": 9,
            "forward_checkpoint": str(tmp_path / "forward"),
            "num_samples": 5,
            "batch_size": 4,
            "output": str(output),
            "experiment": {"sde": {"steps": 2}},
        }
    )
    result = sample_sde.run(config)
    assert result == output
    with np.load(output, allow_pickle=False) as archive:
        assert archive["R"].shape == (5, 1)
        assert archive["sampler_kind"].item() == "sde"
        assert not {"logp", "logq", "logq_ambient", "logw", "weights"}.intersection(archive.files)


def test_pretrain_sde_sampler_is_explicitly_diagnostic_and_has_no_logq(
    tmp_path, monkeypatch
):
    fake_checkpoint = SimpleNamespace(
        variables={},
        digest="7" * 64,
        metadata=SimpleNamespace(
            role="forward_pretrain",
            to_dict=lambda: {"role": "forward_pretrain"},
        ),
    )
    monkeypatch.setattr(
        sample_sde, "build_runtime_system", lambda *args, **kwargs: _FakeSystem()
    )
    monkeypatch.setattr(
        sample_sde,
        "load_controller_initializer_checkpoint",
        lambda *args, **kwargs: fake_checkpoint,
    )
    output = tmp_path / "warm_proposal.npz"
    config = OmegaConf.create(
        {
            "seed": 9,
            "forward_checkpoint": str(tmp_path / "forward_pretrain"),
            "controller_kind": "forward_pretrain",
            "num_samples": 5,
            "batch_size": 4,
            "progress_every_batches": 1,
            "output": str(output),
            "experiment": {"sde": {"steps": 2}},
        }
    )
    sample_sde.run(config)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["sampler_kind"].item() == "sde_pretrain"
        assert not {"logp", "logq", "logq_ambient", "logw", "weights"}.intersection(
            archive.files
        )


def test_experiment_entrypoint_sources_do_not_import_torch():
    root = Path(sample_sde.__file__).resolve().parent
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "import torch" not in source
        assert "from torch" not in source


def test_warm_endpoint_provider_is_key_deterministic_and_requires_18d_augmentations(
    tmp_path,
):
    from cg_bms_jax.data import Ala2WarmStartDataset

    coordinates = np.arange(12 * 6 * 3, dtype=np.float32).reshape(12, 6, 3)
    coordinates *= 1.0e-3
    dataset = Ala2WarmStartDataset(
        path=tmp_path / "flow_b" / "data.npz",
        sha256="1" * 64,
        coordinates_nm=coordinates,
        box_nm=np.eye(3) * 3.7,
        species=np.arange(6),
        mask=np.ones(6, dtype=bool),
        standardization_std_nm=float(
            (coordinates - coordinates.mean(axis=1, keepdims=True)).std()
        ),
    )
    indices = np.arange(10, dtype=np.int64)
    provider = pretrain_bridge.make_ala2_endpoint_provider(dataset, indices)
    first = provider(jax.random.PRNGKey(8), 7)
    second = provider(jax.random.PRNGKey(8), 7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (7, 6, 3)
    with np.testing.assert_raises_regex(ValueError, "SO\\(3\\)"):
        pretrain_bridge.make_ala2_endpoint_provider(
            dataset, indices, random_rotation=False
        )
    with np.testing.assert_raises_regex(ValueError, "COM noise"):
        pretrain_bridge.make_ala2_endpoint_provider(
            dataset, indices, add_com_noise=False
        )


def test_console_main_does_not_return_path_as_a_nonzero_exit_code(tmp_path, monkeypatch, capsys):
    result = tmp_path / "proposal.npz"
    monkeypatch.setattr(sample_sde, "run_hydra_entry", lambda *args, **kwargs: result)
    assert sample_sde.main() is None
    assert str(result) in capsys.readouterr().out
