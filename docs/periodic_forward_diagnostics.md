# Periodic forward checkpoints and BMS-style replay diagnostics

The public `BridgeMatchingSampler` trainer performs the following sequence
after every configured epoch/outer boundary:

1. populate the replay buffer with fresh forward-SDE endpoints;
2. run all optimizer updates for the outer loop;
3. increment the completed-outer counter;
4. save a checkpoint when the counter is divisible by 50;
5. synchronously evaluate replay-buffer `endpoint_1` samples and log figures.

The train-time visualization is not a fresh rollout from the just-saved
checkpoint. With the Ala2 defaults it contains at most 2,048 endpoints, i.e.
roughly the most recent eight outer populations of 256 samples each.

`cg-bms-jax` reproduces that cadence with the root configuration fields:

```yaml
save_checkpoint_interval_outer: 50
save_vis_interval_outer: 50
save_vis_max_samples: 2048
save_vis_target_batch_size: 64
save_vis_n_bootstraps: 1
save_vis_seed: 0
save_vis_include_implicit_reference: false
history_every: 100
```

For the 700,000-update cold-start experiment:

```bash
cg-bms-train-forward \
  experiment=ala2_ambient18_300k_bms_eta10_official700k_additive_v3 \
  save_checkpoint_interval_outer=50 \
  save_vis_interval_outer=50 \
  history_every=100 \
  output_dir=outputs/ala2_bms_official700k_cold_s0/forward
```

With 100 updates per outer, outer 50 maps to global step 5,000. Intermediate
checkpoints are therefore named `forward_step_00005000`,
`forward_step_00010000`, and so on. The final outer is saved exactly once by
the normal final-result path.

Each visualization boundary writes:

```text
training_replay_eval/
└── outer_000050_step_00005000/
    ├── replay_endpoints.npz
    ├── replay_evaluation_summary.json
    └── formal/
        ├── ala2_cb_energy_distribution.png
        ├── ala2_cb_density.png
        ├── ala2_cb_free_energy.png
        ├── ala2_cb_ramachandran_fes.png
        ├── ala2_cb_metrics.json
        └── ala2_cb_metrics.txt
```

The archive is explicitly marked with
`sampler_kind=training_replay_endpoint` and
`fresh_checkpoint_proposal=false`. It must not be used as a formal independent
proposal or for importance reweighting. Generate a fixed-seed forward-SDE
archive with `cg-bms-sample-sde` when an independent checkpoint comparison is
required.

Frequent diagnostics intentionally skip the much larger implicit reference by
default. Set `save_vis_include_implicit_reference=true` only when the extra
comparison is required and sufficient host memory is available.
