from __future__ import annotations

from pathlib import Path

import torch

from chimera.trials.venture import run_venture_trial


def test_small_venture_trial_writes_safe_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "trial.yaml"
    config.write_text(
        """trial_id: CHM-V-T999
hypothesis_id: CHM-V-H001
model:
  hidden_dim: 32
  num_heads: 4
  encoder_layers: 1
  decoder_layers: 1
  transition_layers: 1
  feedforward_multiplier: 2
  max_nodes: 64
  max_edits: 8
  dropout: 0.0
training:
  seed: 7
  batch_size: 2
  steps: 1
  learning_rate: 0.001
  weight_decay: 0.0
  max_grad_norm: 1.0
  target_ema_decay: 0.9
  argument_loss_mode: operation_conditioned
  learning_rate_schedule: cosine
  warmup_steps: 0
  minimum_learning_rate: 0.0001
  device: cpu
evaluation:
  corpus_manifest: datasets/venture_corpus_c0/manifest.json
  eval_interval: 1
  evaluation_batch_size: 128
  candidates_per_case: 1
  generation_temperature: 0.0
  generation_seed: 11
  min_edits: 1
  max_edits: 1
  archive_bins: [2, 2]
  checkpoint_selection: validation_exact_graph
  memorization_exact_graph_min: 0.0
  invalid_candidate_rate_max: 0.0
""",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    checkpoints = tmp_path / "checkpoints"
    result = run_venture_trial(config, output, checkpoint_dir=checkpoints)
    checkpoint = checkpoints / result["checkpoint"]["file"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["trial_id"] == "CHM-V-T999"
    assert checkpoint.name.startswith("chimera-venture-m0-t999-")
    assert result["metrics"]["generation"]["invalid_candidate_rate"] == 0.0
    assert (output / "result.json").is_file()
    assert (output / "candidates.jsonl").is_file()
