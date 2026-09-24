# Artifact publication plan

This project separates source code, reusable ML artifacts and immutable research
records. Large files are not omitted from the research record; they are placed in
repositories designed for their type and linked by version and checksum.

## Publication map

| Artifact | Destination | Policy |
| --- | --- | --- |
| Training/evaluation code, configs, manifests, report JSON, documentation and manuscript | GitHub | Version with the source. Keep commands, seeds, hardware, dependency versions and exact artifact links close to each result. |
| T4 Talker and MTP LoRA adapters (`adapter_config.json`, `adapter_model.safetensors`) | Hugging Face model repository | Publish the selected checkpoint plus the checkpoints needed to support the scaling/selection claims. Include a model card, base-model identifier and revision, dataset reference, metrics, limitations, license review and SHA-256 checksums. |
| Qwen3-Omni base weights, Code2Wav and downloaded tokenizer assets | Upstream model repository only | Do not duplicate them. Record stable upstream identifiers, revisions, expected file hashes and a download/setup command. |
| Romanian training audio | Upstream dataset repository, or a dedicated Hugging Face dataset repository only after release review | Prefer deterministic row IDs and preprocessing scripts over re-hosting a copied subset. Redistribute audio only after confirming the exact source revision, CC-BY-SA obligations, attribution, performer/voice rights and privacy. |
| Held-out prompts and evaluation specification | GitHub; Hugging Face dataset repository if broadly reusable | Publish stable IDs, categories, splits and checksums. Prevent train/evaluation leakage. Do not include third-party audio unless redistribution is permitted. |
| Generated evaluation audio | Hugging Face dataset repository | Publish a curated representative set first. For a full benchmark release, pair every audio file with prompt ID, checkpoint, inference settings, ASR transcript, metrics and checksum. Clearly label the audio as synthetic. |
| Complete report bundle and final research snapshot | Zenodo linked to a tagged GitHub release | Archive an immutable version and obtain a DOI. Include checksums and links to the corresponding Hugging Face model/dataset revisions rather than duplicating upstream base weights. |
| Virtual environment, caches, temporary logs and intermediate downloads | Local/cold storage only | Recreate the environment from a lock file; never publish `.venv`, caches or machine-specific paths. |

## Recommended Hugging Face layout

Use separate repositories so licenses, cards and access controls remain explicit:

```text
dragoshont/qwen3-omni-romanian-t4          # model repo: Talker + MTP adapters
dragoshont/qwen3-omni-romanian-eval        # dataset repo: prompts, metadata, generated audio
dragoshont/qwen3-omni-romanian-data        # optional dataset repo, only after rights review
```

The model repository should contain the final selected checkpoint and any
intermediate checkpoints cited in the paper. Do not label `final/` as the best
model unless the completed held-out evaluation supports that selection.

## Platform decision

**Primary model host: Hugging Face Hub.** It is the best fit for these PEFT
artifacts because the release already uses the conventional
`adapter_config.json` + `adapter_model.safetensors` structure. The Hub supports
PEFT adapters, versioned model repositories, model cards, structured metrics and
custom model code. Publish the Talker and MTP adapters together with an explicit
loader because this project attaches two adapters to different Qwen3-Omni
components.

**Unsloth: compatibility target, not the canonical artifact host.** Unsloth's
documented publishing flow ultimately uploads models to Hugging Face. Its public
catalog documents Qwen3 and Qwen2.5-Omni support, but not this custom Qwen3-Omni
Talker+MTP dual-adapter path; an open feature request exists for Qwen3-Omni TTS
support. Do not claim Unsloth compatibility until a clean-machine load and
inference test passes. If it does, publish a small example notebook and request
catalog/community inclusion while keeping Hugging Face as the source of truth.

**Zenodo: archival record, not day-to-day model distribution.** Use it for the
tagged final research release and DOI. Reference immutable Hugging Face revisions
from the record.

**ModelScope: optional secondary mirror.** It may improve discoverability in the
Qwen ecosystem, but should mirror a versioned Hugging Face release rather than
become an independently drifting copy.

**GitHub Releases: optional convenience bundle.** Small, final adapter bundles
can be attached to a tagged release, but GitHub should not be the only model
host. Avoid committing weights to ordinary Git history.

## Minimum record for every released checkpoint

- artifact name and semantic version;
- Git commit that produced it;
- base-model repository and immutable revision;
- training dataset repository/revision and deterministic selection manifest;
- preprocessing and target-code construction version;
- full hyperparameters, random seed, hardware and software environment;
- Talker and MTP adapter checksums;
- exact evaluation command, evaluation-set revision and raw report;
- intended use, limitations, known failure modes and voice/synthetic-audio notice;
- license and attribution for code, base model, training data and released weights.

## Release gates

1. Complete T4 and preserve the original reports.
2. Repeat Full-200 with explicit recorded seeds and complete blinded native-listener evaluation.
3. Freeze the checkpoint using the declared selection rule, not a preferred anecdotal sample.
4. Run a secret, path, PII and data-leakage scan.
5. Verify the source dataset revision and all redistribution/voice obligations.
6. Generate SHA-256 manifests for every released binary and dataset shard.
7. Publish GitHub code/results and Hugging Face adapters/evaluation artifacts.
8. Tag the matching GitHub commit and archive that release on Zenodo for a DOI.

## Community conventions behind this split

- GitHub enforces a 100 MB per-file Git limit and recommends Git LFS for large
  binaries, while its LFS storage and bandwidth are quota-metered.
- Hugging Face repositories are optimized for model and dataset binaries and
  provide Model Cards, Dataset Cards, structured evaluation metadata and large-file
  storage.
- Zenodo can archive tagged GitHub software releases and issue persistent DOIs.
- ML reproducibility guidance consistently emphasizes exact commands, data and
  preprocessing descriptions, hyperparameters, run counts and raw results.

References:

- https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits
- https://docs.github.com/en/billing/concepts/product-billing/git-lfs
- https://huggingface.co/docs/hub/repositories
- https://huggingface.co/docs/hub/model-cards
- https://huggingface.co/docs/hub/datasets-cards
- https://huggingface.co/docs/hub/storage-limits
- https://help.zenodo.org/docs/github/
- https://arxiv.org/abs/2003.12206
