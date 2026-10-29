# The whole Kaggle submission notebook — one cell. Kaggle → Create → New Notebook, then:
#   + Add Input → Your Datasets → gemma4-submission (made by `UPLOAD=1 … make_submission.py`), Internet OFF.
#   Paste this cell → Save Version → "Save & Run All (Commit)" (not Quick Save) → wait for Completed →
#   open that version → Output shows submission.zip → Submit to Competition from that version.
# New dataset version later: in the notebook, Input panel → ⋮ → Check for updates / pick the latest version, then commit again.
import hashlib, json, pathlib, shutil, zipfile

ADAPTER = None      # LoRA entry: scripts/kaggle_sub_kernel.py --adapter <name> sets this (2026-10-29); None = the no-LoRA trap check
ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
found = [p for p in pathlib.Path("/kaggle/input").rglob("*") if "gemma4-submission" in str(p)]
zips = [p for p in found if p.name == "submission.zip"]
out = pathlib.Path("/kaggle/working/submission.zip")
if zips:                                          # the dataset kept our zip as a file
    shutil.copy(zips[0], out)
else:                                             # Kaggle unpacked it (seen 2026-10-16): re-zip from the agent.yaml dir,
    root = next(p.parent for p in found if p.name == "agent.yaml")   # same fixed timestamps as make_submission.py
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:       # → the same sha256 as its card.txt
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "dataset-metadata.json":
                zi = zipfile.ZipInfo(p.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
                z.writestr(zi, p.read_bytes())
names = zipfile.ZipFile(out).namelist()
assert "agent.yaml" in names, f"agent.yaml not at the zip root: {names[:5]}"
if ADAPTER is None:
    assert not any(n.startswith("adapters/") for n in names), "adapters/ in the zip (LoRA trap)"
else:                                             # adapter name ≠ base alias · rank ≤ 128 (scorer: --max-lora-rank 128) · agent.yaml uses it
    with zipfile.ZipFile(out) as z:
        cfg = json.loads(z.read(f"adapters/{ADAPTER}/adapter_config.json"))
        assert f"adapters/{ADAPTER}/adapter_model.safetensors" in names, f"adapters/{ADAPTER}/adapter_model.safetensors missing"
        assert ADAPTER != ALIAS and cfg.get("r", 0) <= 128, f"adapter {ADAPTER} r {cfg.get('r')}"
        assert f"adapter: {ADAPTER}" in z.read("agent.yaml").decode(), f"agent.yaml has no 'adapter: {ADAPTER}'"
    print(f"adapter {ADAPTER}: r {cfg['r']} alpha {cfg['lora_alpha']} target_modules {cfg['target_modules']}")
print(out, f"{out.stat().st_size / 2**20:.2f} MiB · sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
print(zipfile.ZipFile(out).read("eval_config.yaml").decode())
print("\n".join(names))
