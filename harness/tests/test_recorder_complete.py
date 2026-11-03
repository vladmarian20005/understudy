"""A dataset folder counts as resumable only with a whole first episode: else lerobot's resume() goes to the Hub."""
import json

from harness.recorder import complete


def test_only_a_dataset_with_a_whole_episode_is_complete(tmp_path):
    root = tmp_path / "understudy_ui"
    (root / "meta").mkdir(parents=True)
    json.dump({}, open(root / "meta" / "info.json", "w"))
    assert not complete(str(root))  # begun, robotd stopped before the first episode was saved
    (root / "meta" / "tasks.parquet").write_bytes(b"")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    assert not complete(str(root))
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    assert complete(str(root)) and not complete(str(tmp_path / "nothing"))
