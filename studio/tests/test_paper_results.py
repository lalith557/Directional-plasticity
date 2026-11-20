from studio.engine import paper_results as pr
from studio.engine.paths import PAPER_TEX


def test_every_paper_reference_value_is_quoted_from_the_tex():
    tex = PAPER_TEX.read_text(encoding="utf-8")
    anchors = ([v["tex"] for v in pr.PAPER["table1"].values()] + [pr.PAPER["association"]["tex"]]
               + [r["tex"] for r in pr.PAPER["regimes"]]
               + [pr.PAPER[k]["tex"] for k in ("anr_ablation", "radial", "paired", "cor1")])
    missing = [a for a in anchors if a not in tex]
    assert not missing, f"paper reference values drifted from the .tex: {missing}"


def test_all_paper_values_are_consistent_with_artifacts():
    f = pr.findings()
    assert f["consistency"]["checks"] >= 60
    assert f["consistency"]["artifact_consistent"] == f["consistency"]["checks"]


def test_labels_keep_value_kinds_separate():
    f = pr.findings()
    assert set(f["labels"]) == {"paper", "artifact", "reproduced", "generated"}
    f5 = f["findings"][4]
    assert [r["paper"]["r"] for r in f5["regimes"]] == [0.97, 0.45, -0.52, -0.94, -0.82]
    assert all("method-level" in r["scope_note"] for r in f5["regimes"])
