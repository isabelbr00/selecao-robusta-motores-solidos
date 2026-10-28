import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "run_pipeline.py"


@pytest.mark.parametrize("stage", [1, 2])
def test_cli_ponta_a_ponta_dados_sinteticos(tmp_path, stage):
    out_dir = tmp_path / f"out_{stage}"
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", str(stage), "--out", str(out_dir), "--seed", "0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resultado.returncode == 0, resultado.stderr

    esperados = [
        "scores_W.csv",
        "scores_S.csv",
        "scores_H.csv",
        "ranking_final.csv",
        "criterion_weights.csv",
        "consistency.csv",
        "sens_alpha.csv",
        "sens_pi.csv",
        "sens_anp.csv",
        "sens_lambda.csv",
        "sens_restricoes.csv",
        "admissibilidade.csv",
        "admissibilidade_resumo.csv",
        "vencedores_por_perfil.csv",
        "run_config.json",
    ]
    for nome in esperados:
        assert (out_dir / nome).exists(), f"arquivo ausente: {nome}"

    with open(out_dir / "run_config.json") as f:
        cfg = json.load(f)
    assert cfg["dados_sinteticos"] is True
    assert cfg["n_designs"] == {1: 632, 2: 1089}[stage]
    assert cfg["b_cer_definido"] is False  # placeholder até b ser preenchido
    assert set(cfg["tamanho_F"]) == {"W", "S", "H"}
    assert "d_star_inadmissivel_em" in cfg

    with open(out_dir / "ranking_final.csv") as f:
        linhas = f.readlines()
    assert len(linhas) <= 51  # cabeçalho + até 50 designs
