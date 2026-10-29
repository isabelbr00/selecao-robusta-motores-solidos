import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
        "consistency.csv",
        "sens_alpha.csv",
        "sens_pi.csv",
        "sens_anp.csv",
        "sens_lambda.csv",
        "sens_restricoes.csv",
        "sens_residuo.csv",
        "admissibilidade.csv",
        "admissibilidade_resumo.csv",
        "vencedores_por_perfil.csv",
        "pesos_efetivos.csv",
        "run_config.json",
    ]
    for nome in esperados:
        assert (out_dir / nome).exists(), f"arquivo ausente: {nome}"

    with open(out_dir / "run_config.json") as f:
        cfg = json.load(f)
    assert cfg["dados_sinteticos"] is True
    assert cfg["n_designs"] == {1: 632, 2: 1089}[stage]
    assert cfg["b_cer_provisorio"] is True  # b = 1.0 provisório até ser trocado
    assert cfg["referencia_custo"] == "mediana"
    assert cfg["sens_lambda_executada"] is True  # modelo substituto reexecuta os estados
    assert set(cfg["tamanho_F"]) == {"W", "S", "H"}
    assert "d_star_inadmissivel_em" in cfg

    venc = pd.read_csv(out_dir / "vencedores_por_perfil.csv")
    assert list(venc.columns) == [
        "estagio", "perfil",
        "vencedor_anp", "vencedor_pesos_iguais", "divergem",
        "vencedor_anp_sem_restricoes", "vencedor_pesos_iguais_sem_restricoes", "divergem_sem_restricoes",
        "excluido_da_sintese",
    ]
    assert set(venc["divergem"]) <= {"sim", "não"}
    pesos = pd.read_csv(out_dir / "pesos_efetivos.csv")
    assert list(pesos.columns) == ["criterio", "q0", "q", "q_sobre_q0"]
    assert pesos["q"].sum() == pytest.approx(1.0)

    with open(out_dir / "ranking_final.csv") as f:
        linhas = f.readlines()
    assert len(linhas) <= 51  # cabeçalho + até 50 designs


GERADOR = RAIZ.parent / "gerador_pareto"
requer_gerador = pytest.mark.skipif(not GERADOR.exists(), reason="gerador_pareto/ ausente")


def _rodar(tmp_path, *extra):
    out_dir = tmp_path / "out"
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out_dir), *extra],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert resultado.returncode == 0, resultado.stderr
    with open(out_dir / "run_config.json") as f:
        # JSON estrito: falha se houver Infinity/NaN literais
        cfg = json.load(f, parse_constant=lambda c: pytest.fail(f"constante não-JSON {c}"))
    return out_dir, cfg, resultado.stdout


@requer_gerador
def test_cli_front_de_teste_com_estados_do_csv_pula_lambda(tmp_path):
    csv = GERADOR / "dados" / "pareto_estagio1.csv"
    out_dir, cfg, stdout = _rodar(tmp_path, "--stage", "1", "--csv", str(csv))
    assert cfg["modelo_balistico"] == "EstadosPrecomputados"
    assert cfg["sens_lambda_executada"] is False
    assert "varredura de λ pulada" in stdout
    assert cfg["d_star_design_id"].startswith("S1-")


@requer_gerador
def test_cli_front_de_teste_com_balistica_executa_lambda(tmp_path):
    csv = GERADOR / "dados" / "pareto_estagio2.csv"
    _, cfg, _ = _rodar(tmp_path, "--stage", "2", "--csv", str(csv), "--balistica", str(GERADOR / "balistica.py"))
    assert cfg["modelo_balistico"] == "AdaptadorSimulador"
    assert cfg["sens_lambda_executada"] is True


def test_cli_sem_teto_custo_gera_json_estrito(tmp_path):
    _, cfg, _ = _rodar(tmp_path, "--stage", "1", "--seed", "0", "--sem-teto-custo")
    assert all(p["rho"] == "inf" for p in cfg["config"]["perfis"])


CSV_REAL_1 = RAIZ.parent / "data" / "modeFrontier 1st Stage Us Propellants Designs Table.csv"


@pytest.mark.skipif(not CSV_REAL_1.exists(), reason="tabela real do 1º estágio não disponível")
def test_cli_dados_reais_um_estagio_e_figuras(tmp_path):
    out_dir, cfg, _ = _rodar(tmp_path, "--stage", "1", "--csv", str(CSV_REAL_1))
    assert cfg["modelo_balistico"] == "EscalaQuaseEstacionaria"
    assert cfg["sens_lambda_executada"] is True
    assert cfg["n_designs"] == 437 and cfg["n_duplicatas_removidas"] == 195

    figuras = tmp_path / "figuras"
    resultado = subprocess.run(
        [sys.executable, str(RAIZ / "gerar_figuras_cap4.py"), "--dados1", str(CSV_REAL_1),
         "--resultados1", str(out_dir), "--out", str(figuras)],
        capture_output=True, text=True, timeout=300,
    )
    assert resultado.returncode == 0, resultado.stderr
    for nome in ("coord_paralelas", "custo_pressao", "top10_final", "sensibilidade"):
        assert (figuras / f"{nome}.pdf").exists()
    assert (pd.read_csv(figuras / "top5_final.csv")["estagio"] == 1).all()
