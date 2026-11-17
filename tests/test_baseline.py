from pathlib import Path

import numpy as np
import pytest

from selecao_robusta import baseline, config, data


def test_ahp_baseline_retorna_ranking_valido():
    df = data.synthetic_designs(1, seed=4, n=20)
    req = config.REQUISITOS_ESTAGIO[1]
    resultado = baseline.ahp_baseline(df, req)
    assert set(resultado.columns) == {"score_ahp_baseline", "rank_ahp_baseline"}
    assert sorted(resultado["rank_ahp_baseline"]) == list(range(1, len(df) + 1))


# ─── Validação contra Nascimento et al. (2022), 1º estágio ────────────────
# a tabela não é distribuída: procurada em data/ do repositório ou em ../data/
_NOME_CSV_REAL_1 = "modeFrontier 1st Stage Us Propellants Designs Table.csv"
CSV_REAL_1 = next((d / "data" / _NOME_CSV_REAL_1 for d in Path(__file__).resolve().parents[1:3] if (d / "data" / _NOME_CSV_REAL_1).exists()), Path("ausente"))
requer_dados_reais = pytest.mark.skipif(not CSV_REAL_1.exists(), reason="tabela real do 1º estágio não disponível")
ID_AHP_ARTIGO = 5763  # design escolhido pelo AHP do artigo (Tabela 7)

# Tabela 7 do artigo, design 5763, com a precisão impressa (pcmed em MPa)
TABELA_7 = {"Mp": (8180.96, 0.005), "tb": (111.62, 0.005), "It": (2.38e7, 0.005e7),
            "F": (2.0839e5, 0.00005e5), "rhoIsp": (5.6446e6, 0.00005e6), "pcmed": (4.534, 0.0005)}


@pytest.fixture(scope="module")
def designs_reais():
    return data.load_designs(CSV_REAL_1)


@requer_dados_reais
def test_design_5763_reproduz_tabela_7(designs_reais):
    linha = designs_reais.set_index("design_id").loc[ID_AHP_ARTIGO]
    for c, (valor, meia_unidade) in TABELA_7.items():
        assert abs(linha[c] - valor) <= meia_unidade, c


@requer_dados_reais
def test_ahp_baseline_escolhe_o_design_do_artigo(designs_reais):
    bl = baseline.ahp_baseline(designs_reais, config.REQUISITOS_ESTAGIO[1])
    escolhido = designs_reais.loc[bl["rank_ahp_baseline"].idxmin(), "design_id"]
    assert escolhido == ID_AHP_ARTIGO


def test_beneficio_artigo_residuo_ao_quadrado():
    req = config.REQUISITOS_ESTAGIO[1]
    v = baseline.beneficio_artigo("Mp", np.array([req.Mp * 1.1, req.Mp * 0.8]), req)
    np.testing.assert_allclose(v, [-0.01, -0.04])
