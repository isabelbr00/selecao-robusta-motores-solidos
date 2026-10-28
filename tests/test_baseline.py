import pytest

from tg_robust import baseline, config, data


def test_ahp_baseline_retorna_ranking_valido():
    df = data.synthetic_designs(1, seed=4, n=20)
    req = config.REQUISITOS_ESTAGIO[1]
    resultado = baseline.ahp_baseline(df, req)
    assert set(resultado.columns) == {"score_ahp_baseline", "rank_ahp_baseline"}
    assert sorted(resultado["rank_ahp_baseline"]) == list(range(1, len(df) + 1))


@pytest.mark.xfail(
    reason=(
        "Calibração contra a Tabela 7 de Nascimento et al. (2022) requer os "
        "dados reais e os resultados do AHP original, ainda não disponíveis "
        "nesta pasta (ver README.md e decisões em aberto)."
    ),
    strict=False,
)
def test_ahp_baseline_calibrado_contra_tabela_7_do_artigo():
    raise NotImplementedError(
        "Comparar baseline.ahp_baseline() com os valores da Tabela 7 de "
        "Nascimento et al. (2022) quando os dados reais estiverem disponíveis."
    )
