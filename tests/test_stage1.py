import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from tg_robust import config, data, stage1
from tg_robust.scenarios import SurrogateBallistics


@pytest.fixture
def designs_n_constante():
    """Designs sintéticos com nexp constante (mesmo propelente), usados para
    demonstrar a degeneração da normalização por estado em pcmax/rhoIsp."""

    df = data.synthetic_designs(1, seed=5, n=40)
    df = df.copy()
    df["propelente"] = 1
    df["nexp"] = config.TIPO_PROPELENTE_PARA_N[1]
    return df


@pytest.fixture
def tensor(designs_n_constante):
    req = config.REQUISITOS_ESTAGIO[1]
    model = SurrogateBallistics()
    X, _ = stage1.build_payoff_tensor(
        designs_n_constante, req, config.KAPPA_C_REF, config.KAPPA_CF_REF, model, criterios=config.ATRIBUTOS
    )
    return X


def test_beneficio_residuo():
    req = config.REQUISITOS_ESTAGIO[1]
    attrs = {"Mp": np.array([req.Mp, req.Mp * 1.1, req.Mp * 0.8])}
    np.testing.assert_allclose(stage1.beneficio("Mp", attrs, req), [0.0, -0.01, -0.04], atol=1e-12)  # padrão: -δ²
    np.testing.assert_allclose(stage1.beneficio("Mp", attrs, req, "quadrado"), [0.0, -0.01, -0.04], atol=1e-12)
    np.testing.assert_allclose(stage1.beneficio("Mp", attrs, req, "absoluto"), [0.0, -0.1, -0.2], atol=1e-12)
    with pytest.raises(ValueError):
        stage1.beneficio("Mp", attrs, req, "cubo")


def test_beneficio_maximizar_e_minimizar():
    req = config.REQUISITOS_ESTAGIO[1]
    attrs = {"rhoIsp": np.array([1.0, 2.0]), "pcmax": np.array([1.0, 2.0]), "eta": np.array([0.8, 0.9])}
    assert list(stage1.beneficio("rhoIsp", attrs, req)) == [1.0, 2.0]
    assert list(stage1.beneficio("pcmax", attrs, req)) == [-1.0, -2.0]
    assert list(stage1.beneficio("eta", attrs, req)) == [0.8, 0.9]


def test_payoff_tensor_shape(tensor, designs_n_constante):
    n = len(designs_n_constante)
    assert tensor.shape == (len(config.ATRIBUTOS), n, 3)


def test_common_minmax_nao_degenera_para_rhoIsp_e_pcmax(tensor):
    Xn = stage1.common_minmax(tensor)
    r_rho = config.ATRIBUTOS.index("rhoIsp")
    r_pc = config.ATRIBUTOS.index("pcmax")
    # variação entre colunas (estados) preservada: colunas distintas
    assert not np.allclose(Xn[r_rho, :, 0], Xn[r_rho, :, 1])
    assert not np.allclose(Xn[r_pc, :, 0], Xn[r_pc, :, 2])


def test_per_state_minmax_degenera_para_rhoIsp_e_pcmax(tensor):
    Xn = stage1.per_state_minmax(tensor)
    r_rho = config.ATRIBUTOS.index("rhoIsp")
    r_pc = config.ATRIBUTOS.index("pcmax")
    # fator multiplicativo comum por estado é cancelado pela normalização
    # por estado: colunas tornam-se idênticas (efeito do cenário desaparece)
    assert np.allclose(Xn[r_rho, :, 0], Xn[r_rho, :, 1], atol=1e-9)
    assert np.allclose(Xn[r_rho, :, 0], Xn[r_rho, :, 2], atol=1e-9)
    assert np.allclose(Xn[r_pc, :, 0], Xn[r_pc, :, 1], atol=1e-9)
    assert np.allclose(Xn[r_pc, :, 0], Xn[r_pc, :, 2], atol=1e-9)


def test_common_minmax_zeros_quando_constante():
    X = np.ones((1, 5, 3)) * 3.0
    Xn = stage1.common_minmax(X)
    assert np.allclose(Xn, 0.0)


def test_scores_em_0_1(tensor):
    Xn = stage1.common_minmax(tensor)
    scores = stage1.compute_profile_scores(Xn, config.PERFIS_REFERENCIA)
    for v in scores.values():
        assert (v >= -1e-12).all() and (v <= 1 + 1e-12).all()


def test_hurwicz_alpha_0_eq_wald(tensor):
    Xn = stage1.common_minmax(tensor)
    assert np.allclose(stage1.hurwicz(Xn, 0.0), stage1.wald(Xn))


def test_hurwicz_alpha_1_eq_maximo_por_linha(tensor):
    Xn = stage1.common_minmax(tensor)
    assert np.allclose(stage1.hurwicz(Xn, 1.0), Xn.max(axis=2).T)


def test_savage_e_wald_mesma_ordenacao_para_criterio_invariante(tensor):
    Xn = stage1.common_minmax(tensor)
    v_w = stage1.wald(Xn)
    v_s = stage1.savage(Xn)
    v_h = stage1.hurwicz(Xn, 0.5)
    for criterio in config.CRITERIOS_INVARIANTES:
        r = config.ATRIBUTOS.index(criterio)
        ordem_w = rankdata(v_w[:, r])
        ordem_s = rankdata(v_s[:, r])
        ordem_h = rankdata(v_h[:, r])
        assert np.array_equal(ordem_w, ordem_s)
        assert np.array_equal(ordem_w, ordem_h)


def test_preference_vector_media_simples(tensor):
    Xn = stage1.common_minmax(tensor)
    v = stage1.wald(Xn)
    V = stage1.preference_vector(v)
    assert V == pytest.approx(v.mean(axis=1))
