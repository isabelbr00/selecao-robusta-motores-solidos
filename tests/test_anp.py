import numpy as np
import pytest

from tg_robust import anp, config, data, judgments as J, stage1
from tg_robust.scenarios import SurrogateBallistics


@pytest.fixture
def cenario_pequeno():
    df = data.synthetic_designs(1, seed=11, n=12)
    req = config.REQUISITOS_ESTAGIO[1]
    model = SurrogateBallistics()
    X, _ = stage1.build_payoff_tensor(df, req, config.KAPPA_C_REF, config.KAPPA_CF_REF, model)
    Xn = stage1.common_minmax(X)
    scores = stage1.compute_profile_scores(Xn, alpha=0.8)
    with pytest.warns(UserWarning):
        W_TT, _ = J.build_WTT_placeholder()
    q0 = np.full(config.N_CRITERIOS, 1.0 / config.N_CRITERIOS)
    return scores, W_TT, q0


def test_build_W_AT_colunas_somam_1():
    v = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    W_AT = anp.build_W_AT(v)
    assert W_AT.sum(axis=0) == pytest.approx([1.0, 1.0])
    # coluna nula -> uniforme
    assert np.allclose(W_AT[:, 1], 1.0 / 3)


def test_WTT_estocastica_e_raio_espectral(cenario_pequeno):
    _, W_TT, _ = cenario_pequeno
    assert np.allclose(W_TT.sum(axis=0), 1.0)
    autovalores = np.linalg.eigvals(0.5 * W_TT)
    assert np.max(np.abs(autovalores)) < 1.0


def test_limite_fechado_igual_limite_por_potencias(cenario_pequeno):
    scores, W_TT, q0 = cenario_pequeno
    W_AT = anp.build_W_AT(scores["W"])
    g_fechado, _ = anp.profile_limit_closed(W_AT, W_TT, 0.5, q0)
    g_potencias = anp.profile_limit_power(W_AT, W_TT, 0.5, q0, tol=1e-12)
    assert np.allclose(g_fechado, g_potencias, atol=1e-8)


def test_soma_g_e_g_por_perfil_igual_1(cenario_pequeno):
    scores, W_TT, q0 = cenario_pequeno
    pi = {"W": 1 / 3, "S": 1 / 3, "H": 1 / 3}
    resultado = anp.run_anp(scores, W_TT, gamma=0.5, pi=pi, q0=q0)
    assert resultado.g.sum() == pytest.approx(1.0)
    for g_p in resultado.g_por_perfil.values():
        assert g_p.sum() == pytest.approx(1.0)


def test_sintese_por_mistura_de_subredes(cenario_pequeno):
    scores, W_TT, q0 = cenario_pequeno
    pi = {"W": 0.5, "S": 0.3, "H": 0.2}
    resultado = anp.run_anp(scores, W_TT, gamma=0.5, pi=pi, q0=q0)
    W_AT_bar = anp.W_AT_mixture(resultado.W_AT_por_perfil, pi)
    g_via_mistura = W_AT_bar @ resultado.q
    assert np.allclose(resultado.g, g_via_mistura)


def test_pesos_efetivos_somam_1(cenario_pequeno):
    _, W_TT, q0 = cenario_pequeno
    q = anp.effective_criterion_weights(W_TT, 0.5, q0)
    assert q.sum() == pytest.approx(1.0)


def test_run_anp_verificacao_limite_potencias_nao_levanta_erro(cenario_pequeno):
    scores, W_TT, q0 = cenario_pequeno
    pi = {"W": 1 / 3, "S": 1 / 3, "H": 1 / 3}
    anp.run_anp(scores, W_TT, gamma=0.5, pi=pi, q0=q0, verificar_limite_por_potencias=True)
