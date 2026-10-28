import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from tg_robust import admissibility, anp, config, data, pipeline, sensitivity, stage1
from tg_robust import judgments as J


def _atributos(pc_por_estado):
    """atributos_por_estado mínimo: só pcmax importa para a admissibilidade."""

    return {e: {"pcmax": np.asarray(pc, dtype=float)} for e, pc in zip(config.ESTADOS_NOMES, pc_por_estado)}


@pytest.fixture
def caso_manual():
    # 4 designs; o design 3 tem a menor razão de massa mas viola a pressão
    designs = pd.DataFrame({"Rc": [1.0, 1.0, 1.0, 0.5], "Lt": [1.0, 1.0, 2.0, 1.0]})
    atributos = _atributos(
        [
            [11.0, 11.5, 10.0, 13.0],  # θ1
            [10.0, 11.0, 9.0, 12.0],  # θ2
            [11.2, 10.0, 9.0, 11.0],  # θ3 (design 0: máximo fora de θ1)
        ]
    )
    return designs, atributos


def test_pressao_usa_maximo_entre_estados(caso_manual):
    _, atributos = caso_manual
    np.testing.assert_allclose(admissibility.pressao_maxima_entre_estados(atributos), [11.2, 11.5, 10.0, 13.0])


def test_restricao_estrutural_por_perfil(caso_manual):
    designs, atributos = caso_manual
    adm = admissibility.conjuntos_admissiveis(
        designs, atributos, fs_perfis=(1.40, 1.25, 1.25), rho_perfis=(math.inf,) * 3, b_cer=1.0
    )
    # p_lim/FS: W -> 11.43; S, H -> 12.8
    assert list(adm.adm_pressao["W"]) == [True, False, True, False]
    assert list(adm.adm_pressao["S"]) == [True, True, True, False]
    assert list(adm.adm_pressao["H"]) == list(adm.adm_pressao["S"])


def test_indice_custo_minimo_sobre_todos_os_designs_e_estado_nominal(caso_manual):
    designs, atributos = caso_manual
    adm = admissibility.conjuntos_admissiveis(
        designs, atributos, fs_perfis=(1.25,) * 3, rho_perfis=(math.inf,) * 3, b_cer=2.0
    )
    # pcmax(θ1)·Rc²·Lt = [11, 11.5, 20, 3.25]; mínimo = design 3 (inadmissível por pressão)
    produto = np.array([11.0, 11.5, 20.0, 13.0 * 0.25])
    np.testing.assert_allclose(adm.razao_massa, produto / produto.min())
    np.testing.assert_allclose(adm.c, (produto / produto.min()) ** 2)
    assert adm.c[3] == pytest.approx(1.0)
    assert not adm.chi["S"][3]


def test_teto_de_custo_independe_do_fs():
    designs = pd.DataFrame({"Rc": [1.0, 1.0, 1.0], "Lt": [1.0, 1.1, 1.3]})
    atributos = _atributos([[10.0] * 3, [9.0] * 3, [8.0] * 3])
    adm = admissibility.conjuntos_admissiveis(
        designs, atributos, fs_perfis=(1.40, 1.25, 1.0), rho_perfis=(1.50, 1.15, 1.25), b_cer=1.0
    )
    np.testing.assert_allclose(adm.c, [1.0, 1.1, 1.3])
    assert list(adm.adm_custo["W"]) == [True, True, True]
    assert list(adm.adm_custo["S"]) == [True, True, False]
    assert list(adm.adm_custo["H"]) == [True, True, False]


def test_b_none_desativa_teto_de_custo_com_aviso(caso_manual):
    designs, atributos = caso_manual
    with pytest.warns(UserWarning, match="b da CER"):
        adm = admissibility.conjuntos_admissiveis(
            designs, atributos, fs_perfis=(1.25,) * 3, rho_perfis=(1.05,) * 3, b_cer=None
        )
    assert np.isnan(adm.c).all()
    assert all(adm.adm_custo[p].all() for p in config.PERFIS)
    assert np.isfinite(adm.razao_massa).all()


def test_resumo_conta_motivos_de_exclusao():
    designs = pd.DataFrame({"Rc": [1.0] * 4, "Lt": [1.0, 1.0, 2.0, 2.0]})
    # design 1: só pressão; design 2: só custo; design 3: ambos
    atributos = _atributos([[10.0, 13.0, 10.0, 13.0], [9.0] * 4, [9.0] * 4])
    adm = admissibility.conjuntos_admissiveis(
        designs, atributos, fs_perfis=(1.25,) * 3, rho_perfis=(1.5,) * 3, b_cer=1.0
    )
    r = admissibility.resumo(adm).set_index("perfil")
    assert r.loc["W", "n_admissiveis"] == 1
    assert r.loc["W", "excluidos_so_pressao"] == 1
    assert r.loc["W", "excluidos_so_custo"] == 1
    assert r.loc["W", "excluidos_ambos"] == 1
    assert r.loc["intersecao", "n_admissiveis"] == 1


def test_preferred_design_restrito_ao_conjunto_admissivel():
    V = np.array([0.9, 0.5, 0.7])
    assert stage1.preferred_design(V) == 0
    assert stage1.preferred_design(V, np.array([False, True, True])) == 2
    assert stage1.preferred_design(V, np.array([False, False, False])) is None


def test_W_AT_zera_inadmissiveis_e_renormaliza():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.1, 1.0, size=(6, config.N_CRITERIOS))
    chi = np.array([True, False, True, True, False, True])
    W_AT = anp.build_W_AT(v, chi)
    assert np.all(W_AT[~chi] == 0.0)
    np.testing.assert_allclose(W_AT.sum(axis=0), 1.0)
    esperado = v[chi] / v[chi].sum(axis=0)
    np.testing.assert_allclose(W_AT[chi], esperado)


def test_W_AT_chi_todo_verdadeiro_igual_sem_restricao():
    v = np.random.default_rng(1).uniform(size=(5, config.N_CRITERIOS))
    np.testing.assert_allclose(anp.build_W_AT(v, np.ones(5, dtype=bool)), anp.build_W_AT(v))


def test_perfil_vazio_excluido_da_sintese_com_pi_renormalizado():
    rng = np.random.default_rng(2)
    n = 8
    scores = {p: rng.uniform(size=(n, config.N_CRITERIOS)) for p in config.PERFIS}
    W_TT, _ = J.build_WTT_placeholder()
    pi = {"W": 0.5, "S": 0.3, "H": 0.2}
    chi = {"W": np.zeros(n, dtype=bool), "S": np.ones(n, dtype=bool), "H": rng.uniform(size=n) > 0.3}

    with pytest.warns(UserWarning, match="vazio"):
        res = anp.run_anp(scores, W_TT, gamma=0.5, pi=pi, chi_por_perfil=chi)

    assert res.perfis_excluidos == ("W",)
    assert set(res.g_por_perfil) == {"S", "H"}
    esperado = (0.3 * res.g_por_perfil["S"] + 0.2 * res.g_por_perfil["H"]) / 0.5
    np.testing.assert_allclose(res.g, esperado)
    assert res.g.sum() == pytest.approx(1.0)


def test_todos_os_perfis_vazios_levanta_erro():
    n = 4
    scores = {p: np.ones((n, config.N_CRITERIOS)) for p in config.PERFIS}
    W_TT, _ = J.build_WTT_placeholder()
    chi = {p: np.zeros(n, dtype=bool) for p in config.PERFIS}
    with pytest.raises(ValueError, match="todos os perfis"):
        anp.run_anp(scores, W_TT, gamma=0.5, pi=dict(zip(config.PERFIS, config.PI_PERFIS_REF)), chi_por_perfil=chi)


@pytest.fixture(scope="module")
def resultados_pipeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        designs = data.synthetic_designs(1, seed=5, n=120)
        cfg = config.Config(estagio=1, seed=5, b_cer=0.6)
        ref = pipeline.run(cfg, designs)
        sem = pipeline.run(
            dataclasses.replace(cfg, fs_perfis=(1.0,) * 3, rho_perfis=(math.inf,) * 3), designs
        )
    return ref, sem


def test_restricoes_nao_alteram_payoff_normalizacao_nem_pontuacoes(resultados_pipeline):
    ref, sem = resultados_pipeline
    np.testing.assert_array_equal(ref.X, sem.X)
    np.testing.assert_array_equal(ref.X_norm, sem.X_norm)
    for p in config.PERFIS:
        np.testing.assert_array_equal(ref.scores[p], sem.scores[p])
        np.testing.assert_array_equal(ref.preference_vectors[p], sem.preference_vectors[p])
    np.testing.assert_array_equal(ref.anp_resultado.q, sem.anp_resultado.q)


def test_pipeline_vencedores_e_prioridades_respeitam_F(resultados_pipeline):
    ref, _ = resultados_pipeline
    adm = ref.admissibilidade
    for p in config.PERFIS:
        k = ref.vencedor_perfil(p)
        assert k is not None and adm.chi[p][k]
        assert np.all(ref.anp_resultado.g_por_perfil[p][~adm.chi[p]] == 0.0)
    # design fora de todos os F^(p) tem prioridade global nula
    fora_de_todos = ~np.logical_or.reduce([adm.chi[p] for p in config.PERFIS])
    assert np.all(ref.g[fora_de_todos] == 0.0)
    assert ref.d_star_inadmissivel_em == adm.perfis_inadmissiveis(ref.d_star)


def test_sweep_restricoes_estrutura_e_caso_sem_restricoes(resultados_pipeline):
    ref, sem = resultados_pipeline
    df = sensitivity.sweep_restricoes(ref)
    assert len(df) == 3 * 3 + 3 * 5 + 1
    linha_sem = df[df["tipo"] == "sem_restricoes"].iloc[0]
    assert linha_sem["d_star"] == sem.d_star
    # rho = inf para um perfil nunca reduz F^(p) frente à referência
    for p in config.PERFIS:
        linha = df[(df["tipo"] == "rho") & (df["perfil_variado"] == p) & (df["valor"] == math.inf)].iloc[0]
        assert linha[f"n_F_{p}"] >= int(ref.admissibilidade.chi[p].sum())
    # FS de referência reproduz a referência
    linha_ref = df[(df["tipo"] == "FS") & (df["perfil_variado"] == "W") & (df["valor"] == 1.40)].iloc[0]
    assert linha_ref["mesmo_d_star"] and linha_ref["kendall_tau"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho_perfis": (0.9, 1.15, 1.25)},
        {"fs_perfis": (0.0, 1.25, 1.25)},
        {"b_cer": 0.0},
        {"b_cer": -1.0},
        {"p_lim": 0.0},
    ],
)
def test_config_restricoes_invalidas(kwargs):
    with pytest.raises(ValueError):
        config.Config(estagio=1, **kwargs)
