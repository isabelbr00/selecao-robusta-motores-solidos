import dataclasses
import importlib.util
import inspect
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tg_robust import admissibility, anp, config, data, pipeline, scenarios, sensitivity, stage1
from tg_robust import judgments as J

GERADOR = Path(__file__).resolve().parents[2] / "gerador_pareto"
requer_gerador = pytest.mark.skipif(not GERADOR.exists(), reason="gerador_pareto/ ausente")

# O gabarito e a referência foram construídos com o mínimo como referência de custo e b = 1
GLOBAIS_GABARITO = config.ConfigGlobal(b_cer=1.0, referencia_custo="minimo")
PERFIS = config.PERFIS_REFERENCIA


def _estados(designs, model=None):
    model = model or scenarios.EstadosPrecomputados()
    return scenarios.propagate_states(designs, config.KAPPA_C_REF, config.KAPPA_CF_REF, model)


def _atributos(pc_por_estado):
    """atributos_por_estado mínimo: só pcmax importa para a admissibilidade."""

    return {e: {"pcmax": np.asarray(pc, dtype=float)} for e, pc in zip(config.ESTADOS_NOMES, pc_por_estado)}


def _carregar(csv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return data.load_designs(csv)


# ─────────────────────────────────────────────────────────────────────────
# Fórmulas (casos construídos à mão)
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture
def caso_manual():
    # 4 designs; o design 3 tem a menor base de custo mas viola a pressão
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
    perfis = tuple(dataclasses.replace(p, rho=math.inf) for p in PERFIS)
    adm = admissibility.conjuntos_admissiveis(designs, atributos, perfis, GLOBAIS_GABARITO)
    # p_lim/FS: W -> 11.43; S, H -> 12.8
    assert list(adm.adm_pressao["W"]) == [True, False, True, False]
    assert list(adm.adm_pressao["S"]) == [True, True, True, False]
    assert list(adm.adm_pressao["H"]) == list(adm.adm_pressao["S"])


@pytest.mark.parametrize("referencia, ref_fn", [("minimo", np.min), ("mediana", np.median)])
def test_indice_custo_sobre_todos_os_designs_e_estado_nominal(caso_manual, referencia, ref_fn):
    designs, atributos = caso_manual
    globais = config.ConfigGlobal(b_cer=2.0, referencia_custo=referencia)
    adm = admissibility.conjuntos_admissiveis(designs, atributos, PERFIS, globais)
    # base = pcmax(θ1)·Rc²·Lt = [11, 11.5, 20, 3.25], inclui o design 3 (inadmissível por pressão)
    base = np.array([11.0, 11.5, 20.0, 13.0 * 0.25])
    np.testing.assert_allclose(adm.c, (base / ref_fn(base)) ** 2)
    assert not adm.chi["S"][3]


def test_teto_de_custo_independe_do_fs():
    designs = pd.DataFrame({"Rc": [1.0, 1.0, 1.0], "Lt": [1.0, 1.1, 1.3]})
    atributos = _atributos([[10.0] * 3, [9.0] * 3, [8.0] * 3])
    perfis = config.substituir_perfil(PERFIS, "H", FS=1.0)
    adm = admissibility.conjuntos_admissiveis(designs, atributos, perfis, GLOBAIS_GABARITO)
    np.testing.assert_allclose(adm.c, [1.0, 1.1, 1.3])
    assert list(adm.adm_custo["W"]) == [True, True, True]
    assert list(adm.adm_custo["S"]) == [True, True, False]
    assert list(adm.adm_custo["H"]) == [True, True, False]


def test_b_provisorio_emite_aviso(caso_manual):
    designs, atributos = caso_manual
    with pytest.warns(UserWarning, match="PROVISÓRIO"):
        admissibility.conjuntos_admissiveis(designs, atributos, PERFIS, config.ConfigGlobal())


def test_resumo_conta_motivos_de_exclusao():
    designs = pd.DataFrame({"Rc": [1.0] * 4, "Lt": [1.0, 1.0, 2.0, 2.0]})
    # design 1: só pressão; design 2: só custo; design 3: ambos
    atributos = _atributos([[10.0, 13.0, 10.0, 13.0], [9.0] * 4, [9.0] * 4])
    perfis = tuple(dataclasses.replace(p, FS=1.25, rho=1.5) for p in PERFIS)
    adm = admissibility.conjuntos_admissiveis(designs, atributos, perfis, GLOBAIS_GABARITO)
    r = admissibility.resumo(adm).set_index("perfil")
    assert r.loc["W", "n_admissiveis"] == 1
    assert r.loc["W", "excluidos_so_pressao"] == 1
    assert r.loc["W", "excluidos_so_custo"] == 1
    assert r.loc["W", "excluidos_ambos"] == 1
    assert r.loc["intersecao", "n_admissiveis"] == 1


# ─────────────────────────────────────────────────────────────────────────
# rho = inf equivale a retirar a restrição de custo
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("referencia", config.REFERENCIAS_CUSTO)
@pytest.mark.parametrize("b", [0.5, 1.0, 3.0])
def test_rho_infinito_equivale_a_so_restricao_estrutural(caso_manual, referencia, b):
    designs, atributos = caso_manual
    perfis = tuple(dataclasses.replace(p, rho=math.inf) for p in PERFIS)
    adm = admissibility.conjuntos_admissiveis(designs, atributos, perfis, config.ConfigGlobal(b_cer=b, referencia_custo=referencia))
    for s in adm.siglas:
        assert adm.adm_custo[s].all()
        np.testing.assert_array_equal(adm.chi[s], adm.adm_pressao[s])


# ─────────────────────────────────────────────────────────────────────────
# W_AT, argmax restrito e exclusão de perfil vazio
# ─────────────────────────────────────────────────────────────────────────
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
    np.testing.assert_allclose(W_AT[chi], v[chi] / v[chi].sum(axis=0))


def test_W_AT_chi_todo_verdadeiro_igual_sem_restricao():
    v = np.random.default_rng(1).uniform(size=(5, config.N_CRITERIOS))
    np.testing.assert_array_equal(anp.build_W_AT(v, np.ones(5, dtype=bool)), anp.build_W_AT(v))


def test_perfil_vazio_excluido_da_sintese_com_pi_renormalizado():
    rng = np.random.default_rng(2)
    n = 8
    scores = {p: rng.uniform(size=(n, config.N_CRITERIOS)) for p in "WSH"}
    W_TT, _ = J.build_WTT_validada()
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
    scores = {p: np.ones((n, config.N_CRITERIOS)) for p in "WSH"}
    W_TT, _ = J.build_WTT_validada()
    chi = {p: np.zeros(n, dtype=bool) for p in "WSH"}
    with pytest.raises(ValueError, match="todos os perfis"):
        anp.run_anp(scores, W_TT, gamma=0.5, pi={"W": 0.4, "S": 0.3, "H": 0.3}, chi_por_perfil=chi)


# ─────────────────────────────────────────────────────────────────────────
# Gabarito do conjunto sintético (gerador_pareto/dados/sintetico_*.csv)
# ─────────────────────────────────────────────────────────────────────────
CENARIOS_GABARITO = {
    "referencia": PERFIS,
    "sem_restricoes": config.perfis_sem_restricoes(PERFIS),
    "W_vazio": config.substituir_perfil(PERFIS, "W", FS=2.10),
}


@pytest.fixture(scope="module")
def sintetico():
    designs = _carregar(GERADOR / "dados" / "sintetico_designs.csv")
    gabarito = pd.read_csv(GERADOR / "dados" / "sintetico_gabarito.csv")
    return designs, gabarito


@requer_gerador
@pytest.mark.parametrize("cenario", list(CENARIOS_GABARITO))
def test_gabarito_conjuntos_admissiveis(sintetico, cenario):
    designs, gabarito = sintetico
    esperado = gabarito[gabarito["cenario"] == cenario].set_index("design_id").loc[designs["design_id"]]
    adm = admissibility.conjuntos_admissiveis(designs, _estados(designs), CENARIOS_GABARITO[cenario], GLOBAIS_GABARITO)
    np.testing.assert_allclose(adm.c, esperado["c_k"].to_numpy(), rtol=1e-9)
    for s in "WSH":
        np.testing.assert_array_equal(adm.chi[s], esperado[f"admissivel_{s}"].to_numpy(), err_msg=f"perfil {s}")


@requer_gerador
def test_gabarito_design_I_cai_em_W_pela_pressao_em_theta2(sintetico):
    designs, _ = sintetico
    k = int(np.flatnonzero(designs["design_id"] == "I")[0])
    adm = admissibility.conjuntos_admissiveis(designs, _estados(designs), PERFIS, GLOBAIS_GABARITO)
    assert designs.loc[k, "pcmax_theta2"] > designs.loc[k, "pcmax_theta1"]
    assert adm.pcmax_max[k] == designs.loc[k, "pcmax_theta2"]
    assert not adm.adm_pressao["W"][k]
    assert 1.40 * designs.loc[k, "pcmax_theta1"] <= 16.0  # só θ1 deixaria passar


@pytest.fixture(scope="module")
def pipelines_sintetico(sintetico):
    designs, _ = sintetico
    resultados = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nome, perfis in CENARIOS_GABARITO.items():
            cfg = config.Config(estagio=2, perfis=perfis, globais=GLOBAIS_GABARITO)
            resultados[nome] = pipeline.run(cfg, designs, model=scenarios.EstadosPrecomputados())
    return resultados


@requer_gerador
def test_pipeline_inadmissivel_tem_prioridade_zero_so_no_seu_perfil(pipelines_sintetico):
    ref = pipelines_sintetico["referencia"]
    k = int(np.flatnonzero(ref.design_ids == "C")[0])  # gabarito: C admissível só para H
    g_p = ref.anp_resultado.g_por_perfil
    assert g_p["W"][k] == 0.0 and g_p["S"][k] == 0.0
    assert g_p["H"][k] > 0.0 and ref.g[k] > 0.0
    for s in "WSH":
        np.testing.assert_allclose(ref.anp_resultado.W_AT_por_perfil[s].sum(axis=0), 1.0)
        assert np.all(ref.anp_resultado.W_AT_por_perfil[s][~ref.admissibilidade.chi[s]] == 0.0)
        assert ref.admissibilidade.chi[s][ref.vencedor_perfil(s)]


@requer_gerador
def test_pipeline_W_vazio_exclui_subrede_e_renormaliza_pi(pipelines_sintetico):
    res = pipelines_sintetico["W_vazio"]
    assert res.anp_resultado.perfis_excluidos == ("W",)
    assert res.vencedor_perfil("W") is None
    g_p = res.anp_resultado.g_por_perfil
    np.testing.assert_allclose(res.g, 0.5 * g_p["S"] + 0.5 * g_p["H"])
    assert res.g.sum() == pytest.approx(1.0)
    assert "W" in res.d_star_inadmissivel_em


# ─────────────────────────────────────────────────────────────────────────
# Equivalência com gerador_pareto/restricoes_referencia.py
# ─────────────────────────────────────────────────────────────────────────
def _modulo_referencia():
    spec = importlib.util.spec_from_file_location("restricoes_referencia", GERADOR / "restricoes_referencia.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requer_gerador
@pytest.mark.parametrize("estagio", [1, 2])
@pytest.mark.parametrize("referencia", ["minimo", "mediana"])
def test_equivalencia_com_implementacao_de_referencia(estagio, referencia):
    ref = _modulo_referencia()
    aceita_referencia = "referencia" in inspect.signature(ref.conjuntos_admissiveis).parameters
    if referencia != "minimo" and not aceita_referencia:
        pytest.skip(
            "restricoes_referencia.py em gerador_pareto/ ainda não aceita referencia='mediana' "
            "(versão atualizada não encontrada no disco)"
        )

    csv = GERADOR / "dados" / f"pareto_estagio{estagio}.csv"
    df_bruto = pd.read_csv(csv)
    kwargs = dict(ref.PADRAO)
    if aceita_referencia:
        kwargs["referencia"] = referencia
    esperado = ref.conjuntos_admissiveis(df_bruto, **kwargs)

    designs = _carregar(csv)
    perfis = tuple(
        dataclasses.replace(p, FS=kwargs["FS"][p.sigla], rho=kwargs["rho"][p.sigla]) for p in PERFIS
    )
    globais = config.ConfigGlobal(p_lim=kwargs["p_lim"], b_cer=kwargs["b"], referencia_custo=referencia)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm = admissibility.conjuntos_admissiveis(designs, _estados(designs), perfis, globais)

    np.testing.assert_allclose(adm.c, esperado["c_k"].to_numpy(), rtol=1e-12)
    for s in "WSH":
        np.testing.assert_array_equal(adm.adm_pressao[s], esperado[f"estrutural_{s}"].to_numpy())
        np.testing.assert_array_equal(adm.adm_custo[s], esperado[f"custo_{s}"].to_numpy())
        np.testing.assert_array_equal(adm.chi[s], esperado[f"admissivel_{s}"].to_numpy())
    np.testing.assert_array_equal(adm.intersecao, esperado["admissivel_todos"].to_numpy())


# ─────────────────────────────────────────────────────────────────────────
# Pipeline completo com dados sintéticos do pacote
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def resultados_pipeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        designs = data.synthetic_designs(1, seed=5, n=120)
        cfg = config.Config(estagio=1, seed=5)
        ref = pipeline.run(cfg, designs)
        sem = pipeline.run(dataclasses.replace(cfg, perfis=config.perfis_sem_restricoes(cfg.perfis)), designs)
        sem_custo = pipeline.run(
            dataclasses.replace(cfg, perfis=tuple(dataclasses.replace(p, rho=math.inf) for p in cfg.perfis)), designs
        )
    return ref, sem, sem_custo


def test_restricoes_nao_alteram_payoff_normalizacao_nem_pontuacoes(resultados_pipeline):
    ref, sem, _ = resultados_pipeline
    np.testing.assert_array_equal(ref.X, sem.X)
    np.testing.assert_array_equal(ref.X_norm, sem.X_norm)
    for s in ref.cfg.siglas:
        np.testing.assert_array_equal(ref.scores[s], sem.scores[s])
        np.testing.assert_array_equal(ref.preference_vectors[s], sem.preference_vectors[s])
    np.testing.assert_array_equal(ref.anp_resultado.q, sem.anp_resultado.q)


def test_pipeline_rho_infinito_igual_a_anp_so_com_pressao(resultados_pipeline):
    ref, _, sem_custo = resultados_pipeline
    adm = sem_custo.admissibilidade
    esperado = anp.run_anp(
        ref.scores, ref.W_TT, gamma=ref.cfg.gamma_anp, pi=ref.cfg.pi, q0=ref.anp_resultado.q0,
        chi_por_perfil=adm.adm_pressao,
    )
    np.testing.assert_array_equal(sem_custo.g, esperado.g)
    for s in ref.cfg.siglas:
        assert adm.adm_custo[s].all()


def test_pipeline_vencedores_e_prioridades_respeitam_F(resultados_pipeline):
    ref, _, _ = resultados_pipeline
    adm = ref.admissibilidade
    for s in ref.cfg.siglas:
        k = ref.vencedor_perfil(s)
        assert k is not None and adm.chi[s][k]
        assert np.all(ref.anp_resultado.g_por_perfil[s][~adm.chi[s]] == 0.0)
    fora_de_todos = ~np.logical_or.reduce([adm.chi[s] for s in ref.cfg.siglas])
    assert np.all(ref.g[fora_de_todos] == 0.0)
    assert ref.d_star_inadmissivel_em == adm.perfis_inadmissiveis(ref.d_star)


def test_sweep_restricoes_estrutura_e_caso_sem_restricoes(resultados_pipeline):
    ref, sem, sem_custo = resultados_pipeline
    df = sensitivity.sweep_restricoes(ref)
    assert len(df) == 3 * 3 + 3 * 5 + 2 + 1
    assert set(df.loc[df["tipo"] == "referencia_custo", "valor"]) == {"minimo", "mediana"}
    assert df[df["tipo"] == "sem_restricoes"].iloc[0]["d_star"] == ref.design_ids[sem.d_star]
    # rho = inf para um perfil: F^(p) = conjunto só com a restrição de pressão
    for s in ref.cfg.siglas:
        linha = df[(df["tipo"] == "rho") & (df["perfil_variado"] == s) & (df["valor"] == math.inf)].iloc[0]
        assert linha[f"n_F_{s}"] == int(ref.admissibilidade.adm_pressao[s].sum())
    # parâmetros de referência reproduzem a referência
    for filtro in ((df["tipo"] == "FS") & (df["perfil_variado"] == "W") & (df["valor"] == 1.40),
                   (df["tipo"] == "referencia_custo") & (df["valor"] == "mediana")):
        linha = df[filtro].iloc[0]
        assert linha["mesmo_d_star"] and linha["kendall_tau"] == pytest.approx(1.0)


@requer_gerador
def test_sweep_lambda_pulada_sem_modelo_reexecutavel(sintetico, pipelines_sintetico):
    with pytest.warns(UserWarning, match="PULADA"):
        df = sensitivity.sweep_lambda(pipelines_sintetico["referencia"])
    assert df.empty


# ─────────────────────────────────────────────────────────────────────────
# Modelos de balística para dados por estado
# ─────────────────────────────────────────────────────────────────────────
@requer_gerador
def test_adaptador_balistica_reproduz_estados_do_csv():
    designs = _carregar(GERADOR / "dados" / "pareto_estagio2.csv").head(40)
    modelo = scenarios.AdaptadorSimulador(scenarios.carregar_simulador(GERADOR / "balistica.py"))
    simulados = _estados(designs, modelo)
    lidos = _estados(designs)
    for e in config.ESTADOS_NOMES:
        for c in config.ATRIBUTOS:
            np.testing.assert_allclose(simulados[e][c], lidos[e][c], rtol=1e-10, err_msg=f"{e}/{c}")


def test_estados_precomputados_recusa_kappa_fora_da_referencia():
    designs = pd.DataFrame({c: [1.0] for c in config.CRITERIOS_INVARIANTES})
    with pytest.raises(ValueError, match="reexecutar"):
        scenarios.EstadosPrecomputados()(designs, kc=0.95, kcf=0.9)


# ─────────────────────────────────────────────────────────────────────────
# Vencedor de cada perfil vem do ANP: argmax de g^(p) em F^(p)
# ─────────────────────────────────────────────────────────────────────────
def _argmax_restrito(score, chi):
    idx = np.flatnonzero(chi)
    return int(idx[np.argmax(score[idx])])


def test_vencedor_do_perfil_e_argmax_de_g_em_F(resultados_pipeline):
    ref, _, _ = resultados_pipeline
    for s in ref.cfg.siglas:
        chi = ref.admissibilidade.chi[s]
        k = ref.vencedor_perfil(s)
        assert chi[k]
        assert k == _argmax_restrito(ref.anp_resultado.g_por_perfil[s], chi)
        # comparação de pesos iguais continua disponível e também respeita F^(p)
        k_iguais = ref.vencedor_pesos_iguais(s)
        assert chi[k_iguais] and k_iguais == _argmax_restrito(ref.preference_vectors[s], chi)


def test_vencedor_sem_restricoes_e_argmax_de_g_sem_mascara(resultados_pipeline):
    ref, sem, _ = resultados_pipeline
    for s in ref.cfg.siglas:
        g_livre = ref.g_perfil_sem_restricoes(s)
        np.testing.assert_allclose(g_livre.sum(), 1.0)
        assert ref.vencedor_perfil(s, com_restricoes=False) == int(np.argmax(g_livre))
        # sem restrições e sem perfis inadmissíveis: igual ao pipeline com FS = 1, rho = inf
        np.testing.assert_array_equal(g_livre, sem.anp_resultado.g_por_perfil[s])


@requer_gerador
def test_vencedor_do_perfil_no_gabarito(pipelines_sintetico):
    ref = pipelines_sintetico["referencia"]
    for s in "WSH":
        chi = ref.admissibilidade.chi[s]
        k = ref.vencedor_perfil(s)
        assert chi[k] and k == _argmax_restrito(ref.anp_resultado.g_por_perfil[s], chi)


@requer_gerador
def test_F_vazio_vencedor_nenhum_e_tabela_nao_quebra(pipelines_sintetico):
    import run_pipeline

    res = pipelines_sintetico["W_vazio"]
    with pytest.warns(UserWarning, match="nenhum"):
        assert res.vencedor_perfil("W") is None
    assert res.vencedor_pesos_iguais("W") is None
    tabela = run_pipeline.tabela_vencedores(res).set_index("perfil")
    assert tabela.loc["W", "vencedor_anp"] == "nenhum"
    assert tabela.loc["W", "vencedor_pesos_iguais"] == "nenhum"
    assert tabela.loc["W", "divergem"] == "não"
    assert bool(tabela.loc["W", "excluido_da_sintese"])
    assert tabela.loc["W", "vencedor_anp_sem_restricoes"] != "nenhum"
    for s in "SH":
        assert tabela.loc[s, "vencedor_anp"] == res.design_ids[res.vencedor_perfil(s)]


def test_base_custo_pelo_volume_da_camara_quando_ha_rho_e_eta(caso_manual):
    _, atributos = caso_manual
    # Vc = Mp/(rho·eta) = [1, 1, 2, 0.25]; base = pcmax(θ1)·Vc
    designs = pd.DataFrame({"Mp": [1.0, 2.0, 4.0, 1.0], "rho": [1.0, 2.0, 2.0, 2.0], "eta": [1.0, 1.0, 1.0, 2.0],
                            "Rc": [9.0] * 4, "Lt": [9.0] * 4})  # Rc/Lt ignorados quando há rho e eta
    adm = admissibility.conjuntos_admissiveis(designs, atributos, PERFIS, config.ConfigGlobal())
    np.testing.assert_allclose(adm.base_custo, [11.0, 11.5, 20.0, 3.25])


def test_base_custo_sem_rho_usa_envelope(caso_manual):
    designs, atributos = caso_manual
    adm = admissibility.conjuntos_admissiveis(designs, atributos, PERFIS, config.ConfigGlobal())
    np.testing.assert_allclose(adm.base_custo, [11.0, 11.5, 20.0, 3.25])


def test_pcmax_fora_dos_criterios_continua_no_filtro_e_no_custo():
    designs = data.synthetic_designs(1, seed=5, n=120)
    r6 = pipeline.run(config.Config(estagio=1), designs)
    r7 = pipeline.run(config.Config(estagio=1, criterios=config.VARIANTES_CRITERIOS[7]), designs)
    assert r6.X.shape[0] == 6 and r7.X.shape[0] == 7
    assert list(r6.scores["W"].shape) == [120, 6]
    for s in r6.cfg.siglas:
        np.testing.assert_array_equal(r6.admissibilidade.chi[s], r7.admissibilidade.chi[s])
    np.testing.assert_array_equal(r6.admissibilidade.c, r7.admissibilidade.c)
    # o custo e a pressão máxima entre estados continuam dependendo de pcmax
    d = designs.copy()
    d["pcmax"] = d["pcmax"] * 1.1
    np.testing.assert_allclose(
        pipeline.run(config.Config(estagio=1), d).admissibilidade.pcmax_max, r6.admissibilidade.pcmax_max * 1.1
    )


@pytest.mark.parametrize("n_criterios", sorted(config.VARIANTES_CRITERIOS))
def test_limite_por_potencias_confere_em_cada_variante(n_criterios):
    designs = data.synthetic_designs(1, seed=5, n=40)
    criterios = config.VARIANTES_CRITERIOS[n_criterios]
    r = pipeline.run(config.Config(estagio=1, criterios=criterios, verificar_limite_por_potencias=True), designs)
    assert np.isclose(r.g.sum(), 1.0) and np.isclose(r.anp_resultado.q.sum(), 1.0)
