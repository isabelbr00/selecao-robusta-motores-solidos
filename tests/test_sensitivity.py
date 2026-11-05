import numpy as np
import pytest

from tg_robust import config, data, pipeline, sensitivity


@pytest.fixture
def resultado_pequeno():
    df = data.synthetic_designs(1, seed=21, n=16)
    cfg = config.Config(estagio=1, seed=21)
    return pipeline.run(cfg, df)


def test_kendall_tau_1_para_ranking_identico(resultado_pequeno):
    g = resultado_pequeno.g
    metricas = sensitivity.metricas_estabilidade(g, g)
    assert metricas["kendall_tau"] == pytest.approx(1.0)
    assert metricas["mesmo_d_star"] is True
    assert metricas["posicao_d_star_ref"] == 1
    assert metricas["intersecao_top10"] == min(10, len(g))


def test_reprodutibilidade_mesma_semente():
    df1 = data.synthetic_designs(1, seed=99, n=20)
    df2 = data.synthetic_designs(1, seed=99, n=20)
    cfg = config.Config(estagio=1, seed=99)
    r1 = pipeline.run(cfg, df1)
    r2 = pipeline.run(cfg, df2)
    assert np.allclose(r1.g, r2.g)
    assert r1.d_star == r2.d_star


def test_sweep_alpha_inclui_extremos_e_reproduz_wald(resultado_pequeno):
    df = sensitivity.sweep_alpha(resultado_pequeno, alphas=(0.0, 0.5, 1.0))
    assert set(df["alpha"]) == {0.0, 0.5, 1.0}
    assert set(df.columns) >= {"alpha", "mesmo_d_star", "kendall_tau", "intersecao_top10", "posicao_d_star_ref"}


def test_sweep_pi_inclui_canonicos(resultado_pequeno):
    df = sensitivity.sweep_pi(resultado_pequeno)
    canonicos = df[df["tipo"] == "canonico"]
    assert len(canonicos) == 3
    assert set(canonicos["perfil_variado"]) == set(resultado_pequeno.cfg.siglas)


def test_sweep_anp_inclui_gamma_zero(resultado_pequeno):
    df = sensitivity.sweep_anp(resultado_pequeno, gammas=(0.3, 0.5, 0.7), incluir_gamma_zero=True)
    assert (df.loc[df["tipo"] == "gamma", "passo_saaty"] == 0.0).any()
    assert (df["tipo"] == "julgamento").any()


def test_sweep_lambda_lambda_1_reproduz_referencia(resultado_pequeno):
    df = sensitivity.sweep_lambda(resultado_pequeno, lambdas=(1.0,))
    linha = df.iloc[0]
    assert bool(linha["mesmo_d_star"])
    assert linha["kendall_tau"] == pytest.approx(1.0)


def test_regiao_robustez_filtra_por_mesmo_d_star(resultado_pequeno):
    df = sensitivity.sweep_alpha(resultado_pequeno)
    regiao = sensitivity.regiao_robustez(df, "alpha")
    assert all(df.loc[df["alpha"].isin(regiao), "mesmo_d_star"])


def test_sweep_residuo_referencia_reproduz_e_alternativa_reexecuta(resultado_pequeno):
    import dataclasses

    df = sensitivity.sweep_residuo(resultado_pequeno)
    assert list(df["forma_residuo"]) == list(config.FORMAS_RESIDUO)
    ref = df[df["referencia"]].iloc[0]
    assert ref["forma_residuo"] == "quadrado"
    assert ref["kendall_tau"] == pytest.approx(1.0) and bool(ref["mesmo_d_star"])

    # a linha "absoluto" equivale a rodar o pipeline com forma_residuo="absoluto"
    cfg_abs = dataclasses.replace(
        resultado_pequeno.cfg, globais=dataclasses.replace(resultado_pequeno.cfg.globais, forma_residuo="absoluto")
    )
    r_abs = pipeline.run(cfg_abs, resultado_pequeno.designs)
    alt = df[df["forma_residuo"] == "absoluto"].iloc[0]
    assert alt["d_star"] == resultado_pequeno.design_ids[r_abs.d_star]
    assert alt["kendall_tau"] == pytest.approx(sensitivity.metricas_estabilidade(resultado_pequeno.g, r_abs.g)["kendall_tau"])


def test_forma_residuo_invalida():
    with pytest.raises(ValueError):
        config.ConfigGlobal(forma_residuo="cubo")


def test_cenarios_fs_referencia_reproduz_a_execucao(resultado_pequeno):
    df = sensitivity.sweep_cenarios_fs(resultado_pequeno).set_index("cenario")
    assert list(df.index) == ["referencia", "todos_1,25", "todos_1,40"]
    ref = df.loc["referencia"]
    ids = resultado_pequeno.design_ids
    assert ref["d_star"] == ids[resultado_pequeno.d_star]
    assert ref["kendall_tau"] == pytest.approx(1.0)
    assert ref["top10"] == ";".join(str(i) for i in ids[np.argsort(-resultado_pequeno.g)[:10]])
    for s in resultado_pequeno.cfg.siglas:
        assert ref[f"vencedor_{s}"] == ids[resultado_pequeno.vencedor_perfil(s)]
    assert df.loc["todos_1,25", "n_F_W"] >= ref["n_F_W"] >= 0
    assert df.loc["todos_1,40", "n_F_S"] <= ref["n_F_S"]
