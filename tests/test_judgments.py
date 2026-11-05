import warnings

import numpy as np
import pytest

from tg_robust import config, judgments as J


def test_snap_to_saaty_scale_valores_exatos():
    for v in J.SAATY_SCALE:
        assert J.snap_to_saaty_scale(v) == pytest.approx(v)


def test_snap_to_saaty_scale_reciprocidade_simetrica():
    assert J.snap_to_saaty_scale(1 / 2.4) == pytest.approx(J.snap_to_saaty_scale(1 / 2))


def test_from_intensities_reciprocidade():
    j = J.from_intensities("Mp", {"It": 9, "rhoIsp": 5, "tb": 3, "eta": 3})
    n = len(j.labels)
    for i in range(n):
        assert j.A[i, i] == pytest.approx(1.0)
        for k in range(n):
            assert j.A[i, k] == pytest.approx(1.0 / j.A[k, i])


def test_from_intensities_razoes_exatas_sem_arredondar():
    intens = {"It": 9, "rhoIsp": 7, "tb": 5, "eta": 3}
    j = J.from_intensities("Mp", intens)
    s = np.array([intens[l] for l in j.labels], dtype=float)
    np.testing.assert_allclose(j.A, s[:, None] / s[None, :])
    assert j.A[0, 1] == pytest.approx(9 / 7)  # antes arredondado para 1


def test_from_intensities_prioridades_sao_intensidades_normalizadas_e_RC_zero():
    intens = {"It": 9, "rhoIsp": 7, "F": 5, "tb": 5}
    j = J.from_intensities("It", intens)
    w, lam = J.eigen_priorities(j.A)
    s = np.array([intens[l] for l in j.labels], dtype=float)
    np.testing.assert_allclose(w, s / s.sum(), atol=1e-12)
    _, _, CR = J.consistency(j.A)
    assert abs(CR) < 1e-12
    assert lam == pytest.approx(len(intens))


def test_from_intensities_um_unico_influenciador():
    j = J.from_intensities("eta", {"Mp": 7})
    w, _ = J.eigen_priorities(j.A)
    assert w == pytest.approx([1.0])
    assert J.consistency(j.A)[2] == 0.0


def test_perturb_valor_entre_niveis_vai_ao_nivel_vizinho():
    por_alvo = {"X": J.from_intensities("X", {"a": 9, "b": 5})}  # A_ab = 1,8
    acima = J.perturb(por_alvo, "X", "a", "b", step=1)["X"].A
    abaixo = J.perturb(por_alvo, "X", "a", "b", step=-1)["X"].A
    assert acima[0, 1] == pytest.approx(2.0) and acima[1, 0] == pytest.approx(0.5)
    assert abaixo[0, 1] == pytest.approx(1.0)


def test_build_WTT_validada_nao_emite_aviso_e_retorna_estocastica_por_coluna():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        W_TT, juls = J.build_WTT_validada()
    assert W_TT.shape == (config.N_CRITERIOS, config.N_CRITERIOS)
    assert np.allclose(W_TT.sum(axis=0), 1.0)
    assert (W_TT >= 0).all()


def test_build_WTT_validada_consistencia_dentro_do_limite():
    _, juls = J.build_WTT_validada()
    relatorio = J.consistency_report(juls)
    excedentes = relatorio[relatorio["cr_excede_limite"]]
    assert len(excedentes) == 0, (
        f"Julgamento(s) de W_TT excedem CR>0,10, reportar e ajustar: "
        f"{excedentes[['alvo', 'CR']].to_dict('records')}"
    )


def test_perturb_preserva_reciprocidade_e_escala():
    _, juls = J.build_WTT_validada()
    por_alvo = J.julgamentos_por_alvo(juls)

    novo = J.perturb(por_alvo, "Mp", "It", "tb", step=1)
    labels = novo["Mp"].labels
    i, k = labels.index("It"), labels.index("tb")
    A = novo["Mp"].A
    assert A[i, k] == pytest.approx(1.0 / A[k, i])
    assert A[i, k] in J.SAATY_SCALE


def test_perturb_nao_sai_dos_limites_da_escala():
    _, juls = J.build_WTT_validada()
    por_alvo = J.julgamentos_por_alvo(juls)

    atual = por_alvo
    for _ in range(50):  # muito mais passos que a escala permite
        atual = J.perturb(atual, "Mp", "It", "tb", step=1)
    labels = atual["Mp"].labels
    i, k = labels.index("It"), labels.index("tb")
    assert atual["Mp"].A[i, k] == pytest.approx(max(J.SAATY_SCALE))


def test_perturb_coluna_invalida_levanta_erro():
    _, juls = J.build_WTT_validada()
    por_alvo = J.julgamentos_por_alvo(juls)
    with pytest.raises(KeyError):
        J.perturb(por_alvo, "nao_existe", "It", "tb", step=1)


def test_WTT_de_julgamentos_reconstroi_matriz_original():
    W_TT, juls = J.build_WTT_validada()
    por_alvo = J.julgamentos_por_alvo(juls)
    W_TT_reconstruida = J.WTT_de_julgamentos(por_alvo)
    assert np.allclose(W_TT, W_TT_reconstruida)


def test_toda_relacao_de_WTT_tem_justificativa_e_vice_versa():
    relacoes = {(alvo, infl) for alvo, d in J.INFLUENCIAS.items() for infl in d}
    assert set(J.JUSTIFICATIVAS) == relacoes
    assert all(J.JUSTIFICATIVAS[r].strip() for r in relacoes)


def test_WTT_padrao_sem_pcmax_e_colunas_renormalizadas():
    W_TT, _ = J.build_WTT_validada()
    assert config.CRITERIOS == ("Mp", "tb", "It", "F", "rhoIsp", "eta")
    assert W_TT.shape == (6, 6)
    assert np.allclose(W_TT.sum(axis=0), 1.0)
    i = {c: k for k, c in enumerate(config.CRITERIOS)}
    # tb: {F: 5, It: 3, Mp: 3} sem pcmax; F: {It: 5, tb: 5, rhoIsp: 3} sem pcmax
    assert W_TT[i["F"], i["tb"]] == pytest.approx(5 / 11)
    assert W_TT[i["It"], i["tb"]] == pytest.approx(3 / 11)
    assert W_TT[i["Mp"], i["F"]] == 0.0
    assert W_TT[i["It"], i["F"]] == pytest.approx(5 / 13)
    assert W_TT[i["rhoIsp"], i["F"]] == pytest.approx(3 / 13)


def test_rhoIsp_sem_influenciadores_recebe_auto_laco():
    W_TT, juls = J.build_WTT_validada()
    r = config.CRITERIOS.index("rhoIsp")
    esperado = np.zeros(len(config.CRITERIOS))
    esperado[r] = 1.0
    np.testing.assert_array_equal(W_TT[:, r], esperado)
    assert np.allclose(J.WTT_de_julgamentos(J.julgamentos_por_alvo(juls)), W_TT)
    assert J.consistency_report(juls).set_index("alvo").loc["rhoIsp", "CR"] == 0.0


def test_variante_7_criterios_mantem_todas_as_relacoes():
    W_TT, _ = J.build_WTT_validada(config.VARIANTES_CRITERIOS[7])
    relacoes = {(alvo, infl) for alvo, d in J.INFLUENCIAS.items() for infl in d}
    i = {c: k for k, c in enumerate(config.ATRIBUTOS)}
    assert {(a, b) for a in i for b in i if W_TT[i[b], i[a]] > 0} == relacoes


def test_variante_5_criterios_tem_auto_laco_em_rhoIsp():
    criterios = config.VARIANTES_CRITERIOS[5]
    W_TT, _ = J.build_WTT_validada(criterios)
    r = criterios.index("rhoIsp")
    assert W_TT.shape == (5, 5) and W_TT[r, r] == 1.0
    assert np.allclose(W_TT.sum(axis=0), 1.0)
