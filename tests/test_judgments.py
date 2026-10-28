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


def test_from_intensities_valores_na_escala():
    j = J.from_intensities("Mp", {"It": 9, "rhoIsp": 5, "tb": 3, "eta": 3})
    for i in range(len(j.labels)):
        for k in range(len(j.labels)):
            assert j.A[i, k] in J.SAATY_SCALE


def test_build_WTT_placeholder_emite_aviso_e_retorna_estocastica_por_coluna():
    with pytest.warns(UserWarning, match="PLACEHOLDER"):
        W_TT, juls = J.build_WTT_placeholder()
    assert W_TT.shape == (config.N_CRITERIOS, config.N_CRITERIOS)
    assert np.allclose(W_TT.sum(axis=0), 1.0)
    assert (W_TT >= 0).all()


def test_build_WTT_placeholder_consistencia_dentro_do_limite():
    with pytest.warns(UserWarning):
        _, juls = J.build_WTT_placeholder()
    relatorio = J.consistency_report(juls)
    excedentes = relatorio[relatorio["cr_excede_limite"]]
    assert len(excedentes) == 0, (
        f"Julgamento(s) do rascunho PLACEHOLDER excedem CR>0,10, reportar e ajustar: "
        f"{excedentes[['alvo', 'CR']].to_dict('records')}"
    )


def test_perturb_preserva_reciprocidade_e_escala():
    with pytest.warns(UserWarning):
        _, juls = J.build_WTT_placeholder()
    por_alvo = J.julgamentos_por_alvo(juls)

    novo = J.perturb(por_alvo, "Mp", "It", "tb", step=1)
    labels = novo["Mp"].labels
    i, k = labels.index("It"), labels.index("tb")
    A = novo["Mp"].A
    assert A[i, k] == pytest.approx(1.0 / A[k, i])
    assert A[i, k] in J.SAATY_SCALE


def test_perturb_nao_sai_dos_limites_da_escala():
    with pytest.warns(UserWarning):
        _, juls = J.build_WTT_placeholder()
    por_alvo = J.julgamentos_por_alvo(juls)

    atual = por_alvo
    for _ in range(50):  # muito mais passos que a escala permite
        atual = J.perturb(atual, "Mp", "It", "tb", step=1)
    labels = atual["Mp"].labels
    i, k = labels.index("It"), labels.index("tb")
    assert atual["Mp"].A[i, k] == pytest.approx(max(J.SAATY_SCALE))


def test_perturb_coluna_invalida_levanta_erro():
    with pytest.warns(UserWarning):
        _, juls = J.build_WTT_placeholder()
    por_alvo = J.julgamentos_por_alvo(juls)
    with pytest.raises(KeyError):
        J.perturb(por_alvo, "nao_existe", "It", "tb", step=1)


def test_WTT_de_julgamentos_reconstroi_matriz_original():
    with pytest.warns(UserWarning):
        W_TT, juls = J.build_WTT_placeholder()
    por_alvo = J.julgamentos_por_alvo(juls)
    W_TT_reconstruida = J.WTT_de_julgamentos(por_alvo)
    assert np.allclose(W_TT, W_TT_reconstruida)
