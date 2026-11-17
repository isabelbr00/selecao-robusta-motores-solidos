import numpy as np
import pytest

from selecao_robusta import data
from selecao_robusta.scenarios import SurrogateBallistics, propagate_states
from selecao_robusta import config


@pytest.fixture
def designs():
    return data.synthetic_designs(1, seed=3, n=40)


def test_mp_e_eta_invariantes(designs):
    model = SurrogateBallistics()
    saida = model(designs, kc=0.98, kcf=0.93)
    assert np.allclose(saida["Mp"], designs["Mp"])
    assert np.allclose(saida["eta"], designs["eta"])


def test_identidade_F_igual_It_sobre_tb(designs):
    model = SurrogateBallistics()
    saida = model(designs, kc=0.99, kcf=0.965)
    assert np.allclose(saida["F"], saida["It"] / saida["tb"])


def test_identidade_It_igual_It_vezes_kc_kcf(designs):
    model = SurrogateBallistics()
    kc, kcf = 0.98, 0.93
    saida = model(designs, kc=kc, kcf=kcf)
    assert np.allclose(saida["It"], designs["It"].to_numpy() * kc * kcf)


def test_identidade_rhoIsp(designs):
    model = SurrogateBallistics()
    kc, kcf = 0.99, 0.965
    saida = model(designs, kc=kc, kcf=kcf)
    assert np.allclose(saida["rhoIsp"], designs["rhoIsp"].to_numpy() * kc * kcf)


def test_identidade_pcmax_depende_apenas_de_kc(designs):
    """pcmax depende de kp (função de kc), não de kcf."""
    model = SurrogateBallistics()
    n = designs["nexp"].to_numpy()
    kc = 0.98
    kp = kc ** (1.0 / (1.0 - n))
    for kcf in (1.0, 0.93, 0.5):
        saida = model(designs, kc=kc, kcf=kcf)
        assert np.allclose(saida["pcmax"], designs["pcmax"].to_numpy() * kp)


def test_estado_nominal_nao_altera_nada(designs):
    model = SurrogateBallistics()
    saida = model(designs, kc=1.0, kcf=1.0)
    for criterio in config.ATRIBUTOS:
        assert np.allclose(saida[criterio], designs[criterio].to_numpy())


def test_formula_errada_do_prototipo_antigo_diverge(designs):
    """Um protótipo antigo usava It' = It*kcf*kp (errado). Verifica que isso
    difere da identidade correta It' = It*kc*kcf implementada aqui."""

    model = SurrogateBallistics()
    kc, kcf = 0.98, 0.93
    saida = model(designs, kc=kc, kcf=kcf)

    n = designs["nexp"].to_numpy()
    kp = kc ** (1.0 / (1.0 - n))
    It_formula_errada = designs["It"].to_numpy() * kcf * kp

    assert not np.allclose(saida["It"], It_formula_errada)


def test_propagate_states_tres_estados(designs):
    resultado = propagate_states(designs, config.KAPPA_C_REF, config.KAPPA_CF_REF, SurrogateBallistics())
    assert set(resultado.keys()) == set(config.ESTADOS_NOMES)
    for estado in config.ESTADOS_NOMES:
        assert set(resultado[estado].keys()) == set(config.ATRIBUTOS)


# ─── EscalaQuaseEstacionaria (tabela do modeFRONTIER) ─────────────────────
from selecao_robusta.scenarios import EscalaQuaseEstacionaria  # noqa: E402


@pytest.fixture
def designs_mf(designs):
    # F fora de It/tb e ta/pcmed presentes, como na tabela do modeFRONTIER
    d = designs.copy()
    d["F"] = d["F"] * 1.03
    d["ta"] = d["tb"] * 1.07
    d["pcmed"] = d["pcmax"] * 0.45
    return d


def test_escala_theta1_identica_aos_dados(designs_mf):
    saida = EscalaQuaseEstacionaria()(designs_mf, kc=1.0, kcf=1.0)
    for c, v in saida.items():
        assert np.array_equal(v, designs_mf[c].to_numpy(dtype=float)), c


def test_escala_leis(designs_mf):
    kc, kcf = 0.98, 0.93
    n = designs_mf["nexp"].to_numpy()
    s = EscalaQuaseEstacionaria()(designs_mf, kc=kc, kcf=kcf)
    col = lambda c: designs_mf[c].to_numpy(dtype=float)  # noqa: E731
    assert np.allclose(s["rhoIsp"], col("rhoIsp") * kc * kcf)
    assert np.allclose(s["It"], col("It") * kc * kcf)
    assert np.allclose(s["pcmax"], col("pcmax") * kc ** (1 / (1 - n)))
    assert np.allclose(s["pcmed"], col("pcmed") * kc ** (1 / (1 - n)))
    assert np.allclose(s["tb"], col("tb") * kc ** (-n / (1 - n)))
    assert np.allclose(s["ta"], col("ta") * kc ** (-n / (1 - n)))
    assert np.allclose(s["F"], col("F") * kcf * kc ** (1 / (1 - n)))
    assert np.array_equal(s["Mp"], col("Mp")) and np.array_equal(s["eta"], col("eta"))


def test_escala_coincide_com_substituto_quando_F_igual_It_sobre_tb(designs):
    # sem ta/pcmed e com F = It/tb, as leis de escala coincidem com SurrogateBallistics
    a = EscalaQuaseEstacionaria()(designs, kc=0.99, kcf=0.965)
    b = SurrogateBallistics()(designs, kc=0.99, kcf=0.965)
    assert set(a) == set(b)
    for c in b:
        assert np.allclose(a[c], b[c]), c
