import numpy as np
import pytest

from tg_robust import data
from tg_robust.scenarios import SurrogateBallistics, propagate_states
from tg_robust import config


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
    for criterio in config.CRITERIOS:
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
        assert set(resultado[estado].keys()) == set(config.CRITERIOS)
