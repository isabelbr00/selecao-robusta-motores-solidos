import pytest

from tg_robust import config


def test_requisitos_por_estagio():
    assert config.REQUISITOS_ESTAGIO[1].Mp == pytest.approx(8780.34)
    assert config.REQUISITOS_ESTAGIO[2].It == pytest.approx(4.926e6)


def test_config_valido():
    cfg = config.Config(estagio=1)
    assert cfg.estagio == 1
    assert cfg.requisitos.Mp == pytest.approx(8780.34)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estagio": 3},
        {"estagio": 1, "alpha_hurwicz": 1.5},
        {"estagio": 1, "gamma_anp": 0.0},
        {"estagio": 1, "gamma_anp": 1.0},
        {"estagio": 1, "pi_perfis": (0.5, 0.5, 0.5)},
        {"estagio": 1, "kappa_c": (1.0, 0.99)},
    ],
)
def test_config_invalido_levanta_erro(kwargs):
    with pytest.raises(ValueError):
        config.Config(**kwargs)


def test_config_e_imutavel():
    cfg = config.Config(estagio=1)
    with pytest.raises(Exception):
        cfg.estagio = 2  # type: ignore[misc]


def test_kappa_escalado_lambda_1_reproduz_referencia():
    novo = config.kappa_escalado(config.KAPPA_C_REF, 1.0)
    assert novo == pytest.approx(config.KAPPA_C_REF)


def test_kappa_escalado_lambda_0_colapsa_ao_nominal():
    novo = config.kappa_escalado(config.KAPPA_C_REF, 0.0)
    assert novo == pytest.approx((1.0, 1.0, 1.0))


def test_kappa_escalado_lambda_2_dobra_degradacao():
    ref = (1.0, 0.99, 0.98)
    novo = config.kappa_escalado(ref, 2.0)
    # kappa_j = 1 - 2*(1-kappa_ref) => degradação duplicada
    assert novo[1] == pytest.approx(1.0 - 2 * 0.01)
    assert novo[2] == pytest.approx(1.0 - 2 * 0.02)
