import dataclasses
import math

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
        {"estagio": 1, "gamma_anp": 0.0},
        {"estagio": 1, "gamma_anp": 1.0},
        {"estagio": 1, "kappa_c": (1.0, 0.99)},
        {"estagio": 1, "perfis": ()},
        # pesos pi que não somam 1
        {"estagio": 1, "perfis": tuple(dataclasses.replace(p, peso_pi=0.5) for p in config.PERFIS_REFERENCIA)},
        # siglas repetidas
        {"estagio": 1, "perfis": (config.PERFIS_REFERENCIA[0], dataclasses.replace(config.PERFIS_REFERENCIA[0], peso_pi=0.0))},
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


def test_perfis_referencia():
    por_sigla = {p.sigla: p for p in config.PERFIS_REFERENCIA}
    assert (por_sigla["W"].FS, por_sigla["S"].FS, por_sigla["H"].FS) == (1.40, 1.25, 1.25)
    assert (por_sigla["W"].rho, por_sigla["S"].rho, por_sigla["H"].rho) == (1.50, 1.15, 1.25)
    assert por_sigla["H"].regra == "hurwicz" and por_sigla["H"].alpha == 0.8
    assert sum(p.peso_pi for p in config.PERFIS_REFERENCIA) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "campos",
    [
        {"regra": "laplace"},
        {"regra": "hurwicz", "alpha": None},
        {"regra": "hurwicz", "alpha": 1.5},
        {"regra": "wald", "alpha": 0.5},
        {"FS": 0.0},
        {"rho": 0.0},
        {"peso_pi": -0.1},
    ],
)
def test_perfil_invalido_levanta_erro(campos):
    base = dict(nome="X", sigla="X", regra="wald", alpha=None, FS=1.25, rho=1.25, peso_pi=1.0)
    with pytest.raises(ValueError):
        config.Perfil(**{**base, **campos})


def test_perfil_rho_infinito_e_valido():
    p = config.Perfil("X", "X", "savage", None, FS=1.0, rho=math.inf, peso_pi=1.0)
    assert p.rho == math.inf


def test_substituir_perfil_e_sem_restricoes():
    novos = config.substituir_perfil(config.PERFIS_REFERENCIA, "H", FS=1.15)
    assert {p.sigla: p.FS for p in novos} == {"W": 1.40, "S": 1.25, "H": 1.15}
    assert config.PERFIS_REFERENCIA[2].FS == 1.25  # original intacto
    livres = config.perfis_sem_restricoes(config.PERFIS_REFERENCIA)
    assert all(p.FS == 1.0 and p.rho == math.inf for p in livres)
    with pytest.raises(KeyError):
        config.substituir_perfil(config.PERFIS_REFERENCIA, "Z", FS=1.0)


@pytest.mark.parametrize(
    "kwargs", [{"p_lim": 0.0}, {"b_cer": 0.0}, {"b_cer": math.inf}, {"referencia_custo": "media"}]
)
def test_config_global_invalida(kwargs):
    with pytest.raises(ValueError):
        config.ConfigGlobal(**kwargs)


def test_criterios_padrao_sem_pcmax_mas_atributo_mantido():
    assert config.Config(estagio=1).criterios == config.CRITERIOS == config.VARIANTES_CRITERIOS[6]
    assert "pcmax" not in config.CRITERIOS and "pcmax" in config.ATRIBUTOS
    assert config.REQUISITOS_ESTAGIO[1].Mp == 8780.34


@pytest.mark.parametrize("criterios", [(), ("Mp", "Mp"), ("Mp", "pc_med")])
def test_config_recusa_criterios_invalidos(criterios):
    with pytest.raises(ValueError, match="criterios"):
        config.Config(estagio=1, criterios=criterios)


def test_config_global_padrao_mediana_e_b_provisorio():
    g = config.ConfigGlobal()
    assert g.referencia_custo == "mediana" and g.p_lim == 16.0
    assert g.b_provisorio
    assert not config.ConfigGlobal(b_cer=0.7).b_provisorio
