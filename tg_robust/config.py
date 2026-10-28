"""Configuração e constantes do pipeline de seleção robusta de motores sólidos.

Centraliza, em um único lugar, os parâmetros de referência descritos no
Capítulo 3 da TG (``cap3_metodologia_revisado.tex``): requisitos propulsivos
por estágio, fatores dos estados da natureza, parâmetros de decisão
(γ do ANP), os perfis organizacionais (:class:`Perfil`: regra de decisão,
α de Hurwicz, FS, teto de custo ρ e peso π — Seção 3.3.4 de
``TG_revisao_FS_custo.tex``), os parâmetros globais das restrições
(:class:`ConfigGlobal`: p_lim, expoente ``b`` da CER, referência do custo) e o mapeamento Tabela 6 de tipo de
propelente → expoente da lei de queima ``n``.

Não há estado global mutável: ``Config`` é uma dataclass congelada
(``frozen=True``), e qualquer variação de parâmetro (análise de
sensibilidade, perturbação de julgamentos) é feita criando uma nova
instância com ``dataclasses.replace``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# Ordem fixa dos 7 critérios de desempenho (Seção 3.2 / Tabela "criterios_payoff")
# ─────────────────────────────────────────────────────────────────────────
CRITERIOS: tuple[str, ...] = ("Mp", "tb", "It", "F", "rhoIsp", "pcmax", "eta")
N_CRITERIOS = len(CRITERIOS)

# Critérios cujo objetivo é aproximar um requisito (resíduo relativo, Eq. residuo)
CRITERIOS_RESIDUO: tuple[str, ...] = ("Mp", "tb", "It", "F")
# Critério a maximizar diretamente
CRITERIOS_MAXIMIZAR: tuple[str, ...] = ("rhoIsp", "eta")
# Critério a minimizar diretamente (benefício = -valor)
CRITERIOS_MINIMIZAR: tuple[str, ...] = ("pcmax",)

# Critérios invariantes à propagação de cenários (Seção 3.3.1): dependem da
# geometria do grão/câmara, não de c* nem de C_F.
CRITERIOS_INVARIANTES: tuple[str, ...] = ("Mp", "eta")

# Restrições de projeto do artigo de referência (Nascimento et al., 2022)
PCMAX_LIMITE_SUPERIOR: float = 16.0  # MPa, restrição: pcmax < 16
ETA_LIMITE_INFERIOR: float = 0.8  # restrição: eta_v >= 0.8

# Variáveis de projeto do envelope (Nascimento et al., 2022), usadas apenas no
# índice de custo (Seções 3.2 e 3.3.4): raio do grão R_c e comprimento total L_t.
COLUNAS_GEOMETRIA: tuple[str, ...] = ("Rc", "Lt")

# ─────────────────────────────────────────────────────────────────────────
# Requisitos propulsivos por estágio (Tabela "requisitos", MSGRA)
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Requisitos:
    """Requisitos propulsivos de referência de um estágio."""

    Mp: float
    tb: float
    It: float
    F: float


REQUISITOS_ESTAGIO: Mapping[int, Requisitos] = {
    1: Requisitos(Mp=8780.34, tb=129.4, It=2.602e7, F=2.0107e5),
    2: Requisitos(Mp=1655.20, tb=25.4, It=4.926e6, F=1.9394e5),
}

# Número de designs Pareto reportados por Nascimento et al. (2022) por estágio
# (apenas para referência/validação de tamanho de amostra sintética; não é
# usado para forçar um valor nos dados reais).
N_DESIGNS_REFERENCIA: Mapping[int, int] = {1: 632, 2: 1089}

# ─────────────────────────────────────────────────────────────────────────
# Tabela 6 do artigo: tipo de propelente -> expoente da lei de queima n
# ─────────────────────────────────────────────────────────────────────────
TIPO_PROPELENTE_PARA_N: Mapping[int, float] = {1: 0.31, 2: 0.30, 3: 0.35, 4: 0.39}
N_PADRAO: float = 0.31  # usado quando nem 'propelente' nem 'nexp' estão presentes

# ─────────────────────────────────────────────────────────────────────────
# Estados da natureza (Tabela "cenarios"): fatores multiplicativos sobre
# c* (kappa_c*) e C_F (kappa_CF). Ordem fixa: nominal, moderado, severo.
# ─────────────────────────────────────────────────────────────────────────
ESTADOS_NOMES: tuple[str, ...] = ("nominal", "moderado", "severo")
KAPPA_C_REF: tuple[float, float, float] = (1.0, 0.99, 0.98)
KAPPA_CF_REF: tuple[float, float, float] = (1.0, 0.965, 0.93)

# ─────────────────────────────────────────────────────────────────────────
# Parâmetros de decisão de referência que não pertencem a um perfil
# ─────────────────────────────────────────────────────────────────────────
GAMMA_ANP_REF: float = 0.5

# ─────────────────────────────────────────────────────────────────────────
# Perfis organizacionais (Seções 2.4.3 e 3.3.4 de TG_revisao_FS_custo.tex).
# Cada perfil reúne sua regra de decisão e suas restrições de admissibilidade
# (FS e teto de custo rho, aplicados como LIMIARES, não como critérios
# pontuados). O código trata qualquer lista de perfis com a mesma lógica.
# ─────────────────────────────────────────────────────────────────────────
REGRAS_DECISAO: tuple[str, ...] = ("wald", "savage", "hurwicz")


@dataclass(frozen=True)
class Perfil:
    """Perfil organizacional: regra de decisão + restrições de admissibilidade."""

    nome: str  # "Governamental", "Comercial", "New Space"
    sigla: str  # "W", "S", "H"
    regra: str  # "wald", "savage", "hurwicz"
    alpha: float | None  # coeficiente de otimismo, só para "hurwicz"
    FS: float  # fator de segurança mínimo sobre max_j pcmax
    rho: float  # teto do índice de custo relativo c_k (math.inf = sem teto)
    peso_pi: float  # peso na síntese do ANP

    def __post_init__(self) -> None:
        if self.regra not in REGRAS_DECISAO:
            raise ValueError(f"regra deve ser uma de {REGRAS_DECISAO}, recebido {self.regra!r}")
        if self.regra == "hurwicz":
            if self.alpha is None or not (0.0 <= self.alpha <= 1.0):
                raise ValueError(f"perfil {self.sigla}: alpha de Hurwicz deve estar em [0, 1]")
        elif self.alpha is not None:
            raise ValueError(f"perfil {self.sigla}: alpha só se aplica à regra 'hurwicz'")
        if not self.FS > 0.0:
            raise ValueError(f"perfil {self.sigla}: FS deve ser positivo")
        if not self.rho > 0.0:
            # rho = math.inf é válido e equivale a retirar o teto de custo
            raise ValueError(f"perfil {self.sigla}: rho deve ser positivo (ou math.inf)")
        if not self.peso_pi >= 0.0:
            raise ValueError(f"perfil {self.sigla}: peso_pi deve ser não negativo")


PERFIS_REFERENCIA: tuple[Perfil, ...] = (
    Perfil("Governamental", "W", "wald", None, FS=1.40, rho=1.50, peso_pi=1.0 / 3.0),
    Perfil("Comercial", "S", "savage", None, FS=1.25, rho=1.15, peso_pi=1.0 / 3.0),
    Perfil("New Space", "H", "hurwicz", 0.8, FS=1.25, rho=1.25, peso_pi=1.0 / 3.0),
)


def substituir_perfil(perfis: tuple[Perfil, ...], sigla: str, **campos) -> tuple[Perfil, ...]:
    """Nova lista de perfis com os ``campos`` do perfil ``sigla`` alterados."""

    if sigla not in {p.sigla for p in perfis}:
        raise KeyError(f"perfil {sigla!r} não está na lista")
    return tuple(replace(p, **campos) if p.sigla == sigla else p for p in perfis)


def perfis_sem_restricoes(perfis: tuple[Perfil, ...]) -> tuple[Perfil, ...]:
    """Mesmos perfis com ``FS = 1`` e ``rho = inf``: diferem só pela regra de decisão."""

    return tuple(replace(p, FS=1.0, rho=math.inf) for p in perfis)


# ─────────────────────────────────────────────────────────────────────────
# Parâmetros globais das restrições (não dependem do perfil)
# ─────────────────────────────────────────────────────────────────────────
# Expoente b da CER de motores sólidos (TransCost, Koelle 2013).
# PROVISÓRIO: 1.0 até ser trocado pelo valor da CER (aviso emitido enquanto isso).
# TODO: preencher com o valor da CER de motores sólidos do Koelle (2013), com página/tabela.
B_CER_PROVISORIO: float = 1.0
REFERENCIAS_CUSTO: tuple[str, ...] = ("mediana", "minimo")


@dataclass(frozen=True)
class ConfigGlobal:
    """Parâmetros das restrições de admissibilidade comuns a todos os perfis."""

    # Capacidade estrutural de referência do envelope: a restrição pcmax < 16 MPa
    # do artigo reinterpretada como p_lim (FS unitário implícito no modelo original).
    p_lim: float = PCMAX_LIMITE_SUPERIOR  # MPa
    b_cer: float = B_CER_PROVISORIO
    # Referência do índice de custo: c_k = (base_k / ref)^b, com ref = mediana ou
    # mínimo de base_d sobre TODOS os designs do estágio.
    referencia_custo: str = "mediana"

    def __post_init__(self) -> None:
        if not self.p_lim > 0.0:
            raise ValueError("p_lim deve ser positivo")
        if not (self.b_cer > 0.0 and math.isfinite(self.b_cer)):
            raise ValueError("b_cer deve ser positivo e finito")
        if self.referencia_custo not in REFERENCIAS_CUSTO:
            raise ValueError(f"referencia_custo deve ser uma de {REFERENCIAS_CUSTO}")

    @property
    def b_provisorio(self) -> bool:
        return self.b_cer == B_CER_PROVISORIO

# Índice de consistência aleatória de Saaty por ordem da matriz (RI)
RI_SAATY: Mapping[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
}
CR_LIMITE: float = 0.10

# Constante gravitacional padrão usada em Isp = c*.CF/g0 e It = Mp.Isp.g0
G0: float = 9.80665  # m/s^2


@dataclass(frozen=True)
class Config:
    """Configuração imutável de uma execução do pipeline.

    Qualquer variação paramétrica (sensibilidade, perturbação) deve criar uma
    nova instância via ``dataclasses.replace(cfg, campo=novo_valor)`` — nunca
    mutar uma instância existente.
    """

    estagio: int  # 1 ou 2
    seed: int = 0
    perfis: tuple[Perfil, ...] = PERFIS_REFERENCIA
    globais: ConfigGlobal = field(default_factory=ConfigGlobal)
    gamma_anp: float = GAMMA_ANP_REF
    kappa_c: tuple[float, float, float] = KAPPA_C_REF
    kappa_cf: tuple[float, float, float] = KAPPA_CF_REF
    verificar_limite_por_potencias: bool = False

    def __post_init__(self) -> None:
        if self.estagio not in (1, 2):
            raise ValueError(f"estagio deve ser 1 ou 2, recebido {self.estagio!r}")
        if not (0.0 < self.gamma_anp < 1.0):
            raise ValueError("gamma_anp deve estar em (0, 1)")
        if len(self.kappa_c) != 3 or len(self.kappa_cf) != 3:
            raise ValueError("kappa_c e kappa_cf devem ter 3 elementos (um por estado)")
        validar_perfis(self.perfis)

    @property
    def requisitos(self) -> Requisitos:
        return REQUISITOS_ESTAGIO[self.estagio]

    @property
    def siglas(self) -> tuple[str, ...]:
        return tuple(p.sigla for p in self.perfis)

    @property
    def pi(self) -> dict[str, float]:
        """Pesos ``pi`` da síntese, por sigla."""

        return {p.sigla: p.peso_pi for p in self.perfis}


def validar_perfis(perfis: tuple[Perfil, ...]) -> None:
    """Uma lista de perfis válida: não vazia, siglas únicas e ``peso_pi`` somando 1."""

    if len(perfis) == 0:
        raise ValueError("a lista de perfis não pode ser vazia")
    siglas = [p.sigla for p in perfis]
    if len(set(siglas)) != len(siglas):
        raise ValueError(f"siglas de perfil repetidas: {siglas}")
    if abs(sum(p.peso_pi for p in perfis) - 1.0) > 1e-9:
        raise ValueError("os pesos peso_pi dos perfis devem somar 1")


def kappa_escalado(kappa_ref: tuple[float, float, float], lam: float) -> tuple[float, float, float]:
    """Escala a magnitude de degradação dos estados 2 e 3 por ``lam``.

    Implementa ``kappa_j = 1 - lam*(1 - kappa_j_ref)`` (análise de
    sensibilidade, item 4): ``lam=1`` reproduz o valor de referência,
    ``lam=0`` colapsa todos os estados ao nominal (kappa=1).
    """

    return tuple(1.0 - lam * (1.0 - float(k)) for k in kappa_ref)  # type: ignore[return-value]


def aviso_b_cer_provisorio() -> str:
    """Emite (e devolve) o aviso de que o expoente ``b`` da CER ainda é o provisório."""

    aviso = (
        f"Expoente b da CER de motores sólidos (Koelle, TransCost) PROVISÓRIO "
        f"(b={B_CER_PROVISORIO}): o índice de custo c_k e o teto rho_p usam um valor "
        "não definitivo. Troque config.B_CER_PROVISORIO / --b-cer pelo expoente da "
        "CER antes de reportar resultados."
    )
    warnings.warn(aviso, stacklevel=2)
    return aviso


__all__ = [
    "CRITERIOS",
    "N_CRITERIOS",
    "CRITERIOS_RESIDUO",
    "CRITERIOS_MAXIMIZAR",
    "CRITERIOS_MINIMIZAR",
    "CRITERIOS_INVARIANTES",
    "PCMAX_LIMITE_SUPERIOR",
    "ETA_LIMITE_INFERIOR",
    "COLUNAS_GEOMETRIA",
    "Requisitos",
    "REQUISITOS_ESTAGIO",
    "N_DESIGNS_REFERENCIA",
    "TIPO_PROPELENTE_PARA_N",
    "N_PADRAO",
    "ESTADOS_NOMES",
    "KAPPA_C_REF",
    "KAPPA_CF_REF",
    "GAMMA_ANP_REF",
    "REGRAS_DECISAO",
    "Perfil",
    "PERFIS_REFERENCIA",
    "substituir_perfil",
    "perfis_sem_restricoes",
    "B_CER_PROVISORIO",
    "REFERENCIAS_CUSTO",
    "ConfigGlobal",
    "RI_SAATY",
    "CR_LIMITE",
    "G0",
    "Config",
    "validar_perfis",
    "kappa_escalado",
    "aviso_b_cer_provisorio",
]
