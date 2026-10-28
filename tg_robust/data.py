"""Carregamento e geração de designs Pareto para a seleção robusta de motores.

Duas fontes de dados são suportadas:

* :func:`load_designs` — carrega um CSV real de designs Pareto (7 critérios
  + variáveis de projeto ``Rc``/``Lt`` do envelope) e valida as restrições
  de projeto do artigo de referência. Os arquivos ``TG/data/pareto_estagio_*.csv``
  pertencem à metodologia anterior e **não** devem ser usados (não têm
  ``Rc``/``Lt`` e trazem ``pc_med``, não ``pcmax``).
* :func:`synthetic_designs` — gera, de forma determinística (semente
  explícita), um conjunto de designs sintéticos plausíveis para permitir o
  desenvolvimento e teste do pipeline enquanto os dados reais de
  Nascimento et al. (2022) não estão disponíveis. A saída é marcada
  explicitamente como sintética (coluna ``sintetico=True`` + aviso de log).

Nenhum valor do artigo de referência (em particular a Tabela 7, resultados
do AHP) é usado nesta pasta.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, scenarios

logger = logging.getLogger(__name__)

# Colunas canônicas exigidas em qualquer DataFrame de designs: os 7 critérios
# e as variáveis de projeto do envelope (índice de custo, Seção 3.3.4).
COLUNAS_CANONICAS: tuple[str, ...] = config.CRITERIOS + config.COLUNAS_GEOMETRIA

# Aliases de nomes de coluna. ``pc_med`` (pressão média, usada nos CSVs da
# metodologia anterior) NÃO é aceito como ``pcmax``: a restrição de FS exige
# a pressão máxima de câmara.
ALIASES: dict[str, str] = {
    "F_med": "F",
    "Fmed": "F",
    "rho_Isp": "rhoIsp",
    "rho_isp": "rhoIsp",
    "pc_max": "pcmax",
    "eta_v": "eta",
    "etav": "eta",
    "R_c": "Rc",
    "L_t": "Lt",
}

COLUNA_PROPELENTE = "propelente"
COLUNA_NEXP = "nexp"
COLUNA_SINTETICO = "sintetico"


def _renomear_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica :data:`ALIASES` às colunas de ``df`` (sem alterar o original)."""

    renomeio = {c: ALIASES[c] for c in df.columns if c in ALIASES}
    # colunas por estado: <alias>_theta{j} -> <canônico>_theta{j}
    for c in df.columns:
        base, sep, sufixo = c.rpartition("_theta")
        if sep and sufixo in ("1", "2", "3") and base in ALIASES:
            renomeio[c] = f"{ALIASES[base]}_theta{sufixo}"
    return df.rename(columns=renomeio)


def tem_estados_precomputados(df: pd.DataFrame) -> bool:
    """Verdadeiro se ``df`` traz colunas ``<criterio>_theta{j}`` (estados já calculados)."""

    return any(f"{c}_theta1" in df.columns for c in scenarios.CRITERIOS_POR_ESTADO)


def _completar_nominal_dos_estados(df: pd.DataFrame) -> pd.DataFrame:
    """No formato por estado, exige os 3 estados e usa ``theta1`` como valor nominal."""

    faltantes = [
        scenarios.coluna_estado(c, j)
        for c in scenarios.CRITERIOS_POR_ESTADO
        for j in (1, 2, 3)
        if scenarios.coluna_estado(c, j) not in df.columns
    ]
    if faltantes:
        raise ValueError(f"Formato por estado incompleto; colunas ausentes: {faltantes}")

    df = df.copy()
    for c in scenarios.CRITERIOS_POR_ESTADO:
        nominal = df[scenarios.coluna_estado(c, 1)]
        if c in df.columns and not np.allclose(df[c], nominal):
            raise ValueError(f"Coluna {c!r} difere de {c}_theta1 (valor nominal)")
        df[c] = nominal
    return df


def resolver_nexp(df: pd.DataFrame) -> pd.Series:
    """Resolve o expoente ``n`` da lei de queima para cada design.

    Ordem de prioridade:

    1. coluna ``nexp`` explícita;
    2. coluna ``propelente`` (tipo 1-4), mapeada pela Tabela 6 do artigo
       (:data:`config.TIPO_PROPELENTE_PARA_N`);
    3. nenhuma das duas presentes: usa :data:`config.N_PADRAO` para todos os
       designs e emite um aviso explícito (``warnings.warn`` + log).
    """

    if COLUNA_NEXP in df.columns:
        return df[COLUNA_NEXP].astype(float)

    if COLUNA_PROPELENTE in df.columns:
        tipos = df[COLUNA_PROPELENTE].astype(int)
        invalidos = sorted(set(tipos.unique()) - set(config.TIPO_PROPELENTE_PARA_N))
        if invalidos:
            raise ValueError(
                f"Tipo(s) de propelente inválido(s) (esperado 1-4): {invalidos}"
            )
        return tipos.map(config.TIPO_PROPELENTE_PARA_N).astype(float)

    aviso = (
        "Nenhuma coluna 'nexp' ou 'propelente' encontrada nos dados; "
        f"usando n={config.N_PADRAO} (padrão) para todos os designs. "
        "Isso afeta apenas a propagação de cenários (scenarios.py)."
    )
    warnings.warn(aviso, stacklevel=2)
    logger.warning(aviso)
    return pd.Series(config.N_PADRAO, index=df.index, dtype=float)


def validar_restricoes(df: pd.DataFrame, *, levantar_erro: bool = False) -> dict[str, int]:
    """Verifica as restrições de projeto do artigo de referência.

    Restrições: ``pcmax < 16`` MPa e ``eta >= 0,8``. Retorna um dicionário
    com a contagem de violações de cada restrição. Se ``levantar_erro`` for
    verdadeiro, lança ``ValueError`` quando houver qualquer violação; caso
    contrário, apenas registra um aviso.
    """

    viol_pcmax = int((df["pcmax"] >= config.PCMAX_LIMITE_SUPERIOR).sum())
    viol_eta = int((df["eta"] < config.ETA_LIMITE_INFERIOR).sum())
    relatorio = {"pcmax_violacoes": viol_pcmax, "eta_violacoes": viol_eta}

    if viol_pcmax or viol_eta:
        msg = (
            f"Restrições violadas: {viol_pcmax} design(s) com pcmax >= "
            f"{config.PCMAX_LIMITE_SUPERIOR} MPa; {viol_eta} design(s) com "
            f"eta < {config.ETA_LIMITE_INFERIOR}."
        )
        if levantar_erro:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)

    return relatorio


def _validar_colunas(df: pd.DataFrame) -> None:
    faltantes = [c for c in COLUNAS_CANONICAS if c not in df.columns]
    if faltantes:
        msg = (
            f"Colunas obrigatórias ausentes no CSV de designs: {faltantes}. "
            f"Esperado (após aliases): {COLUNAS_CANONICAS}"
        )
        if set(faltantes) & set(config.COLUNAS_GEOMETRIA):
            msg += (
                ". Rc (raio do grão) e Lt (comprimento total do grão) são variáveis "
                "de projeto de Nascimento et al. (2022) exigidas pelo índice de custo; "
                "não há valor padrão."
            )
        raise ValueError(msg)

    geometria = df[list(config.COLUNAS_GEOMETRIA)].to_numpy(dtype=float)
    if not (np.isfinite(geometria).all() and (geometria > 0.0).all()):
        raise ValueError("Rc e Lt devem ser positivos e finitos em todos os designs")


def hash_arquivo(path: str | Path) -> str:
    """Hash SHA-256 do conteúdo do arquivo, para registro em ``run_config.json``."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def load_designs(path: str | Path, *, levantar_erro_restricoes: bool = False) -> pd.DataFrame:
    """Carrega um CSV real de designs Pareto.

    O CSV deve conter, após a resolução de aliases (:data:`ALIASES`), as
    colunas canônicas :data:`COLUNAS_CANONICAS`. No formato por estado
    (colunas ``<criterio>_theta{1,2,3}``, como em ``gerador_pareto/dados``),
    os três estados são exigidos e ``theta1`` vira o valor nominal; os
    estados podem então ser lidos por
    :class:`tg_robust.scenarios.EstadosPrecomputados`. Uma coluna opcional
    ``propelente`` (tipo 1-4) ou ``nexp`` é usada para a propagação de
    cenários; na ausência de ambas, aplica-se o padrão com aviso (ver
    :func:`resolver_nexp`). As restrições de projeto são validadas via
    :func:`validar_restricoes`.
    """

    path = Path(path)
    df = pd.read_csv(path)
    df = _renomear_aliases(df)
    if tem_estados_precomputados(df):
        df = _completar_nominal_dos_estados(df)
    _validar_colunas(df)

    df = df.copy()
    df[COLUNA_NEXP] = resolver_nexp(df)
    df[COLUNA_SINTETICO] = False
    validar_restricoes(df, levantar_erro=levantar_erro_restricoes)

    logger.info("Carregados %d designs reais de %s", len(df), path)
    return df.reset_index(drop=True)


def synthetic_designs(stage: int, seed: int, n: int | None = None) -> pd.DataFrame:
    """Gera designs Pareto sintéticos plausíveis para o estágio dado.

    Os dados são explicitamente marcados como sintéticos (coluna
    ``sintetico=True`` + aviso de log) e **não** devem ser confundidos com
    dados reais de Nascimento et al. (2022); servem apenas para desenvolver
    e testar o pipeline antes da obtenção dos dados reais.

    Construção (determinística dado ``seed``):

    * ``Mp``, ``tb`` dentro de ±6% do requisito do estágio;
    * ``Isp`` ≈ 303 s (pequena variação gaussiana, σ=3 s);
    * ``It = Mp · Isp · g0``; ``F = It / tb``;
    * ``rho_p ~ U[1700, 1800]`` kg/m³; critério ``rhoIsp = rho_p · Isp``;
    * ``pcmax ~ U[5.5, 15.5]`` MPa (restrição de projeto: < 16);
    * ``eta ~ U[0.8, 0.95]``;
    * ``propelente`` sorteado uniformemente em {1,2,3,4} (Tabela 6);
    * ``Rc ~ U[0.50, 0.60]`` m e ``Lt = Mp / (rho_p · eta · π · Rc²)``, de modo
      que a geometria seja coerente com ``Mp`` e ``eta`` (sorteados por último,
      para não alterar a sequência aleatória das colunas acima).

    Parameters
    ----------
    stage:
        1 ou 2.
    seed:
        Semente do gerador pseudoaleatório (reprodutibilidade).
    n:
        Número de designs; por padrão usa a contagem de referência do
        artigo (632 para o estágio 1, 1089 para o estágio 2) apenas como
        tamanho de amostra plausível — os valores numéricos são sintéticos.
    """

    if stage not in (1, 2):
        raise ValueError(f"stage deve ser 1 ou 2, recebido {stage!r}")

    n = n if n is not None else config.N_DESIGNS_REFERENCIA[stage]
    req = config.REQUISITOS_ESTAGIO[stage]
    rng = np.random.default_rng(seed)

    Mp = req.Mp * (1.0 + rng.uniform(-0.06, 0.06, size=n))
    tb = req.tb * (1.0 + rng.uniform(-0.06, 0.06, size=n))
    Isp = rng.normal(303.0, 3.0, size=n)
    It = Mp * Isp * config.G0
    F = It / tb

    rho_p = rng.uniform(1700.0, 1800.0, size=n)
    rhoIsp = rho_p * Isp
    pcmax = rng.uniform(5.5, 15.5, size=n)
    eta = rng.uniform(0.8, 0.95, size=n)
    propelente = rng.integers(1, 5, size=n)  # {1,2,3,4}
    Rc = rng.uniform(0.50, 0.60, size=n)
    Lt = Mp / (rho_p * eta * np.pi * Rc**2)

    df = pd.DataFrame(
        {
            "Mp": Mp,
            "tb": tb,
            "It": It,
            "F": F,
            "rhoIsp": rhoIsp,
            "pcmax": pcmax,
            "eta": eta,
            "Rc": Rc,
            "Lt": Lt,
            COLUNA_PROPELENTE: propelente,
        }
    )
    df[COLUNA_NEXP] = resolver_nexp(df)
    df[COLUNA_SINTETICO] = True

    aviso = (
        f"Gerando {n} designs SINTÉTICOS (estágio {stage}, seed={seed}). "
        "Estes NÃO são dados reais de Nascimento et al. (2022); use "
        "load_designs() quando os dados reais estiverem disponíveis."
    )
    warnings.warn(aviso, stacklevel=2)
    logger.warning(aviso)

    validar_restricoes(df, levantar_erro=False)
    return df


__all__ = [
    "COLUNAS_CANONICAS",
    "ALIASES",
    "COLUNA_PROPELENTE",
    "COLUNA_NEXP",
    "COLUNA_SINTETICO",
    "tem_estados_precomputados",
    "resolver_nexp",
    "validar_restricoes",
    "hash_arquivo",
    "load_designs",
    "synthetic_designs",
]
