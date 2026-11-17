"""Regressão: com FS = 1 e rho = inf, o código atual reproduz o anterior às restrições.

Requer a versão anterior do pacote em um diretório indicado pela variável de
ambiente ``TG_ROBUST_ANTES`` (raiz contendo o pacote ``tg_robust/``); sem ela, os
testes são pulados. Cada versão roda em um subprocesso próprio
(``_exportar_regressao.py``) sobre os mesmos designs, e os resultados devem
ser IDÊNTICOS (igualdade exata de arrays e tabelas).
"""

import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selecao_robusta import data

RAIZ = Path(__file__).resolve().parent.parent
EXPORTADOR = Path(__file__).resolve().parent / "_exportar_regressao.py"
GERADOR = RAIZ.parent / "gerador_pareto"
ANTES = os.environ.get("TG_ROBUST_ANTES")

pytestmark = pytest.mark.skipif(not ANTES, reason="defina TG_ROBUST_ANTES com a raiz da versão anterior")


def _exportar(tmp_path, raiz, versao, arq_designs, modo):
    saida = tmp_path / f"{versao}.pkl"
    proc = subprocess.run(
        [sys.executable, str(EXPORTADOR), str(raiz), versao, str(arq_designs), str(saida), modo],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    with open(saida, "rb") as f:
        return pickle.load(f)


def _comparar(antes, depois):
    for chave in ("X_norm", "g", "q"):
        np.testing.assert_array_equal(depois[chave], antes[chave], err_msg=chave)
    assert depois["d_star"] == antes["d_star"]
    for chave in ("scores", "V", "W_AT", "g_por_perfil"):
        assert set(depois[chave]) == set(antes[chave]), chave
        for s in antes[chave]:
            np.testing.assert_array_equal(depois[chave][s], antes[chave][s], err_msg=f"{chave}/{s}")
    for chave in ("sens_alpha", "sens_pi", "sens_anp", "sens_lambda"):
        if chave not in antes:
            continue
        comuns = [c for c in antes[chave].columns if c in depois[chave].columns]
        assert comuns == list(antes[chave].columns), f"{chave}: colunas antigas ausentes"
        pd.testing.assert_frame_equal(depois[chave][comuns], antes[chave][comuns], check_exact=True, obj=chave)


def _casos():
    casos = [pytest.param(("sintetico", e), id=f"sintetico-estagio{e}") for e in (1, 2)]
    casos += [pytest.param(("front", e), id=f"front-estagio{e}") for e in (1, 2)]
    return casos


@pytest.mark.parametrize("caso", _casos())
def test_sem_restricoes_identico_a_versao_anterior(tmp_path, caso):
    fonte, estagio = caso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fonte == "sintetico":
            designs = data.synthetic_designs(estagio, seed=0)
            modo = "substituto"
        else:
            if not GERADOR.exists():
                pytest.skip("gerador_pareto/ ausente")
            designs = data.load_designs(GERADOR / "dados" / f"pareto_estagio{estagio}.csv")
            modo = "estados"

    arq_designs = tmp_path / "designs.pkl"
    with open(arq_designs, "wb") as f:
        pickle.dump((designs, estagio), f)

    antes = _exportar(tmp_path, ANTES, "antes", arq_designs, modo)
    depois = _exportar(tmp_path, RAIZ, "depois", arq_designs, modo)
    _comparar(antes, depois)
