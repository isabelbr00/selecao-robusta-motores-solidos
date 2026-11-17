"""Exporta os resultados de uma versão do pacote para o teste de regressão.

Uso (chamado por ``tests/test_regressao.py`` em subprocesso, um por versão):

    python _exportar_regressao.py <raiz_do_pacote> <antes|depois> <designs.pkl> <saida.pkl> <estados|substituto>

``antes`` roda o código anterior às restrições com sua configuração padrão;
``depois`` roda o código atual com ``FS = 1`` e ``rho = inf`` em todos os
perfis. Os mesmos ``designs`` (DataFrame em pickle) entram nas duas versões.
"""

import math
import pickle
import sys
import warnings

import numpy as np

raiz, versao, arq_designs, arq_saida, modo_modelo = sys.argv[1:6]
sys.path.insert(0, raiz)
warnings.simplefilter("ignore")

# a versão anterior às restrições ainda se chama ``tg_robust``
try:
    from selecao_robusta import config, pipeline, sensitivity  # noqa: E402
    from selecao_robusta import judgments as J  # noqa: E402
    from selecao_robusta.scenarios import SurrogateBallistics  # noqa: E402
except ImportError:
    from tg_robust import config, pipeline, sensitivity  # noqa: E402
    from tg_robust import judgments as J  # noqa: E402
    from tg_robust.scenarios import SurrogateBallistics  # noqa: E402

KAPPAS = list(zip((1.0, 0.99, 0.98), (1.0, 0.965, 0.93)))

# W_TT congelada (rascunho original, razões arredondadas à Escala de Saaty),
# imposta às DUAS versões: a regressão verifica só a camada de restrições, não
# as mudanças posteriores nos julgamentos de W_TT.
INFLUENCIAS_REGRESSAO = {
    "Mp": {"It": 9, "rhoIsp": 5, "tb": 3, "eta": 3},
    "tb": {"pcmax": 5, "F": 5, "It": 3, "Mp": 3},
    "It": {"Mp": 9, "rhoIsp": 7, "F": 5, "tb": 5},
    "F": {"pcmax": 9, "It": 5, "tb": 5, "rhoIsp": 3},
    "rhoIsp": {"pcmax": 5, "F": 3},
    "pcmax": {"F": 9, "tb": 3, "rhoIsp": 3, "eta": 3},
    "eta": {"Mp": 7, "pcmax": 3, "It": 3},
}


def wtt_congelada(criterios=getattr(config, "ATRIBUTOS", config.CRITERIOS)):
    julgamentos = []
    for alvo in criterios:
        intens = INFLUENCIAS_REGRESSAO[alvo]
        labels = tuple(intens)
        A = np.ones((len(labels), len(labels)))
        for i, li in enumerate(labels):
            for k, lk in enumerate(labels):
                if i < k:
                    A[i, k] = J.snap_to_saaty_scale(intens[li] / intens[lk])
                    A[k, i] = 1.0 / A[i, k]
        julgamentos.append(J.ColumnJudgment(target=alvo, labels=labels, A=A))
    return J.WTT_de_julgamentos(J.julgamentos_por_alvo(julgamentos), criterios), julgamentos


J.build_WTT_placeholder = wtt_congelada  # versão anterior
J.build_WTT_validada = wtt_congelada  # versão atual


class EstadosDasColunas:
    """Lê <criterio>_theta{j}; implementado aqui para servir às duas versões."""

    suporta_reexecucao = False

    def __call__(self, designs, kc, kcf):
        j = next(i for i, (c, f) in enumerate(KAPPAS, 1) if math.isclose(kc, c) and math.isclose(kcf, f))
        saida = {c: designs[c].to_numpy(dtype=float).copy() for c in ("Mp", "eta")}
        for c in ("tb", "It", "F", "rhoIsp", "pcmax"):
            saida[c] = designs[f"{c}_theta{j}"].to_numpy(dtype=float).copy()
        return saida


with open(arq_designs, "rb") as f:
    designs, estagio = pickle.load(f)

if versao == "antes":
    cfg = config.Config(estagio=estagio)
else:
    # a versão anterior usava -|δ| nos resíduos
    cfg = config.Config(
        estagio=estagio,
        criterios=config.ATRIBUTOS,  # a versão anterior tinha pcmax como critério
        perfis=config.perfis_sem_restricoes(config.PERFIS_REFERENCIA),
        globais=config.ConfigGlobal(forma_residuo="absoluto"),
    )

modelo = EstadosDasColunas() if modo_modelo == "estados" else SurrogateBallistics()
res = pipeline.run(cfg, designs, model=modelo)

saida = {
    "X_norm": res.X_norm,
    "scores": res.scores,
    "V": res.preference_vectors,
    "W_AT": res.anp_resultado.W_AT_por_perfil,
    "g_por_perfil": res.anp_resultado.g_por_perfil,
    "g": res.g,
    "q": res.anp_resultado.q,
    "d_star": res.d_star,
    "sens_alpha": sensitivity.sweep_alpha(res),
    "sens_pi": sensitivity.sweep_pi(res).drop(columns=["pi"]),
    "sens_anp": sensitivity.sweep_anp(res),
}
if modo_modelo == "substituto":
    saida["sens_lambda"] = sensitivity.sweep_lambda(res, model=SurrogateBallistics())

with open(arq_saida, "wb") as f:
    pickle.dump(saida, f)
