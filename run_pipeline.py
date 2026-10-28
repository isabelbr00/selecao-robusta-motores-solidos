#!/usr/bin/env python
"""CLI do pipeline de seleção robusta de motores sólidos (ANP + Wald/Savage/Hurwicz).

Uso:
    python run_pipeline.py --stage 1 --out resultados_estagio1 --seed 0
    python run_pipeline.py --stage 2 --csv caminho/para/designs.csv --out resultados_estagio2

Sem ``--csv``, gera designs sintéticos (:func:`tg_robust.data.synthetic_designs`)
marcados explicitamente como tal — ver aviso no terminal e a coluna
``sintetico`` / o campo ``dados_sinteticos`` em ``run_config.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tg_robust import admissibility, baseline, config, data, pipeline, sensitivity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True, help="Estágio (1 ou 2)")
    parser.add_argument("--csv", type=str, default=None, help="CSV real de designs Pareto; se omitido, usa dados sintéticos")
    parser.add_argument("--out", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=0, help="Semente aleatória (reprodutibilidade)")
    parser.add_argument(
        "--b-cer",
        type=float,
        default=config.B_CER_REF,
        help="Expoente b da CER de motores sólidos (Koelle, TransCost); sem ele, o teto de custo não é aplicado",
    )
    parser.add_argument(
        "--verificar-limite-potencias",
        action="store_true",
        help="Verifica o limite por potências contra o limite fechado (mais lento)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dados_sinteticos = args.csv is None
    csv_hash = None
    if args.csv is not None:
        designs = data.load_designs(args.csv)
        csv_hash = data.hash_arquivo(args.csv)
        logger.info("Designs reais carregados de %s (n=%d)", args.csv, len(designs))
    else:
        designs = data.synthetic_designs(args.stage, seed=args.seed)
        logger.warning(
            "Nenhum --csv fornecido: usando %d designs SINTÉTICOS (estágio %d, seed=%d). "
            "Resultados não representam os dados reais de Nascimento et al. (2022).",
            len(designs),
            args.stage,
            args.seed,
        )

    cfg = config.Config(
        estagio=args.stage,
        seed=args.seed,
        b_cer=args.b_cer,
        verificar_limite_por_potencias=args.verificar_limite_potencias,
    )

    logger.info("Executando pipeline (Estágio 1 + ANP) para %d designs...", len(designs))
    resultado = pipeline.run(cfg, designs)

    logger.info("Calculando baseline AHP (aproximação)...")
    bl = baseline.ahp_baseline(designs, cfg.requisitos)

    logger.info("Executando análise de sensibilidade (5 varreduras)...")
    sens_alpha = sensitivity.sweep_alpha(resultado)
    sens_pi = sensitivity.sweep_pi(resultado)
    sens_anp = sensitivity.sweep_anp(resultado)
    sens_lambda = sensitivity.sweep_lambda(resultado)
    sens_restricoes = sensitivity.sweep_restricoes(resultado)

    adm = resultado.admissibilidade
    if cfg.b_cer is None:
        logger.warning(config.aviso_b_cer_ausente())

    # ─── admissibilidade.csv (por design) e admissibilidade_resumo.csv ──
    df_adm = pd.DataFrame(
        {
            "design_id": designs.index,
            "pcmax_max_estados": adm.pcmax_max,
            "razao_massa_envelope": adm.razao_massa,
            "indice_custo_c": adm.c,
        }
    )
    for p in config.PERFIS:
        df_adm[f"adm_pressao_{p}"] = adm.adm_pressao[p]
        df_adm[f"adm_custo_{p}"] = adm.adm_custo[p]
        df_adm[f"F_{p}"] = adm.chi[p]
    df_adm["F_intersecao"] = adm.intersecao
    df_adm.to_csv(out_dir / "admissibilidade.csv", index=False)
    resumo_adm = admissibility.resumo(adm)
    resumo_adm.to_csv(out_dir / "admissibilidade_resumo.csv", index=False)

    # ─── vencedores_por_perfil.csv (com e sem restrições) ───────────────
    linhas_venc = []
    for p in config.PERFIS:
        sem = resultado.vencedor_perfil(p, com_restricoes=False)
        com = resultado.vencedor_perfil(p, com_restricoes=True)
        V = resultado.preference_vectors[p]
        linhas_venc.append(
            {
                "perfil": p,
                "vencedor_sem_restricoes": int(designs.index[sem]),
                "V_sem_restricoes": V[sem],
                "vencedor_com_restricoes": None if com is None else int(designs.index[com]),
                "V_com_restricoes": None if com is None else V[com],
                "excluido_da_sintese": p in resultado.anp_resultado.perfis_excluidos,
            }
        )
    df_venc = pd.DataFrame(linhas_venc)
    df_venc.to_csv(out_dir / "vencedores_por_perfil.csv", index=False)

    # ─── scores_<perfil>.csv ────────────────────────────────────────────
    for perfil, v in resultado.scores.items():
        df_scores = pd.DataFrame(v, columns=list(config.CRITERIOS))
        df_scores.insert(0, "design_id", designs.index)
        df_scores["V"] = resultado.preference_vectors[perfil]
        df_scores.to_csv(out_dir / f"scores_{perfil}.csv", index=False)

    # ─── ranking_final.csv (top-50) ─────────────────────────────────────
    g_por_perfil = resultado.anp_resultado.g_por_perfil
    ranking = pd.DataFrame(
        {
            "design_id": designs.index,
            "g": resultado.g,
            "g_W": g_por_perfil["W"],
            "g_S": g_por_perfil["S"],
            "g_H": g_por_perfil["H"],
            "V_W": resultado.preference_vectors["W"],
            "V_S": resultado.preference_vectors["S"],
            "V_H": resultado.preference_vectors["H"],
            "F_W": adm.chi["W"],
            "F_S": adm.chi["S"],
            "F_H": adm.chi["H"],
        }
    ).join(bl, how="left")
    ranking = ranking.sort_values("g", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank_integrado", np.arange(1, len(ranking) + 1))
    ranking.head(50).to_csv(out_dir / "ranking_final.csv", index=False)

    # ─── criterion_weights.csv ──────────────────────────────────────────
    pd.DataFrame(
        {"criterio": list(config.CRITERIOS), "q0": resultado.anp_resultado.q0, "q_efetivo": resultado.anp_resultado.q}
    ).to_csv(out_dir / "criterion_weights.csv", index=False)

    # ─── consistency.csv ────────────────────────────────────────────────
    resultado.consistency_df.to_csv(out_dir / "consistency.csv", index=False)

    # ─── sensibilidade ──────────────────────────────────────────────────
    sens_alpha.to_csv(out_dir / "sens_alpha.csv", index=False)
    sens_pi.drop(columns=["pi"]).to_csv(out_dir / "sens_pi.csv", index=False)
    sens_anp.to_csv(out_dir / "sens_anp.csv", index=False)
    sens_lambda.to_csv(out_dir / "sens_lambda.csv", index=False)
    sens_restricoes.to_csv(out_dir / "sens_restricoes.csv", index=False)

    # ─── run_config.json ────────────────────────────────────────────────
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "n_designs": len(designs),
        "dados_sinteticos": dados_sinteticos,
        "csv_path": args.csv,
        "csv_sha256": csv_hash,
        "config": dataclasses.asdict(cfg),
        "verificar_limite_por_potencias": args.verificar_limite_potencias,
        "cr_excede_limite": bool(resultado.consistency_df["cr_excede_limite"].any()),
        "d_star_design_id": int(designs.index[resultado.d_star]),
        "b_cer_definido": cfg.b_cer is not None,
        "teto_custo_aplicado": cfg.b_cer is not None,
        "tamanho_F": {p: int(adm.chi[p].sum()) for p in config.PERFIS},
        "tamanho_intersecao_F": int(adm.intersecao.sum()),
        "perfis_excluidos_da_sintese": list(resultado.anp_resultado.perfis_excluidos),
        "d_star_inadmissivel_em": resultado.d_star_inadmissivel_em,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # ─── resumo no terminal ─────────────────────────────────────────────
    print()
    print(f"=== Resumo — Estágio {args.stage} (n={len(designs)}, "
          f"{'SINTÉTICO' if dados_sinteticos else 'dados reais'}) ===")
    print(f"Design selecionado (d*): design_id={run_config['d_star_design_id']}, g={resultado.g[resultado.d_star]:.4f}")
    inadm = run_config["d_star_inadmissivel_em"]
    print(f"d* inadmissível para os perfis: {', '.join(inadm) if inadm else 'nenhum'}")
    print("Conjuntos admissíveis F^(p):")
    print(resumo_adm.to_string(index=False))
    if run_config["perfis_excluidos_da_sintese"]:
        print(f"Perfis excluídos da síntese (F vazio): {run_config['perfis_excluidos_da_sintese']}")
    print("Vencedor por perfil (argmax V^(p)), sem -> com restrições:")
    for _, linha in df_venc.iterrows():
        print(f"  {linha['perfil']}: {linha['vencedor_sem_restricoes']} -> {linha['vencedor_com_restricoes']}")
    if not run_config["b_cer_definido"]:
        print("ATENÇÃO: expoente b da CER não definido; teto de custo NÃO aplicado (use --b-cer).")
    print(f"CR do rascunho de julgamentos excede 0,10? {run_config['cr_excede_limite']}")
    print(f"Saídas gravadas em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
