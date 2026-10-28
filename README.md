# tg_robust — Seleção Robusta Pós-Pareto de Motores Sólidos

Implementação computacional da metodologia do Capítulo 3 da TG
(`../cap3_metodologia_revisado.tex`, fonte da verdade): seleção pós-Pareto de
motores sólidos combinando critérios de decisão sob incerteza estrita
(**Wald**, **Savage**, **Hurwicz** — Estágio 1) com o **Processo Analítico em
Rede (ANP)** (Estágio 2), aplicada aos designs Pareto de
Nascimento et al. (2022).

Este pacote é independente do restante de `TG/` — nenhum arquivo existente
(`tratamento_dados.py`, `ANP.ipynb`, `data/`, `resultados/`, capítulos, .bib)
foi modificado, movido ou apagado. `tratamento_dados.py` é um **protótipo
anterior**, com uma metodologia diferente (fatores de cenário aplicados
diretamente a todas as métricas, pesos fixos por persona) — não é reutilizado
aqui; a metodologia correta é a descrita no `.tex` e resumida abaixo.

## Instalação

```bash
conda activate projects   # Python 3.11, numpy/scipy/pandas já instalados
pip install pytest        # se ainda não instalado no ambiente
```

Dependências: `numpy`, `scipy`, `pandas` (execução) e `pytest` (testes).
Nenhuma outra dependência externa.

## Uso rápido

```bash
cd tg_robust
python run_pipeline.py --stage 1 --out resultados_estagio1 --seed 0
python run_pipeline.py --stage 2 --out resultados_estagio2 --seed 0

# fronts de TESTE (gerador_pareto/, modelo simplificado — não usar no Cap. 4),
# reexecutando a balística (habilita a varredura de λ):
python run_pipeline.py --stage 2 --csv ../gerador_pareto/dados/pareto_estagio2.csv \
    --balistica ../gerador_pareto/balistica.py --out resultados_teste2

# opções das restrições de admissibilidade:
#   --referencia-custo {mediana,minimo}   (padrão: mediana)
#   --b-cer B                             (padrão: 1.0 PROVISÓRIO, com aviso)
#   --sem-teto-custo                      (rho = inf em todos os perfis)
```

Sem `--csv`, o pipeline usa **dados sintéticos** (`tg_robust/data.py:synthetic_designs`),
explicitamente marcados como tal (aviso no terminal, coluna `sintetico` e
`run_config.json:dados_sinteticos=true`). **Não são dados reais do artigo.**

### Saídas (`--out`)

| Arquivo | Conteúdo |
|---|---|
| `scores_W.csv`, `scores_S.csv`, `scores_H.csv` | Pontuações $v^{(p)}_{k,r}$ por critério e, para comparação, $V^{(p)}$ (pesos iguais), por perfil |
| `ranking_final.csv` | Top-50 por $g$, com $g, g_W, g_S, g_H, V_W, V_S, V_H$ e rank do baseline AHP |
| `consistency.csv` | CR por coluna de julgamento de $W_{TT}$ |
| `sens_alpha.csv`, `sens_pi.csv`, `sens_anp.csv`, `sens_lambda.csv`, `sens_restricoes.csv` | As 5 varreduras de sensibilidade (`sens_lambda.csv` fica vazio se não houver modelo de balística que reexecute os estados) |
| `admissibilidade.csv` | Por design: `pcmax` em θ1 e máximo entre estados, base e índice de custo $c_k$, e as máscaras de pressão, custo e $\mathcal{F}^{(p)}$ |
| `admissibilidade_resumo.csv` | Tamanho de $\mathcal{F}^{(p)}$, exclusões só por pressão / só por custo / por ambos, e a interseção |
| `vencedores_por_perfil.csv` | Por perfil: `vencedor_anp` (argmax de $g^{(p)}$ em $\mathcal{F}^{(p)}$, resultado principal), `vencedor_pesos_iguais` (argmax de $V^{(p)}$ em $\mathcal{F}^{(p)}$, comparação), `divergem`, e as mesmas três colunas sem restrições; "nenhum" se $\mathcal{F}^{(p)}$ for vazio |
| `pesos_efetivos.csv` | $q_0$, $q = (1-\gamma)(I - \gamma W_{TT})^{-1} q_0$ e a razão $q/q_0$ |
| `run_config.json` | Configuração completa (perfis, parâmetros globais), semente, modelo de balística, correlação $c_k \times p_{c,\max}(\theta_1)$, perfis em que $d^*$ é inadmissível, hash SHA-256 do CSV |

## Restrições de admissibilidade por perfil (Seção 3.3.4)

Cada perfil é um `config.Perfil` (regra de decisão, α, FS, teto de custo ρ,
peso π); os parâmetros globais ficam em `config.ConfigGlobal` (p_lim = 16 MPa,
expoente `b` da CER — **1,0 provisório**, com aviso —, referência do custo
`mediana`/`minimo`). O pipeline e a sensibilidade tratam qualquer lista de
perfis da mesma forma. As restrições atuam só no bloco $W_{AT}^{(p)}$ e na
escolha do vencedor de cada perfil; `rho = inf` retira o teto de custo.

O **vencedor de cada perfil** é o argmax de $g^{(p)}$ (sub-rede ANP do perfil,
a mesma que entra na síntese de $d^*$) restrito a $\mathcal{F}^{(p)}$.
$V^{(p)}$ (média simples dos 7 critérios, pesos iguais) é mantido só como
referência "sem interdependência", como no AHP original.

## Onde plugar os dados reais

Os arquivos `../data/pareto_estagio_*.csv` são da metodologia anterior e
**não devem ser usados**. Os dados reais de Nascimento et al. (2022) precisam
trazer, por design:

1. os 7 critérios por estado, em colunas `<criterio>_theta{1,2,3}`
   (`tb`, `It`, `F`, `rho_isp`, `pcmax`) mais `Mp` e `eta_v` — mesmo
   formato de `../gerador_pareto/dados/` —, **ou** só os valores nominais e um
   modelo de balística (`--balistica`) para calcular os estados;
2. `Rc` e `Lt` (raio e comprimento do grão; a unidade se cancela no índice de
   custo) — sem eles, `load_designs` recusa o arquivo;
3. `propelente` (tipo 1-4) ou `nexp`.

`pc_med` (pressão média) **não** é aceito no lugar de `pcmax`.

## Onde plugar o modelo real de balística interna

`tg_robust/scenarios.py` define a interface `BallisticModel`:

```python
class BallisticModel(Protocol):
    def __call__(self, designs: pd.DataFrame, kc: float, kcf: float) -> dict[str, np.ndarray]:
        ...
```

Implemente uma classe com essa assinatura (recalculando a curva de empuxo
por inteiro a partir de $c^{*}_j = \kappa_{c^{*},j} c^{*}$ e
$C_{F,j} = \kappa_{C_F,j} C_F$, conforme a Seção "Definição dos Estados da
Natureza" do `.tex`) e passe-a ao pipeline:

```python
resultado = pipeline.run(cfg, designs, model=ModeloRealDeBalistica())
```

Um simulador por design no formato `simular(x, k_cstar, k_cf) -> dict` (como
`../gerador_pareto/balistica.py`) é adaptado por `scenarios.AdaptadorSimulador`
(`--balistica arquivo.py` no CLI). Se os estados vêm prontos no CSV e nenhum
modelo é fornecido, usa-se `scenarios.EstadosPrecomputados` e a varredura de λ
é pulada com aviso. Sem CSV por estado nem modelo, o pipeline usa
`SurrogateBallistics`, que aplica as identidades analíticas da Seção 3.3.1 —
o "FALLBACK" mencionado no `.tex`.

## Regressão contra a versão anterior às restrições

`tests/test_regressao.py` compara, com FS = 1 e ρ = ∞, os resultados atuais com
os da versão anterior (igualdade exata). Aponte a variável `TG_ROBUST_ANTES`
para a raiz dessa versão: `TG_ROBUST_ANTES=/caminho/tg_robust_antes python -m pytest tests/test_regressao.py`.

## Arquitetura

```
tg_robust/
  tg_robust/
    config.py       # Config (dataclass imutável) + constantes dos critérios/requisitos
    data.py         # load_designs, synthetic_designs, validação de restrições
    scenarios.py    # BallisticModel (interface) + SurrogateBallistics + propagate_states
    stage1.py       # benefício, normalização comum/por-estado, Wald/Savage/Hurwicz
    judgments.py    # Saaty: from_intensities, consistência (CR), perturb, W_TT placeholder
    anp.py          # W_AT, limite fechado (solve), limite por potências (verificação), síntese
    pipeline.py      # run(cfg, designs, model) -> Result
    baseline.py     # AHP de pesos iguais (aproximação, 5 objetivos, estado nominal)
    admissibility.py # restrições por perfil (FS, teto de custo), F^(p) e resumo
    sensitivity.py  # 5 varreduras (alpha, pi, ANP/gamma, lambda, restrições) + estabilidade
  run_pipeline.py   # CLI
  tests/            # pytest (69 testes + 1 xfail)
```

Sem estado global mutável; qualquer variação de parâmetro cria uma nova
`Config` via `dataclasses.replace`. Semente aleatória sempre explícita
(`--seed`, `synthetic_designs(stage, seed)`).

## Testes

```bash
cd tg_robust
python -m pytest -q
```

Cobertura: identidades de propagação, normalização comum vs. por-estado,
Wald/Savage/Hurwicz (casos-limite e invariância), consistência e
perturbação de Saaty, limite fechado vs. por potências (`atol=1e-9`),
síntese por mistura, reprodutibilidade, e o CLI ponta a ponta com $n=632$
e $n=1089$ designs sintéticos (ambos em menos de 2s).

## Decisões em aberto (dependem do autor e/ou do orientador)

Ver a lista completa na mensagem final da sessão em que este pacote foi
criado. Resumo:

1. **Dados reais** — os CSVs de `../data/` parecem ser dados de
   desenvolvimento do protótipo antigo (violam `eta>=0,8` massivamente), não
   o *front* de Pareto validado do artigo. Os dados reais de
   Nascimento et al. (2022) ainda precisam ser obtidos/confirmados.
2. **Scripts de balística interna** do orientador — opcional; na ausência,
   usa-se `SurrogateBallistics` (identidades analíticas).
3. **Julgamentos de `W_TT`** (`judgments.PLACEHOLDER_INFLUENCIAS`) — rascunho
   provisório, a validar com o orientador (acoplamentos físicos reais entre
   os 7 critérios).
4. **$\alpha$ de Hurwicz** (referência 0,8) — confirmar a postura de risco do
   perfil *New Space* adotada.
5. **Margens de $c^{*}$ e $C_F$** — valores de $\kappa$ (Tabela "cenarios")
   vêm diretamente do texto do artigo reportado no `.tex`; confirmar se são
   as margens finais a usar.
