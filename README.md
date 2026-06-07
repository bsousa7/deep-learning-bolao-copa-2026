# Bolão Copa do Mundo FIFA 2026 — Previsão de placares com Rede Neural

Pipeline reprodutível em **TensorFlow/Keras** que prevê os **placares exatos dos 24 jogos da
1ª rodada da fase de grupos** da Copa do Mundo FIFA 2026, exportando o resultado em JSON.

> Trabalho da disciplina **Deep Learning e Processamento de Linguagem Natural** — Mestrado · 2º BIM 2026
> Aluno: **Bruno Aires**

## Abordagem

O número de gols de cada seleção é modelado como uma **distribuição de Poisson** cujo parâmetro
λ (gols esperados) é aprendido por uma rede neural a partir da força relativa das equipes:

- **Rating Elo** construído cronologicamente sobre ~150 anos de jogos (sem vazamento — estado pré-jogo);
- **Ranking e pontos FIFA** vigentes na data da partida;
- **Forma recente** (gols pró/contra e aproveitamento nos últimos jogos);
- **Probabilidade implícita do mercado** (odds 1/X/2 convertidas, sem overround) como *feature direta*.

A rede tem saída de 2 unidades com ativação `softplus` → (λ_A, λ_B), treinada com **log-verossimilhança
negativa de Poisson** e **decay weighting** temporal (meia-vida de 2 anos). Seleções com histórico
esparso recebem **transfer learning** por confederação (encolhimento empírico-Bayes). O placar final
é a **moda da Poisson** de cada λ, restrita ao intervalo realista [0,5] — um inteiro determinístico.

> Todas as previsões derivam **exclusivamente da rede neural** — sem regras, médias diretas ou
> ensemble com o mercado. Nenhum ajuste manual de mando de campo (apenas a feature `neutral`/sede).

## Resultados (validação holdout — 18 meses mais recentes)

| Modelo | MAE gols | Acerto 1X2 | Placar exato |
|---|---|---|---|
| Baseline (média histórica) | 1.113 | 47.5% | 6.5% |
| **Rede Neural (Poisson)** | **0.980** | **59.0%** | **13.1%** |

## Estrutura

```
bolao_copa_2026_v3.ipynb   # notebook principal (8 seções, com saídas)
build_notebook.py          # script gerador do notebook
bolao_resultado.json       # previsões finais (formato do bolão)
bolao_copa.txt             # template/fixtures oficiais (formato de saída)
Copa_do_Mundo_Cotacoes.pdf # odds 1/X/2 (fonte das probabilidades implícitas)
data/                      # results.csv, fifa_ranking.csv, shootouts.csv (fallback offline)
```

## Como executar

Requer Python 3.13 com `tensorflow`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `requests`, `beautifulsoup4`, `nbformat`, `jupyter`.

```bash
# Executa o notebook de ponta a ponta e regenera o JSON
jupyter nbconvert --to notebook --execute --inplace bolao_copa_2026_v3.ipynb
```

A execução é **totalmente reprodutível**: com seeds fixas (NumPy/TensorFlow) e operações
determinísticas do TF, dois runs produzem o JSON com hash idêntico.

## Fontes de dados

- **results.csv** — partidas internacionais 1872→presente ([martj42/international_results](https://github.com/martj42/international_results))
- **fifa_ranking.csv** — ranking FIFA histórico (rank, pontos, confederação)
- **Copa_do_Mundo_Cotacoes.pdf** — odds 1/X/2 dos 24 jogos (ponto único no tempo)

## Limitações

A moda da Poisson é conservadora (tende a placares baixos, típico de fase de grupos, mas subestima
goleadas); as odds são de fonte genérica e ponto único; seleções estreantes dependem do prior de
confederação. Melhorias futuras: odds multi-bookmaker, modelo bivariado (Dixon–Coles), embeddings
de seleção/jogadores e dados de viagem/clima/estádio.
