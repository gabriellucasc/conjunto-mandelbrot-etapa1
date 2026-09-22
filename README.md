# Projeto 1 — Conjunto de Mandelbrot · Etapa 1 (OpenMP)

DEC107 — Processamento Paralelo · UESC · 2026.2

**Autores:** Gabriel Oliveira Lucas César · Ronaldo Ribeiro Porto Filho

## Sobre

Implementação do conjunto de Mandelbrot em C utilizando o método de *escape-time*, com versões sequencial e paralela em OpenMP.

O projeto avalia diferentes números de threads, políticas de escalonamento e tamanhos de *chunk*, além de escalabilidade forte e fraca.

## Estrutura do projeto

```text
conjunto-mandelbrot-etapa1/
├── src/
│   ├── mandelbrot_sequencial.c
│   └── mandelbrot_openmp.c
│
├── codigos_auxiliares/
│   ├── benchmark.py
│   ├── analise.py
│   ├── gera_mandelbrot.py
│   └── validar_professor.py
│
├── resultados/
│   ├── resultados_consolidados.csv
│   ├── ambiente.md
│   ├── validacao_referencia_professor.md
│   └── tabelas/
│
├── graficos/
├── relatorio_etapa1.pdf
├── README.md
└── .gitignore
```

`gera_mandelbrot.py` foi fornecido pelo professor e é utilizado apenas na validação da implementação.

A pasta `output/` contém executáveis e arquivos `.bin`/`.ppm` gerados localmente e não é versionada.

## Ambiente

- Intel Core i5-12400F
- 32 GB de RAM
- Windows 11
- GCC 6.3.0 com OpenMP
- Python 3
- NumPy, pandas e matplotlib

Mais detalhes estão em `resultados/ambiente.md`.

## Como executar

Os comandos devem ser executados a partir da raiz do repositório `conjunto-mandelbrot-etapa1/`.

### Teste rápido

```bash
python codigos_auxiliares/benchmark.py --preset smoke
```

### Bateria experimental

```bash
python codigos_auxiliares/benchmark.py --preset padrao
python codigos_auxiliares/benchmark.py --preset zoom --resume
python codigos_auxiliares/benchmark.py --preset weak --resume
```

### Gerar tabelas e gráficos

```bash
python codigos_auxiliares/analise.py
```

### Validar contra a referência do professor

```bash
python codigos_auxiliares/validar_professor.py
```

A validação externa é registrada em `resultados/validacao_referencia_professor.md`.

## Resultados

Os dados consolidados estão em `resultados/` e os gráficos utilizados na análise estão em `graficos/`.

As versões sequencial e OpenMP desenvolvidas produzem matrizes idênticas nos casos oficiais testados. A validação contra a referência fornecida pelo professor e seus resultados estão documentados em `resultados/validacao_referencia_professor.md`.

## Relatório

A metodologia, os resultados experimentais, a análise de desempenho e a discussão completa estão descritos em [`relatorio_etapa1.pdf`](relatorio_etapa1.pdf).

## Nota de Transparência sobre o Uso de IA

Declaro que este projeto contou com o auxílio das ferramentas de IA ChatGPT e Claude exclusivamente para as tarefas de revisão, correção e padronização do código C/OpenMP, organização da metodologia experimental, geração e revisão dos scripts auxiliares de benchmark e análise, interpretação dos resultados e apoio na documentação, no relatório e na preparação da apresentação.

Como autores, revisamos, testamos e validamos criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.

**Gabriel Oliveira Lucas César — 22/09/2026**  
**Ronaldo Ribeiro Porto Filho — 22/09/2026**
