# Validação contra a referência do professor

## Objetivo

Este documento registra uma validação reprodutível das implementações sequencial e OpenMP em C contra a função `mandelbrot()` presente em `codigos_auxiliares/gera_mandelbrot.py`.

A matriz da referência do professor foi gerada **somente em memória**. Nenhuma cópia dessa matriz foi persistida em disco.

A implementação C adota a convenção de armazenar o **número de iterações executadas** (`i + 1` para um pixel que escapa), enquanto a referência do professor registra o **índice da iteração** (`i`, iniciado em zero). Para permitir a comparação conceitual das duas convenções, este script aplica apenas durante o diagnóstico a transformação `professor + 1` nos pixels em que `professor < MAX_ITER`; pixels que atingem `MAX_ITER` permanecem inalterados.

> **Importante:** essa normalização é apenas diagnóstica. Ela não modifica os binários das implementações C e não transforma automaticamente a comparação em aprovação pelo critério formal do enunciado.

## Ambiente da validação

- Python: `3.13.3`
- NumPy: `2.4.3`
- Plataforma: `Windows-11-10.0.26200-SP0`
- Referência: `codigos_auxiliares\gera_mandelbrot.py`
- SHA-256 de `gera_mandelbrot.py`: `eb9a1ede4a344029822c1766ab5bb18decdeeaa7052e661264c27afd1bc93634`

## 1. Consistência interna: Sequencial × OpenMP

| Caso | Pixels diferentes | Maior diferença | Resultado |
|---|---:|---:|---|
| Padrão | 0 | 0 | **IDÊNTICOS** |
| Zoom — vale dos cavalos-marinhos | 0 | 0 | **IDÊNTICOS** |

A comparação acima verifica diretamente os `.bin` preservados em `output/ref`. Igualdade aqui demonstra que a paralelização OpenMP preservou o resultado da implementação sequencial desenvolvida.

## 2. Comparação normalizada de convenção com a referência

| Caso / implementação | Total de pixels | Divergentes | % divergente | Dif. ≤ 1 | Dif. > 1 | Maior diferença |
|---|---:|---:|---:|---:|---:|---:|
| Padrão — Sequencial | 16.777.216 | 786 | 0,004685% | 187 | 599 | 460 |
| Padrão — OpenMP | 16.777.216 | 786 | 0,004685% | 187 | 599 | 460 |
| Zoom — vale dos cavalos-marinhos — Sequencial | 16.777.216 | 24.242 | 0,144494% | 5.975 | 18.267 | 1.415 |
| Zoom — vale dos cavalos-marinhos — OpenMP | 16.777.216 | 24.242 | 0,144494% | 5.975 | 18.267 | 1.415 |

## 3. Critério formal do enunciado

O critério verificado neste relatório é: no máximo **0,01%** de pixels divergentes e, entre eles, diferença máxima de **1 iteração**.

| Caso | Sequencial | OpenMP | Observação |
|---|---|---|---|
| Padrão | **NÃO ATENDE** | **NÃO ATENDE** | diferença por pixel acima de 1 |
| Zoom — vale dos cavalos-marinhos | **NÃO ATENDE** | **NÃO ATENDE** | percentual acima do limite; diferença por pixel acima de 1 |

## 4. Evidências dos arquivos C

| Caso | Arquivo sequencial | SHA-256 | Arquivo OpenMP | SHA-256 |
|---|---|---|---|---|
| Padrão | `output\ref\sequencial_padrao_4096.bin` | `e50c4473b1256734f2a689a350cd056774c7a3aa5461a598bb12055c19c5c817` | `output\ref\openmp_padrao_4096.bin` | `e50c4473b1256734f2a689a350cd056774c7a3aa5461a598bb12055c19c5c817` |
| Zoom — vale dos cavalos-marinhos | `output\ref\sequencial_zoom_4096.bin` | `e69dba7173ce24c6a7da469a73d8ffbc979bc58101ec20458f333f9b8208dba2` | `output\ref\openmp_zoom_4096.bin` | `e69dba7173ce24c6a7da469a73d8ffbc979bc58101ec20458f333f9b8208dba2` |

## 5. Conclusão

- **Consistência interna atestada:** os binários sequencial e OpenMP são idênticos nos dois casos oficiais.
- **Equivalência de convenção tratada explicitamente:** a diferença `i` versus `i + 1` é normalizada somente na comparação diagnóstica, sem alterar os arquivos C.
- **Critério formal não integralmente satisfeito:** apesar da forte consistência interna e da equivalência de convenção, pelo menos um caso ultrapassa a tolerância formal configurada. O relatório deve manter essa distinção de forma explícita.

Este arquivo foi produzido automaticamente por `codigos_auxiliares/validar_professor.py`.
