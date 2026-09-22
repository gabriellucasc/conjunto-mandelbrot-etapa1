# Ambiente de execucao

Parte automatica atualizada por `benchmark.py` em 2026-09-20T22:58:08

## Hardware

- CPU: 12th Gen Intel(R) Core(TM) i5-12400F
- Nucleos fisicos: 6
- Threads logicas: 12
- RAM: 31.8 GB

## Sistema e software

- SO: Windows-11-10.0.26200-SP0
- Python: 3.13.3
- Compilador: gcc (MinGW.org GCC-6.3.0-1) 6.3.0
- Alvo do GCC: `mingw32` (32 bits)
- Flags: `-O2 -msse2 -mfpmath=sse -fopenmp -lm` (+ `-DWIDTH`/`-DHEIGHT`/`-DCASO_ZOOM` por variante)
- Timer dos programas C: `omp_get_wtime()`
- SHA-256 (12 primeiros) das fontes:
  - sequencial: `dba0b7e33100`
  - OpenMP: `271751a61931`
- Plano de energia durante a bateria: Equilibrado

## Parametros da ultima execucao

- Repeticoes por configuracao OpenMP: 5
- Baseline sequencial: 7 execucoes antes + 3 depois
- Variaveis `OMP_*` previamente definidas no ambiente: nenhuma
- `OMP_NUM_THREADS` e `OMP_SCHEDULE` definidos pelo `benchmark.py` em cada execucao

## Condicoes (preencher a mao)

- Sincronizacao do OneDrive pausada durante a bateria oficial.
- Navegadores, Discord e outros programas de uso geral permaneceram fechados durante as medicoes.
- VS Code e utilitarios de perifericos permaneceram abertos.
- O computador nao foi utilizado para outras tarefas durante os benchmarks.
- Plano de energia mantido em `Equilibrado` durante toda a coleta.
- Nao houve controle explicito de afinidade das threads (`OMP_PROC_BIND` / `OMP_PLACES`).

## Limitacoes

- O compilador utilizado possui alvo `mingw32` (32 bits).
- Por precaucao (alvo de 32 bits, ~2 GB de espaco de enderecamento; a matriz 14189x14189 ocuparia ~805 MB), o script ignorou resolucoes acima de 11585x11585; o weak scaling foi executado ate 8 threads.
- O ponto de 12 threads no weak scaling (`14189x14189`) nao foi testado.
- O temporizador apresentou granularidade observada da ordem de 1 ms; valores muito pequenos de `F_LB` devem ser interpretados considerando essa resolucao.
- A afinidade das threads aos nucleos nao foi fixada explicitamente.

## Historico de comandos (a bateria pode ser feita em etapas)

- 2026-09-20T22:41:56: `codigos_auxiliares\benchmark.py --preset padrao --recomecar`
- 2026-09-20T22:53:48: `codigos_auxiliares\benchmark.py --preset zoom --resume`
- 2026-09-20T22:58:08: `codigos_auxiliares\benchmark.py --preset weak --resume`
- 2026-09-20: `codigos_auxiliares\benchmark.py --preset imagens`
