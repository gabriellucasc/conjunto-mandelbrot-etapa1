#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validação reprodutível das implementações C (sequencial e OpenMP)
contra a função mandelbrot() fornecida pelo professor.

Local esperado:
    Etapa 1/
    ├── codigos_auxiliares/
    │   ├── validar_professor.py
    │   ├── gera_mandelbrot.py
    │   ├── benchmark.py
    │   └── analise.py
    ├── output/
    │   └── ref/
    │       ├── sequencial_padrao_4096.bin
    │       ├── openmp_padrao_4096.bin
    │       ├── sequencial_zoom_4096.bin
    │       └── openmp_zoom_4096.bin
    └── resultados/

Saída persistente:
    resultados/validacao_referencia_professor.md

IMPORTANTE:
- O script NÃO modifica os binários das implementações.
- A matriz produzida pela referência do professor permanece somente em memória.
- A normalização i -> i+1 é aplicada SOMENTE para diagnóstico de equivalência
  de convenção, preservando MAX_ITER para pixels que não escapam.
- O script distingue "consistência interna" de "conformidade formal" com o
  critério do enunciado. Não declara aprovação formal quando ela não ocorre.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Configuração do projeto
# ---------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
ARQ_PROFESSOR = AQUI / "gera_mandelbrot.py"
PASTA_REF = RAIZ / "output" / "ref"
PASTA_RESULTADOS = RAIZ / "resultados"
ARQ_RELATORIO = PASTA_RESULTADOS / "validacao_referencia_professor.md"

WIDTH = 4096
HEIGHT = 4096

CRITERIO_PERCENTUAL_MAX = 0.01  # %
CRITERIO_DIF_MAX = 1            # iteração


@dataclass(frozen=True)
class Caso:
    nome: str
    max_iter: int
    re_min: float
    re_max: float
    im_min: float
    im_max: float
    seq_nome: str
    omp_nome: str


@dataclass
class Metricas:
    total: int
    diferentes: int
    percentual: float
    ate_1: int
    acima_1: int
    maior_diferenca: int

    @property
    def iguais(self) -> int:
        return self.total - self.diferentes

    @property
    def percentual_igual(self) -> float:
        return 100.0 - self.percentual


def casos_oficiais() -> list[Caso]:
    # Caso padrão.
    padrao = Caso(
        nome="Padrão",
        max_iter=1000,
        re_min=-2.0,
        re_max=1.0,
        im_min=-1.5,
        im_max=1.5,
        seq_nome="sequencial_padrao_4096.bin",
        omp_nome="openmp_padrao_4096.bin",
    )

    # Zoom: calculado da mesma forma conceitual usada pela referência do professor,
    # a partir de centro e largura, em vez de escrever diretamente o limite mínimo.
    cx = -0.743643887
    cy = 0.131825904
    largura_real = 3.0e-3
    half = largura_real / 2.0

    # Como a resolução é quadrada, a largura imaginária preservando o aspecto
    # também é 3e-3.
    zoom = Caso(
        nome="Zoom — vale dos cavalos-marinhos",
        max_iter=5000,
        re_min=cx - half,
        re_max=cx + half,
        im_min=cy - half,
        im_max=cy + half,
        seq_nome="sequencial_zoom_4096.bin",
        omp_nome="openmp_zoom_4096.bin",
    )

    return [padrao, zoom]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def sha256_arquivo(path: Path, bloco: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            dados = f.read(bloco)
            if not dados:
                break
            h.update(dados)
    return h.hexdigest()


def caminho_relativo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RAIZ.resolve()))
    except ValueError:
        return str(path.resolve())


def validar_binario(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo binário não encontrado: {path}")

    esperado = WIDTH * HEIGHT * np.dtype(np.int32).itemsize
    tamanho = path.stat().st_size
    if tamanho != esperado:
        raise ValueError(
            f"Tamanho inválido em {path.name}: {tamanho} bytes; "
            f"esperado {esperado} bytes para {WIDTH}x{HEIGHT} int32."
        )

    return np.memmap(path, dtype=np.int32, mode="r", shape=(HEIGHT, WIDTH), order="C")


# ---------------------------------------------------------------------------
# Carregamento seguro da função do professor
# ---------------------------------------------------------------------------

def carregar_funcao_professor(path: Path) -> Callable[..., Any]:
    """
    Carrega apenas a definição de mandelbrot() do arquivo do professor.

    Isso evita executar blocos de plotagem, geração de imagens ou código de
    nível superior que possa existir em gera_mandelbrot.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Referência do professor não encontrada: {path}\n"
            "Coloque gera_mandelbrot.py na mesma pasta deste script."
        )

    fonte = path.read_text(encoding="utf-8")
    arvore = ast.parse(fonte, filename=str(path))

    func_node = None
    for no in arvore.body:
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef)) and no.name == "mandelbrot":
            func_node = no
            break

    if func_node is None:
        raise RuntimeError(
            f"Não encontrei uma função chamada mandelbrot() em {path.name}."
        )

    modulo = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(modulo)

    namespace: dict[str, Any] = {
        "np": np,
        "numpy": np,
    }
    exec(compile(modulo, str(path), "exec"), namespace, namespace)

    fn = namespace.get("mandelbrot")
    if not callable(fn):
        raise RuntimeError("A função mandelbrot() não pôde ser carregada.")

    return fn


def _valor_por_nome(nome: str, caso: Caso) -> Any:
    n = nome.lower().strip()

    aliases = {
        # Limites reais
        "re_min": caso.re_min,
        "real_min": caso.re_min,
        "xmin": caso.re_min,
        "x_min": caso.re_min,
        "x0": caso.re_min,

        "re_max": caso.re_max,
        "real_max": caso.re_max,
        "xmax": caso.re_max,
        "x_max": caso.re_max,
        "x1": caso.re_max,

        # Limites imaginários
        "im_min": caso.im_min,
        "imag_min": caso.im_min,
        "ymin": caso.im_min,
        "y_min": caso.im_min,
        "y0": caso.im_min,

        "im_max": caso.im_max,
        "imag_max": caso.im_max,
        "ymax": caso.im_max,
        "y_max": caso.im_max,
        "y1": caso.im_max,

        # Dimensões
        "width": WIDTH,
        "largura": WIDTH,
        "nx": WIDTH,
        "n_x": WIDTH,
        "cols": WIDTH,
        "ncol": WIDTH,
        "ncols": WIDTH,

        "height": HEIGHT,
        "altura": HEIGHT,
        "ny": HEIGHT,
        "n_y": HEIGHT,
        "rows": HEIGHT,
        "nlin": HEIGHT,
        "nrows": HEIGHT,

        # Iterações
        "max_iter": caso.max_iter,
        "maxiter": caso.max_iter,
        "max_iters": caso.max_iter,
        "iterations": caso.max_iter,
        "iteracoes": caso.max_iter,
    }

    if n in aliases:
        return aliases[n]

    raise KeyError(nome)


def executar_referencia(fn: Callable[..., Any], caso: Caso) -> np.ndarray:
    """
    Chama mandelbrot() usando os nomes dos parâmetros da função do professor.

    Há também um fallback para a assinatura posicional convencional:
        mandelbrot(re_min, re_max, im_min, im_max, width, height, max_iter)
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    kwargs: dict[str, Any] = {}
    falharam: list[str] = []

    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        try:
            kwargs[p.name] = _valor_por_nome(p.name, caso)
        except KeyError:
            if p.default is inspect.Parameter.empty:
                falharam.append(p.name)

    if not falharam:
        resultado = fn(**kwargs)
    elif len(params) == 7:
        # Fallback para a ordem mais comum da referência utilizada na disciplina.
        resultado = fn(
            caso.re_min,
            caso.re_max,
            caso.im_min,
            caso.im_max,
            WIDTH,
            HEIGHT,
            caso.max_iter,
        )
    else:
        raise RuntimeError(
            "Não consegui mapear automaticamente a assinatura de mandelbrot().\n"
            f"Assinatura encontrada: {sig}\n"
            f"Parâmetros não reconhecidos: {', '.join(falharam)}"
        )

    # A referência pode retornar diretamente a matriz ou uma tupla contendo-a.
    candidatos: list[Any]
    if isinstance(resultado, tuple):
        candidatos = list(resultado)
    else:
        candidatos = [resultado]

    for obj in candidatos:
        if isinstance(obj, np.ndarray) and obj.shape == (HEIGHT, WIDTH):
            return obj

    # Aceita objeto conversível, desde que tenha exatamente o shape esperado.
    for obj in candidatos:
        try:
            arr = np.asarray(obj)
        except Exception:
            continue
        if arr.shape == (HEIGHT, WIDTH):
            return arr

    shapes = []
    for obj in candidatos:
        shapes.append(getattr(obj, "shape", type(obj).__name__))
    raise RuntimeError(
        "A função do professor executou, mas não retornou uma matriz "
        f"{HEIGHT}x{WIDTH}. Retornos observados: {shapes}"
    )


# ---------------------------------------------------------------------------
# Comparações
# ---------------------------------------------------------------------------

def comparar_matrizes(
    a: np.ndarray,
    b: np.ndarray,
    *,
    normalizar_professor: bool = False,
    max_iter: int | None = None,
    linhas_por_bloco: int = 128,
) -> Metricas:
    """
    Compara duas matrizes em blocos de linhas para não criar temporários gigantes.

    Se normalizar_professor=True, considera que `a` é a referência do professor
    e aplica, SOMENTE NO BLOCO TEMPORÁRIO:
        esperado = professor + 1, quando professor < MAX_ITER
        esperado = MAX_ITER, quando professor == MAX_ITER
    """
    if a.shape != (HEIGHT, WIDTH) or b.shape != (HEIGHT, WIDTH):
        raise ValueError("As matrizes devem ter shape 4096x4096.")

    if normalizar_professor and max_iter is None:
        raise ValueError("max_iter é obrigatório quando normalizar_professor=True.")

    total = WIDTH * HEIGHT
    diferentes = 0
    ate_1 = 0
    acima_1 = 0
    maior = 0

    for y0 in range(0, HEIGHT, linhas_por_bloco):
        y1 = min(HEIGHT, y0 + linhas_por_bloco)

        aa = np.asarray(a[y0:y1], dtype=np.int64)
        bb = np.asarray(b[y0:y1], dtype=np.int64)

        if normalizar_professor:
            # Cria somente um temporário do bloco corrente.
            esperado = aa.copy()
            mascara_escape = esperado < int(max_iter)
            esperado[mascara_escape] += 1
            aa = esperado

        dif = np.abs(aa - bb)
        mask = dif != 0
        qtd = int(np.count_nonzero(mask))

        if qtd:
            diferentes += qtd
            d = dif[mask]
            ate_1 += int(np.count_nonzero(d <= 1))
            acima_1 += int(np.count_nonzero(d > 1))
            maior = max(maior, int(d.max()))

    percentual = 100.0 * diferentes / total

    return Metricas(
        total=total,
        diferentes=diferentes,
        percentual=percentual,
        ate_1=ate_1,
        acima_1=acima_1,
        maior_diferenca=maior,
    )


def atende_criterio_formal(m: Metricas) -> bool:
    return (
        m.percentual <= CRITERIO_PERCENTUAL_MAX
        and m.maior_diferenca <= CRITERIO_DIF_MAX
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return f"{x:.6f}%".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def gerar_markdown(
    resultados: list[dict[str, Any]],
    hash_professor: str,
) -> str:
    linhas: list[str] = []

    linhas += [
        "# Validação contra a referência do professor",
        "",
        "## Objetivo",
        "",
        "Este documento registra uma validação reprodutível das implementações "
        "sequencial e OpenMP em C contra a função `mandelbrot()` presente em "
        "`codigos_auxiliares/gera_mandelbrot.py`.",
        "",
        "A matriz da referência do professor foi gerada **somente em memória**. "
        "Nenhuma cópia dessa matriz foi persistida em disco.",
        "",
        "A implementação C adota a convenção de armazenar o **número de iterações "
        "executadas** (`i + 1` para um pixel que escapa), enquanto a referência do "
        "professor registra o **índice da iteração** (`i`, iniciado em zero). "
        "Para permitir a comparação conceitual das duas convenções, este script "
        "aplica apenas durante o diagnóstico a transformação `professor + 1` nos "
        "pixels em que `professor < MAX_ITER`; pixels que atingem `MAX_ITER` "
        "permanecem inalterados.",
        "",
        "> **Importante:** essa normalização é apenas diagnóstica. Ela não modifica "
        "os binários das implementações C e não transforma automaticamente a "
        "comparação em aprovação pelo critério formal do enunciado.",
        "",
        "## Ambiente da validação",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- Plataforma: `{platform.platform()}`",
        f"- Referência: `{caminho_relativo(ARQ_PROFESSOR)}`",
        f"- SHA-256 de `gera_mandelbrot.py`: `{hash_professor}`",
        "",
        "## 1. Consistência interna: Sequencial × OpenMP",
        "",
        "| Caso | Pixels diferentes | Maior diferença | Resultado |",
        "|---|---:|---:|---|",
    ]

    for r in resultados:
        m: Metricas = r["interno"]
        ok = m.diferentes == 0
        linhas.append(
            f"| {r['caso'].nome} | {fmt_int(m.diferentes)} | "
            f"{fmt_int(m.maior_diferenca)} | "
            f"{'**IDÊNTICOS**' if ok else '**DIVERGENTES**'} |"
        )

    linhas += [
        "",
        "A comparação acima verifica diretamente os `.bin` preservados em "
        "`output/ref`. Igualdade aqui demonstra que a paralelização OpenMP "
        "preservou o resultado da implementação sequencial desenvolvida.",
        "",
        "## 2. Comparação normalizada de convenção com a referência",
        "",
        "| Caso / implementação | Total de pixels | Divergentes | % divergente | Dif. ≤ 1 | Dif. > 1 | Maior diferença |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in resultados:
        for rotulo, chave in [("Sequencial", "seq_prof"), ("OpenMP", "omp_prof")]:
            m: Metricas = r[chave]
            linhas.append(
                f"| {r['caso'].nome} — {rotulo} | {fmt_int(m.total)} | "
                f"{fmt_int(m.diferentes)} | {fmt_pct(m.percentual)} | "
                f"{fmt_int(m.ate_1)} | {fmt_int(m.acima_1)} | "
                f"{fmt_int(m.maior_diferenca)} |"
            )

    linhas += [
        "",
        "## 3. Critério formal do enunciado",
        "",
        f"O critério verificado neste relatório é: no máximo "
        f"**{str(CRITERIO_PERCENTUAL_MAX).replace('.', ',')}%** de pixels "
        f"divergentes e, entre eles, diferença máxima de "
        f"**{CRITERIO_DIF_MAX} iteração**.",
        "",
        "| Caso | Sequencial | OpenMP | Observação |",
        "|---|---|---|---|",
    ]

    for r in resultados:
        ms: Metricas = r["seq_prof"]
        mo: Metricas = r["omp_prof"]
        oks = atende_criterio_formal(ms)
        oko = atende_criterio_formal(mo)

        obs = []
        if ms.percentual > CRITERIO_PERCENTUAL_MAX:
            obs.append("percentual acima do limite")
        if ms.maior_diferenca > CRITERIO_DIF_MAX:
            obs.append("diferença por pixel acima de 1")
        if not obs:
            obs.append("dentro da tolerância verificada")

        linhas.append(
            f"| {r['caso'].nome} | "
            f"{'**ATENDE**' if oks else '**NÃO ATENDE**'} | "
            f"{'**ATENDE**' if oko else '**NÃO ATENDE**'} | "
            f"{'; '.join(obs)} |"
        )

    linhas += [
        "",
        "## 4. Evidências dos arquivos C",
        "",
        "| Caso | Arquivo sequencial | SHA-256 | Arquivo OpenMP | SHA-256 |",
        "|---|---|---|---|---|",
    ]

    for r in resultados:
        linhas.append(
            f"| {r['caso'].nome} | `{caminho_relativo(r['seq_path'])}` | "
            f"`{r['seq_hash']}` | `{caminho_relativo(r['omp_path'])}` | "
            f"`{r['omp_hash']}` |"
        )

    linhas += [
        "",
        "## 5. Conclusão",
        "",
    ]

    internos_ok = all(r["interno"].diferentes == 0 for r in resultados)
    formais_ok = all(
        atende_criterio_formal(r["seq_prof"]) and atende_criterio_formal(r["omp_prof"])
        for r in resultados
    )

    if internos_ok:
        linhas.append(
            "- **Consistência interna atestada:** os binários sequencial e OpenMP "
            "são idênticos nos dois casos oficiais."
        )
    else:
        linhas.append(
            "- **Consistência interna NÃO atestada:** há divergência entre "
            "sequencial e OpenMP em pelo menos um caso."
        )

    linhas.append(
        "- **Equivalência de convenção tratada explicitamente:** a diferença "
        "`i` versus `i + 1` é normalizada somente na comparação diagnóstica, "
        "sem alterar os arquivos C."
    )

    if formais_ok:
        linhas.append(
            "- **Critério formal verificado:** após a comparação definida acima, "
            "os dois casos atendem aos limites configurados neste script."
        )
    else:
        linhas.append(
            "- **Critério formal não integralmente satisfeito:** apesar da forte "
            "consistência interna e da equivalência de convenção, pelo menos um "
            "caso ultrapassa a tolerância formal configurada. O relatório deve "
            "manter essa distinção de forma explícita."
        )

    linhas += [
        "",
        "Este arquivo foi produzido automaticamente por "
        "`codigos_auxiliares/validar_professor.py`.",
        "",
    ]

    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

        fn_professor = carregar_funcao_professor(ARQ_PROFESSOR)
        hash_professor = sha256_arquivo(ARQ_PROFESSOR)

        resultados: list[dict[str, Any]] = []

        for caso in casos_oficiais():
            seq_path = PASTA_REF / caso.seq_nome
            omp_path = PASTA_REF / caso.omp_nome

            seq = validar_binario(seq_path)
            omp = validar_binario(omp_path)

            print(f"[{caso.nome}] gerando referência do professor em memória...")
            professor = executar_referencia(fn_professor, caso)

            # A matriz do professor nunca é salva.
            # As comparações são feitas por blocos para limitar temporários.
            interno = comparar_matrizes(seq, omp)

            seq_prof = comparar_matrizes(
                professor,
                seq,
                normalizar_professor=True,
                max_iter=caso.max_iter,
            )

            omp_prof = comparar_matrizes(
                professor,
                omp,
                normalizar_professor=True,
                max_iter=caso.max_iter,
            )

            resultados.append(
                {
                    "caso": caso,
                    "seq_path": seq_path,
                    "omp_path": omp_path,
                    "seq_hash": sha256_arquivo(seq_path),
                    "omp_hash": sha256_arquivo(omp_path),
                    "interno": interno,
                    "seq_prof": seq_prof,
                    "omp_prof": omp_prof,
                }
            )

            # Libera explicitamente a matriz grande antes do próximo caso.
            del professor
            del seq
            del omp

        md = gerar_markdown(resultados, hash_professor)
        ARQ_RELATORIO.write_text(md, encoding="utf-8")

        print()
        print(f"Validação concluída.")
        print(f"Relatório: {ARQ_RELATORIO}")
        print("Nenhuma matriz da referência do professor foi salva em disco.")
        return 0

    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
