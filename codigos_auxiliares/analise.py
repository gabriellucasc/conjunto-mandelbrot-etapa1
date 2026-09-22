#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analise.py - Etapa 1 (OpenMP) - tabelas e graficos do Benchmark do Mandelbrot (DEC107)

Le resultados/resultados_consolidados.csv (gerado pelo benchmark.py) e produz:
  - tabelas (Speedup, Eficiencia, F_LB, weak scaling, corretude ...);
  - PREVISAO do desbalanceamento a partir do custo por linha da matriz de contagens;
  - graficos em graficos/.

DOIS USOS (a mesma logica, sem duplicar codigo):
  1) terminal :  python codigos_auxiliares/analise.py        -> gera TODOS os PNGs e tabelas
  2) notebook :  import analise as an ; an.fig_strong(...)   -> uma funcao por pergunta

DEFINICOES (as do professor, Aula 03)
  Speedup     S = T_seq / T_p                (T_seq = mediana do sequencial na mesma resolucao)
  Eficiencia  E = S / p
  F_LB        (Tmax - Tmin) / Tmax           (por thread; 0 = perfeito, 1 = pessimo)
  Karp-Flatt  e = (1/S - 1/p) / (1 - 1/p)    (fracao serial EFETIVA; inclui overheads e hardware)
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACAO
# =============================================================================

RAIZ = Path(__file__).resolve().parent.parent
CSV_OFICIAL = RAIZ / "resultados" / "resultados_consolidados.csv"
PASTA_GRAFICOS = RAIZ / "graficos"
PASTA_TABELAS = RAIZ / "resultados" / "tabelas"
REF_DIR = RAIZ / "output" / "ref"

RES_BASE = 4096                 # resolucao oficial (strong scaling)
NUCLEOS_FISICOS = 6             # i5-12400F (edite se mudar de maquina)
POLITICA_WEAK = ("dynamic", 1)  # politica usada no weak scaling

CASOS = {"padrao": "Região padrão", "zoom": "Zoom (cavalos-marinhos)"}
POLITICAS_PLOT = ["static (blocos)", "static,1", "dynamic,1", "dynamic,16", "guided,1"]
COR = {"static (blocos)": "#d62728", "static,1": "#ff7f0e", "dynamic,1": "#1f77b4",
       "dynamic,16": "#17becf", "guided,1": "#2ca02c"}
CHUNKS_GRADE = [0, 1, 4, 16, 64, 256]     # 0 = static em blocos (sem chunk)
MSG_SEM_BIN = ("Nao encontrei output/ref/sequencial_padrao_4096.bin e sequencial_zoom_4096.bin "
               "(o benchmark.py os guarda). Rode o benchmark ou copie esses arquivos para output/ref/.")


def rotulo(schedule: str, chunk: int) -> str:
    if schedule == "nenhum":
        return "sequencial"
    if schedule == "static" and int(chunk) == 0:
        return "static (blocos)"
    return f"{schedule},{int(chunk)}"


# =============================================================================
# 2. CARREGAMENTO E TABELAS BASICAS
# =============================================================================

def carregar(csv_path=CSV_OFICIAL) -> pd.DataFrame:
    """Le o CSV consolidado (1 linha = 1 execucao)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nao encontrei {csv_path}. Rode antes o benchmark.py.")
    df = pd.read_csv(csv_path)
    df["hash_ok"] = df["hash_ok"].astype(str) == "True"
    df["config_ok"] = df["config_ok"].astype(str) == "True"
    df["pol"] = [rotulo(s, c) for s, c in zip(df["schedule"], df["chunk"])]
    return df


def baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo sequencial de referencia por (caso, resolucao).
    Em 4096: mediana das execucoes baseline (antes + depois). Nas demais resolucoes
    (weak scaling) so existe 1 execucao de referencia."""
    s = df[df["versao"] == "sequencial"]
    base = s[s["fase"].isin(["baseline_ini", "baseline_fim"])].groupby(["caso", "width"])["tempo_calculo_s"].median()
    ref = s[s["fase"] == "referencia"].groupby(["caso", "width"])["tempo_calculo_s"].median()
    out = pd.concat([base, ref[~ref.index.isin(base.index)]])
    return out.rename("t_seq").reset_index()


def tabela_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Estabilidade do baseline: mediana no inicio x no fim da bateria (deriva)."""
    s = df[(df["versao"] == "sequencial") & (df["width"] == RES_BASE)]
    linhas = []
    for caso, g in s.groupby("caso"):
        ini = g[g["fase"] == "baseline_ini"]["tempo_calculo_s"]
        fim = g[g["fase"] == "baseline_fim"]["tempo_calculo_s"]
        todos = pd.concat([ini, fim])
        linhas.append({"caso": caso, "n_ini": len(ini), "mediana_ini_s": ini.median(),
                       "n_fim": len(fim), "mediana_fim_s": fim.median(),
                       "deriva_pct": 100 * (fim.median() - ini.median()) / ini.median(),
                       "baseline_s": todos.median(), "cv_pct": 100 * todos.std() / todos.mean()})
    return pd.DataFrame(linhas)


def resumo(df: pd.DataFrame, descartar_primeira: bool = False) -> pd.DataFrame:
    """Uma linha por configuracao OpenMP (caso, resolucao, threads, schedule, chunk).
    Usa a MEDIANA das repeticoes. Nao descarta o warm-up por padrao: nos dados medidos a
    1a repeticao ficou em 1,000 da mediana das demais (sem efeito de warm-up)."""
    o = df[(df["versao"] == "openmp") & (df["fase"] == "medicao")].copy()
    if descartar_primeira:
        o = o[o["rep"] > 1]
    o["razao_tmax_tmed"] = o["carga_max"] / o["carga_media"]
    chaves = ["caso", "width", "threads", "schedule", "chunk"]
    r = (o.groupby(chaves)
           .agg(n=("tempo_calculo_s", "count"), t_med=("tempo_calculo_s", "median"),
                t_min=("tempo_calculo_s", "min"), t_max=("tempo_calculo_s", "max"),
                t_media=("tempo_calculo_s", "mean"), t_desvio=("tempo_calculo_s", "std"),
                flb=("fator_balanceamento", "median"), razao_tmax_tmed=("razao_tmax_tmed", "median"),
                w_max=("trabalho_max", "median"), w_media=("trabalho_media", "median"),
                escrita_s=("tempo_escrita_s", "median"))
           .reset_index())
    r["cv_pct"] = 100 * r["t_desvio"] / r["t_media"]
    r["pol"] = [rotulo(s, c) for s, c in zip(r["schedule"], r["chunk"])]
    r = r.merge(baseline(df), on=["caso", "width"], how="left")
    r["speedup"] = r["t_seq"] / r["t_med"]
    r["speedup_min"] = r["t_seq"] / r["t_max"]      # pior repeticao paralela
    r["speedup_max"] = r["t_seq"] / r["t_min"]      # melhor repeticao paralela
    r["eficiencia"] = 100 * r["speedup"] / r["threads"]
    p = r["threads"].astype(float)
    r["karp_flatt"] = np.where(p > 1, (1 / r["speedup"] - 1 / p) / (1 - 1 / p), np.nan)
    return r


def tabela_corretude(df: pd.DataFrame) -> pd.DataFrame:
    """Quantas execucoes tiveram o .bin identico ao do sequencial (hash) e a configuracao pedida."""
    t = (df.groupby(["caso", "versao"])
           .agg(execucoes=("run_id", "count"), hash_ok=("hash_ok", "sum"), config_ok=("config_ok", "sum"))
           .reset_index())
    t["todas_validas"] = (t["execucoes"] == t["hash_ok"]) & (t["execucoes"] == t["config_ok"])
    return t


def tabela_strong(r: pd.DataFrame, caso: str) -> pd.DataFrame:
    """Speedup e Eficiencia (%) por politica x numero de threads, 4096x4096."""
    d = r[(r["caso"] == caso) & (r["width"] == RES_BASE) & (r["threads"] >= 2) & r["pol"].isin(POLITICAS_PLOT)]
    s = d.pivot(index="pol", columns="threads", values="speedup").reindex(POLITICAS_PLOT)
    e = d.pivot(index="pol", columns="threads", values="eficiencia").reindex(POLITICAS_PLOT)
    return pd.concat({"Speedup": s.round(2), "Eficiência (%)": e.round(0)}, axis=1)


def tabela_melhores(r: pd.DataFrame, caso: str, n: int = 8) -> pd.DataFrame:
    """As n configuracoes mais rapidas. Atencao: diferencas pequenas entre as primeiras
    estao dentro do ruido (veja a faixa min-max)."""
    d = r[(r["caso"] == caso) & (r["width"] == RES_BASE)].sort_values("t_med").head(n)
    return pd.DataFrame({"threads": d["threads"], "politica": d["pol"], "tempo_s": d["t_med"].round(3),
                         "faixa_min_max_s": [f"{a:.3f}–{b:.3f}" for a, b in zip(d["t_min"], d["t_max"])],
                         "speedup": d["speedup"].round(2), "eficiencia_%": d["eficiencia"].round(0),
                         "F_LB": d["flb"].round(3), "cv_%": d["cv_pct"].round(1)}).reset_index(drop=True)


def tabela_qualidade(r: pd.DataFrame) -> pd.DataFrame:
    """Ruido das medicoes (coeficiente de variacao das repeticoes) por numero de threads."""
    d = r[r["width"] == RES_BASE]
    return (d.groupby("threads")["cv_pct"].agg(configuracoes="count", cv_mediano="median", cv_maximo="max")
             .round(2).reset_index())


def tabela_escrita(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo de ESCRITA do .bin, reportado a parte (o enunciado exclui do tempo principal)."""
    return (df.groupby(["versao", "width"])["tempo_escrita_s"].median().round(3)
              .rename("escrita_mediana_s").reset_index())


# =============================================================================
# 3. PREVISAO DO DESBALANCEAMENTO (a partir do custo por linha)
# =============================================================================

def custo_por_linha(bin_path) -> np.ndarray:
    """Trabalho (soma das iteracoes) de cada LINHA da matriz de contagens.
    E exatamente o que o laco externo distribui entre as threads."""
    n = Path(bin_path).stat().st_size // 4
    lado = int(round(math.sqrt(n)))
    w = np.empty(lado, dtype=np.int64)
    with open(bin_path, "rb") as f:
        for y0 in range(0, lado, 256):
            nl = min(256, lado - y0)
            a = np.fromfile(f, dtype=np.int32, count=nl * lado).reshape(nl, lado)
            w[y0:y0 + nl] = a.sum(axis=1, dtype=np.int64)
    return w


def carregar_custos() -> dict:
    """Custo por linha do caso padrao e do zoom (usa os .bin de output/ref, se existirem)."""
    out = {}
    for caso in CASOS:
        p = REF_DIR / f"sequencial_{caso}_{RES_BASE}.bin"
        if p.exists():
            out[caso] = custo_por_linha(p)
    return out


def prever(w: np.ndarray, p: int, sched: str, chunk: int) -> np.ndarray:
    """Trabalho que CADA thread receberia, supondo threads de mesma velocidade e sem overhead.
    static: exato (distribuicao deterministica). dynamic: simulacao gulosa.
    guided: aproximado (blocos de ceil(restante / threads), minimo = chunk)."""
    n = len(w)
    if sched == "static" and chunk == 0:                      # blocos contiguos
        q, resto = divmod(n, p)
        out, i = [], 0
        for t in range(p):
            k = q + 1 if t < resto else q
            out.append(w[i:i + k].sum())
            i += k
        return np.array(out)
    if sched == "static":                                     # blocos de 'chunk' em rodizio
        out = np.zeros(p, dtype=np.int64)
        for k, i in enumerate(range(0, n, chunk)):
            out[k % p] += w[i:i + chunk].sum()
        return out
    fila = [(0, t) for t in range(p)]                         # dynamic / guided
    heapq.heapify(fila)
    total, i = np.zeros(p, dtype=np.int64), 0
    while i < n:
        resto = n - i
        q = max(chunk, math.ceil(resto / p)) if sched == "guided" else chunk
        q = min(q, resto)
        custo = int(w[i:i + q].sum())
        i += q
        livre, t = heapq.heappop(fila)
        total[t] += custo
        heapq.heappush(fila, (livre + custo, t))
    return total


def adicionar_previsao(r: pd.DataFrame, custos: dict) -> pd.DataFrame:
    """Acrescenta flb_prev (F_LB previsto) e efic_bal_prev (% maximo so por balanceamento)."""
    r = r.copy()
    r["flb_prev"], r["efic_bal_prev"] = np.nan, np.nan
    for i, x in r.iterrows():
        if x["width"] != RES_BASE or x["threads"] < 2 or x["caso"] not in custos:
            continue
        tot = prever(custos[x["caso"]], int(x["threads"]), x["schedule"], int(x["chunk"]))
        r.at[i, "flb_prev"] = (tot.max() - tot.min()) / tot.max()
        r.at[i, "efic_bal_prev"] = 100 * tot.mean() / tot.max()
    return r


# =============================================================================
# 4. WEAK SCALING E LEIS DE AMDAHL / GUSTAFSON
# =============================================================================

def weak(r: pd.DataFrame) -> pd.DataFrame:
    """Pontos de weak scaling: p threads em uma imagem de lado ~ RES_BASE*raiz(p)
    (pixels por thread constantes). E_weak = T(1 thread, 4096) / T(p, lado)."""
    s, c = POLITICA_WEAK
    x = r[(r["caso"] == "padrao") & (r["schedule"] == s) & (r["chunk"] == c)].copy()
    x = x[[w == int(round(RES_BASE * math.sqrt(t))) for t, w in zip(x["threads"], x["width"])]]
    x = x.sort_values("threads")
    if x.empty or 1 not in set(x["threads"]):
        return x
    x["pixels_por_thread_M"] = (x["width"] ** 2 / x["threads"] / 1e6).round(2)
    x["e_weak_pct"] = 100 * x.loc[x["threads"] == 1, "t_med"].iloc[0] / x["t_med"]
    x["speedup_escalado"] = x["threads"] * x["e_weak_pct"] / 100
    return x[["threads", "width", "pixels_por_thread_M", "t_med", "t_min", "t_max", "e_weak_pct",
              "speedup_escalado", "t_seq"]].reset_index(drop=True)


def alfa_gustafson(w: pd.DataFrame) -> float:
    """Ajusta alpha em  E_weak(p) = 1 - alpha*(1 - 1/p)   (Gustafson: S_esc = p - alpha*(p-1))."""
    w = w[w["threads"] > 1]
    x = 1 - 1 / w["threads"].to_numpy(dtype=float)
    y = 1 - w["e_weak_pct"].to_numpy(dtype=float) / 100
    return float((x * y).sum() / (x * x).sum())


def f_efetivo(r: pd.DataFrame, caso: str, pol: str = "dynamic,1") -> float:
    """Fracao serial EFETIVA (Karp-Flatt mediana) da politica de referencia."""
    d = r[(r["caso"] == caso) & (r["width"] == RES_BASE) & (r["pol"] == pol) & (r["threads"] >= 2)]
    return float(d["karp_flatt"].median())


def amdahl(f: float, p):
    return 1 / (f + (1 - f) / np.asarray(p, dtype=float))


# =============================================================================
# 5. GRAFICOS  (cada funcao: uma pergunta, uma figura; salva em graficos/ e devolve a Figure)
# =============================================================================

def _fim(fig, nome, salvar):
    fig.tight_layout()
    if salvar:
        PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PASTA_GRAFICOS / nome, dpi=150)
    plt.close(fig)          # no notebook a Figure devolvida e exibida uma vez so
    return fig


def _xthreads(ax, threads):
    ax.set_xticks(sorted(threads))
    ax.axvline(NUCLEOS_FISICOS, color="gray", ls=":", lw=1)
    ax.grid(alpha=0.25)


def fig_strong(r, caso, metrica, salvar=True):
    """Como o desempenho evolui com as threads? metrica: 'tempo' | 'speedup' | 'eficiencia'."""
    d = r[(r["caso"] == caso) & (r["width"] == RES_BASE)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for pol in POLITICAS_PLOT:
        x = d[(d["pol"] == pol) & (d["threads"] >= 2)].sort_values("threads")
        if x.empty:
            continue
        if metrica == "tempo":
            ax.plot(x["threads"], x["t_med"], "o-", color=COR[pol], label=pol)
        elif metrica == "speedup":
            yerr = [x["speedup"] - x["speedup_min"], x["speedup_max"] - x["speedup"]]
            ax.errorbar(x["threads"], x["speedup"], yerr=yerr, fmt="o-", capsize=3, color=COR[pol], label=pol)
        else:
            ax.plot(x["threads"], x["eficiencia"], "o-", color=COR[pol], label=pol)
    pmax = int(d["threads"].max())
    if metrica == "tempo":
        ax.axhline(d["t_seq"].iloc[0], color="k", ls="--", label=f"sequencial ({d['t_seq'].iloc[0]:.2f} s)")
        ax.set_ylabel("Tempo de cálculo (s)")
    elif metrica == "speedup":
        ax.plot([1, pmax], [1, pmax], "k--", label="ideal (S = p)")
        f = f_efetivo(r, caso)
        ps = np.linspace(1, pmax, 100)
        ax.plot(ps, amdahl(f, ps), ":", color="gray", lw=2, label=f"Amdahl, f = {100 * f:.1f}% (Karp–Flatt)")
        ax.set_ylabel("Speedup")
    else:
        ax.axhline(100, color="k", ls="--", label="ideal (100%)")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Eficiência (%)")
    ax.set_xlabel("Threads (linha pontilhada = 6 núcleos físicos)")
    nome_metrica = {"tempo": "Tempo", "speedup": "Speedup", "eficiencia": "Eficiência"}[metrica]
    ax.set_title(f"{nome_metrica} × threads — {CASOS[caso]}")
    _xthreads(ax, d["threads"].unique())
    ax.legend(fontsize=8)
    return _fim(fig, f"strong_{metrica}_{caso}.png", salvar)


def fig_custo_por_linha(custos, salvar=True):
    """Por que o problema e desbalanceado? Custo (iteracoes) de cada linha da imagem."""
    if not custos:
        raise FileNotFoundError(MSG_SEM_BIN)
    fig, eixos = plt.subplots(1, len(custos), figsize=(6.2 * len(custos), 4.2))
    for ax, (caso, w) in zip(np.atleast_1d(eixos), custos.items()):
        n = len(w)
        central = w[n // 4:3 * n // 4].sum() / w.sum()
        ax.fill_between(range(n), w / 1e3, step="mid", alpha=0.6, color="#1f77b4")
        for k in (1, 2, 3):
            ax.axvline(k * n / 4, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{CASOS[caso]}\n50% central das linhas = {100 * central:.0f}% do trabalho")
        ax.set_xlabel("linha da imagem (y)\n(tracejado: blocos do static com 4 threads)")
        ax.set_ylabel("trabalho da linha (milhares de iterações)")
        ax.grid(alpha=0.25)
    return _fim(fig, "custo_por_linha.png", salvar)


def fig_previsto_medido(r, salvar=True):
    """O modelo do custo por linha explica o F_LB medido? (cada ponto = 1 configuracao)"""
    if r["flb_prev"].isna().all():
        raise FileNotFoundError(MSG_SEM_BIN)
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, caso in zip(eixos, CASOS):
        d = r[(r["caso"] == caso) & r["flb_prev"].notna()]
        # destaca os "static com chunk" cujo F_LB medido foge do previsto (o modelo supoe threads iguais)
        fora = ((d["flb"] - d["flb_prev"]).abs() > 0.05) & d["pol"].str.startswith("static") & (d["pol"] != "static (blocos)")
        sc = ax.scatter(d.loc[~fora, "flb_prev"], d.loc[~fora, "flb"], c=d.loc[~fora, "threads"], cmap="viridis", s=28)
        if fora.any():
            ax.scatter(d.loc[fora, "flb_prev"], d.loc[fora, "flb"], marker="x", color="red", s=50,
                       label="static com chunk: desvio > 0,05")
            ax.legend(fontsize=8, loc="upper left")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        erro = (d["flb"] - d["flb_prev"]).abs().median()
        ax.set_title(f"{CASOS[caso]}  (erro absoluto mediano = {erro:.3f})")
        ax.set_xlabel("F_LB PREVISTO (trabalho por linha)")
        ax.set_ylabel("F_LB MEDIDO (tempo por thread)")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="threads")
    return _fim(fig, "previsto_x_medido.png", salvar)


def _matriz(d, coluna):
    m = np.full((3, len(CHUNKS_GRADE)), np.nan)
    for i, s in enumerate(["static", "dynamic", "guided"]):
        for j, c in enumerate(CHUNKS_GRADE):
            x = d[(d["schedule"] == s) & (d["chunk"] == c)]
            if len(x):
                m[i, j] = x[coluna].iloc[0]
    return m


def fig_heatmap(r, p=12, metrica="tempo", salvar=True):
    """Qual combinacao schedule x chunk e melhor? metrica: 'tempo' | 'flb' | 'speedup'."""
    coluna, cmap, titulo = {"tempo": ("t_med", "YlOrRd", "Tempo (s)"), "flb": ("flb", "YlOrRd", "F_LB"),
                            "speedup": ("speedup", "YlGn", "Speedup")}[metrica]
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, caso in zip(eixos, CASOS):
        d = r[(r["caso"] == caso) & (r["width"] == RES_BASE) & (r["threads"] == p)]
        m = _matriz(d, coluna)
        im = ax.imshow(np.ma.masked_invalid(m), cmap=plt.get_cmap(cmap).copy(), aspect="auto")
        im.get_cmap().set_bad("lightgray")
        vmax = np.nanmax(m)
        for (i, j), v in np.ndenumerate(m):
            if not np.isnan(v):
                cor = "white" if v > 0.65 * vmax else "black"      # texto legivel nas celulas escuras
                ax.text(j, i, f"{v:.2f}" if metrica != "tempo" else f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color=cor)
        ax.set_xticks(range(len(CHUNKS_GRADE)), ["blocos" if c == 0 else str(c) for c in CHUNKS_GRADE])
        ax.set_yticks(range(3), ["static", "dynamic", "guided"])
        ax.set_xlabel("chunk (blocos = static sem chunk; cinza = não medido)")
        ax.set_title(f"{titulo} — {CASOS[caso]} ({p} threads)")
        fig.colorbar(im, ax=ax)
    return _fim(fig, f"heatmap_{metrica}.png", salvar)


def fig_flb_politicas(r, p=12, salvar=True):
    """Qual politica equilibra melhor? Barras = F_LB medido; losango = previsto pelo modelo."""
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, caso in zip(eixos, CASOS):
        d = r[(r["caso"] == caso) & (r["width"] == RES_BASE) & (r["threads"] == p)].set_index("pol").reindex(POLITICAS_PLOT)
        ax.bar(range(len(d)), d["flb"], color=[COR[x] for x in d.index], label="medido")
        if "flb_prev" in d and d["flb_prev"].notna().any():
            ax.plot(range(len(d)), d["flb_prev"], "kD", label="previsto (modelo)")
        ax.set_xticks(range(len(d)), d.index, rotation=20, fontsize=8)
        ax.set_title(f"{CASOS[caso]} — {p} threads")
        ax.set_ylabel("F_LB  (0 = perfeito)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    return _fim(fig, "flb_politicas.png", salvar)


def fig_chunk(r, caso, p=12, salvar=True):
    """Qual o efeito do chunk? Esquerda: tempo. Direita: F_LB (overhead x balanceamento)."""
    d = r[(r["caso"] == caso) & (r["width"] == RES_BASE) & (r["threads"] == p)]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, coluna, rot in zip(eixos, ["t_med", "flb"], ["Tempo de cálculo (s)", "F_LB"]):
        for s, cor in (("static", "#ff7f0e"), ("dynamic", "#1f77b4"), ("guided", "#2ca02c")):
            x = d[(d["schedule"] == s) & (d["chunk"] >= 1)].sort_values("chunk")
            ax.plot(x["chunk"], x[coluna], "o-", color=cor, label=s)
        blocos = d[(d["schedule"] == "static") & (d["chunk"] == 0)]
        if len(blocos):
            ax.axhline(blocos[coluna].iloc[0], color="#d62728", ls="--", label="static (blocos)")
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.set_xlabel("chunk (linhas por pedaço)")
        ax.set_ylabel(rot)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Efeito do chunk — {CASOS[caso]} ({p} threads)")
    return _fim(fig, f"chunk_{caso}.png", salvar)


def fig_weak(r, salvar=True):
    """O desempenho se mantem quando o problema cresce junto com as threads?"""
    w = weak(r)
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = eixos[0]
    ax.plot(w["threads"], w["t_med"], "o-", label="medido")
    ax.axhline(w["t_med"].iloc[0], color="gray", ls="--", label="ideal (tempo constante)")
    for x in w.itertuples():
        ax.annotate(f"{x.width}²", (x.threads, x.t_med), fontsize=8, xytext=(4, 5), textcoords="offset points")
    ax.set_xlabel("Threads (pixels por thread constantes)")
    ax.set_ylabel("Tempo de cálculo (s)")
    ax.set_title("Weak scaling — tempo")
    _xthreads(ax, w["threads"])
    ax.legend(fontsize=8)
    ax = eixos[1]
    a = alfa_gustafson(w)
    ps = np.linspace(1, w["threads"].max(), 100)
    ax.plot(w["threads"], w["e_weak_pct"], "o-", color="darkorange", label="medida")
    ax.plot(ps, 100 * (1 - a * (1 - 1 / ps)), ":", color="gray", lw=2, label=f"Gustafson, α = {100 * a:.1f}% (ajustado)")
    ax.axhline(100, color="k", ls="--", lw=1)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Eficiência weak (%)")
    ax.set_title("Weak scaling — eficiência")
    _xthreads(ax, w["threads"])
    ax.legend(fontsize=8)
    return _fim(fig, "weak_scaling.png", salvar)


# =============================================================================
# 6. VALIDACAO PIXEL A PIXEL E MINIATURAS (usadas pelo notebook)
# =============================================================================

def comparar_binarios(bin_a, bin_b) -> dict:
    """Compara duas matrizes de contagens em blocos (sem manter arquivos abertos)."""
    a, b = Path(bin_a), Path(bin_b)
    if a.stat().st_size != b.stat().st_size:
        raise ValueError("Os .bin tem tamanhos diferentes (resolucoes diferentes?).")
    n = a.stat().st_size // 4
    lado = int(round(math.sqrt(n)))
    diferentes, maior = 0, 0
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x = np.fromfile(fa, dtype=np.int32, count=1 << 22)
            if x.size == 0:
                break
            y = np.fromfile(fb, dtype=np.int32, count=x.size)
            m = x != y
            if m.any():
                diferentes += int(m.sum())
                maior = max(maior, int(np.abs(x[m].astype(np.int64) - y[m]).max()))
    return {"dimensao": f"{lado}x{lado}", "pixels": n, "pixels_diferentes": diferentes,
            "maior_diferenca": maior, "igualdade_exata": diferentes == 0}


def _miniatura(path, lado, alvo=700):
    passo = max(1, lado // alvo)
    linhas = []
    with open(path, "rb") as f:
        for y in range(0, lado, passo):
            f.seek(y * lado * 4)
            linhas.append(np.fromfile(f, dtype=np.int32, count=lado)[::passo])
    return np.array(linhas)


def miniaturas(bin_a, bin_b, titulos=("Sequencial", "OpenMP"), salvar_como=None):
    """Mostra as duas matrizes lado a lado (escala log) para inspecao visual."""
    lado = int(round(math.sqrt(Path(bin_a).stat().st_size // 4)))
    fig, eixos = plt.subplots(1, 2, figsize=(10, 5))
    for ax, caminho, t in zip(eixos, (bin_a, bin_b), titulos):
        ax.imshow(np.log1p(_miniatura(caminho, lado)), cmap="magma")
        ax.set_title(f"{t} ({lado}×{lado})")
        ax.axis("off")
    fig.tight_layout()
    if salvar_como:
        PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PASTA_GRAFICOS / salvar_como, dpi=130)
    plt.close(fig)
    return fig


def par_final(caso: str):
    """(sequencial, openmp) guardados em output/ref pelo benchmark.py, se existirem."""
    a = REF_DIR / f"sequencial_{caso}_{RES_BASE}.bin"
    b = REF_DIR / f"openmp_{caso}_{RES_BASE}.bin"
    return (a, b) if a.exists() and b.exists() else (None, None)


# =============================================================================
# 7. GERAR TUDO (terminal)
# =============================================================================

def gerar_tudo(csv_path=CSV_OFICIAL) -> None:
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    df = carregar(csv_path)
    custos = carregar_custos()
    r = adicionar_previsao(resumo(df), custos) if custos else resumo(df)

    print(f"{len(df)} execucoes lidas de {csv_path}")
    corr = tabela_corretude(df)
    print("Corretude:", "TODAS as execucoes validas" if corr["todas_validas"].all() else "HA EXECUCOES INVALIDAS!")

    tabelas = {"corretude": corr, "baseline": tabela_baseline(df), "qualidade": tabela_qualidade(r),
               "escrita": tabela_escrita(df), "resumo_completo": r}
    for caso in CASOS:
        tabelas[f"strong_{caso}"] = tabela_strong(r, caso)
        tabelas[f"melhores_{caso}"] = tabela_melhores(r, caso)
    w = weak(r)
    if not w.empty:
        tabelas["weak"] = w
    for nome, t in tabelas.items():
        # as tabelas de strong scaling tem a politica como indice; as demais nao
        t.to_csv(PASTA_TABELAS / f"{nome}.csv", index=nome.startswith("strong_"))
    print(f"{len(tabelas)} tabelas em {PASTA_TABELAS}")

    n = 0
    for caso in CASOS:
        for m in ("tempo", "speedup", "eficiencia"):
            fig_strong(r, caso, m); n += 1
        fig_chunk(r, caso); n += 1
    for m in ("tempo", "flb", "speedup"):
        fig_heatmap(r, 12, m); n += 1
    fig_flb_politicas(r); n += 1
    if custos:
        fig_custo_por_linha(custos); fig_previsto_medido(r); n += 2
    else:
        print("AVISO: sem output/ref/sequencial_*_4096.bin: pulei 'custo por linha' e 'previsto x medido'.")
    if not w.empty:
        fig_weak(r); n += 1
    for caso in CASOS:
        a, b = par_final(caso)
        if a:
            miniaturas(a, b, salvar_como=f"miniaturas_{caso}.png"); n += 1
    print(f"{n} figuras em {PASTA_GRAFICOS}")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser(description="Gera tabelas e graficos da Etapa 1.")
    ap.add_argument("--csv", default=str(CSV_OFICIAL), help="CSV consolidado (padrao: resultados/resultados_consolidados.csv)")
    gerar_tudo(ap.parse_args().csv)
