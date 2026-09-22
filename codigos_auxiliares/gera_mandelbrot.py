#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geração de imagens do conjunto de Mandelbrot para ilustrar o enunciado
da atividade avaliativa da disciplina DEC107 - Processamento Paralelo (UESC).

Gera quatro figuras:
  1. Vista completa (input padrão do enunciado)
  2. Zoom no vale dos cavalos-marinhos (caso de desbalanceamento de carga)
  3. Mapa de custo por pixel (escape-time = trabalho por pixel)
  4. Zoom em região de espirais (detalhe da fronteira fractal)

Requisitos: numpy, matplotlib
    pip install numpy matplotlib

Uso:
    python3 gera_mandelbrot.py
As imagens sao salvas na pasta indicada por OUT (por padrao, ./saida).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica (salva em arquivo)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Pasta de saída das imagens (pode ser alterada livremente)
OUT = "saida"

# Paleta "Malha & Fluxo": azul-núcleo -> ciano -> âmbar (identidade dos slides)
malha_fluxo = LinearSegmentedColormap.from_list(
    "malha_fluxo",
    ["#0b1d3a", "#12386e", "#1f6fb4", "#22b6c8", "#f0b429", "#fff3d1"],
)


def mandelbrot(re_min, re_max, im_min, im_max, width, height, max_iter):
    """Calcula a matriz de escape-time (numero de iteracoes ate o escape).

    Para cada pixel, mapeia (px, py) para um ponto complexo c da regiao dada
    e itera z_{n+1} = z_n^2 + c a partir de z_0 = 0, com criterio de escape
    |z|^2 > 4. Pontos que nao escapam em max_iter iteracoes recebem max_iter.

    Retorna um array (height x width) de inteiros (contagens de iteracao).
    Implementacao vetorizada em numpy (referencia serial conceitual; a versao
    C dos alunos deve reproduzir estas contagens).
    """
    xs = np.linspace(re_min, re_max, width, dtype=np.float64)
    ys = np.linspace(im_min, im_max, height, dtype=np.float64)
    C = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    Z = np.zeros_like(C)
    count = np.zeros(C.shape, dtype=np.int32)
    alive = np.ones(C.shape, dtype=bool)  # pixels que ainda nao escaparam
    for i in range(max_iter):
        Z[alive] = Z[alive] * Z[alive] + C[alive]
        escaped = alive & (Z.real * Z.real + Z.imag * Z.imag > 4.0)
        count[escaped] = i
        alive &= ~escaped
    count[alive] = max_iter  # nao escaparam: pertencem ao conjunto
    return count


def _normaliza_suave(count, max_iter):
    """Normalizacao logaritmica para colorir agradavelmente o exterior.

    Retorna (valores_normalizados_em_[0,1], mascara_do_interior).
    """
    c = count.astype(np.float64)
    interior = (count >= max_iter)
    c = np.log1p(c)
    c = c / c.max()
    c[interior] = 0.0
    return c, interior


def salva_colorida(count, max_iter, path, titulo=None):
    """Salva a imagem colorida por escape-time com a paleta Malha & Fluxo."""
    c, interior = _normaliza_suave(count, max_iter)
    rgb = malha_fluxo(c)
    rgb[interior] = np.array([0.043, 0.114, 0.227, 1.0])  # interior azul-escuro
    fig_h = 6.0
    fig_w = fig_h * count.shape[1] / count.shape[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.imshow(rgb, origin="lower", interpolation="bilinear")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=9, color="#12386e", pad=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("  salvo:", path)


def salva_custo(count, max_iter, path):
    """Salva o mapa de custo por pixel (escape-time direto, com barra de escala).

    Serve para ilustrar o desbalanceamento de carga: o interior (custo maximo)
    concentra o trabalho, enquanto o exterior escapa cedo.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    im = ax.imshow(count, origin="lower", cmap="inferno",
                   interpolation="nearest", vmin=0, vmax=max_iter)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Iteracoes ate o escape  (=  custo do pixel)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Custo por pixel: interior (amarelo) = MAX_ITER = trabalho maximo",
                 fontsize=8.5, color="#333333", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("  salvo:", path)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) Vista completa (input padrão do enunciado)
    print("Gerando vista completa...")
    inicio = time.perf_counter()
    full = mandelbrot(-2.0, 1.0, -1.5, 1.5, 4096, 4096, 1000)
    print(f"  tempo de mandelbrot (vista completa): {time.perf_counter() - inicio:.3f} s")
    salva_colorida(
        full, 1000, os.path.join(OUT, "mandelbrot_1_vista_completa.png"),
        "Conjunto de Mandelbrot - vista completa  (Re in [-2,1], Im in [-1.5,1.5])")

    # 2) Zoom do vale dos cavalos-marinhos (caso de desbalanceamento)
    print("Gerando vale dos cavalos-marinhos...")
    cx, cy = -0.743643887, 0.131825904
    half = 3.0e-3 / 2.0  # largura 3e-3 no eixo real
    inicio = time.perf_counter()
    sea = mandelbrot(cx - half, cx + half, cy - half, cy + half, 4096, 4096, 5000)
    print(f"  tempo de mandelbrot (vale dos cavalos-marinhos): {time.perf_counter() - inicio:.3f} s")
    salva_colorida(
        sea, 5000, os.path.join(OUT, "mandelbrot_2_vale_cavalos_marinhos.png"),
        "Zoom: vale dos cavalos-marinhos  (centro -0.743643887 + 0.131825904i, MAX_ITER=5000)")

    # 3) Mapa de custo por pixel (ilustra o desbalanceamento de carga)
    print("Gerando mapa de custo...")
    inicio = time.perf_counter()
    custo = mandelbrot(-2.0, 1.0, -1.5, 1.5, 4096, 4096, 1000)
    print(f"  tempo de mandelbrot (mapa de custo): {time.perf_counter() - inicio:.3f} s")
    salva_custo(custo, 1000, os.path.join(OUT, "mandelbrot_3_mapa_de_custo.png"))

    # 4) Zoom classico "espiral" para variedade visual
    print("Gerando zoom espiral...")
    sx, sy = -0.16070135, 1.0375665
    h2 = 0.004
    inicio = time.perf_counter()
    spiral = mandelbrot(sx - h2, sx + h2, sy - h2, sy + h2, 4096, 4096, 3000)
    print(f"  tempo de mandelbrot (zoom espiral): {time.perf_counter() - inicio:.3f} s")
    salva_colorida(
        spiral, 3000, os.path.join(OUT, "mandelbrot_4_zoom_espiral.png"),
        "Zoom: regiao de espirais  (detalhe da fronteira fractal)")

    print("Concluido. Imagens em:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
