#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py - Etapa 1 (OpenMP) - Benchmark do Conjunto de Mandelbrot (DEC107)
 
O QUE ESTE SCRIPT FAZ
  1. compila as variantes necessarias (sequencial/OpenMP x padrao/zoom x resolucao);
  2. executa cada configuracao (threads, schedule, chunk) varias vezes;
  3. valida CADA execucao: compara o hash (SHA-256) do .bin com o do sequencial;
  4. grava uma linha por execucao em resultados/resultados_consolidados.csv.
 
O QUE ELE NAO FAZ
  Nao calcula Speedup/Eficiencia nem desenha graficos (isso e do analise.py).
  Nao altera os programas C: eles medem, este script apenas os comanda.
 
COMO USAR (a partir da pasta "Etapa 1")
  python codigos_auxiliares/benchmark.py --preset smoke      # teste rapido (512x512)
  python codigos_auxiliares/benchmark.py --dry-run --preset padrao   # mostra o plano
  python codigos_auxiliares/benchmark.py --preset padrao     # bateria do caso padrao
  python codigos_auxiliares/benchmark.py --preset zoom
  python codigos_auxiliares/benchmark.py --preset weak
  python codigos_auxiliares/benchmark.py --preset oficial    # padrao + zoom + weak
  python codigos_auxiliares/benchmark.py --preset imagens    # gera os .ppm (secao 5.4 do enunciado)
  python codigos_auxiliares/benchmark.py --preset padrao --resume     # continua de onde parou
  python codigos_auxiliares/benchmark.py --casos zoom --threads 12 --schedules static,dynamic --chunks 0,1,16 --reps 5
"""
 
from __future__ import annotations
 
import argparse
import csv
import hashlib
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
 
# =============================================================================
# 1. CONFIGURACAO  (a parte que voce edita para mudar o experimento)
# =============================================================================
 
# --- Compilacao: as MESMAS flags nos dois programas (as que foram validadas) ---
GCC = "gcc"
FLAGS = ["-O2", "-msse2", "-mfpmath=sse", "-fopenmp"]
LIBS = ["-lm"]
 
# --- Hardware do autor (usado so para ESTIMAR o tempo e desenhar a matriz) ---
NUCLEOS_FISICOS = 6
THREADS_LOGICAS = 12
 
# --- Matriz de experimentos ---
RES_BASE = 4096                     # resolucao oficial (enunciado, secao 5.2)
THREADS_STRONG = [2, 4, 6, 8, 12]   # strong scaling
THREADS_SCHEDULE = [6, 12]          # comparacao schedule x chunk
POLITICAS_STRONG = [("static", 0), ("static", 1), ("dynamic", 1),
                    ("dynamic", 16), ("guided", 1)]     # chunk 0 = static em BLOCOS
CHUNKS = {"static": [0, 1, 4, 16, 64, 256],
          "dynamic": [1, 4, 16, 64, 256],
          "guided": [1, 4, 16, 64]}
THREADS_WEAK = [1, 2, 4, 6, 8, 12]  # weak scaling: lado = RES_BASE * raiz(p)
POLITICA_WEAK = ("dynamic", 1)
 
# --- Repeticoes ---
REPS = 5              # repeticoes de cada configuracao OpenMP
BASELINE_INI = 7      # execucoes do sequencial ANTES da bateria
BASELINE_FIM = 3      # execucoes do sequencial DEPOIS (detecta deriva de clock/temperatura)
 
# --- Casos (mesmos parametros do enunciado; MAX_ITER vem do proprio C) ---
MAX_ITER_POR_CASO = {"padrao": 1000, "zoom": 5000}
T_SEQ_ESTIMADO = {"padrao": 7.5, "zoom": 3.2}   # so para a estimativa do --dry-run/ETA
 
# --- Limites de seguranca ---
TIMEOUT_PADRAO = 1800        # segundos por execucao
MAX_REF_MB = 100             # so guarda .bin de referencia ate este tamanho
RES_MAX_32BITS = 11585       # acima disso (537 MB) a matriz pode nao caber em GCC de 32 bits
 
# =============================================================================
# 2. CAMINHOS  (tudo relativo a pasta "Etapa 1", onde este script mora em ../)
# =============================================================================
 
RAIZ = Path(__file__).resolve().parent.parent
SRC_DIR = RAIZ / "src"
RESULTADOS = RAIZ / "resultados"
OUTPUT = RAIZ / "output"
EXE_DIR, REF_DIR, TRAB_DIR = OUTPUT / "exe", OUTPUT / "ref", OUTPUT / "trabalho"
CSV_OFICIAL = RESULTADOS / "resultados_consolidados.csv"
CSV_SMOKE = OUTPUT / "smoke_resultados.csv"       # o teste rapido NAO suja o CSV oficial
LOG_PATH = OUTPUT / "benchmark.log"
 
SUFIXO_EXE = ".exe" if os.name == "nt" else ""
NOMES_FONTE = {"sequencial": "mandelbrot_sequencial.c", "openmp": "mandelbrot_openmp.c"}
NOME_CURTO = {"sequencial": "seq", "openmp": "omp"}
 
# Colunas que os programas C gravam (20). Se o C mudar, o script avisa.
COLUNAS_C = ["versao", "threads", "schedule", "chunk", "width", "height", "max_iter",
             "real_a", "real_b", "imag_a", "imag_b", "tempo_calculo_s", "tempo_escrita_s",
             "carga_min", "carga_media", "carga_max", "fator_balanceamento",
             "trabalho_min", "trabalho_media", "trabalho_max"]
COLUNAS_ANTES = ["run_id", "data_hora", "caso", "fase", "rep"]
COLUNAS_DEPOIS = ["pedido_threads", "pedido_schedule", "pedido_chunk",
                  "hash", "hash_ok", "config_ok", "tempo_wall_s", "exe"]
COLUNAS_CSV = COLUNAS_ANTES + COLUNAS_C + COLUNAS_DEPOIS
 
 
# =============================================================================
# 3. ESTRUTURAS DO PLANO
# =============================================================================
 
@dataclass(frozen=True)
class Execucao:
    """Uma execucao planejada de um dos programas."""
    versao: str    # "sequencial" | "openmp"
    caso: str      # "padrao" | "zoom"
    res: int       # lado da imagem (largura = altura)
    threads: int
    sched: str     # "nenhum" no sequencial
    chunk: int
    fase: str      # baseline_ini | referencia | medicao | baseline_fim
    rep: int
 
    @property
    def run_id(self) -> str:
        return (f"{self.fase}|{self.versao}|{self.caso}|{self.res}|"
                f"t{self.threads}|{self.sched},{self.chunk}|r{self.rep}")
 
 
@dataclass
class Grupo:
    """Configuracoes OpenMP de um (caso, resolucao)."""
    caso: str
    res: int
    configs: list          # [(threads, sched, chunk), ...]
    com_baseline: bool     # True: mede baseline do sequencial (para Speedup)
 
 
@dataclass
class Plano:
    grupos: list
    reps: int
    base_ini: int
    base_fim: int
    csv: Path
    oficial: bool          # False no smoke
 
 
class ErroExecucao(Exception):
    pass
 
 
# =============================================================================
# 4. UTILITARIOS
# =============================================================================
 
LOG_ATIVO = False
 
 
def log(msg: str = "") -> None:
    """Imprime e (se ativo) acrescenta ao output/benchmark.log."""
    print(msg, flush=True)
    if LOG_ATIVO:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
 
 
def com_retentativas(func, *args, tentativas: int = 40, espera: float = 0.25):
    """No Windows, antivirus/OneDrive podem travar um arquivo recem-criado por instantes."""
    for _ in range(tentativas - 1):
        try:
            return func(*args)
        except PermissionError:
            time.sleep(espera)
    return func(*args)
 
 
def sha256_arquivo(caminho: Path) -> str:
    def _hash():
        h = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(bloco)
        return h.hexdigest()
    return com_retentativas(_hash)
 
 
def normalizar(sched: str, chunk: int):
    """static: chunk 0 = blocos. dynamic/guided: chunk 0 equivale a 1 (o libgomp faz isso)."""
    if sched == "static":
        return "static", max(0, chunk)
    return sched, max(1, chunk)
 
 
def hhmm(segundos: float) -> str:
    segundos = int(round(segundos))
    return f"{segundos // 3600}h{(segundos % 3600) // 60:02d}min" if segundos >= 3600 \
        else f"{segundos // 60}min{segundos % 60:02d}s"
 
 
def detectar_32bits():
    """(eh_32bits, alvo). GCC do MinGW.org informa 'mingw32'; o de 64 bits, 'x86_64-...'."""
    try:
        r = subprocess.run([GCC, "-dumpmachine"], capture_output=True, text=True, timeout=30)
        alvo = r.stdout.strip().lower()
        return alvo.startswith(("i686", "i586", "i386", "mingw32")), alvo
    except Exception:
        return None, ""
 
 
# =============================================================================
# 5. CONSTRUCAO DO PLANO (o que sera executado)
# =============================================================================
 
def sem_duplicatas(lista):
    return list(dict.fromkeys(lista))
 
 
def configs_padrao_ou_zoom():
    """Overhead (1 thread) + strong scaling + schedule x chunk, sem repeticoes."""
    cfg = [(1, "dynamic", 1)]
    cfg += [(t, s, c) for t in THREADS_STRONG for (s, c) in POLITICAS_STRONG]
    cfg += [(t, s, c) for t in THREADS_SCHEDULE for s, lista in CHUNKS.items() for c in lista]
    return sem_duplicatas([(t,) + normalizar(s, c) for (t, s, c) in cfg])
 
 
def grupos_weak(res_base, threads_weak, politica):
    s, c = normalizar(*politica)
    return [Grupo("padrao", int(round(res_base * math.sqrt(p))), [(p, s, c)], False)
            for p in threads_weak]
 
 
def mesclar_grupos(grupos):
    """Junta grupos do mesmo (caso, resolucao); ex.: weak p=1 coincide com o caso padrao."""
    por_chave = {}
    for g in grupos:
        k = (g.caso, g.res)
        if k not in por_chave:
            por_chave[k] = Grupo(g.caso, g.res, [], False)
        por_chave[k].configs = sem_duplicatas(por_chave[k].configs + g.configs)
        por_chave[k].com_baseline = por_chave[k].com_baseline or g.com_baseline
    return list(por_chave.values())
 
 
def construir_plano(args) -> Plano:
    custom = any(x is not None for x in (args.threads, args.schedules, args.chunks))
 
    if args.preset and custom:
        sys.exit("Erro: use --preset OU as opcoes personalizadas (--threads/--schedules/--chunks), nao os dois.")
    if not args.preset and not custom and not args.casos:
        sys.exit("Erro: escolha --preset (smoke|padrao|zoom|weak|oficial) ou informe --casos/--threads/--schedules/--chunks.")
 
    reps, ini, fim = REPS, BASELINE_INI, BASELINE_FIM
    oficial, grupos = True, []
    res = args.res or RES_BASE
 
    if args.preset == "smoke":
        res = args.res or 512
        reps, ini, fim, oficial = 2, 2, 1, False
        cfg = sem_duplicatas([(1, "dynamic", 1)] +
                             [(t,) + normalizar(s, c) for t in (2, 4) for (s, c) in [("static", 0), ("dynamic", 1)]])
        grupos = [Grupo(c, res, cfg, True) for c in ("padrao", "zoom")]
        grupos += grupos_weak(res, [1, 2, 4], POLITICA_WEAK)
    elif args.preset in ("padrao", "zoom", "oficial"):
        casos = ["padrao", "zoom"] if args.preset == "oficial" else [args.preset]
        grupos = [Grupo(c, res, configs_padrao_ou_zoom(), True) for c in casos]
        if args.preset in ("oficial",):
            grupos += grupos_weak(res, THREADS_WEAK, POLITICA_WEAK)
    elif args.preset == "weak":
        # weak scaling: cada resolucao so precisa de 1 sequencial de REFERENCIA (para o hash)
        grupos = grupos_weak(res, THREADS_WEAK, POLITICA_WEAK)
    else:  # modo personalizado
        casos = [c.strip() for c in (args.casos or "padrao").split(",")]
        threads = [int(x) for x in args.threads.split(",")] if args.threads else THREADS_STRONG
        scheds = [x.strip() for x in (args.schedules or "static,dynamic,guided").split(",")]
        chunks = [int(x) for x in args.chunks.split(",")] if args.chunks else [0, 1, 4, 16]
        cfg = sem_duplicatas([(t,) + normalizar(s, c) for t in threads for s in scheds for c in chunks])
        grupos = [Grupo(c, res, cfg, True) for c in casos]
 
    for g in grupos:
        if g.caso not in MAX_ITER_POR_CASO:
            sys.exit(f"Erro: caso desconhecido '{g.caso}' (use padrao ou zoom).")
        for (_, s, _) in g.configs:
            if s not in ("static", "dynamic", "guided"):
                sys.exit(f"Erro: schedule desconhecido '{s}'.")
 
    if args.reps is not None:
        reps = args.reps
    if args.baseline_ini is not None:
        ini = args.baseline_ini
    if args.baseline_fim is not None:
        fim = args.baseline_fim
 
    csv_path = Path(args.csv) if args.csv else (CSV_OFICIAL if oficial else CSV_SMOKE)
    return Plano(mesclar_grupos(grupos), reps, ini, fim, csv_path, oficial)
 
 
def construir_execucoes(plano: Plano, seed: int):
    """Ordem: baseline inicial -> medicoes (repeticoes no laco EXTERNO, configs embaralhadas)
    -> baseline final. Embaralhar e repetir por rodadas espalha qualquer deriva de clock ou
    temperatura por todas as configuracoes, em vez de contaminar so as ultimas."""
    exs = []
    for g in plano.grupos:
        if g.com_baseline:
            exs += [Execucao("sequencial", g.caso, g.res, 1, "nenhum", 0, "baseline_ini", r)
                    for r in range(1, plano.base_ini + 1)]
        else:  # so precisa de UMA execucao sequencial para ter o hash de referencia
            exs.append(Execucao("sequencial", g.caso, g.res, 1, "nenhum", 0, "referencia", 1))
    for r in range(1, plano.reps + 1):
        for g in plano.grupos:
            cfgs = list(g.configs)
            random.Random(f"{seed}-{r}-{g.caso}-{g.res}").shuffle(cfgs)
            exs += [Execucao("openmp", g.caso, g.res, t, s, c, "medicao", r) for (t, s, c) in cfgs]
    for g in plano.grupos:
        if g.com_baseline:
            exs += [Execucao("sequencial", g.caso, g.res, 1, "nenhum", 0, "baseline_fim", r)
                    for r in range(1, plano.base_fim + 1)]
    return exs
 
 
def estimar_s(ex: Execucao, t_seq: dict) -> float:
    """Estimativa GROSSEIRA (so para --dry-run e ETA). Speedup ~ linear ate os nucleos fisicos."""
    base = t_seq[ex.caso] * (ex.res / 4096) ** 2
    if ex.versao == "sequencial":
        t = base
    else:
        p = ex.threads
        s = float(p) if p <= NUCLEOS_FISICOS else NUCLEOS_FISICOS + 0.35 * (min(p, THREADS_LOGICAS) - NUCLEOS_FISICOS)
        t = base / s
    return t + 0.3 + 0.4 * (ex.res / 4096) ** 2     # abrir processo + gravar .bin + hash
 
 
def imprimir_plano(plano: Plano, execs, t_seq):
    log("PLANO")
    for g in plano.grupos:
        n_omp = len(g.configs) * plano.reps
        base = (plano.base_ini + plano.base_fim) if g.com_baseline else 1
        log(f"  {g.caso:7s} {g.res}x{g.res}: {len(g.configs):3d} configuracoes x {plano.reps} reps = {n_omp:4d} execucoes OpenMP"
            f" + {base} sequencial ({'baseline' if g.com_baseline else 'so referencia'})")
    total_s = sum(estimar_s(e, t_seq) for e in execs)
    log(f"  TOTAL: {len(execs)} execucoes | tempo estimado (grosseiro): {hhmm(total_s)}")
    log(f"  CSV de saida: {plano.csv}")
    log("  (estimativa usa T_seq padrao=%.1fs, zoom=%.1fs; ajuste com --tseq-padrao/--tseq-zoom)" %
        (t_seq["padrao"], t_seq["zoom"]))
 
 
# =============================================================================
# 6. COMPILACAO
# =============================================================================
 
def garantir_exe(versao: str, caso: str, res: int, src_dir: Path) -> Path:
    """Compila uma variante (so se a fonte, as flags ou os -D mudaram)."""
    fonte = src_dir / NOMES_FONTE[versao]
    exe = EXE_DIR / f"mandelbrot_{NOME_CURTO[versao]}_{caso}_{res}{SUFIXO_EXE}"
    defines = [f"-DWIDTH={res}", f"-DHEIGHT={res}"] + (["-DCASO_ZOOM"] if caso == "zoom" else [])
    cmd = [GCC, *FLAGS, *defines, "-o", str(exe), str(fonte), *LIBS]
 
    # a "chave" muda se a fonte, as flags ou os -D mudarem -> recompila so quando preciso
    chave = hashlib.sha256(fonte.read_bytes() + " ".join(FLAGS + defines + LIBS).encode()).hexdigest()
    marca = Path(str(exe) + ".build")
    if exe.exists() and marca.exists() and marca.read_text() == chave:
        return exe
 
    log(f"  compilando {exe.name} ...")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"Erro de compilacao ({fonte.name}):\n{r.stderr}")
    marca.write_text(chave)
    return exe
 
 
# =============================================================================
# 7. EXECUCAO DE UMA RODADA
# =============================================================================
 
def limpar_trabalho():
    """Apaga sobras da execucao anterior (o C acrescenta ao CSV e sobrescreve o .bin)."""
    for p in TRAB_DIR.iterdir():
        if p.is_file():
            com_retentativas(p.unlink)
 
 
def ler_linha_csv(caminho: Path) -> dict:
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = [l for l in csv.reader(f) if l]
    if len(linhas) != 2:
        raise ErroExecucao(f"{caminho.name}: esperava cabecalho + 1 linha, veio {len(linhas)} linhas")
    if linhas[0] != COLUNAS_C:
        raise ErroExecucao(f"{caminho.name}: cabecalho diferente do esperado (o C mudou?)\n"
                           f"  esperado: {COLUNAS_C}\n  veio:     {linhas[0]}")
    return dict(zip(linhas[0], linhas[1]))
 
 
def executar_uma(ex: Execucao, exe: Path, refs: dict, args) -> dict:
    limpar_trabalho()
 
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ex.threads)
    if ex.versao == "openmp":
        # static sem chunk = blocos; nos demais o chunk e informado
        env["OMP_SCHEDULE"] = ex.sched if (ex.sched == "static" and ex.chunk == 0) else f"{ex.sched},{ex.chunk}"
    else:
        env.pop("OMP_SCHEDULE", None)
 
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(exe)], cwd=TRAB_DIR, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=args.timeout)
    except subprocess.TimeoutExpired:
        raise ErroExecucao(f"timeout de {args.timeout}s")
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise ErroExecucao(f"codigo de retorno {r.returncode}\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
 
    linha = ler_linha_csv(TRAB_DIR / f"metricas_{ex.versao}.csv")
 
    # --- a configuracao efetiva e a pedida? (pega -D esquecido, OMP_SCHEDULE invalido, etc.) ---
    sched_ok, chunk_ok = normalizar(ex.sched, ex.chunk) if ex.versao == "openmp" else ("nenhum", 0)
    config_ok = (linha["schedule"] == sched_ok and int(linha["chunk"]) == chunk_ok
                 and int(linha["threads"]) == ex.threads
                 and int(linha["width"]) == ex.res and int(linha["height"]) == ex.res
                 and int(linha["max_iter"]) == MAX_ITER_POR_CASO[ex.caso])
 
    # --- corretude: hash do .bin contra o do sequencial do mesmo (caso, resolucao) ---
    bin_path = TRAB_DIR / f"mandelbrot_{ex.versao}.bin"
    h, hash_ok = "", "NA"
    if bin_path.exists() and not args.nao_validar:
        h = sha256_arquivo(bin_path)
        chave = (ex.caso, ex.res)
        if chave not in refs:
            if ex.versao == "sequencial":
                refs[chave] = h                       # 1o sequencial = referencia
                hash_ok = "True"
                guardar(bin_path, f"sequencial_{ex.caso}_{ex.res}.bin")
            # OpenMP sem referencia: fica "NA"
        else:
            hash_ok = str(h == refs[chave])
    guardar(bin_path, f"openmp_{ex.caso}_{ex.res}.bin") if ex.versao == "openmp" else None
 
    linha.update({
        "run_id": ex.run_id, "data_hora": datetime.now().isoformat(timespec="seconds"),
        "caso": ex.caso, "fase": ex.fase, "rep": ex.rep,
        "pedido_threads": ex.threads, "pedido_schedule": ex.sched, "pedido_chunk": ex.chunk,
        "hash": h, "hash_ok": hash_ok, "config_ok": str(config_ok),
        "tempo_wall_s": f"{wall:.3f}", "exe": exe.name,
    })
    return linha
 
 
def guardar(bin_path: Path, nome_destino: str) -> None:
    """Guarda o .bin em output/ref se ele for pequeno (o notebook usa o par final).
    Arquivos grandes (weak scaling) so contribuem com o hash."""
    if not bin_path.exists():
        return
    if bin_path.stat().st_size <= MAX_REF_MB * 1024 * 1024:
        com_retentativas(os.replace, str(bin_path), str(REF_DIR / nome_destino))
 
 
# =============================================================================
# 8. CSV CONSOLIDADO (gravacao linha a linha) E RETOMADA
# =============================================================================
 
def preparar_csv(plano: Plano, args):
    """Devolve (run_ids ja feitos, refs). Recusa misturar com um CSV existente por engano."""
    caminho = plano.csv
    caminho.parent.mkdir(parents=True, exist_ok=True)
    existe = caminho.exists() and caminho.stat().st_size > 0
 
    if existe and args.recomecar:
        backup = caminho.with_name(caminho.name + f".bak.{int(time.time())}")
        shutil.move(str(caminho), str(backup))
        log(f"CSV existente arquivado em {backup.name}; comecando do zero.")
        existe = False
    if existe and not args.resume:
        sys.exit(f"Erro: {caminho.name} ja existe. Use --resume para continuar ou --recomecar "
                 f"para arquivar o antigo e comecar do zero.")
    if not existe:
        return set(), {}
 
    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        if leitor.fieldnames != COLUNAS_CSV:
            sys.exit(f"Erro: o cabecalho de {caminho.name} nao bate com o esperado "
                     f"(versao antiga do script?). Use --recomecar.")
        feitos, refs = set(), {}
        for row in leitor:
            feitos.add(row["run_id"])
            if row["versao"] == "sequencial" and row["hash"]:
                refs.setdefault((row["caso"], int(row["width"])), row["hash"])
    log(f"Retomando: {len(feitos)} execucoes ja registradas em {caminho.name}.")
    return feitos, refs
 
 
def anexar_linha(caminho: Path, linha: dict) -> None:
    novo = not caminho.exists() or caminho.stat().st_size == 0
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow(linha)
 
 
# =============================================================================
# 9. AMBIENTE (o relatorio exige registrar hardware, SO, compilador e parametros)
# =============================================================================
 
def _cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        return r.stdout.strip()
    except Exception:
        return ""
 
 
def info_cpu():
    """(nome, nucleos fisicos, threads logicas) - '(preencher)' quando nao der para descobrir."""
    logicas = str(os.cpu_count() or "(preencher)")
    if os.name == "nt":
        s = _cmd(["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Processor | ForEach-Object { $_.Name + '|' + $_.NumberOfCores + '|' + $_.NumberOfLogicalProcessors }"])
        if "|" in s:
            partes = s.splitlines()[0].split("|")
            return partes[0].strip(), partes[1].strip(), partes[2].strip()
    else:
        try:
            for linha in open("/proc/cpuinfo", encoding="utf-8"):
                if linha.startswith("model name"):
                    return linha.split(":", 1)[1].strip(), "(preencher)", logicas
        except Exception:
            pass
    return platform.processor() or "(preencher)", "(preencher)", logicas
 
 
def info_ram_gb():
    try:
        if os.name == "nt":
            import ctypes
 
            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30
        for linha in open("/proc/meminfo", encoding="utf-8"):
            if linha.startswith("MemTotal"):
                return int(linha.split()[1]) / 2 ** 20
    except Exception:
        pass
    return None
 
 
CAB_MANUAL = "## Condicoes (preencher a mao)"
 
 
def registrar_ambiente(plano: Plano, args, alvo_gcc: str, eh_32: bool):
    """Escreve o ambiente.md. A parte automatica e refeita a cada execucao; a parte manual
    (condicoes + historico de comandos) e PRESERVADA, porque a bateria pode ser feita em etapas."""
    caminho = plano.csv.with_name("ambiente.md") if plano.oficial else plano.csv.with_suffix(".ambiente.md")
    nome, nucleos, logicas = info_cpu()
    ram = info_ram_gb()
    gcc_v = (_cmd([GCC, "--version"]).splitlines() or ["(nao encontrado)"])[0]
    fontes = {k: hashlib.sha256((Path(args.src_dir) / v).read_bytes()).hexdigest()[:12]
              for k, v in NOMES_FONTE.items()}
    omp_env = {k: v for k, v in os.environ.items() if k.startswith("OMP_")}
    energia = _cmd(["powercfg", "/getactivescheme"]) if os.name == "nt" else ""
    bits = ("32 bits" if eh_32 else "64 bits") if eh_32 is not None else "(desconhecido)"
    agora = datetime.now().isoformat(timespec="seconds")
 
    automatica = [
        "# Ambiente de execucao", "",
        f"Parte automatica atualizada por `benchmark.py` em {agora}", "",
        "## Hardware",
        f"- CPU: {nome}", f"- Nucleos fisicos: {nucleos}", f"- Threads logicas: {logicas}",
        f"- RAM: {ram:.1f} GB" if ram else "- RAM: (preencher)", "",
        "## Sistema e software",
        f"- SO: {platform.platform()}", f"- Python: {sys.version.split()[0]}",
        f"- Compilador: {gcc_v}", f"- Alvo do GCC: `{alvo_gcc}` ({bits})",
        f"- Flags: `{' '.join(FLAGS + LIBS)}` (+ `-DWIDTH`/`-DHEIGHT`/`-DCASO_ZOOM` por variante)",
        f"- Timer dos programas C: `omp_get_wtime()`",
        f"- SHA-256 (12 primeiros) das fontes: sequencial `{fontes['sequencial']}`, openmp `{fontes['openmp']}`",
        f"- Plano de energia (no inicio desta execucao): {energia if energia else '(preencher)'}", "",
        "## Parametros da ultima execucao",
        f"- Repeticoes por configuracao OpenMP: {plano.reps}",
        f"- Baseline sequencial: {plano.base_ini} antes + {plano.base_fim} depois",
        f"- Variaveis `OMP_*` ja definidas no ambiente: {omp_env if omp_env else 'nenhuma'}",
        "- `OMP_NUM_THREADS` e `OMP_SCHEDULE` sao definidos pelo script em cada execucao", "",
    ]
 
    # parte manual: se o arquivo ja existe, preserva tudo a partir do cabecalho manual
    manual = None
    if caminho.exists():
        antigo = caminho.read_text(encoding="utf-8")
        if CAB_MANUAL in antigo:
            manual = antigo[antigo.index(CAB_MANUAL):].rstrip("\n")
    if manual is None:
        manual = "\n".join([
            CAB_MANUAL,
            "- Programas em segundo plano / temperatura / pasta fora do OneDrive: (preencher)", "",
            "## Historico de comandos (a bateria pode ser feita em etapas)"])
    manual += f"\n- {agora}: `{' '.join(sys.argv)}`"
 
    caminho.write_text("\n".join(automatica) + "\n" + manual + "\n", encoding="utf-8")
    log(f"Ambiente registrado em {caminho}")
 
 
def impedir_suspensao(ativar: bool):
    """Windows: evita que o PC entre em suspensao durante a bateria (vale so enquanto o script roda)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if ativar else 0))
    except Exception:
        pass
 
 
# =============================================================================
# 10. IMAGENS (secao 5.4 do enunciado pede uma imagem PPM para inspecao visual)
# =============================================================================
 
def gerar_imagens(args):
    res = args.res or RES_BASE
    threads = min(os.cpu_count() or 1, THREADS_LOGICAS)
    for caso in ("padrao", "zoom"):
        exe = garantir_exe("openmp", caso, res, Path(args.src_dir))
        limpar_trabalho()
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": str(threads), "OMP_SCHEDULE": "dynamic,1"})
        log(f"  gerando PPM do caso {caso} ({res}x{res}, {threads} threads) ...")
        r = subprocess.run([str(exe), "ppm"], cwd=TRAB_DIR, env=env, capture_output=True, text=True,
                           timeout=args.timeout)
        if r.returncode != 0:
            sys.exit(f"Falha ao gerar imagem: {r.stderr[-300:]}")
        destino = REF_DIR / f"openmp_{caso}_{res}.ppm"
        com_retentativas(os.replace, str(TRAB_DIR / "mandelbrot_openmp.ppm"), str(destino))
        log(f"  -> {destino}")
    limpar_trabalho()
 
 
# =============================================================================
# 11. PROGRAMA PRINCIPAL
# =============================================================================
 
def ler_argumentos():
    ap = argparse.ArgumentParser(description="Automatiza o benchmark do Mandelbrot (sequencial x OpenMP).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("COMO USAR")[1])
    ap.add_argument("--preset", choices=["smoke", "padrao", "zoom", "weak", "oficial", "imagens"],
                    help="conjunto pronto de experimentos")
    ap.add_argument("--casos", help="modo personalizado: padrao,zoom")
    ap.add_argument("--threads", help="modo personalizado: ex. 2,4,6,12")
    ap.add_argument("--schedules", help="modo personalizado: static,dynamic,guided")
    ap.add_argument("--chunks", help="modo personalizado: ex. 0,1,16 (0 = static em blocos)")
    ap.add_argument("--res", type=int, help=f"lado da imagem (padrao {RES_BASE})")
    ap.add_argument("--reps", type=int, help=f"repeticoes por configuracao (padrao {REPS})")
    ap.add_argument("--baseline-ini", type=int, help=f"sequenciais antes (padrao {BASELINE_INI})")
    ap.add_argument("--baseline-fim", type=int, help=f"sequenciais depois (padrao {BASELINE_FIM})")
    ap.add_argument("--csv", help="CSV de saida (padrao: resultados/resultados_consolidados.csv)")
    ap.add_argument("--dry-run", action="store_true", help="mostra o plano e o tempo estimado, sem executar")
    ap.add_argument("--resume", action="store_true", help="continua um CSV existente, pulando o que ja foi feito")
    ap.add_argument("--recomecar", action="store_true", help="arquiva o CSV existente e comeca do zero")
    ap.add_argument("--nao-validar", action="store_true", help="nao calcula hash (so para testes rapidos)")
    ap.add_argument("--forcar-memoria", action="store_true", help="ignora o limite de resolucao do GCC de 32 bits")
    ap.add_argument("--seed", type=int, default=2026, help="semente da ordem embaralhada")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_PADRAO, help="segundos por execucao")
    ap.add_argument("--tseq-padrao", type=float, default=T_SEQ_ESTIMADO["padrao"])
    ap.add_argument("--tseq-zoom", type=float, default=T_SEQ_ESTIMADO["zoom"])
    ap.add_argument("--src-dir", default=str(SRC_DIR), help="pasta com os .c (padrao: src/)")
    return ap.parse_args()
 
 
def main() -> int:
    global LOG_ATIVO
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # evita erro de acentos no Windows
    except Exception:
        pass
 
    args = ler_argumentos()
    t_seq = {"padrao": args.tseq_padrao, "zoom": args.tseq_zoom}
 
    if args.preset == "imagens":
        for d in (EXE_DIR, REF_DIR, TRAB_DIR):
            d.mkdir(parents=True, exist_ok=True)
        gerar_imagens(args)
        return 0
 
    plano = construir_plano(args)
    execs = construir_execucoes(plano, args.seed)
 
    if args.dry_run:
        imprimir_plano(plano, execs, t_seq)
        return 0
 
    for nome in NOMES_FONTE.values():
        if not (Path(args.src_dir) / nome).exists():
            sys.exit(f"Erro: nao encontrei {Path(args.src_dir) / nome}")
    if shutil.which(GCC) is None:
        sys.exit(f"Erro: '{GCC}' nao esta no PATH. Abra o terminal onde o gcc funciona.")
 
    for d in (EXE_DIR, REF_DIR, TRAB_DIR, RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)
    LOG_ATIVO = True
 
    # --- limite de memoria em GCC de 32 bits (weak scaling) ---
    eh_32, alvo = detectar_32bits()
    if eh_32 and not args.forcar_memoria:
        grandes = {(g.caso, g.res) for g in plano.grupos if g.res > RES_MAX_32BITS}
        if grandes:
            log(f"AVISO: GCC de 32 bits ({alvo}): pulando resolucoes > {RES_MAX_32BITS}: "
                f"{sorted(r for _, r in grandes)} (use --forcar-memoria para tentar mesmo assim)")
            plano.grupos = [g for g in plano.grupos if (g.caso, g.res) not in grandes]
            execs = construir_execucoes(plano, args.seed)
 
    feitos, refs = preparar_csv(plano, args)
    registrar_ambiente(plano, args, alvo, eh_32)
    imprimir_plano(plano, execs, t_seq)
 
    log("Compilando variantes ...")
    exes = {(v, c, r): garantir_exe(v, c, r, Path(args.src_dir))
            for (v, c, r) in sorted({(e.versao, e.caso, e.res) for e in execs})}
 
    pendentes = [e for e in execs if e.run_id not in feitos]
    log(f"\n{len(pendentes)} execucoes a fazer ({len(execs) - len(pendentes)} ja registradas).\n")
 
    registradas, invalidos, cfg_erradas, falhas = 0, 0, 0, []
    inicio = time.perf_counter()
    est_feito = 0.0
    est_total = sum(estimar_s(e, t_seq) for e in pendentes)
    impedir_suspensao(True)
    interrompido = False
 
    try:
        for i, ex in enumerate(pendentes, 1):
            est = estimar_s(ex, t_seq)
            try:
                linha = executar_uma(ex, exes[(ex.versao, ex.caso, ex.res)], refs, args)
            except ErroExecucao as e:
                falhas.append((ex.run_id, str(e).splitlines()[0]))
                log(f"[{i:4d}/{len(pendentes)}] FALHA {ex.run_id}: {str(e).splitlines()[0]}")
                est_feito += est
                continue
 
            anexar_linha(plano.csv, linha)
            registradas += 1
            est_feito += est
            decorrido = time.perf_counter() - inicio
            eta = (est_total - est_feito) * (decorrido / est_feito) if est_feito > 0 else 0
 
            if linha["hash_ok"] == "False":
                invalidos += 1
            if linha["config_ok"] == "False":
                cfg_erradas += 1
            marca_hash = {"True": "hash OK", "False": "HASH DIFERENTE!", "NA": "sem hash"}[linha["hash_ok"]]
            marca_cfg = "" if linha["config_ok"] == "True" else " | CONFIG DIFERENTE DA PEDIDA!"
            log(f"[{i:4d}/{len(pendentes)}] {NOME_CURTO[ex.versao]:3s} {ex.caso:6s} {ex.res:5d} "
                f"t={ex.threads:<2d} {ex.sched:>7s},{ex.chunk:<3d} {ex.fase[:10]:10s} r{ex.rep} | "
                f"calc {float(linha['tempo_calculo_s']):7.3f}s F_LB {float(linha['fator_balanceamento']):.3f} | "
                f"{marca_hash}{marca_cfg} | ETA {hhmm(eta)}")
    except KeyboardInterrupt:
        interrompido = True
        log("\nInterrompido (Ctrl+C). O que foi feito ja esta no CSV; use --resume para continuar.")
    finally:
        impedir_suspensao(False)
        try:
            limpar_trabalho()
        except Exception:
            pass
 
    log("\n" + "=" * 70)
    log(f"Concluido em {hhmm(time.perf_counter() - inicio)}." if not interrompido else "Execucao interrompida.")
    log(f"  execucoes registradas nesta rodada : {registradas} de {len(pendentes)}")
    log(f"  hash DIFERENTE da referencia       : {invalidos}")
    log(f"  configuracao diferente da pedida   : {cfg_erradas}")
    log(f"  falhas de execucao                 : {len(falhas)}")
    for rid, msg in falhas[:10]:
        log(f"     - {rid}: {msg}")
    log(f"  CSV: {plano.csv}")
    if invalidos or cfg_erradas:
        log("  ATENCAO: ha execucoes invalidas; NAO use esses dados no relatorio sem investigar.")
        return 2
    return 1 if (falhas or interrompido) else 0
 
 
if __name__ == "__main__":
    sys.exit(main())