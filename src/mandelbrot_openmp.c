#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// IDENTIFICACAO DA VERSAO: openmp
// ---------------------------------------------------------------------------
#ifndef _OPENMP
#error "Esta versao precisa ser compilada com -fopenmp"
#endif

#define VERSAO "openmp"
#define ARQ_BIN "mandelbrot_openmp.bin"
#define ARQ_PPM "mandelbrot_openmp.ppm"
#define ARQ_CSV "metricas_openmp.csv"

// ---------------------------------------------------------------------------
// PARAMETROS DO PROBLEMA (identicos nas versoes sequencial e OpenMP)
// ---------------------------------------------------------------------------

// Resolucao sobrescrevivel na compilacao: -DWIDTH=8192 -DHEIGHT=8192
#ifndef WIDTH
#define WIDTH 4096
#endif
#ifndef HEIGHT
#define HEIGHT 4096
#endif

// Caso obrigatorio (vale dos cavalos-marinhos): compile com -DCASO_ZOOM
#ifdef CASO_ZOOM
#define MAX_ITER 5000
#define REAL_A -0.745143887
#define REAL_B -0.742143887
#define IMAG_A 0.130325904
#define IMAG_B 0.133325904
#else
#define MAX_ITER 1000
#define REAL_A -2.0 // INTERVALO REAL
#define REAL_B 1.0 // INTERVALO REAL
#define IMAG_A -1.5 // INTERVALO IMAGINÁRIO
#define IMAG_B 1.5 // INTERVALO IMAGINÁRIO
#endif

typedef struct {
    int R;
    int G;
    int B;
} Cor;

typedef struct {
    double tempoRegiao;                 // T(n,p): regiao de calculo inteira
    int threads;                        // threads que realmente participaram
    double tMin, tMedio, tMax, flb;     // tempo por thread; flb = (tMax - tMin) / tMax
    long long wMin, wMax;               // iteracoes (trabalho) por thread
    double wMedio;                      // media (pode ser fracionaria)
} Metricas;

double mapear_real(int x);
double mapear_imaginario(int y);
int* alocaImagem1D(int nLin, int nCol);
int** alocaImagem2D(int* img1D, int nLin, int nCol);
void mandelbrot(int** img2D, int nLin, int nCol, Metricas *m);
int calculaMandelbrot(double cr, double ci);
void obterSchedule(const char **nome, int *chunk);
void salvarPPM(int **img2D, int nLin, int nCol);
Cor mapear_pixel(int iteracoes);
void salvarBinario(int *img1D, int nLin, int nCol);
void salvarMetricas(const Metricas *m, double tempoEscrita);
double tempoAgora(void);

// Uso: <programa>        -> grava so o .bin
//      <programa> ppm    -> grava .bin e .ppm
int main(int argc, char **argv) {
    int gravarPPM = (argc > 1 && strcmp(argv[1], "ppm") == 0);

    int* img1D = alocaImagem1D(HEIGHT, WIDTH);
    int** img2D = alocaImagem2D(img1D, HEIGHT, WIDTH);

    if (img1D == NULL || img2D == NULL) {
        printf("Erro ao alocar imagem.\n");
        free(img2D);
        free(img1D);
        return 1;
    }

    Metricas m;
    mandelbrot(img2D, HEIGHT, WIDTH, &m);

    double inicioEscrita = tempoAgora();
    salvarBinario(img1D, HEIGHT, WIDTH);
    if (gravarPPM) salvarPPM(img2D, HEIGHT, WIDTH);
    double tempoEscrita = tempoAgora() - inicioEscrita;

    printf("Versao: %s\n", VERSAO);
    printf("Threads: %d\n", m.threads);
    printf("Tempo de calculo: %.6f segundos\n", m.tempoRegiao);
    printf("Tempo de escrita: %.6f segundos\n", tempoEscrita);
    printf("F_LB: %.4f (Tmin=%.6f s, Tmax=%.6f s)\n", m.flb, m.tMin, m.tMax);

    salvarMetricas(&m, tempoEscrita);

    free(img2D);
    free(img1D);

    return 0;
}

double mapear_real(int x) {
    return REAL_A + (((REAL_B - REAL_A) / (WIDTH - 1)) * x);
}

double mapear_imaginario(int y) {
    return IMAG_A + (((IMAG_B - IMAG_A) / (HEIGHT - 1)) * y);
}

int* alocaImagem1D(int nLin, int nCol) {
    int* img1D = (int*)calloc(nLin*nCol, sizeof(int));
    
    if(img1D != NULL) return img1D;
    
    return NULL;
}

int** alocaImagem2D(int* img1D, int nLin, int nCol) {
    int** img2D = (int**)calloc(nLin, sizeof(int*));

    if (img1D == NULL || img2D == NULL) {
        free(img2D);
        return NULL;
    }

    for (int i = 0; i < nLin; i++) {
        img2D[i] = &img1D[i * nCol];
    }

    return img2D;
}

int calculaMandelbrot(double cr, double ci) {
    double zReal = 0;
    double zImag = 0;
    double zRealN = 0;
    double zImagN = 0;
    int iteracoes;

    for(int i = 0; i < MAX_ITER; i++) {
        iteracoes = i;
        zRealN = (zReal * zReal) - (zImag * zImag) + cr;
        zImagN = (2 * zReal * zImag) + ci;
        if (((zRealN * zRealN) + (zImagN * zImagN)) > 4) break;
        zReal = zRealN;
        zImag = zImagN;
    }

    return iteracoes + 1;
}

// ---------------------------------------------------------------------------
// PARTE ESPECIFICA DA VERSAO: unica regiao que difere entre os dois arquivos
// ---------------------------------------------------------------------------
void mandelbrot(int** img2D, int nLin, int nCol, Metricas *m) {
    int maxThreads = omp_get_max_threads();

    double *tempoThread = malloc(maxThreads * sizeof(double));
    long long *trabalhoThread = calloc(maxThreads, sizeof(long long));

    if (tempoThread == NULL || trabalhoThread == NULL) {
        printf("Erro ao alocar vetores de metricas.\n");
        exit(1);
    }

    // -1 marca "thread que nao participou" (evita Tmin = 0 falso)
    for (int i = 0; i < maxThreads; i++) tempoThread[i] = -1.0;

    double inicioRegiao = tempoAgora();

    #pragma omp parallel default(none) shared(img2D, tempoThread, trabalhoThread, nLin, nCol)
    {
        int tid = omp_get_thread_num();
        long long trabalhoLocal = 0;

        #pragma omp barrier

        double inicioThread = tempoAgora();

        #pragma omp for schedule(runtime) nowait
        for (int y = 0; y < nLin; y++) {
            double ci = mapear_imaginario(y);

            for (int x = 0; x < nCol; x++) {
                double cr = mapear_real(x);
                int iter = calculaMandelbrot(cr, ci);

                img2D[y][x] = iter;

                trabalhoLocal += iter;
            }
        }
        double fimThread = tempoAgora();

        tempoThread[tid] = fimThread - inicioThread;
        trabalhoThread[tid] = trabalhoLocal;
    }

    m->tempoRegiao = tempoAgora() - inicioRegiao;

    // Agregacao (so threads que participaram)
    int usadas = 0;
    double tSoma = 0.0;
    long long wSoma = 0;

    for (int i = 0; i < maxThreads; i++) {
        if (tempoThread[i] < 0.0) continue;

        if (usadas == 0) {
            m->tMin = m->tMax = tempoThread[i];
            m->wMin = m->wMax = trabalhoThread[i];
        } else {
            if (tempoThread[i] < m->tMin) m->tMin = tempoThread[i];
            if (tempoThread[i] > m->tMax) m->tMax = tempoThread[i];
            if (trabalhoThread[i] < m->wMin) m->wMin = trabalhoThread[i];
            if (trabalhoThread[i] > m->wMax) m->wMax = trabalhoThread[i];
        }

        tSoma += tempoThread[i];
        wSoma += trabalhoThread[i];
        usadas++;
    }

    m->threads = usadas;
    m->tMedio = (usadas > 0) ? tSoma / usadas : 0.0;
    m->wMedio = (usadas > 0) ? (double)wSoma / usadas : 0.0;
    m->flb = (m->tMax > 0.0) ? (m->tMax - m->tMin) / m->tMax : 0.0;

    free(tempoThread);
    free(trabalhoThread);
}

// Schedule efetivo (definido por OMP_SCHEDULE por causa do schedule(runtime)).
// Sem OMP_SCHEDULE o padrao do libgomp e dynamic,1 (nao static!).
void obterSchedule(const char **nome, int *chunk) {
    omp_sched_t kind;
    omp_get_schedule(&kind, chunk);

    // GCC >= 9 pode devolver o bit "monotonic" (0x80000000) junto do tipo
    int tipo = (int)kind & 0x7FFFFFFF;

    *nome = "outro";
    if (tipo == (int)omp_sched_static) *nome = "static";
    else if (tipo == (int)omp_sched_dynamic) *nome = "dynamic";
    else if (tipo == (int)omp_sched_guided) *nome = "guided";
    else if (tipo == (int)omp_sched_auto) *nome = "auto";
}

// ---------------------------------------------------------------------------
// SAIDAS E METRICAS (identicas nas versoes sequencial e OpenMP)
// ---------------------------------------------------------------------------

void salvarPPM(int **img2D, int nLin, int nCol) {
    FILE *arquivo = fopen(ARQ_PPM, "w");

    if (arquivo == NULL) {
        printf("Erro ao criar arquivo PPM.\n");
        return;
    }

    fprintf(arquivo, "P3\n");
    fprintf(arquivo, "%d %d\n", nCol, nLin);
    fprintf(arquivo, "255\n");

    for (int y = 0; y < nLin; y++) {
        for (int x = 0; x < nCol; x++) {
            int iteracoes = img2D[y][x];
            Cor pixel;

            if (iteracoes == MAX_ITER) {
                pixel.R = 0;
                pixel.G = 0;
                pixel.B = 0;
            }
            else pixel = mapear_pixel(iteracoes);

            fprintf(arquivo, "%d %d %d ", pixel.R, pixel.G, pixel.B);
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);
}

Cor mapear_pixel(int iteracoes) {
    Cor pixel;

    double t = log(iteracoes + 1.0) / log(MAX_ITER + 1.0);

    pixel.R = (int)(255 * t * t);
    pixel.G = (int)(100 * t * t * t);
    pixel.B = (int)(255 * t);

    return pixel;
}

void salvarBinario(int *img1D, int nLin, int nCol) {
    FILE *arquivo = fopen(ARQ_BIN, "wb");

    if (arquivo == NULL) {
        printf("Erro ao criar arquivo binario.\n");
        return;
    }

    size_t total = (size_t)nLin * nCol;
    size_t escritos = fwrite(img1D, sizeof(int), total, arquivo);

    if (escritos != total) {
        printf("Erro ao escrever matriz no arquivo binario.\n");
    }

    fclose(arquivo);
}

void salvarMetricas(const Metricas *m, double tempoEscrita) {
    FILE *arquivo = fopen(ARQ_CSV, "a+");

    if (arquivo == NULL) {
        printf("Erro ao abrir arquivo de metricas.\n");
        return;
    }

    const char *nomeSchedule;
    int chunk;
    obterSchedule(&nomeSchedule, &chunk);

    fseek(arquivo, 0, SEEK_END);
    long tamanho = ftell(arquivo);

    if (tamanho == 0) {
        fprintf(
            arquivo,
            "versao,threads,schedule,chunk,width,height,max_iter,"
            "real_a,real_b,imag_a,imag_b,"
            "tempo_calculo_s,tempo_escrita_s,"
            "carga_min,carga_media,carga_max,fator_balanceamento,"
            "trabalho_min,trabalho_media,trabalho_max\n"
        );
    }

    // trabalho impresso como double (%.0f): evita depender de %lld no printf do MinGW
    fprintf(
        arquivo,
        "%s,%d,%s,%d,%d,%d,%d,"
        "%.17g,%.17g,%.17g,%.17g,"
        "%.9f,%.9f,"
        "%.9f,%.9f,%.9f,%.6f,"
        "%.0f,%.3f,%.0f\n",
        VERSAO,
        m->threads,
        nomeSchedule,
        chunk,
        WIDTH,
        HEIGHT,
        MAX_ITER,
        REAL_A,
        REAL_B,
        IMAG_A,
        IMAG_B,
        m->tempoRegiao,
        tempoEscrita,
        m->tMin,
        m->tMedio,
        m->tMax,
        m->flb,
        (double)m->wMin,
        m->wMedio,
        (double)m->wMax
    );

    fclose(arquivo);

    printf("Metricas salvas em %s\n", ARQ_CSV);
}

// Mesmo relogio nas duas versoes (as duas exigem -fopenmp; o sequencial continua
// serial porque nao ha diretivas).
double tempoAgora(void) {
    return omp_get_wtime();
}
