#define _POSIX_C_SOURCE 200809L

#include "fft.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HARNESS_PI 3.141592653589793238462643383279502884

typedef struct {
    int run_tests;
    int run_benchmarks;
    unsigned min_power;
    unsigned max_power;
    double target_ms;
    const char *csv_path;
} options;

typedef struct {
    double median_seconds;
    double minimum_seconds;
    size_t samples;
    double checksum;
} benchmark_result;

typedef struct {
    long double re;
    long double im;
} reference_complex;

static volatile double benchmark_sink;

static uint64_t random_state = UINT64_C(0x8f3c1d72a94b650e);

static uint64_t next_random_u64(void)
{
    uint64_t x = random_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    random_state = x;
    return x * UINT64_C(2685821657736338717);
}

static double random_signed_unit(void)
{
    const uint64_t bits = next_random_u64() >> 11;
    const double unit = (double)bits * (1.0 / 9007199254740992.0);
    return 2.0 * unit - 1.0;
}

static void fill_random(fft_complex *data, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        data[i].re = random_signed_unit();
        data[i].im = random_signed_unit();
    }
}

static void reference_dft(const fft_complex *input,
                          reference_complex *output,
                          size_t n)
{
    size_t k;
    for (k = 0; k < n; ++k) {
        long double sum_re = 0.0L;
        long double sum_im = 0.0L;
        size_t j;
        for (j = 0; j < n; ++j) {
            const long double angle =
                -2.0L * (long double)HARNESS_PI * (long double)j *
                (long double)k / (long double)n;
            const long double c = cosl(angle);
            const long double s = sinl(angle);
            sum_re += (long double)input[j].re * c -
                      (long double)input[j].im * s;
            sum_im += (long double)input[j].re * s +
                      (long double)input[j].im * c;
        }
        output[k].re = sum_re;
        output[k].im = sum_im;
    }
}

static double relative_max_error(const fft_complex *actual,
                                 const reference_complex *expected,
                                 size_t n)
{
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        const long double error_re =
            (long double)actual[i].re - expected[i].re;
        const long double error_im =
            (long double)actual[i].im - expected[i].im;
        const double error = (double)hypotl(error_re, error_im);
        const double reference =
            (double)hypotl(expected[i].re, expected[i].im);
        if (error > maximum_error) {
            maximum_error = error;
        }
        if (reference > maximum_reference) {
            maximum_reference = reference;
        }
    }

    return maximum_error / fmax(1.0, maximum_reference);
}

static double relative_max_error_float(const fft_complex *actual,
                                       const fft_complex *expected,
                                       size_t n)
{
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        const double error =
            hypot((double)actual[i].re - expected[i].re,
                  (double)actual[i].im - expected[i].im);
        const double reference =
            hypot((double)expected[i].re, (double)expected[i].im);
        maximum_error = fmax(maximum_error, error);
        maximum_reference = fmax(maximum_reference, reference);
    }
    return maximum_error / fmax(1.0, maximum_reference);
}

static int check_vector(size_t n,
                        const fft_complex *input,
                        double maxima[FFT_ALGORITHM_COUNT])
{
    fft_plan *plan = fft_plan_create(n);
    reference_complex *reference =
        (reference_complex *)malloc(n * sizeof(*reference));
    fft_complex *work = (fft_complex *)malloc(n * sizeof(*work));
    int success = 1;
    int algorithm;

    if (plan == NULL || reference == NULL || work == NULL) {
        fprintf(stderr, "allocation failed while testing n=%zu\n", n);
        fft_plan_destroy(plan);
        free(reference);
        free(work);
        return 0;
    }

    reference_dft(input, reference, n);
    for (algorithm = 0; algorithm < FFT_ALGORITHM_COUNT; ++algorithm) {
        const fft_algorithm selected = (fft_algorithm)algorithm;
        const double tolerance =
            3.0e-6 * fmax(1.0, (double)(n > 1 ? log2((double)n) : 1.0));
        double error;

        memcpy(work, input, n * sizeof(*work));
        if (fft_execute(plan, selected, work) != 0) {
            fprintf(stderr,
                    "%s execution failed for n=%zu\n",
                    fft_algorithm_name(selected),
                    n);
            success = 0;
            continue;
        }

        error = relative_max_error(work, reference, n);
        if (error > maxima[algorithm]) {
            maxima[algorithm] = error;
        }
        if (!isfinite(error) || error > tolerance) {
            fprintf(stderr,
                    "FAIL: %-16s n=%-5zu relative max error %.3e "
                    "(limit %.3e)\n",
                    fft_algorithm_name(selected),
                    n,
                    error,
                    tolerance);
            success = 0;
        }
    }

    fft_plan_destroy(plan);
    free(reference);
    free(work);
    return success;
}

static int run_correctness_tests(void)
{
    double maxima[FFT_ALGORITHM_COUNT] = {0};
    unsigned power;
    int success = 1;

    printf("Correctness tests (long-double direct DFT reference)\n");
    for (power = 0; power <= 9; ++power) {
        const size_t n = (size_t)1 << power;
        fft_complex *input = (fft_complex *)calloc(n, sizeof(*input));
        size_t i;

        if (input == NULL) {
            fprintf(stderr, "allocation failed while testing n=%zu\n", n);
            return 0;
        }

        /* Deterministic complex random data. */
        fill_random(input, n);
        success = check_vector(n, input, maxima) && success;

        /* An impulse exercises every output and all twiddle paths. */
        memset(input, 0, n * sizeof(*input));
        input[n > 1 ? n / 3 : 0].re = 0.75;
        input[n > 1 ? n / 3 : 0].im = -0.25;
        success = check_vector(n, input, maxima) && success;

        /* A structured mixture catches ordering/permutation mistakes. */
        for (i = 0; i < n; ++i) {
            input[i].re =
                cos(2.0 * HARNESS_PI * 3.0 * (double)i / (double)n) +
                0.2 * sin(2.0 * HARNESS_PI * 5.0 * (double)i / (double)n);
            input[i].im =
                0.3 * cos(2.0 * HARNESS_PI * 2.0 * (double)i / (double)n);
        }
        success = check_vector(n, input, maxima) && success;
        free(input);
    }

    /*
     * A direct DFT is deliberately limited to small sizes. Exercise the
     * commonly used larger powers independently by cross-checking the two
     * recursive decompositions against radix-2.
     */
    for (power = 10; power <= 22; ++power) {
        const size_t n = (size_t)1 << power;
        fft_plan *plan = fft_plan_create(n);
        fft_complex *input = (fft_complex *)malloc(n * sizeof(*input));
        fft_complex *radix = (fft_complex *)malloc(n * sizeof(*radix));
        fft_complex *work = (fft_complex *)malloc(n * sizeof(*work));
        int algorithm;

        if (plan == NULL || input == NULL || radix == NULL || work == NULL) {
            fprintf(stderr,
                    "allocation failed in large-size test n=%zu\n",
                    n);
            fft_plan_destroy(plan);
            free(input);
            free(radix);
            free(work);
            return 0;
        }

        fill_random(input, n);
        memcpy(radix, input, n * sizeof(*radix));
        (void)fft_execute(plan, FFT_RADIX2, radix);

        for (algorithm = FFT_SPLIT_RADIX;
             algorithm <= FFT_FFMPEG;
             ++algorithm) {
            const double tolerance =
                6.0e-6 * (double)power;
            double error;
            if (!fft_plan_supports(plan, (fft_algorithm)algorithm)) {
                continue;
            }
            memcpy(work, input, n * sizeof(*work));
            (void)fft_execute(plan, (fft_algorithm)algorithm, work);
            error = relative_max_error_float(work, radix, n);
            if (!isfinite(error) || error > tolerance) {
                fprintf(stderr,
                        "FAIL: %-16s n=%-8zu cross-check error %.3e "
                        "(limit %.3e)\n",
                        fft_algorithm_name((fft_algorithm)algorithm),
                        n,
                        error,
                        tolerance);
                success = 0;
            }
        }

        fft_plan_destroy(plan);
        free(input);
        free(radix);
        free(work);
    }

    if (success) {
        int algorithm;
        printf("PASS: direct DFT through 512; cross-checks through 2^22\n");
        for (algorithm = 0; algorithm < FFT_ALGORITHM_COUNT; ++algorithm) {
            printf("  %-16s worst relative max error: %.3e\n",
                   fft_algorithm_name((fft_algorithm)algorithm),
                   maxima[algorithm]);
        }
    }
    return success;
}

static double seconds_between(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) +
           1.0e-9 * (double)(end.tv_nsec - start.tv_nsec);
}

static int compare_doubles(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static benchmark_result benchmark_algorithm(fft_plan *plan,
                                            fft_algorithm algorithm,
                                            const fft_complex *input,
                                            double target_seconds)
{
    const size_t n = fft_plan_size(plan);
    const size_t maximum_samples = 16384;
    fft_complex *work = (fft_complex *)malloc(n * sizeof(*work));
    double *samples =
        (double *)malloc(maximum_samples * sizeof(*samples));
    benchmark_result result = {0.0, 0.0, 0, 0.0};
    double total = 0.0;
    size_t count = 0;
    size_t warmup;

    if (work == NULL || samples == NULL) {
        free(work);
        free(samples);
        return result;
    }

    for (warmup = 0; warmup < 2; ++warmup) {
        memcpy(work, input, n * sizeof(*work));
        (void)fft_execute(plan, algorithm, work);
    }

    while ((total < target_seconds || count < 7) &&
           count < maximum_samples) {
        struct timespec start;
        struct timespec end;
        double elapsed;

        memcpy(work, input, n * sizeof(*work));
        clock_gettime(CLOCK_MONOTONIC, &start);
        (void)fft_execute(plan, algorithm, work);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = seconds_between(start, end);
        samples[count++] = elapsed;
        total += elapsed;
    }

    result.checksum = work[(n * 7) / 11].re + work[(n * 5) / 13].im;
    benchmark_sink += result.checksum;
    qsort(samples, count, sizeof(*samples), compare_doubles);
    result.samples = count;
    result.minimum_seconds = samples[0];
    result.median_seconds =
        (count & 1U) != 0
            ? samples[count / 2]
            : 0.5 * (samples[count / 2 - 1] + samples[count / 2]);

    free(work);
    free(samples);
    return result;
}

static int run_benchmarks(const options *settings)
{
    FILE *csv = NULL;
    unsigned power;

    if (settings->csv_path != NULL) {
        csv = fopen(settings->csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr,
                    "cannot open CSV '%s': %s\n",
                    settings->csv_path,
                    strerror(errno));
            return 0;
        }
        fprintf(csv,
                "n,algorithm,median_us,minimum_us,samples,"
                "theoretical_flops,speedup_vs_radix2,checksum\n");
    }

    printf("\nBenchmarks (median execution time; plan/setup excluded)\n");
    printf("FFmpeg AVTX is reported through N=131072, its direct optimized "
           "x86 kernel range.\n");
    printf("%10s  %-16s %12s %10s %14s %11s %8s\n",
           "N",
           "algorithm",
           "median us",
           "speedup",
           "theory FLOPs",
           "GFLOP/s",
           "samples");

    for (power = settings->min_power; power <= settings->max_power; ++power) {
        const size_t n = (size_t)1 << power;
        fft_plan *plan = fft_plan_create(n);
        fft_complex *input = (fft_complex *)malloc(n * sizeof(*input));
        benchmark_result results[FFT_ALGORITHM_COUNT] = {{0}};
        double radix_time;
        int algorithm;

        if (plan == NULL || input == NULL) {
            fprintf(stderr, "allocation failed while benchmarking n=%zu\n", n);
            fft_plan_destroy(plan);
            free(input);
            if (csv != NULL) {
                fclose(csv);
            }
            return 0;
        }

        fill_random(input, n);
        for (algorithm = 0; algorithm < FFT_ALGORITHM_COUNT; ++algorithm) {
            if (!fft_plan_supports(plan, (fft_algorithm)algorithm)) {
                continue;
            }
            results[algorithm] = benchmark_algorithm(
                plan,
                (fft_algorithm)algorithm,
                input,
                settings->target_ms / 1000.0);
            if (results[algorithm].samples == 0) {
                fprintf(stderr,
                        "benchmark allocation failed for n=%zu\n",
                        n);
                fft_plan_destroy(plan);
                free(input);
                if (csv != NULL) {
                    fclose(csv);
                }
                return 0;
            }
        }

        radix_time = results[FFT_RADIX2].median_seconds;
        for (algorithm = 0; algorithm < FFT_ALGORITHM_COUNT; ++algorithm) {
            const fft_algorithm selected = (fft_algorithm)algorithm;
            if (!fft_plan_supports(plan, selected)) {
                continue;
            }

            {
                const uint64_t flops =
                    fft_theoretical_flops(selected, n);
                const double speedup =
                    radix_time / results[algorithm].median_seconds;
                const double gflops =
                    flops == 0
                        ? 0.0
                        : (double)flops /
                              results[algorithm].median_seconds / 1.0e9;

                if (flops == 0) {
                    printf("%10zu  %-16s %12.3f %9.3fx %14s %11s %8zu\n",
                           n,
                           fft_algorithm_name(selected),
                           results[algorithm].median_seconds * 1.0e6,
                           speedup,
                           "n/a",
                           "n/a",
                           results[algorithm].samples);
                } else {
                    printf("%10zu  %-16s %12.3f %9.3fx %14llu %11.3f %8zu\n",
                           n,
                           fft_algorithm_name(selected),
                           results[algorithm].median_seconds * 1.0e6,
                           speedup,
                           (unsigned long long)flops,
                           gflops,
                           results[algorithm].samples);
                }

                if (csv != NULL) {
                    fprintf(csv,
                            "%zu,%s,%.9f,%.9f,%zu,%llu,%.6f,%.17g\n",
                            n,
                            fft_algorithm_name(selected),
                            results[algorithm].median_seconds * 1.0e6,
                            results[algorithm].minimum_seconds * 1.0e6,
                            results[algorithm].samples,
                            (unsigned long long)flops,
                            speedup,
                            results[algorithm].checksum);
                }
            }
        }
        putchar('\n');
        fft_plan_destroy(plan);
        free(input);
    }

    if (csv != NULL) {
        fclose(csv);
        printf("CSV written to %s\n", settings->csv_path);
    }
    return 1;
}

static void print_usage(const char *program)
{
    printf("Usage: %s [options]\n\n", program);
    printf("With no mode option, correctness tests and benchmarks both run.\n\n");
    printf("  --test             run correctness tests only (unless --bench also given)\n");
    printf("  --bench            run benchmarks only (unless --test also given)\n");
    printf("  --min-power P      smallest size is 2^P (default: 4)\n");
    printf("  --max-power P      largest size is 2^P (default: 22)\n");
    printf("  --target-ms MS     timing budget per algorithm/size (default: 100)\n");
    printf("  --csv PATH         also write machine-readable benchmark results\n");
    printf("  --help             show this help\n");
}

static int parse_unsigned(const char *text, unsigned *result)
{
    char *end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > 30) {
        return 0;
    }
    *result = (unsigned)parsed;
    return 1;
}

static int parse_positive_double(const char *text, double *result)
{
    char *end = NULL;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) ||
        parsed <= 0.0) {
        return 0;
    }
    *result = parsed;
    return 1;
}

static int parse_options(int argc, char **argv, options *settings)
{
    int mode_was_selected = 0;
    int i;

    settings->run_tests = 0;
    settings->run_benchmarks = 0;
    settings->min_power = 4;
    settings->max_power = 13;
    settings->target_ms = 100.0;
    settings->csv_path = NULL;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test") == 0) {
            settings->run_tests = 1;
            mode_was_selected = 1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            settings->run_benchmarks = 1;
            mode_was_selected = 1;
        } else if (strcmp(argv[i], "--min-power") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &settings->min_power)) {
                fprintf(stderr, "invalid --min-power value\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--max-power") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &settings->max_power)) {
                fprintf(stderr, "invalid --max-power value\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--target-ms") == 0 && i + 1 < argc) {
            if (!parse_positive_double(argv[++i], &settings->target_ms)) {
                fprintf(stderr, "invalid --target-ms value\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            settings->csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 0;
        }
    }

    if (!mode_was_selected) {
        settings->run_tests = 1;
        settings->run_benchmarks = 1;
    }
    if (settings->min_power > settings->max_power) {
        fprintf(stderr, "--min-power cannot exceed --max-power\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    options settings;
    int success = 1;

    if (!parse_options(argc, argv, &settings)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (settings.run_tests) {
        success = run_correctness_tests() && success;
    }
    if (settings.run_benchmarks) {
        success = run_benchmarks(&settings) && success;
    }

    /* Make the anti-optimisation sink observable without cluttering output. */
    if (!isfinite(benchmark_sink)) {
        fprintf(stderr, "non-finite benchmark checksum\n");
        success = 0;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
