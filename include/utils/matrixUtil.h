#ifndef UTIL_BASE_INCLUDED
#define UTIL_BASE_INCLUDED
#include "utilityBase.h"
#endif
#include <cstdlib>
#include <unordered_map>
#define ROW 1
#define COL 0

namespace MACHINE_LEARNING {
    using ll = long long;
    template<typename T>
    struct slice {
        const T start, end;
        const size_t size;
        slice(const T& start, const T& end) : start(start), end(end), size(this->end - start) {}
        explicit slice(const T& end) : start(0), end(end), size(this->end - start) {}
    };

    template<typename T>
    struct slice<T*> {
        T* start, * end;
        const size_t size;
        slice(T* start, const size_t end) : start(start), end(start + end), size(this->end - start) {}
    };

    using rangeSlicer = slice<size_t>; 
    using ptrSlicer = slice<size_t*>;

    struct MatrixUtil : public UtilBase {
        template<typename T>
        struct isPTR {
            const static bool val = 0;
        };

        template<typename T>
        struct isPTR<T*> {
            const static bool val = 1;
        };

        template<typename X, typename Y, typename Z>
        struct Mult {
            static void mult(X* x, Y* y, Z*& res, size_t r, size_t k, size_t c) {
                auto batch_mult = [&](size_t from, size_t to) {
                    for (size_t i = from; i < to; ++i)
                        for (size_t m = 0; m < k; ++m)
                            for (size_t j = 0; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
                };
                const size_t pcnt = std::thread::hardware_concurrency();
                std::thread threads[pcnt - 1];
                size_t batch = r / pcnt;
                batch_mult(0, batch);
                for (size_t i = 0, cnt = batch; i < pcnt - 1; ++i) {
                    threads[i] = std::thread(batch_mult, cnt, cnt + batch < r ? cnt + batch : r);
                    threads[i].join();
                    cnt += batch;
                }
            }
        };

        template<>
        struct Mult<double, double, double> {
            static void mult(double* x, double* y, double*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128 sum = _mm_setzero_pd();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_pd(sum, _mm_mul_pd(_mm_set1_pd(x[i * k + m]), _mm_loadu_pd(&y[m * c + j])));
                        _mm_storeu_pd(&res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<float, float, float> {
            static void mult(float* x, float* y, float*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128 sum = _mm_setzero_ps();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_set1_ps(x[i * k + m]), _mm_loadu_ps(&y[m * c + j])));
                        _mm_storeu_ps(&res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<int, int, int> {
            static void mult(int* x, int* y, int*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128i sum = _mm_setzero_si128();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_epi32(sum, _mm_mullo_epi32(_mm_set1_epi32(x[i * k + m]), _mm_loadu_si128((__m128i*) y[m * c + j])));
                        _mm_store_si128((__m128i*) &res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<double, int, double> {
            static void mult(double* x, int* y, double*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128 sum = _mm_setzero_pd();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_pd(sum, _mm_mul_pd(_mm_set1_pd(x[i * k + m]), 
                                            _mm_cvtepi32_pd(_mm_loadu_si128((__m128i*) &y[m * c + j]))));
                        _mm_storeu_pd(&res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<int, double, double> {
            static void mult(int* x, double* y, double*& res, size_t r, size_t k, size_t c) {
                Mult<double, int, double>::mult(y, x, res, r, k, c);
            }
        };

        template<>
        struct Mult<double, float, double> {
            static void mult(double* x, float* y, double*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128 sum = _mm_setzero_pd();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_pd(sum, _mm_mul_pd(_mm_set1_pd(x[i * k + m]), _mm_cvtps_pd(_mm_loadu_ps(&y[m * c + j]))));
                        _mm_storeu_pd(&res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<float, double, double> {
            static void mult(float* x, double* y, double*& res, size_t r, size_t k, size_t c) {
                Mult<double, float, double>::mult(y, x, res, r, k, c);
            }
        };

        template<>
        struct Mult<float, int, float> {
            static void mult(float* x, int* y, float*& res, size_t r, size_t k, size_t c) {
                size_t end = c - c % 4;
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < end; j += 4) {
                        __m128 sum = _mm_setzero_ps();
                        for (size_t m = 0; m < k; ++m)
                            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_set1_ps(x[i * k + m]), 
                                            _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*) &y[m * c + j]))));
                        _mm_storeu_ps(&res[i * c + j], sum);
                    }
                if (end == c) return;
                for (size_t i = 0; i < r; ++i)
                    for (size_t m = 0; m < k; ++m)
                        for (size_t j = end; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
            }
        };

        template<>
        struct Mult<int, float, float> {
            static void mult(int* x, float* y, float*& res, size_t r, size_t k, size_t c) {
                Mult<float, int, float>::mult(y, x, res, r, k, c);
            }
        };
    };
}