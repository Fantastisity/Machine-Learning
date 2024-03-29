#ifndef UTIL_BASE_INCLUDED
#define UTIL_BASE_INCLUDED
#include "utilityBase.h"
#endif
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#define ROW 1
#define COL 0
#define MAX(a, b) (a > b ? a : b)

namespace MACHINE_LEARNING {
    namespace UTIL_BASE {
         namespace MATRIX_UTIL {
            template<typename T>
            struct slice {
                const T start, end;
                const size_t size;
                slice(const T& start, const T& end) : start(start), end(end), size(this->end - start) {}
                explicit slice(const T& end) : start(0), end(end), size(this->end - start) {} // Starting from the 0th index
            };

            template<typename T>
            struct slice<T*> {
                T* start, * end;
                const size_t size;
                slice(T* start, const size_t len) : start(start), end(start + len), size(len) {}
            };

            // Determine whether s is double, with converted value stored in res
            inline bool isDouble(const char* s, double& res) {
                char* end;
                res = strtod(s, &end);
                return *end == '\0';
            }

            /*
                Convert matrix element type
                @template param
                    R: target type
                    T: original type
                @param
                    cap: number of elements
            */
            template<typename R, typename T>
            inline void conv_type(R* res, T* mat, const size_t cap) {
                for (size_t i = 0; i < cap; ++i) {
                    res[i] = static_cast<R>(mat[i]);
                }
            }

            template<typename T>
            struct isPTR {
                const static bool val = 0;
            };

            template<typename T>
            struct isPTR<T*> {
                const static bool val = 1;
            };

            // i: column & row index(mat[i, i])    m: number of rows    na: number of columns of a    nb: number of columns of b
            template<typename T>
            inline bool partial_pivoting(T* a, T* b, const size_t i, const size_t m, const size_t na, const size_t nb) {
                T tmp = a[i * na + i];
                size_t swap_ind = -1;
                for (size_t j = i + 1; j < m; ++j) 
                    if (tmp < std::abs(a[j * na + i]))
                        tmp = a[j * na + i], swap_ind = j; // Get the row index of the maximum value
                if (!tmp) return 0; // Ignore current column
                for (size_t j = 0, n = MAX(na, nb); j < n; ++j) {
                    if (j < na) std::swap(a[i * n + j], a[swap_ind * n + j]); // Exchange row-to-be-swapped with ith row
                    if (b && j < nb) std::swap(b[i * n + j], b[swap_ind * n + j]);
                }
                return 1;
            }

            template<typename T>
            inline bool gauss_elimination(T* a, T* b, const size_t m, const size_t na, const size_t nb, bool check_invertible = 0) {
                assert(a != nullptr);
                for (size_t i = 0, r = 0; i < na - 1 && r < m; ++i, ++r) {
                    while (!partial_pivoting(a, b, i, m, na, nb)) {
                        if (check_invertible) return 0; // Invertible matrices cannot have zero columns
                        ++i;
                        if (i == na - 1) return 1;
                    }
                    for (size_t j = r + 1; j < m; ++j) {
                        if (!a[j * na + i]) continue;
                        double frac = 1.0 * a[r * na + i] / a[j * na + i];
                        for (size_t k = 0, n = MAX(na, nb); k < n; ++k) {
                            if (k < na) a[j * n + k] = a[j * n + k] * frac - a[i * n + k];
                            if (b && k < nb) b[j * n + k] = b[j * n + k] * frac - b[i * n + k];
                        }
                    }
                }
                return 1;
            }

            template<typename T>
            inline bool gauss_jordan_elimination(T* a, T* b, const size_t m, const size_t na, const size_t nb, bool check_invertible = 0) {
                assert(a != nullptr);
                double tmp;
                for (size_t i = 0, r = 0; i < na && r < m; ++i, ++r) {
                    while (!a[r * na + i] && !partial_pivoting(a, b, i, m, na, nb)) {
                        if (check_invertible) return 0;
                        ++i;
                        if (i == na) return 1;
                    }
                    if (a[r * na + i] != 1) {
                        tmp = 1.0 / a[r * na + i];
                        for (size_t j = 0, n = MAX(na, nb); j < n; ++j) {
                            if (j < na) a[r * na + j] *= tmp; 
                            if (b && j < nb) b[r * nb + j] *= tmp;
                        }
                    }
                    for (size_t j = r + 1; j < m; ++j) {
                        if (!a[j * na + i]) continue;
                        tmp = a[j * na + i];
                        for (size_t k = 0, n = MAX(na, nb); k < n; ++k) {
                            if (k < na) a[j * na + k] -= a[r * na + k] * tmp;
                            if (b && k < nb) b[j * nb + k] -= b[r * nb + k] * tmp;
                        }
                    }
                }

                for (size_t i = na - 1, r = m - 1; r; --r, --i) {
                    while (!a[r * na + i]) --i;
                    for (size_t j = 0; j < r; ++j) {
                        tmp = a[j * na + i];
                        for (size_t k = 0, n = MAX(na, nb); k < n; ++k) {
                            if (k < na) a[j * n + k] -= a[r * n + k] * tmp;
                            if (b && k < nb) b[j * n + k] -= b[r * n + k] * tmp;
                        }
                    }
                }
                return 1;
            }

            /*
                General matrix multiplcation using threads
                @param
                    x: left matrix dim: r * k
                    y: right matrix dim: k * c
                    res: stores the multiplication result dim: r * c
            */
            template<typename X, typename Y, typename Z>
            struct Mult {
                static void mult(X* x, Y* y, Z* res, size_t r, size_t k, size_t c) {
                    auto batch_mult = [&](size_t from, size_t to) {
                        for (size_t i = from; i < to; ++i)
                            for (size_t m = 0; m < k; ++m)
                                for (size_t j = 0; j < c; ++j) res[i * c + j] += x[i * k + m] * y[m * c + j];
                    };
                    const size_t pcnt = std::thread::hardware_concurrency(); // Number of cores
                    std::thread threads[pcnt - 1]; // Ignore the main thread
                    size_t batch = r / pcnt; // Determine the batch(no. of rows) size
                    batch_mult(0, batch); // Main thread
                    for (size_t i = 0, cnt = batch; i < pcnt - 1; ++i, cnt += batch) { // Remaining threads
                        threads[i] = std::thread(batch_mult, cnt, cnt + batch < r ? cnt + batch : r);
                        threads[i].join();
                    }
                }
            };

            // SSE multiplication for specific val types (double, int, float)
            template<>
            struct Mult<double, double, double> {
                static void mult(double* x, double* y, double* res, size_t r, size_t k, size_t c) {
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
                static void mult(float* x, float* y, float* res, size_t r, size_t k, size_t c) {
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
                static void mult(int* x, int* y, int* res, size_t r, size_t k, size_t c) {
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
                static void mult(double* x, int* y, double* res, size_t r, size_t k, size_t c) {
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
                static void mult(int* x, double* y, double* res, size_t r, size_t k, size_t c) {
                    Mult<double, int, double>::mult(y, x, res, r, k, c);
                }
            };

            template<>
            struct Mult<double, float, double> {
                static void mult(double* x, float* y, double* res, size_t r, size_t k, size_t c) {
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
                static void mult(float* x, double* y, double* res, size_t r, size_t k, size_t c) {
                    Mult<double, float, double>::mult(y, x, res, r, k, c);
                }
            };

            template<>
            struct Mult<float, int, float> {
                static void mult(float* x, int* y, float* res, size_t r, size_t k, size_t c) {
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
                static void mult(int* x, float* y, float* res, size_t r, size_t k, size_t c) {
                    Mult<float, int, float>::mult(y, x, res, r, k, c);
                }
            };
        }
    }

    /* 
        Used for slicing matrix by supplying a row/column range [start, end)
        eg. Matrix(rngSlicer(2, 3), rngSlicer(1, 2)): get the 2nd row and 1st column
    */
    using rngSlicer = UTIL_BASE::MATRIX_UTIL::slice<size_t>; 
    /* 
        Used for slicing matrix by supplying a list of indices (start, length)
        eg. Matrix(ptrSlicer([1, 3, 5], 3), ptrSlicer([2], 1)): get the 1st, 3rd and 5th row and 2nd column
    */
    using ptrSlicer = UTIL_BASE::MATRIX_UTIL::slice<size_t*>;

    struct CstrFunctor {
        size_t operator()(const char* str) const {
            size_t hash = 5381;
            for (; *str; ++str) hash = ((hash << 5) + hash) + *str;
            return hash;
        }
        bool operator()(const char* a, const char* b) const {
            return !strcmp(a, b);
        }
    };

    // Matrix element(used mainly for convertion b.t double and string)
    struct elem {
        enum class vtype { DBL, STR }; // Element value type
        vtype t;
        double dval{0.0};
        char* sval = nullptr;

        operator bool() const {
            assert(t != vtype::STR);
            return dval;
        }

        operator double() {
            assert(t == vtype::DBL);
            return dval;
        }

        const char* c_str() {
            assert(t == vtype::STR);
            return sval;
        }

        elem& operator= (const elem& e) {
            switch (e.t) {
                case vtype::DBL:
                    dval = e.dval, t = vtype::DBL;
                    break;
                case vtype::STR:
                    sval = strdup(e.sval), t = vtype::STR;
            }
            return *this;
        }

        elem& operator= (elem&& e) {
            switch (e.t) {
                case vtype::DBL:
                    dval = e.dval, t = vtype::DBL;
                    break;
                case vtype::STR:
                    sval = e.sval, t = vtype::STR, e.sval = nullptr;
            }
            return *this;
        }

        elem() = default;
        ~elem() {
            if (sval) {
                free(sval);
                sval = nullptr;
            }
        }

        elem(const elem& e) {
            switch (e.t) {
                case vtype::DBL:
                    dval = e.dval, t = vtype::DBL;
                    break;
                case vtype::STR:
                    sval = strdup(e.sval), t = vtype::STR;
            }
        }

        elem(elem&& e) {
            switch (e.t) {
                case vtype::DBL:
                    dval = e.dval, t = vtype::DBL;
                    break;
                case vtype::STR:
                    sval = e.sval, t = vtype::STR, e.sval = nullptr;
            }
        }

        elem(const double val) : dval(val), t(vtype::DBL) {}
        elem(const char* val) {
            double tmp_dval;
            // Determine the type of the value
            if (UTIL_BASE::MATRIX_UTIL::isDouble(val, tmp_dval)) dval = tmp_dval, t = vtype::DBL;
            else sval = strdup(val), t = vtype::STR;
        }

        elem& operator= (const double val) {
            dval = val, t = vtype::DBL;
            return *this;
        }

        elem& operator= (const char* val) {
            double tmp_dval;
            if (UTIL_BASE::MATRIX_UTIL::isDouble(val, tmp_dval)) dval = tmp_dval, t = vtype::DBL;
            else sval = strdup(val), t = vtype::STR;
            return *this;
        }

        bool operator== (const elem& e) const {
            bool res = t == e.t;
            if (!res) return 0;
            switch (e.t) {
                case vtype::DBL:
                    res = e.dval == dval;
                    break;
                case vtype::STR:
                    res = !strcmp(e.sval, sval);
            }
            return res;
        }

        bool operator!= (const elem& e) const {
            return !(*this == e);
        }

        elem& operator+= (const elem& e) {
            assert(t != vtype::STR && e.t != vtype::STR);
            dval += e.dval;
            return *this;
        }

        template<typename T>
        auto operator+= (const T val) 
        -> typename std::enable_if<UTIL_BASE::isNumerical<T>::val, elem&>::type {
            assert(t != vtype::STR);
            dval += val;
            return *this;
        }

        elem& operator-= (const elem& e) {
            assert(t != vtype::STR && e.t != vtype::STR);
            dval -= e.dval;
            return *this;
        }

        template<typename T>
        auto operator-= (const T val)
        -> typename std::enable_if<UTIL_BASE::isNumerical<T>::val, elem&>::type {
            assert(t != vtype::STR);
            dval -= val;
            return *this;
        }

        elem& operator*= (const elem& e) {
            assert(t != vtype::STR && e.t != vtype::STR);
            dval *= e.dval;
            return *this;
        }

        template<typename T>
        auto operator*= (const T val)
        -> typename std::enable_if<UTIL_BASE::isNumerical<T>::val, elem&>::type {
            assert(t != vtype::STR);
            dval *= val;
            return *this;
        }

        elem& operator/= (const elem& e) {
            assert(t != vtype::STR && (e.t == vtype::DBL && e.dval));
            dval /= e.dval;
            return *this;
        }

        template<typename T>
        auto operator/= (const T val)
        -> typename std::enable_if<UTIL_BASE::isNumerical<T>::val, elem&>::type {
            assert(t != vtype::STR && val);
            dval /= val;
            return *this;
        }

        elem operator-() {
            assert(t != vtype::STR);
            return elem(-dval);
        }

        elem& operator++ () {
            assert(t != vtype::STR);
            ++dval;
            return *this;
        }

        elem operator++ (int) {
            assert(t != vtype::STR);
            elem tmp = *this;
            ++dval;
            return tmp;
        }

        elem& operator-- () {
            assert(t != vtype::STR);
            --dval;
            return *this;
        }

        elem operator-- (int) {
            assert(t != vtype::STR);
            elem tmp = *this;
            --dval;
            return tmp;
        }

        template<typename T>
        elem operator+ (const T& rht) const {
            elem e = *this;
            e += rht;
            return e;
        }

        template<typename T>
        elem operator- (const T& rht) const {
            elem e = *this;
            e -= rht;
            return e;
        }

        template<typename T>
        elem operator* (const T& rht) const {
            elem e = *this;
            e *= rht;
            return e;
        }

        template<typename T>
        elem operator/ (const T& rht) const {
            elem e = *this;
            e /= rht;
            return e;
        }
    };
    using evType = elem::vtype;
    std::ostream& operator<< (std::ostream& os, const elem& e);
}

namespace std {
    template<>
    struct hash<MACHINE_LEARNING::elem> {
        size_t operator()(const MACHINE_LEARNING::elem& e) const;
    };
}