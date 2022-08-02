#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    template<typename M>
    class SVM {
            UTIL_BASE::MODEL_UTIL::KERNEL::Kernel k = UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::LINEAR;
            float gamma = 0.1, tol = 1e-3, eps = 1e-3, sigma = 1;
            double* alpha = nullptr, * error = nullptr, b = 0, C = 100;

            bool takeStep(size_t i, size_t j) {
                size_t n = x.rowNum(), m = x.colNum();
                auto obj_func = [&]() {
                    double lpart = 0, rpart = 0;
                    for (size_t i = 0; i < n; ++i) {
                        lpart += alpha[i];
                        for (size_t j = 0; j < n; ++j) 
                            switch (k) {
                                case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::RBF:
                                    rpart += alpha[i] * alpha[j] * y(i, 0) * y(j, 0) * 
                                             UTIL_BASE::MODEL_UTIL::KERNEL::rbf(&x(i, 0), &x(j, 0), m, gamma);
                                    break;
                                case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::GAUSSIAN:
                                    rpart += alpha[i] * alpha[j] * y(i, 0) * y(j, 0) * 
                                             UTIL_BASE::MODEL_UTIL::KERNEL::gaussian(&x(i, 0), &x(j, 0), m, sigma);
                                    break;
                                case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::LINEAR:
                                    rpart += alpha[i] * alpha[j] * y(i, 0) * y(j, 0) * 
                                             UTIL_BASE::MODEL_UTIL::KERNEL::linear(&x(i, 0), &x(j, 0), m);
                            }
                    }
                    return lpart + 0.5 * rpart;
                };
                double alph1 = alpha[i], alph2 = alpha[j], y1 = y(i, 0), y2 = y(j, 0), s = y1 * y2, 
                       lbound, ubound, k11, k12, k22, eta, a1, a2, b1, b2, E1 = error[i], E2 = error[j], threshold;
                if (std::abs(E1 - E2) < eps) return 0;

                if (y1 != y2) lbound = alph2 > alph1 ? alph2 - alph1 : 0, ubound = alph2 > alph1 ? C : C + alph2 - alph1;
                else          lbound = alph2 + alph1 >= C ? alph2 + alph1 - C : 0, ubound = alph2 + alph1 >= C ? C : alph2 + alph1;
                
                if (lbound == ubound) return 0;

                switch (k) {
                    case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::RBF:
                        k11 = UTIL_BASE::MODEL_UTIL::KERNEL::rbf(&x(i, 0), &x(i, 0), m, gamma),
                        k12 = UTIL_BASE::MODEL_UTIL::KERNEL::rbf(&x(i, 0), &x(j, 0), m, gamma),
                        k22 = UTIL_BASE::MODEL_UTIL::KERNEL::rbf(&x(j, 0), &x(j, 0), m, gamma);
                        break;
                    case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::GAUSSIAN:
                        k11 = UTIL_BASE::MODEL_UTIL::KERNEL::gaussian(&x(i, 0), &x(i, 0), m, sigma),
                        k12 = UTIL_BASE::MODEL_UTIL::KERNEL::gaussian(&x(i, 0), &x(j, 0), m, sigma),
                        k22 = UTIL_BASE::MODEL_UTIL::KERNEL::gaussian(&x(j, 0), &x(j, 0), m, sigma);
                        break;
                    case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::LINEAR:
                        k11 = UTIL_BASE::MODEL_UTIL::KERNEL::linear(&x(i, 0), &x(i, 0), m),
                        k12 = UTIL_BASE::MODEL_UTIL::KERNEL::linear(&x(i, 0), &x(j, 0), m),
                        k22 = UTIL_BASE::MODEL_UTIL::KERNEL::linear(&x(j, 0), &x(j, 0), m);
                }      

                eta = k11 + k22 - 2 * k12;
                if (eta > 0) {
                    a2 = alph2 + y2 * (E1 - E2) / eta;
                    a2 = a2 < lbound ? lbound : (a2 > ubound ? ubound : a2);
                } else {
                    double lobj, hobj;
                    std::swap(alpha[j], lbound);
                    lobj = obj_func();
                    std::swap(alpha[j], lbound);
                    std::swap(alpha[j], ubound);
                    hobj = obj_func();
                    std::swap(alpha[j], ubound);
                    if (lobj > hobj) a2 = lbound;
                    else if (lobj < hobj) a2 = ubound;
                    else a2 = alph2;
                }
                if (std::abs(a2 - alph2) < eps) return 0;
                a1 = alph1 + s * (alph2 - a2);

                b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b,
                b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b;

                threshold = (b1 + b2) / 2;
                alpha[i] = a1, alpha[j] = a2;
                b = threshold;

                error[i] = f(&x(i, 0)) - y(i, 0), error[j] = f(&x(j, 0)) - y(j, 0);
                return 1;
            }

            bool examineExample(size_t j) {
                size_t n = y.rowNum(), i;
                double y2 = y(j, 0), alph2 = alpha[j], E2 = error[j], r2 = y2 * E2;
                if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0)) {
                    bool flg = 0; for (size_t k = 0; k < n; ++k) if (k ^ j && alpha[k] && alpha[k] != C) { flg = 1; break; }
                    if (flg) {
                        if (E2 > 0) {
                            double min_err = std::numeric_limits<double>::max();
                            for (size_t k = 0; k < n; ++k) if (k ^ j && min_err > error[k]) min_err = error[k], i = k;
                        } else {
                            double max_err = std::numeric_limits<double>::min();
                            for (size_t k = 0; k < n; ++k) if (k ^ j && max_err < error[k]) max_err = error[k], i = k;
                        }
                        if (takeStep(i, j)) return 1;
                        for (size_t k = rand() % n; k < n; ++k) if (k ^ j && alpha[k] && alpha[k] != C && takeStep(k, j)) return 1;
                    }
                    for (size_t k = rand() % n; k < n; ++k) if (k ^ j && takeStep(k, j)) return 1;
                }
                return 0;
            }
        protected:
            Matrix<double> x{0}, y{0};
            void train() {
                bool examineAll = 1; size_t numChanged, n = y.rowNum();
                while (numChanged || examineAll) {
                    numChanged = 0;
                    if (examineAll) for (size_t j = 0; j < n; ++j) numChanged += examineExample(j);
                    else for (size_t j = 0; j < n; ++j) if (alpha[j] && alpha[j] != C) numChanged += examineExample(j);
                    examineAll ^= 1;
                }
            }

            double f(const double *const X) {
                double ypred = -b;
                for (size_t i = 0, n = x.rowNum(), m = x.colNum(); i < n; ++i) {
                    switch (k) {
                        case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::RBF:
                            ypred += alpha[i] * y(i, 0) * UTIL_BASE::MODEL_UTIL::KERNEL::rbf(&x(i, 0), X, m, gamma);
                            break;
                        case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::GAUSSIAN:
                            ypred += alpha[i] * y(i, 0) * UTIL_BASE::MODEL_UTIL::KERNEL::gaussian(&x(i, 0), X, m, sigma);
                            break;
                        case UTIL_BASE::MODEL_UTIL::KERNEL::Kernel::LINEAR:
                            ypred += alpha[i] * y(i, 0) * UTIL_BASE::MODEL_UTIL::KERNEL::linear(&x(i, 0), X, m);
                    }             
                }
                return ypred;
            }

            template<typename T, typename R>
            void init(T&& x, R&& y) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);

                size_t n = x.rowNum();
                error = new double[n], alpha = new double[n]{};
                for (size_t i = 0; i < n; ++i) error[i] = f(&(this->x(i, 0))) - this->y(i, 0);
            }
        public:
            ~SVM() {
                if (error) {
                    delete[] error;
                    error = nullptr;
                }
                if (alpha) {
                    delete[] alpha;
                    alpha = nullptr;
                }
            }

            void set_C(const double C) {
                this->C = C;
            }

            void set_gamma(const float gamma) {
                this->gamma = gamma;
            }

            void set_sigma(const float sigma) {
                this->sigma = sigma;
            }

            void set_tol(const float tol) {
                this->tol = tol;
            }

            void set_eps(const float eps) {
                this->eps = eps;
            }

            void set_kernel(UTIL_BASE::MODEL_UTIL::KERNEL::Kernel k) {
                this->k = k;
            }

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                (static_cast<M*>(this))->fit(std::forward<T>(x), std::forward<R>(y), verbose);
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                return (static_cast<M*>(this))->predict(std::forward<T>(xtest));
            }
    };
}