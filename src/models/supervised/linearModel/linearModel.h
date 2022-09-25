#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif
// #define WRITE_TO_FILE

enum class GDType { None, BATCH , SAG, MINI_BATCH }; // Gradient descent type
enum class Regularizor { None, L1, L2, ENet };

namespace MACHINE_LEARNING {
    template<typename M>
    class LinearModel {
        protected:
            std::ofstream output;
            bool* seen = nullptr;
            Matrix<double> x{0}, y{0}, w{0}, gradient_table{0}, gradient_sum{0};
            double eta = 1e-9, lamb, alpha, eps = 1e-2;
            size_t iter = 1000, batch_size;
            Regularizor r = Regularizor::None;
            GDType t = GDType::None;

            template<typename T, typename R>
            void init(T&& x, R&& y) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);

                this->x.addCol(std::vector<double>(x.rowNum(), 1.0).data());

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);

                if (t == GDType::SAG) {
                    if (!(seen = (bool*) calloc(x.rowNum(), 1))) {
                        std::cerr << "error calloc\n"; exit(1);
                    }
                    this->gradient_table = Matrix<double>(this->x.rowNum(), this->x.colNum());
                    this->gradient_sum = Matrix<double>(this->x.colNum(), 1);
                }
            }

            void gradient_descent() {
                switch (t) {
                    case GDType::BATCH: {
                        for (size_t i = 0; i < iter; ++i) {
                            double l = loss();
                            #ifdef WRITE_TO_FILE
                                output << i << '\t' << std::fixed << l << '\n';
                            #endif
                            if (l <= eps) break;
                            w -= gradient(x, y) * eta;
                        }
                        #ifdef WRITE_TO_FILE
                            output.close();
                        #endif
                        return;
                    }
                    case GDType::SAG: {
                        size_t m = x.rowNum(), ind[m];
                        for (size_t i = 0; i < m; ++i) ind[i] = i;
                        #ifdef WRITE_TO_FILE
                            ll cnt = 0;
                        #endif
                        for (size_t i = 0, term = 0; i < iter && !term; ++i) {
                            std::shuffle(ind, ind + m, std::default_random_engine {});
                            for (auto& j : ind) {
                                double l = loss();
                                #ifdef WRITE_TO_FILE
                                    output << cnt++ << '\t' << std::fixed << l << '\n';
                                #endif
                                if (l <= eps) {
                                    term = 1;
                                    break;
                                }
                                auto dL = gradient(x(rngSlicer(j, j + 1), rngSlicer(x.colNum())), y(rngSlicer(j, j + 1), rngSlicer(y.colNum())));
                                
                                if (!seen[j]) { // First time seeing the point, initialize its entry in the gradient table
                                    seen[j] = 1;
                                    for (size_t k = 0, n = x.colNum(); k < n; ++k) gradient_table(j, k) = dL(k, 0);
                                    gradient_sum += dL;
                                }
                                else for (size_t k = 0, n = x.colNum(); k < n; ++k) // Update the gradient for the points in the gradient table
                                    gradient_sum(k, 0) += dL(k, 0) - gradient_table(j, k), gradient_table(j, k) = dL(k, 0);
                                
                                w -= gradient_sum * (eta / m);
                            }
                        }
                        #ifdef WRITE_TO_FILE
                            output.close();
                        #endif
                        return;
                    }
                    case GDType::MINI_BATCH: {
                        size_t n = x.rowNum(), ind[n];
                        while (n / batch_size < 2) batch_size >>= 1;
                        if (!batch_size) {
                            logger("switching to BGD due to insufficient data");
                            t = GDType::BATCH, gradient_descent();
                        }
                        for (size_t i = 0; i < n; ++i) ind[i] = i;
                        #ifdef WRITE_TO_FILE
                            ll cnt = 0;
                        #endif
                        for (size_t i = 0, term = 0; i < iter && !term; ++i) {
                            std::shuffle(ind, ind + n, std::default_random_engine {});
                            for (size_t j = 0, num; j < n; j += num) {
                                double l = loss();
                                #ifdef WRITE_TO_FILE
                                    output << cnt++ << '\t' << std::fixed << l << '\n';
                                #endif
                                if (l <= eps) {
                                    term = 1;
                                    break;
                                }
                                num = std::min(static_cast<size_t>(batch_size), n - j);
                                w -= gradient(x(ptrSlicer(ind + j, num), rngSlicer(x.colNum())), 
                                                y(ptrSlicer(ind + j, num), rngSlicer(y.colNum()))) * eta;
                            }
                        }
                        #ifdef WRITE_TO_FILE
                            output.close();
                        #endif
                        return;
                    }
                }
            }

            virtual double loss(){}
            virtual Matrix<double> gradient(const Matrix<double>& X, const Matrix<double>& Y){}
            void print_params() {
                printf("**********************************************************\n");
                printf("\t\t\t\t\tParameter Settings\n");
                printf("**********************************************************\n");
                switch (t) { 
                    case GDType::BATCH: 
                        printf("gradient descent type:\t\t\t\t\t\t\t\t BGD\n");
                        break;
                    case GDType::SAG:
                        printf("gradient descent type:\t\t\t\t\t\t\t\t SAG\n");
                        break;
                    case GDType::MINI_BATCH:
                        printf("gradient descent type:\t\t\t\t\t\t\t   m-BGD\n");
                        printf("batch size:\t\t\t\t\t\t\t\t\t\t\t  %3u\n", batch_size);
                        break;
                }
                if (r != Regularizor::None) {
                    switch (r) {
                        case Regularizor::L1:
                            printf("regularizor:\t\t\t\t\t\t\t\t\t   Lasso\n");
                            break;
                        case Regularizor::L2:
                            printf("regularizor:\t\t\t\t\t\t\t\t\t   Ridge\n");
                            break;
                        case Regularizor::ENet:
                            printf("regularizor:\t\t\t\t\t\t\t\t Elastic Net\n");
                            printf("alpha:\t\t\t\t\t\t\t\t\t\t\t\t %.1f\n", alpha);
                            break;
                    };
                    printf("lambda:\t\t\t\t\t\t\t\t\t\t\t\t %.1f\n", lamb);
                }
                
                printf("eta:\t\t\t\t\t\t\t\t\t\t\t   %.0e\n", eta);
                printf("epsilon:\t\t\t\t\t\t\t\t\t\t    %.2f\n", eps);
                printf("iterations:\t\t\t\t\t\t\t\t\t\t\t%u\n\n", iter);
            }

            void print_weights() {
                printf("**********************************************************\n");
                printf("\t\t\t\t\t  Final Weights\n");
                printf("**********************************************************\n");
                for (size_t i = 0, r = w.rowNum(); i < r; ++i) printf("%lld\t\t\t\t\t\t\t\t\t\t\t  %+9.5f\n", i, w(i, 0));
                puts("");
            }
        public:
            virtual ~LinearModel(){
                if (seen) free(seen), seen = nullptr;
            }
            void set_eta(const double eta) {
                this->eta = eta;
            }

            void set_epsilon(const double eps) {
                this->eps = eps;
            }

            void set_iteration(const ll iter) {
                this->iter = iter;
            }

            void set_regularizor(const Regularizor r, const double lamb = 1, const double alpha = 0.5) {
                this->r = r;
                this->lamb = lamb;
                this->alpha = alpha;
            }

            void set_gd_type(const GDType t, const ll batch_size = 64) {
                this->t = t;
                this->batch_size = batch_size;
            }

            void set_params(std::vector<std::pair<char const *, elem>>& grid) {
                for (auto& i : grid) {
                    if      (!strcmp(i.first, "eta")) this->eta = i.second;
                    else if (!strcmp(i.first, "epsilon")) this->eps = i.second;
                    else if (!strcmp(i.first, "iteration")) this->iter = i.second;
                    else if (!strcmp(i.first, "regularizor")) this->r = static_cast<Regularizor>((size_t) i.second);
                    else if (!strcmp(i.first, "lambda")) this->lamb = i.second;
                    else if (!strcmp(i.first, "alpha")) this->alpha = i.second;
                    else if (!strcmp(i.first, "gd_type")) this->t = static_cast<GDType>((size_t) i.second);
                    else if (!strcmp(i.first, "batch_size")) this->batch_size = i.second;
                }
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