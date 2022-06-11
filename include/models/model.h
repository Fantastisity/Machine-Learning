#include <vector>
#ifndef DATA_FRAME_INCLUDED
#define DATA_FRAME_INCLUDED
#include "../tabular-data/dataFrame.h"
#endif
// #define WRITE_TO_FILE

enum class GDType { None, BATCH , STOCHASTIC, MINI_BATCH };
enum class Regularizor { None, L1, L2, ENet };

namespace MACHINE_LEARNING {
    template<typename M>
    class SupervisedModel {
        protected:
            std::ofstream output;
            bool* seen = nullptr;
            Matrix<double> x{0}, y{0}, w{0};
            double eta = 1e-9, lamb, alpha, eps = 1e-2;
            ll iter = 1000, batch_size;
            Regularizor r = Regularizor::None;
            GDType t = GDType::None;

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
                    case GDType::STOCHASTIC: {
                        size_t n = x.rowNum(), ind[n];
                        for (size_t i = 0; i < n; ++i) ind[i] = i;
                        #ifdef WRITE_TO_FILE
                            ll cnt = 0;
                        #endif
                        for (size_t i = 0, term = 0; i < iter && !term; ++i) {
                            std::shuffle(ind, ind + n, std::default_random_engine {});
                            for (auto& j : ind) {
                                if (!seen[j]) seen[j] = 1;
                                double l = loss();
                                #ifdef WRITE_TO_FILE
                                    output << cnt++ << '\t' << std::fixed << l << '\n';
                                #endif
                                if (l <= eps) {
                                    term = 1;
                                    break;
                                }
                                auto x_t = x(rngSlicer(j, j + 1), rngSlicer(0, x.colNum())),
                                    y_t = y(rngSlicer(j, j + 1), rngSlicer(0, y.colNum()));
                                w -= gradient(x_t, y_t) * eta;
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
                                auto x_t = x(ptrSlicer(ind + j, num), rngSlicer(x.colNum())), 
                                        y_t = y(ptrSlicer(ind + j,num), rngSlicer(y.colNum()));
                                w -= gradient(x_t, y_t) * eta;
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
            virtual Matrix<double> gradient(Matrix<double>& X, Matrix<double>& Y){}
            void print_params() {
                pretty_print("", '*', 58, '*');
                pretty_print("", ' ', 38, "Parameter Settings");
                pretty_print("", '*', 58, '*');
                switch (t) { 
                    case GDType::BATCH: 
                        pretty_print("gradient descent type:", ' ', 29, "BGD");
                        break;
                    case GDType::STOCHASTIC:
                        pretty_print("gradient descent type:", ' ', 29, "SGD");
                        break;
                    case GDType::MINI_BATCH:
                        pretty_print("gradient descent type:", ' ', 29, "MBGD"),
                        pretty_print("batch size:", ' ', 40, batch_size);
                        break;
                }
                if (r != Regularizor::None) {
                    switch (r) {
                        case Regularizor::L1:
                            pretty_print("regularizor:", ' ', 39, "Lasso");
                            break;
                        case Regularizor::L2:
                            pretty_print("regularizor:", ' ', 39, "Ridge");
                            break;
                        case Regularizor::ENet:
                            pretty_print("regularizor:", ' ', 39, "Elastic Net"),
                            pretty_print("alpha:", ' ', 45, alpha);
                            break;
                    };
                    pretty_print("lambda:", ' ', 44, lamb);
                }
                
                pretty_print("eta:", ' ', 47, eta);
                pretty_print("epsilon:", ' ', 43, eps);
                pretty_print("iterations:", ' ', 40, iter);
                pretty_print("", '*', 58, '*');
            }

            void print_weights() {
                pretty_print("", '*', 58, '*');
                pretty_print("", ' ', 35, "Final Weights");
                pretty_print("", '*', 58, '*');
                for (size_t i = 0, r = w.rowNum(); i < r; ++i) pretty_print(i, ' ', 50, w(i, 0));
                pretty_print("", '*', 58, '*');
            }
        public:
            virtual ~SupervisedModel(){
                if (seen) free(seen);
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

            void set_params(std::vector<std::pair<char*, elem>>& grid) {
                for (auto& i : grid) {
                    if (!strcmp(i.first, "eta")) this->eta = i.second;
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