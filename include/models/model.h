#include <vector>
#ifndef DATA_FRAME_INCLUDED
#define DATA_FRAME_INCLUDED
#include "../tabular-data/dataFrame.h"
#endif

enum class GDType { None, BATCH , STOCHASTIC, MINI_BATCH };
enum class Regularizor { None, L1, L2, ENet };
enum class Param {ETA, EPSILON, ITERATION, REGULARIZOR, LAMBDA, ALPHA, GD_TYPE, BATCH_SIZE};

namespace MACHINE_LEARNING {
    template<typename M>
    class SupervisedModel {
        friend class ModelUtil;
        protected:
            std::ofstream output;
            std::vector<std::pair<Param, std::vector<double>>> param_list;
            Matrix<double> x{0}, y{0}, w{0};
            double eta = 1e-9, lamb, alpha, eps = 1e-2;
            ll iter = 1000, batch_size;
            Regularizor r = Regularizor::None;
            GDType t = GDType::None;

            void gradient_descent() {
                switch (t) {
                    case GDType::BATCH: {
                        for (size_t i = 0; i < iter; ++i) {
                            if (loss() <= eps) break;
                            #ifdef WRITE_TO_FILE
                                output << i << '\t' << std::fixed << l;
                            #endif
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
                                if (loss() <= eps) {
                                    term = 1;
                                    break;
                                }
                                #ifdef WRITE_TO_FILE
                                    output << cnt++ << '\t' << std::fixed << l;
                                #endif
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
                                if (loss() <= eps) {
                                    term = 1;
                                    break;
                                }
                                #ifdef WRITE_TO_FILE
                                    output << cnt++ << '\t' << std::fixed << l;
                                #endif
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
        public:
            virtual ~SupervisedModel(){}
            void set_eta(const double eta) {
                this->eta = eta;
            }
            void set_eta(const std::initializer_list<double>&& eta) {
                if (eta.size() == 1) this->eta = *eta.begin();
                else param_list.emplace_back(std::make_pair(Param::ETA, eta));
            }

            void set_epsilon(const double eps) {
                this->eps = eps;
            }
            void set_epsilon(const std::initializer_list<double>&& eps) {
                if (eps.size() == 1) this->eps = *eps.begin();
                else param_list.emplace_back(std::make_pair(Param::EPSILON, eps));
            }

            void set_iteration(const ll iter) {
                this->iter = iter;
            }
            void set_iteration(const std::initializer_list<double>&& iter) {
                if (iter.size() == 1) this->iter = *iter.begin();
                else param_list.emplace_back(std::make_pair(Param::ITERATION, iter));
            }

            void set_regularizor(const Regularizor r, const double lamb = 1, const double alpha = 0.5) {
                this->r = r;
                this->lamb = lamb;
                this->alpha = alpha;
            }
            void set_regularizor(
                const std::initializer_list<double>&& r, 
                const std::initializer_list<double>&& lamb = {1.0}, 
                const std::initializer_list<double>&& alpha = {0.5}) {
                if (r.size() == 1) this->r = static_cast<Regularizor>((size_t) *r.begin());
                else param_list.emplace_back(std::make_pair(Param::REGULARIZOR, r));
                if (lamb.size() == 1) this->lamb = *lamb.begin();
                else param_list.emplace_back(std::make_pair(Param::LAMBDA, lamb));
                if (alpha.size() == 1) this->alpha = *alpha.begin();
                else param_list.emplace_back(std::make_pair(Param::ALPHA, alpha));
            }

            void set_gd_type(const GDType t, const ll batch_size = 64) {
                this->t = t;
                this->batch_size = batch_size;
            }
            void set_gd_type(const std::initializer_list<double>&& t, const std::initializer_list<double>&& batch_size = {64}) {
                if (t.size() == 1) this->t = static_cast<GDType>((size_t) *t.begin());
                else param_list.emplace_back(std::make_pair(Param::GD_TYPE, t));
                if (batch_size.size() == 1) this->batch_size = *batch_size.begin();
                else param_list.emplace_back(std::make_pair(Param::BATCH_SIZE, batch_size));
            }

            void set_params(std::vector<std::pair<Param, double>>& grid) {
                for (auto& i : grid) {
                    switch (i.first) {
                        case Param::ETA:
                            this->eta = i.second;
                            break;
                        case Param::EPSILON:
                            this->eps = i.second;
                            break;
                        case Param::ITERATION:
                            this->iter = i.second;
                            break;
                        case Param::REGULARIZOR:
                            this->r = static_cast<Regularizor>((size_t) i.second);
                            break;
                        case Param::LAMBDA:
                            this->lamb = i.second;
                            break;
                        case Param::ALPHA:
                            this->alpha = i.second;
                            break;
                        case Param::GD_TYPE:
                            this->t = static_cast<GDType>((size_t) i.second);
                            break;
                        case Param::BATCH_SIZE:
                            this->batch_size = i.second;
                    } 
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