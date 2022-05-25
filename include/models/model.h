#include <vector>
#ifndef DATA_FRAME_INCLUDED
#define DATA_FRAME_INCLUDED
#include "../tabular-data/dataFrame.h"
#endif

enum Type { BATCH , STOCHASTIC, MINI_BATCH };
enum Regularizor { None, L1, L2, ENet };

namespace MACHINE_LEARNING {
    class SupervisedModel {
        protected:
            std::ofstream output;
            std::vector<std::pair<size_t, std::vector<double>>> varying_params;
            Matrix<double> x{0}, y{0};
            double eta = 1e-9, lamb, alpha, eps = 1e-2;
            ll iter = 1000, batch_size;
            Regularizor r = None;
            Type t = BATCH;
        public:
            void set_eta(const double eta) {
                this->eta = eta;
            }
            void set_eta(const std::initializer_list<double>&& eta) {
                if (eta.size() == 1) this->eta = *eta.begin();
                else varying_params.emplace_back(std::make_pair(0, eta));
            }

            void set_epsilon(const double eps) {
                this->eps = eps;
            }
            void set_epsilon(const std::initializer_list<double>&& eps) {
                if (eps.size() == 1) this->eps = *eps.begin();
                else varying_params.emplace_back(std::make_pair(1, eps));
            }

            void set_iteration(const ll iter) {
                this->iter = iter;
            }
            void set_iteration(const std::initializer_list<double>&& iter) {
                if (iter.size() == 1) this->iter = *iter.begin();
                else varying_params.emplace_back(std::make_pair(2, iter));
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
                else varying_params.emplace_back(std::make_pair(3, r));
                if (lamb.size() == 1) this->lamb = *lamb.begin();
                else varying_params.emplace_back(std::make_pair(4, lamb));
                if (alpha.size() == 1) this->alpha = *alpha.begin();
                else varying_params.emplace_back(std::make_pair(5, alpha));
            }

            void set_gd_type(const Type t, const ll batch_size = 64) {
                this->t = t;
                this->batch_size = batch_size;
            }
            void set_gd_type(const std::initializer_list<double>&& t, const std::initializer_list<double>&& batch_size = {64}) {
                if (t.size() == 1) this->t = static_cast<Type>((size_t) *t.begin());
                else varying_params.emplace_back(std::make_pair(6, t));
                if (batch_size.size() == 1) this->batch_size = *batch_size.begin();
                else varying_params.emplace_back(std::make_pair(7, batch_size));
            }

            void set_params(std::vector<std::pair<size_t, double>>& grid) {
                for (auto& i : grid) {
                    switch (i.first) {
                        case 0:
                            this->eta = i.second;
                            break;
                        case 1:
                            this->eps = i.second;
                            break;
                        case 2:
                            this->iter = i.second;
                            break;
                        case 3:
                            this->r = static_cast<Regularizor>((size_t) i.second);
                            break;
                        case 4:
                            this->lamb = i.second;
                            break;
                        case 5:
                            this->alpha = i.second;
                            break;
                        case 6:
                            this->t = static_cast<Type>((size_t) i.second);
                            break;
                        case 7:
                            this->batch_size = i.second;
                    } 
                }
            }
    };
}