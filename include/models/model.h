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
            struct params {
                std::vector<double> eta, lamb, alpha, eps, iter, batch_size, r, t;
                auto gen_param_grid() {
                    std::vector<std::vector<double>> v;
                    if (!eta.empty()) v.push_back(eta);
                    if (!lamb.empty()) v.push_back(lamb);
                    if (!alpha.empty()) v.push_back(alpha);
                    if (!eps.empty()) v.push_back(eps);
                    if (!iter.empty()) v.push_back(iter);
                    if (!batch_size.empty()) v.push_back(batch_size);
                    if (!r.empty()) v.push_back(r);
                    if (!t.empty()) v.push_back(t);
                    return v;
                }
            };
            std::ofstream output;
            params p;
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
                std::copy(eta.begin(), eta.end(), std::back_inserter(p.eta));
            }

            void set_epsilon(const double eps) {
                this->eps = eps;
            }
            void set_epsilon(const std::initializer_list<double>&& eps) {
                std::copy(eps.begin(), eps.end(), std::back_inserter(p.eps));
            }

            void set_iteration(const ll iter) {
                this->iter = iter;
            }
            void set_iteration(const std::initializer_list<double>&& iter) {
                std::copy(iter.begin(), iter.end(), std::back_inserter(p.iter));
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
                std::copy(r.begin(), r.end(), std::back_inserter(p.r));
                std::copy(lamb.begin(), lamb.end(), std::back_inserter(p.lamb));
                std::copy(alpha.begin(), alpha.end(), std::back_inserter(p.alpha));
            }

            void set_gd_type(const Type t, const ll batch_size = 64) {
                this->t = t;
                this->batch_size = batch_size;
            }
            void set_gd_type(const std::initializer_list<double>&& t, const std::initializer_list<double>&& batch_size = {64}) {
                std::copy(t.begin(), t.end(), std::back_inserter(p.t));
                std::copy(batch_size.begin(), batch_size.end(), std::back_inserter(p.batch_size));
            }
    };
}