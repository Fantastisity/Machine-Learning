#ifndef NNMODEL_INCLUDED
#define NNMODEL_INCLUDED
#include "nnModel.h"
#endif

#ifndef BALLTREE_INCLUDED
#define BALLTREE_INCLUDED
#include "balltree.h"
#endif

#ifndef KDTREE_INCLUDED
#define KDTREE_INCLUDED
#include "kdTree.h"
#endif

namespace MACHINE_LEARNING {
    class KNNRegressor : public NNModel<KNNRegressor> {
            NNTREE::TreeBase<double>* tree{};
        public:
            KNNRegressor() = default;

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                switch (algo) {
                    case NNAlgo::BALLTREE:
                        tree = new NNTREE::BallTree<double>(this->x, this->leaf_size, this->m, this->p);
                        break;
                    case NNAlgo::KDTREE:
                        tree = new NNTREE::KDTree<double>(this->x, this->leaf_size);
                }
                if (verbose == 2) print_params();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> ypred(xtest.rowNum(), 1), tmp{0};

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                else tmp = std::forward<T>(xtest);

                std::priority_queue<std::pair<double, size_t>> res;
                double sum;

                for (size_t i = 0, n = tmp.rowNum(); i < n; ++i) {
                    sum = 0;
                    switch (algo) {
                        case NNAlgo::BRUTEFORCE: {
                            for (size_t j = 0, m = this->x.rowNum(), ncol = this->x.colNum(); j < m; ++j) {
                                switch (this->m) { 
                                    case Metric::EUCLIDEAN:
                                        res.push(std::make_pair(UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&tmp(i, 0), &this->x(j, 0), ncol), j));
                                        break;
                                    case Metric::MANHATTAN:
                                        res.push(std::make_pair(UTIL_BASE::MODEL_UTIL::METRICS::manhattan(&tmp(i, 0), &this->x(j, 0), ncol), j));
                                        break;
                                    case Metric::MINKOWSKI:
                                        res.push(std::make_pair(UTIL_BASE::MODEL_UTIL::METRICS::minkowski(&tmp(i, 0), &this->x(j, 0), ncol, this->p), j));
                                        break;
                                }
                                if (res.size() > this->n_neighbors) res.pop();
                            }
                            break;
                        }
                        default: res = tree->query(&tmp(i, 0), this->n_neighbors);
                    }
                    while (!res.empty()) {
                        sum += this->y(res.top().second, 0); res.pop();
                    }
                    ypred(i, 0) = sum / n_neighbors;
                }              
                return ypred;
            }
    };
}