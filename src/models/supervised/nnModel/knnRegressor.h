#ifndef NNMODEL_INCLUDED
#define NNMODEL_INCLUDED
#include "nnModel.h"
#endif

#ifndef BALLTREE_INCLUDED
#define BALLTREE_INCLUDED
#include "balltree.h"
#endif

namespace MACHINE_LEARNING {
    class KNNRegressor : public NNModel<KNNRegressor> {
            BallTree<double> tree{};
        public:
            KNNRegressor() = default;

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                switch (algo) {
                    case NNAlgo::BALLTREE:
                        tree = BallTree<double>(this->x, this->leaf_size, this->m, this->p);
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
                for (size_t i = 0, n = xtest.rowNum(); i < n; ++i) {
                    res = tree.query(&tmp(i, 0), this->n_neighbors), sum = 0;
                    while (!res.empty()) {
                        auto cur = res.top(); res.pop();
                        sum += this->y(cur.second, 0);
                    }
                    ypred(i, 0) = sum / res.size();
                }
                return ypred;
            }
    };
}