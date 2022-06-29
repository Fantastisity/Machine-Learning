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
    class KNNClassifer : public NNModel<KNNClassifer> {
            NNTREE::TreeBase<double>* tree = nullptr;
        public:
            KNNClassifer() = default;
            ~KNNClassifer() {
                if (tree) {
                    delete tree;
                    tree = nullptr;
                }
            }
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
                tree->init_root();
                if (verbose == 2) print_params();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> ypred(xtest.rowNum(), 1), tmp{0};

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();

                auto batch_predict = [&](size_t from, size_t to) {
                    std::priority_queue<std::pair<double, size_t>> res;
                    std::unordered_map<double, size_t> counter;
                    size_t cnt; double label;

                    for (size_t i = 0, n = tmp.rowNum(); i < n; ++i) {
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
                        cnt = 0;
                        while (!res.empty()) {
                            auto cur = res.top(); res.pop();
                            // Majority voting
                            if (++counter[this->y(cur.second, 0)] > cnt) cnt = counter[label = this->y(cur.second, 0)];
                        }
                        ypred(i, 0) = label;
                        counter.clear();
                    }
                };

                const size_t pcnt = std::thread::hardware_concurrency();
                std::thread threads[pcnt - 1];
                size_t n = xtest.rowNum(), batch = n / pcnt;
                batch_predict(0, batch);
                for (size_t i = 0, cnt = batch; i < pcnt - 1; ++i, cnt += batch) {
                    threads[i] = std::thread(batch_predict, cnt, cnt + batch < n ? cnt + batch : n);
                    threads[i].join();
                }
                return ypred;
            }
    };
}