#ifndef CLUSTERINGBASE_INCLUDED
#define CLUSTERINGBASE_INCLUDED
#include "clusteringBase.h"
#endif

enum class KMAlgo {LLYOD, HARTIGAN};

namespace MACHINE_LEARNING {
    class KMeans : public ClusteringBase {
            Matrix<double> centroids{0};

            void kmeanspp_initialization() {
                centroids = x.sample();
                for (size_t k = 1; k < n_clusters; ++k) {
                    double dist = std::numeric_limits<double>::min(); size_t ind;
                    for (size_t i = 0, nrow = x.rowNum(), ncol = x.colNum(); i < nrow; ++i) {
                        double tmp = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&x(i, 0), &centroids(k - 1, 0), ncol);
                        if (tmp > dist) dist = tmp, ind = i;
                    }
                    centroids.addRow(&x(ind, 0));
                }
            }

            void Lloyd() {
                size_t nrow = x.rowNum(), ncol = x.colNum();
                bool terminate = 0;
                for (size_t i = 0; i < iter && !terminate; ++i) {
                    for (size_t j = 0; j < nrow; ++j) {
                        double dist = std::numeric_limits<double>::max(); size_t ind;
                        for (size_t k = 0; k < n_clusters; ++k) {
                            double tmp = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&x(j, 0), &centroids(k, 0), ncol);
                            if (tmp < dist) dist = tmp, ind = k;
                        }
                        clusters[ind].push_back(j);
                    }
                    size_t converged = 0;
                    for (size_t k = 0; k < n_clusters; ++k) {
                        size_t n = clusters[k].size(), cnt = 0;
                        for (size_t c = 0; c < ncol; ++c) {
                            double avg = 0; bool f = 0;
                            for (size_t j = 0; j < n; ++j, f = 1) 
                                avg += x(clusters[k][j], c);
                            if ( f && centroids(k, c) != (avg /= n)) centroids(k, c) = avg, ++cnt;
                        }
                        if (!cnt) ++converged;
                    }
                    terminate = converged == n_clusters;
                }
            }

            void Hartigan() {
                size_t nrow = x.rowNum(), ncol = x.colNum();
                bool updated[n_clusters]; memset(updated, 0, sizeof(updated));
                for (size_t j = 0; j < nrow; ++j) {
                    double dist = std::numeric_limits<double>::max(); size_t ind;
                    for (size_t k = 0; k < n_clusters; ++k) {
                        double tmp = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&x(j, 0), &centroids(k, 0), ncol);
                        if (tmp < dist) dist = tmp, ind = k;
                    }
                    clusters[ind].push_back(j);
                }
                for (size_t k = 0; k < n_clusters; ++k) {
                    size_t n = clusters[k].size(), cnt = 0;
                    for (size_t c = 0; c < ncol; ++c) {
                        double avg = 0; bool f = 0;
                        for (size_t j = 0; j < n; ++j, f = 1) 
                            avg += x(clusters[k][j], c);
                        if (f && centroids(k, c) != (avg /= n)) centroids(k, c) = avg, ++cnt, updated[k] = 1;
                    }
                }
                auto sse = [&](size_t ind) {
                    double sum = 0;
                    for (auto& i : clusters[ind]) {
                        double norm = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&x(i, 0), &centroids(ind, 0), ncol);
                        sum += norm * norm;
                    }
                    size_t n = clusters[ind].size();
                    sum = sum * n / (n - 1);
                    return sum;
                };
                for (size_t j = 0; j < n_clusters; ++j) 
                    if (updated[j]) {
                        double cur_sse = sse(j);
                        for (auto& i : clusters[j]) 
                            for (size_t k = 0; k < n_clusters; ++k) {
                                if (k == j) continue;
                                clusters[k].push_back(i);
                                if (sse(k) >= cur_sse) clusters[k].pop_back();
                            }
                    }
            }
        public:
            template<typename T>
            void fit(T&& x, KMAlgo kmalgo = KMAlgo::LLYOD, bool kmeanspp = 0) {
                init(std::forward<T>(x));
                if (kmeanspp) kmeanspp_initialization();
                else centroids = this->x.sample(n_clusters);

                switch (kmalgo) {
                    case KMAlgo::LLYOD:
                        Lloyd();
                        break;
                    case KMAlgo::HARTIGAN:
                        Hartigan();
                }
            }

            const Matrix<double>& _centroids() {
                return centroids;
            }

            template<typename T>
            size_t* predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                size_t nrow = tmp.rowNum(), index[nrow]; memset(index, 0, sizeof(index));
                for (size_t i = 0, ncol = tmp.colNum(); i < nrow; ++i) {
                    double dist = std::numeric_limits<double>::max(); size_t ind;
                    for (size_t k = 0; k < n_clusters; ++k) {
                        double cur = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(&tmp(i, 0), &centroids(k, 0), ncol);
                        if (cur < dist) dist = cur, ind = k;
                    }
                    index[i] = ind;
                }
                return index;
            }
    };
}