#ifndef CLUSTERINGBASE_INCLUDED
#define CLUSTERINGBASE_INCLUDED
#include "clusteringBase.h"
#endif

enum class KMAlgo {LLYOD, }

namespace MACHINE_LEARNING {
    class KMeans : ClusteringBase {
            Matrix<double> centroids{0};
            std::vector<std::vector<size_t>> clusters(n_clusters);

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
                            double avg = 0;
                            for (size_t j = 0; j < n; ++j) 
                                avg += x(clusters[k][j], c);
                            avg /= n;
                            if (centroids(k, c) != avg) centroids(k, c) = avg, ++cnt;
                        }
                        if (!cnt) ++converged;
                    }
                    terminate = converged == n_clusters;
                }
            }
        public:
            template<typename T>
            void fit(T&& x) {
                init(std::forward<T>(x));
                centroids = this->x.sample(n_clusters);
                
            }
    };
}