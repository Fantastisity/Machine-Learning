#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif
// #define WRITE_TO_FILE

enum class Metric {EUCLIDEAN, MANHATTAN, MINKOWSKI};
enum class NNAlgo {BALLTREE};

namespace MACHINE_LEARNING {
    template<typename M>
    class NNModel {
        protected:
            std::ofstream output;
            size_t n_neighbors = 5, leaf_size = 20;
            uint32_t p = 2;
            Metric m;
            NNAlgo algo;

            void print_params() {
                printf("**********************************************************\n");
                printf("\t\t\t\t\tParameter Settings\n");
                printf("**********************************************************\n");

                printf("n neignbors:\t\t\t\t\t\t\t\t\t\t\t   %u\n", n_neighbors);
                printf("leaf size:\t\t\t\t\t\t\t\t\t\t\t   %u\n", leaf_size);
                switch (m) { 
                    case Metric::EUCLIDEAN: 
                        printf("metric:\t\t\t\t\t\t\t\t\t\t Euclidean\n");
                        break;
                    case Metric::MANHATTAN:
                        printf("metric:\t\t\t\t\t\t\t\t\t\t Manhattan\n");
                        break;
                    case Metric::MINKOWSKI:
                        printf("metric:\t\t\t\t\t\t\t\t\t\t Minkowski\n");
                        printf("p:\t\t\t\t\t\t\t\t\t\t\t   %u\n", p);
                        break;
                }
                switch (algo) {
                    case NNAlgo::BALLTREE:
                        printf("algorithm:\t\t\t\t\t\t\t\t\t   Ball Tree\n");
                        break;
                };
            }
        public:
            void set_n_neighbors(const size_t n_neighbors) {
                this->n_neighbors = n_neighbors;
            }

            void set_leaf_size(const size_t leaf_size) {
                this->leaf_size = leaf_size;
            }

            void set_metric(const Metric m, const size_t p = 2) {
                this->m = m, this->p = p;
            }

            void set_algo(const NNAlgo algo) {
                this->algo = algo;
            }

            void set_params(std::vector<std::pair<char const *, elem>>& grid) {
                for (auto& i : grid) {
                    if      (!strcmp(i.first, "n_neighbors")) this->n_neighbors = i.second;
                    else if (!strcmp(i.first, "leaf_size")) this->leaf_size = i.second;
                    else if (!strcmp(i.first, "metric")) this->m = static_cast<Metric>((size_t) i.second);
                    else if (!strcmp(i.first, "p")) this->p = i.second;
                    else if (!strcmp(i.first, "algorithm")) this->algo = static_cast<NNAlgo>((size_t) i.second);
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