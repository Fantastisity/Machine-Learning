#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif

enum class Metric {EUCLIDEAN, MANHATTAN, MINKOWSKI};
enum class NNAlgo {BRUTEFORCE, KDTREE, BALLTREE};

namespace MACHINE_LEARNING {
    template<typename M>
    class NNModel {
        protected:
            std::ofstream output;
            size_t n_neighbors = 5, leaf_size = 20;
            uint32_t p = 2;
            Metric m = Metric::EUCLIDEAN;
            NNAlgo algo = NNAlgo::BALLTREE;
            Matrix<double> x{0}, y{0};

            template<typename T, typename R>
            void init(T&& x, R&& y) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);
            }

            void print_params() {
                printf("**********************************************************\n");
                printf("\t\t\t\t\tParameter Settings\n");
                printf("**********************************************************\n");

                printf("n neignbors:\t\t\t\t\t\t\t\t\t\t   %2u\n", n_neighbors);
                printf("leaf size:\t\t\t\t\t\t\t\t\t\t\t   %2u\n", leaf_size);
                switch (m) { 
                    case Metric::EUCLIDEAN: 
                        printf("metric:\t\t\t\t\t\t\t\t\t\t\tEuclidean\n");
                        break;
                    case Metric::MANHATTAN:
                        printf("metric:\t\t\t\t\t\t\t\t\t\t\tManhattan\n");
                        break;
                    case Metric::MINKOWSKI:
                        printf("metric:\t\t\t\t\t\t\t\t\t\t\tMinkowski\n");
                        printf("p:\t\t\t\t\t\t\t\t\t\t\t   %u\n", p);
                        break;
                }
                switch (algo) {
                    case NNAlgo::BRUTEFORCE:
                        printf("algorithm:\t\t\t\t\t\t\t\t\t  Brute Force\n");
                        break;
                    case NNAlgo::KDTREE:
                        printf("algorithm:\t\t\t\t\t\t\t\t\t\t  KD Tree\n");
                        break;
                    case NNAlgo::BALLTREE:
                        printf("algorithm:\t\t\t\t\t\t\t\t\t\tBall Tree\n");
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