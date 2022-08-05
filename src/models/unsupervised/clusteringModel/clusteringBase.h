#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    class ClusteringBase {
        protected:
            Matirx<double> x{0};
            uint8_t n_clusters = 5;
            uint64_t iter = 500;
            float tol = 0.0001;

            template<typename T>
            void init(T&& x) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);
            }
        public:
            void set_n_clusters(const uint8_t n_clusters) {
                this->n_clusters = n_clusters;
            }

            void set_iter(const uint32_t iter) {
                this->iter = iter;
            }

            void set_tol(const float tol) {
                this->tol = tol;
            }
    };
}