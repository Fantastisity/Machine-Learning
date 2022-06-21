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
        public:
            KNNRegressor();

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init_matrices(std::forward<T>(x), std::forward<R>(y));
                switch (algo) {
                    case NNAlgo::BALLTREE:
                        BallTree<T> tree(this->x, this->leaf_size, this->m, this->p);
                        
                }
                if (verbose == 2) print_params();
            }
    };

    class KNNClassifier : public NNModel<KNNClassifier> {

    };
}