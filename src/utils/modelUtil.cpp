#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    namespace UTIL_BASE::MODEL_UTIL {
        std::tuple<DataFrame<elem>, DataFrame<elem>, DataFrame<elem>, DataFrame<elem>>
        train_test_split(DataFrame<elem> X, DataFrame<elem> Y, const float test_size, const bool shuffle, size_t random_state) {
            size_t row = X.rowNum(), testRow = row * test_size;
            if (shuffle) X.shuffle(random_state), Y.shuffle(random_state);
            return std::make_tuple(X.iloc(rngSlicer(row - testRow),      rngSlicer(X.colNum())), 
                                   X.iloc(rngSlicer(row - testRow, row), rngSlicer(X.colNum())), 
                                   Y.iloc(rngSlicer(row - testRow),      rngSlicer(Y.colNum())), 
                                   Y.iloc(rngSlicer(row - testRow, row), rngSlicer(Y.colNum())));
        }

        std::tuple<size_t*, size_t*, size_t*, size_t> 
        k_fold(const size_t k, const size_t sample_size) {
            size_t n = static_cast<size_t>(std::ceil(sample_size / (k * 1.0))),
                * indTrain = (size_t*) malloc(sizeof(size_t) * k * n * (k - 1)),
                * indTest = (size_t*) malloc(sizeof(size_t) * k * n),
                * range = (size_t*) malloc(sizeof(size_t) * k * 2);
            if (!indTrain || !indTest || !range) {
                std::cerr << "error malloc\n"; exit(1);
            }
            for (size_t i = 0, cntTrain, cntTest; i < k; ++i) {
                cntTrain = 0, cntTest = 0;
                for (size_t j = 0; j < sample_size; ++j) {
                    if (j >= n * i && cntTest < n) {
                        indTest[i * n + cntTest++] = j;
                        continue;
                    }
                    indTrain[i * n * (k - 1) + cntTrain++] = j;
                }
                range[i << 1] = cntTrain, range[(i << 1) + 1] = cntTest;
            }
            return std::make_tuple(indTrain, indTest, range, n);
        }
    }
}