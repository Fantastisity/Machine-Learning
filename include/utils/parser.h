#ifndef DATA_FRAME_INCLUDED
#define DATA_FRAME_INCLUDED
#include "../tabular-data/dataFrame.h"
#endif
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <sstream>

namespace MACHINE_LEARNING {
    class Parser {
        DataFrame<elem> df;
        ll cols = 0, rows = 0;
        public:
            Parser(std::string&& filename, char delim = ',', bool ignore_header = 0);

            auto getX(size_t startCol, size_t range) {
                return df.iloc(rngSlicer(rows), rngSlicer(startCol, startCol + range), 1);
            }

            auto getY(size_t tarCol) {
                return df.iloc(rngSlicer(rows), rngSlicer(tarCol, tarCol + 1), 1);
            }

            void head(size_t num = 5);
    };
}