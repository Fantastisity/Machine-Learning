#ifndef MATRIX_UTIL_INCLUDED
#define MATRIX_UTIL_INCLUDED
#include "include/utils/matrixUtil.h"
#endif

size_t std::hash<MACHINE_LEARNING::elem>::operator()(const MACHINE_LEARNING::elem& e) const {
    return MACHINE_LEARNING::CstrFunctor()(e.sval) ^ (std::hash<double>()(e.dval) << 2);
}

namespace MACHINE_LEARNING {
    std::ostream& operator<< (std::ostream& os, const elem& e) {
        switch (e.t) {
            case evType::DBL:
                os << std::fixed << e.dval;
                break;
            case evType::STR:
                os << e.sval;
        }
        return os;
    }
}