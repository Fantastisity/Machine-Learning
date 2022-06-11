#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "parser.h"
#endif

namespace MACHINE_LEARNING {
    Parser::Parser(std::string&& filename, char delim, bool ignore_header) {
        std::ifstream input(filename);
        std::string line, col;
        int cnt = 0;
        if (!ignore_header) {
            getline(input, line);
            std::stringstream ss(line);
            while (getline(ss, col, delim)) df.colNameMapping(col.c_str(), cnt++);
            df.init_ind2name(cols = cnt);
        }
        while (getline(input, line)) {
            std::stringstream ss(line);
            cnt = 0;
            while (getline(ss, col, delim)) df.insert(rows, cnt++, col.c_str());
            ++rows;
        }
    }

    void Parser::head(size_t num) {
        assert(rows && cols);
        num = std::min(num, static_cast<size_t>(rows));
        for (ll r = 0; r < num; ++r) {
            for (ll c = 0; c < cols; ++c) 
                std::cout << df(r, c) << " ";
            std::cout << '\n';
        }
    }
}