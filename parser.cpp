#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "include/utils/parser.h"
#endif

namespace MACHINE_LEARNING {
    Parser::Parser(std::string&& filename, char delim) {
        std::ifstream input(filename);
        std::string line, col;
        int cnt;
        while (getline(input, line)) {
            std::stringstream ss(line);
            cnt = 0;
            while (getline(ss, col, delim)) {
                if (rows == -1) df.colNameMapping(std::move(col), cnt++);
                else df.insert(rows, cnt++, stod(col));
            }
            if (rows == -1) cols = cnt;
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