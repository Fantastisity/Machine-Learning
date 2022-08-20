## lightweight c++ implementation of common ML algorithms
### Build
```bash
cd {path to current directory}
g++ -w $(find . -type f -iregex '.*\.cpp') -o model -std=c++17 -pthreads -O2
```
