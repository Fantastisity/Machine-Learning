namespace MACHINE_LEARNING {
    namespace NNTREE {
        template<typename T>
        struct TreeBase {
                std::priority_queue<std::pair<double, size_t>> query(T* point, size_t n_neighbors) {
                    assert(point && n_neighbors);
                    if (!root) {
                        size_t nrow = mat.rowNum(), ind[nrow];
                        for (size_t i = 0; i < nrow; ++i) ind[i] = i;
                        root = partition(ind, nrow);
                    }
                    std::priority_queue<std::pair<double, size_t>> q;
                    query(root, point, n_neighbors, q);
                    return q;
                }
                virtual ~TreeBase() {
                    if (root) {
                        delete root;
                        root = nullptr;
                    }
                }
            protected:
                Matrix<T> mat;
                size_t leaf_size, p;
                Metric m;
                TreeBase(const Matrix<T>& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2)
                : mat(mat), leaf_size(leaf_size), m(m), p(p) {}

                TreeBase(const Matrix<T>&& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2)
                : mat(std::move(mat)), leaf_size(leaf_size), m(m), p(p) {}
                struct node {
                    node* left_child, * right_child;
                    T* split_node, * indices;
                    size_t size;
                    double radius;
                    bool is_leaf;
                    ~node() {
                        if (left_child) {
                            delete left_child;
                            left_child = nullptr;
                        }
                        if (right_child) {
                            delete right_child; 
                            right_child = nullptr;
                        }
                        if (split_node) {
                            delete[] split_node;
                            split_node = nullptr;
                        }
                        if (indices) {
                            delete[] indices;
                            indices = nullptr;
                        }
                    }
                };
                node* init_node(size_t nrow, size_t ncol) {
                    node* nd = new node;
                    nd->left_child = nd->right_child = nullptr;
                    nd->split_node = new T[ncol];
                    nd->indices = new T[nd->size = nrow];
                    nd->is_leaf = 0, nd->radius = 0;
                    return nd;
                }
                node* root = nullptr;
                virtual node* partition(size_t ind[], size_t nrow) = 0;
            private:
                // Ref. https://www.cs.cmu.edu/~agray/clsfnn.pdf
                void query(node* root, const T const * point, const size_t n_neighbors, 
                        std::priority_queue<std::pair<double, size_t>>& point_set, double dist_min = -1) {
                    if (!root) return;
                    if (dist_min == -1) dist_min = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->split_node, mat.colNum());
                    if (point_set.size() >= n_neighbors && dist_min >= point_set.top().first) return;
                    if (root->is_leaf) 
                        for (size_t i = 0, n = root->size; i < n; ++i) {
                            double cur_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, &mat(root->indices[i], 0), mat.colNum());
                            if (point_set.size() < n_neighbors || point_set.top().first > cur_dist) {
                                point_set.push(std::make_pair(cur_dist, root->indices[i]));
                                if (point_set.size() == n_neighbors + 1) point_set.pop();
                            }
                        }
                    else {
                        double left_dist  = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->left_child->split_node, mat.colNum()),
                               right_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->right_child->split_node, mat.colNum());
                        // Child closest to point -> Child furthest to point
                        if (left_dist < right_dist) {
                            query(root->left_child, point, n_neighbors, point_set, left_dist);
                            query(root->right_child, point, n_neighbors, point_set, right_dist);
                        } else {
                            query(root->right_child, point, n_neighbors, point_set, right_dist);
                            query(root->left_child, point, n_neighbors, point_set, left_dist);
                        }
                    }
                }
        };
    }
}