#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    namespace TREES {
        template<typename T>
        struct BallTree {
            BallTree(const Matrix<T>& mat, const size_t leaf_size, const char* metric = "euclid", size_t p = 2) 
            : mat(mat), leaf_size(leaf_size), p(p) {
                strcpy(this->metric, metric);
                size_t nrow = mat.rowNum(), ind[nrow];
                for (size_t i = 0; i < nrow; ++i) ind[i] = i;
                root = partition(ind, nrow);
            }
            BallTree(const Matrix<T>&& mat, const size_t leaf_size, const char* metric = "euclid", size_t p = 2) 
            : mat(mat), leaf_size(leaf_size), p(p) {
                strcpy(this->metric, metric);
                size_t nrow = mat.rowNum(), ind[nrow];
                for (size_t i = 0; i < nrow; ++i) ind[i] = i;
                root = partition(ind, nrow);
            }
            BallTree(const BallTree&) = delete;
            BallTree(BallTree&&) = delete;
            BallTree& operator= (const BallTree&) = delete;
            BallTree& operator= (BallTree&&) = delete;
            ~BallTree() {
                if (root) {
                    free(root);
                    root = nullptr;
                }
            }

            std::priority_queue<std::pair<double, size_t>> 
            query(T* point, size_t n_neighbors) {
                std::priority_queue<std::pair<double, size_t>> q;
                query(root, point, n_neighbors, q);
                return q;
            }

            private:
                Matrix<T> mat;
                size_t leaf_size, p;
                char metric[7];
                struct node {
                    node* left_child, * right_child;
                    T* centroid, * indices;
                    double radius;
                    size_t size;
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
                        if (indices) {
                            delete[] indices;
                            indices = nullptr;
                        }
                        if (centroid) {
                            delete[] centroid;
                            centroid = nullptr;
                        }
                    }
                };
                node* root;
                node* init_node(size_t ncol, size_t nrow) {
                    node* tmp = new node;
                    tmp->left_child = tmp->right_child = nullptr;
                    tmp->indices = new T[tmp->size = nrow];
                    tmp->centroid = new T[ncol]{};
                    tmp->radius = 0, tmp->is_leaf = 0;
                    return tmp;
                }
                
                // Ref. https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html
                node* partition(size_t ind[], size_t nrow) {
                    size_t ncol = mat.colNum();
                    node* root = init_node(ncol, nrow);
                    // Find centroid
                    for (size_t i = 0; i < nrow; ++i) 
                        for (size_t j = 0; j < ncol; ++j) {
                            root->centroid[j] += root->centroid[j];
                            if (i == nrow - 1) root->centroid[j] /= nrow;
                        }
                    // Single node case
                    if (nrow == 1) { 
                        root->radius = 0, root->is_leaf = 1, root->indices[0] = ind[0], root->size = 1;
                        return root;
                    }
                    // Store indices for current subtree
                    std::copy(ind, ind + nrow, root->indices);
                    // Locate the first node that is the furthest from centroid
                    size_t first_node_ind;
                    for (size_t i = 0; i < nrow; ++i) {
                        double dist;
                        if (!strcmp(metric, "euclid")) dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(mat(ind[i], 0), root->centroid, ncol);
                        else if (!strcmp(metric, "manhat")) dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(mat(ind[i], 0), root->centroid, ncol);
                        else dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(mat(ind[i], 0), root->centroid, ncol, p);

                        if (dist > root->radius) root->radius = dist, first_node_ind = ind[i];
                    }
                    if (nrow <= leaf_size) { // Leaf nodes; no further splits are needed
                        root->is_leaf = 1;
                        return root;
                    }
                    // Locate the second node that is the furthest from first node
                    size_t second_node_ind;
                    double max_dist = 0;
                    for (size_t i = 0; i < nrow; ++i) {
                        double dist;
                        if (!strcmp(metric, "euclid")) dist = UTIL_BASE::MODEL_UTIL::METRICS::euclidean(mat(ind[i], 0), mat(first_node_ind, 0), ncol);
                        else if (!strcmp(metric, "manhat")) dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(mat(ind[i], 0), mat(first_node_ind, 0), ncol);
                        else dist = UTIL_BASE::MODEL_UTIL::METRICS::minkowski(mat(ind[i], 0), mat(first_node_ind, 0), ncol, p);

                        if (dist > max_dist) max_dist = dist, second_node_ind = ind[i];
                    }

                    // Project data on to (first_node - second_node)
                    T diff[ncol]; memset(diff, 0, sizeof(diff));
                    std::vector<std::pair<T, size_t>> Z(nrow, std::make_pair(0, 0));
                    for (size_t i = 0; i < ncol; ++i) diff[i] = mat(first_node_ind, i) - mat(second_node_ind, i);
                    for (size_t i = 0; i < nrow; ++i) {
                        for (size_t j = 0; j < ncol; ++i) {
                            Z[i].first += diff[j] * mat(ind[i], j);
                        }
                    }

                    // Construct left and right child indices
                    sort(Z.begin(), Z.end());
                    size_t mid = nrow >> 1;
                    size_t left_child_indices[mid], right_child_indices[nrow - mid];
                    for (size_t i = 0; i < mid; ++i) 
                        left_child_indices[i] = Z[i].second, right_child_indices[i + mid] = Z[i + mid].second;

                    root->left = partition(left_child_indices, mid);
                    root->right = partition(right_child_indices, nrow - mid);

                    return root;
                }

                // Ref. https://www.cs.cmu.edu/~agray/clsfnn.pdf
                void query(node* root, const T const * point, const size_t n_neighbors, 
                            std::priority_queue<std::pair<double, size_t>>& point_set, double dist_min = -1) {
                    if (dist_min == -1) dist_min = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->centroid, mat.colNum());
                    if (!point_set.empty() && dist_min >= point_set.top().first) return;
                    if (root->is_leaf) 
                        for (size_t i = 0, n = root->size; i < n; ++i) {
                            double cur_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, mat(root->indices[i], 0), mat.colNum());
                            if (point_set.top().first > cur_dist) {
                                if (point_set.size() == n_neighbors) point_set.pop();
                                point_set.push(std::make_pair(cur_dist, root->indices[i]));
                            }
                        }
                    else {
                        double left_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->left->centroid, mat.colNum()),
                                right_dist = UTIL_BASE::MODEL_UTIL::METRICS::manhattan(point, root->right->centroid, mat.colNum());
                        // Child closest to point -> Child furthest to point
                        if (left_dist < right_dist) {
                            query(root->left, point, n_neighbors, point_set, left_dist);
                            query(root->right, point, n_neighbors, point_set, right_dist);
                        } else {
                            query(root->right, point, n_neighbors, point_set, right_dist);
                            query(root->left, point, n_neighbors, point_set, left_dist);
                        }
                    }
                }
        };
    }
    class KNNRegressor {

    };

    class KNNClassifier {

    };
}