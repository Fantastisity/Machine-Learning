#ifndef NNTREEBASE_INCLUDED
#define NNTREEBASE_INCLUDED
#include "treeBase.h"
#endif

namespace MACHINE_LEARNING {
    namespace NNTREE {
        template<typename T>
        struct KDTree : TreeBase<T> {
            KDTree(const Matrix<T>& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2) 
            : TreeBase<T>(mat, leaf_size, m, p) {}
            KDTree(const Matrix<T>&& mat, const size_t leaf_size, const Metric m = Metric::EUCLIDEAN, const size_t p = 2) 
            : TreeBase<T>(std::move(mat), leaf_size, m, p) {}
            KDTree(const KDTree& tree) : TreeBase<T>(tree.mat, tree.leaf_size) {
                this->root = tree.root;
            }
            KDTree(KDTree&& tree) : TreeBase<T>(std::move(tree.mat), tree.leaf_size) {
                this->root = tree.root;
                tree.root = nullptr;
            }

            KDTree& operator= (const KDTree& tree) {
                this->mat = tree.mat;
                this->leaf_size = tree.leaf_size;
                this->root = tree.root;
                return *this;
            }
            KDTree& operator= (KDTree&& tree) {
                this->mat = std::move(tree.mat);
                this->leaf_size = tree.leaf_size;
                this->root = tree.root, tree.root = nullptr;
                return *this;
            }

            private:
                // Ref. https://arxiv.org/pdf/1903.04936.pdf
                typename TreeBase<T>::node* partition(size_t ind[], size_t nrow) {
                    size_t ncol = this->mat.colNum();
                    typename TreeBase<T>::node* root = this->init_node(nrow, ncol);
                    // Single node case
                    if (nrow == 1) { 
                        root->is_leaf = 1, root->indices[0] = ind[0], root->size = 1;
                        std::copy(&this->mat(ind[0], 0), &this->mat(ind[0], 0) + ncol, root->split_node);
                        return root;
                    }
                    // Store indices for current subtree
                    std::copy(ind, ind + nrow, root->indices);
                    root->size = nrow;
                    // Determine the feature having largest spread
                    double val_min, val_max, spread = std::numeric_limits<double>::min();
                    size_t feature_index;
                    for (size_t j = 0; j < ncol; ++j) {
                        val_min = std::numeric_limits<double>::max(), val_max = std::numeric_limits<double>::min();
                        for (size_t i = 0; i < nrow; ++i) {
                            if (this->mat(ind[i], j) > val_max) val_max = this->mat(ind[i], j);
                            if (this->mat(ind[i], j) < val_min) val_min = this->mat(ind[i], j);
                        }
                        if (val_max - val_min < spread) spread = val_max - val_min, feature_index = j;
                    }
                    // Determine median of the selected feature
                    std::vector<std::pair<T, size_t>> tmp;
                    for (size_t i = 0; i < nrow; ++i) tmp.push_back(std::make_pair(this->mat(ind[i], feature_index), i));
                    sort(tmp.begin(), tmp.end());
                    size_t mid = nrow >> 1;
                    std::copy(&this->mat(ind[mid], 0), &this->mat(ind[mid], 0) + ncol, root->split_node);

                    if (nrow <= this->leaf_size) { // Leaf nodes; no further splits are needed
                        root->is_leaf = 1;
                        return root;
                    }

                    size_t left_child_indices[mid], right_child_indices[nrow - mid - 1];
                    for (size_t i = 0; i < nrow - mid - 1; ++i) left_child_indices[i] = tmp[i].second, right_child_indices[i] = tmp[i + mid + 1].second;
                    for (size_t i = nrow - mid - 1; i < mid; ++i) left_child_indices[i] = tmp[i].second;
                    root->left_child = partition(left_child_indices, mid), root->right_child = partition(right_child_indices, nrow - mid - 1);
                    
                    return root;
                }
        };
    }
}