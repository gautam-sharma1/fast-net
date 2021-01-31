/**
    Main file that implements functions
    @file Functions.hpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/



#include "../include/Tensor.hpp"
#include <algorithm>
#include <math.h>

// Sigmoid fucntion
// 1/1+e^(-x)
template <class T>
inline constexpr T _sigmoid(T &input)
{
    return 1 / (1 + exp(-input));
}

// calculates sigmoid for each tensor input and returns a tensor
template <class T>
Tensor<T> sigmoid(Tensor<T> &t1)
{
    std::vector<T> vec;
    for (auto &v : t1.GetTensor())
    {
        vec.push_back(_sigmoid(v));
    }
    Tensor<T> ans{vec};
    return ans;
};

// Simple function to calculate python style tensors
Tensor<float> ForInRange(const int &size)
{
    std::vector<float> temp;
    for (int i = 0; i < size; i++)
    {
        temp.push_back(i);
    }
    Tensor<float> ans{temp};
    return ans;
}
