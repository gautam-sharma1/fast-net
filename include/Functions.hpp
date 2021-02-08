/**
    Main file that implements functions
    @file Functions.hpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/

#include "../include/Tensor.hpp"
#include <algorithm>
#include <math.h>
#include <algorithm>


template<class T> 
const T& max(const T& a, const T& b)
{
    return (a < b) ? b : a;
}


// Sigmoid function
// 1/1+e^(-x)
template <class T>
inline constexpr T _SIGMOID(T &input)
{
    return 1 / (1 + exp(-input));
}

// Relu function
// max(0,x)
template <class T>
inline constexpr T _RELU(T &input)
{
    return max((decltype(input))0, input);
}

// calculates sigmoid for each tensor input and returns a tensor
template <class T>
Tensor<T> sigmoid(const Tensor<T> &t1)
{
    std::vector<T> vec;
    for (auto &v : t1.GetTensor())
    {
        vec.push_back(_SIGMOID(v));
    }
    Tensor<T> ans{vec};
    return ans;
};

// calculates relu for each tensor input and return a tensor
template <class T>
Tensor<T> relu(const Tensor<T> &t1)
{
    std::vector<T> vec;
    for (auto &v : t1.GetTensor())
    {
        vec.push_back(_RELU(v));
    }
    Tensor<T> ans{vec};
    return ans;
}

// Simple function to calculate python style tensors
Tensor<float> Range(const int &begin = 0, const int &size=10)
{
    std::vector<float> temp;
    for (int i = begin; i < size; i++)
    {
        temp.push_back(i);
    }
    Tensor<float> ans{temp};
    return ans;
}
