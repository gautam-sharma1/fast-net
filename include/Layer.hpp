/**
    Defines the Layer class
    @file Layer.hpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/



#pragma once
#include "../include/Tensor.hpp"
#include <iostream>
#include <stdlib.h> /* srand, rand */
namespace FastNet
{
namespace Layer
{
class Layer
{
public:
    /**
    Conctructs a laer of a neural network
    @param rows number of neurons in the network
    @param cols equals the shape of the previous input
    @return constructs a layer object
*/
    explicit Layer(int rows, int cols);

    /**
    overloads the () operator
    @param position index of the row you want to index
    @return Tensor at the position
*/
    const FastNet::Tensor::Tensor<float> &operator()(const int &position) const;

    /**
    Prints the layer weights to the console
    @param None
    @return None
*/
    const void Print() const;

    /**
    Implements a dot product b/w a vector of tensor and the current layer
    @param t1 vector of Tensors
    @return vector of Tensors after the dot product
*/
    std::vector<FastNet::Tensor::Tensor<float>> dot(std::vector<FastNet::Tensor::Tensor<float>> t1); // returns a tensor

    /**
    Return number of rows
    @param None
    @return gets the rows
*/
    inline size_t GetRows() const;

    /**
    Return number of columns
    @param None
    @return gets the columns
*/
    inline constexpr size_t GetCols() const;

    /**
    Return the weights
    @param None
    @return weights of the layer
*/
    inline std::vector<FastNet::Tensor::Tensor<float>> GetWeights() const;

private:
    int _Rows, _Cols;
    std::vector<FastNet::Tensor::Tensor<float>> _Weights;
};
} // namespace Layer
} // namespace FastNet