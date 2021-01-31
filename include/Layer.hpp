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
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    explicit Layer(int rows, int cols);

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    const FastNet::Tensor::Tensor<float> &operator()(const int &position) const;

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    const void Print() const;

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    FastNet::Tensor::Tensor<float> dot(std::vector<FastNet::Tensor::Tensor<float>> t1); // returns a tensor

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    inline size_t GetRows() const;

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    inline constexpr size_t GetCols() const;

    /**
    Encodes a single digit of a POSTNET "A" bar code.
    @param digit the single digit to encode.
    @return a bar code of the digit using "|" as the long bar
    and "," as the half bar.
*/
    inline std::vector<FastNet::Tensor::Tensor<float>> GetWeights() const;

private:
    int _Rows, _Cols;
    std::vector<FastNet::Tensor::Tensor<float>> _Weights;
};
} // namespace Layer
} // namespace FastNet