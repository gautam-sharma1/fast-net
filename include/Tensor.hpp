/**
    Declares the Tensor class
    @file Tensor.hpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/

#pragma once
#include <cstdlib>
#include <vector>
#include <iostream>
#include <numeric>

namespace FastNet
{
namespace Tensor
{
template <class T>
class Tensor
{
public:
    /**
    Constructor for Tensor classs 
    @param input takes in a vector to instantiate
    @return Tensor
*/
    Tensor(std::vector<T> input);

    /**
    Default constructor
    @param None 
    @return Tensor
*/
    Tensor() = default;

    /**
    Prints out the Tensor
    @param None
    @return None --> prints out the vector
*/
    inline const void Print() const;

    /**
    Overloads the () operator
    @param position index of the value to access
    @return value of the index
*/
    inline const T operator()(const int &position) const;

    /**
    Overloads the + operator
    @param t1 Tensor to add to the current tensor
    @return Modified tensor 
*/
    inline void operator+(const Tensor &t1);

    /**
    Overloads the - operator
    @param t1 Tensor to subtract from the current tensor
    @return Modified tensor 
*/
    inline void operator-(const Tensor &t1);

    /**
    Overloads the * operator
    @param gain multiplies each entry of the Tesnor with gain
    @return Modified tensor 
*/
    inline void operator*(const T &gain); // element wise

    /**
    To get the current Tensor
    @param None
    @return current Tensor object
*/
    inline const std::vector<T> GetTensor() const;

    /**
    Copy Conctructor
    @param t1 Tensor to copy 
    @return Constructs new tensor from t1
*/
    Tensor(const Tensor &t1); //copy constructor

    /**
    Assignment Constructor
    @param rhs Assigns right hand side tensor to current tensor
    @return Modified tensor
*/
    Tensor &operator=(const Tensor &rhs); // assignment constructor

    /**
    Performs dot product
    @param t1 tensor to perform dot product with
    @return dot product
*/
    inline T dot(const Tensor &t1); // inplace

    /**
    Multiplies two tensors
    @param t1 tensor to multiply with
    @return new tensor object
*/
    inline Tensor multiply(const Tensor &t1); // gives new object

    /**
    To get the size of the tensor
    @param None
    @return number of elements in the tensor
*/
    inline constexpr size_t GetSize() const;

    /**
    Sums up the tensor
    @param None
    @return sum of the tensor
*/

    inline T Sum();

    std::size_t SIZE;

private:
    std::vector<T> _Tensor;
    std::vector<std::vector<float>> _Matrix;
};
} // namespace Tensor
} // namespace FastNet