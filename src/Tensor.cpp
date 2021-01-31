/**
    Defines the Tensor class
    @file Tensor.cpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/

#include "../include/Tensor.hpp"

using std::cout;
using std::endl;
using std::size_t;
using std::vector;
using namespace FastNet::Tensor;

template <class T>
Tensor<T>::Tensor(std::vector<T> input) : _Tensor{input}
{
    this->SIZE = input.size();
};

// prints with nice formatting
template <class T>
const void Tensor<T>::Print() const
{
    cout << "Tensor( ";
    for (auto &t : _Tensor)
    {
        cout << t << " ";
    }
    cout << ")" << endl;
}

// overloads call
template <class T>
const T Tensor<T>::operator()(const int &position) const
{
    return this->_Tensor.at(position);
}

// overloads addition
template <class T>
void Tensor<T>::operator+(const Tensor &t1)
{
    assert(this->SIZE == t1.SIZE);
    for (int i = 0; i < this->SIZE; i++)
    {
        this->_Tensor.at(i) += t1.GetTensor().at(i);
    }
}

// overloads subtraction
template <class T>
void Tensor<T>::operator-(const Tensor &t1)
{
    assert(this->SIZE == t1.SIZE);
    for (int i = 0; i < this->SIZE; i++)
    {
        this->_Tensor.at(i) -= t1.GetTensor().at(i);
    }
}

// overloads element multiplication
template <class T>
void Tensor<T>::operator*(const T &gain)
{

    for (int i = 0; i < this->SIZE; i++)
    {
        this->_Tensor.at(i) *= gain;
    }
}

// Copy constructor
template <class T>
Tensor<T>::Tensor(const Tensor &t1)
{
    this->_Tensor = t1.GetTensor();
    this->SIZE = t1.SIZE;
}

// Assignment constructor
template <class T>
Tensor<T> &Tensor<T>::operator=(const Tensor &rhs)
{
    this->_Tensor = rhs.GetTensor();
    this->SIZE = rhs.SIZE;
    return *this;
}

// returns the current tensor
template <class T>
const vector<T> Tensor<T>::GetTensor() const
{
    return this->_Tensor;
};

template <class T>
inline constexpr size_t Tensor<T>::GetSize() const
{
    return this->SIZE;
}

////////////////////////////////////////////////////////////////////////////////
// Functions

// Inplace multiplication
// multiplies two tensors
template <class T>
T Tensor<T>::dot(const Tensor &t1)
{
    //cout<<this->SIZE << " "<<  t1.SIZE<<endl;
    assert(this->SIZE == t1.SIZE);
    return std::inner_product(std::begin(this->_Tensor), std::end(this->_Tensor), std::begin(t1.GetTensor()), (decltype(t1(1)))0.0);
}

// Sum
struct _Sum
{
    void operator()(float n) { sum += n; }
    float sum{0};
};

template <class T>
T Tensor<T>::Sum()
{
    _Sum s = std::for_each(this->_Tensor.begin(), this->_Tensor.end(), _Sum());
    return s.sum;
}
