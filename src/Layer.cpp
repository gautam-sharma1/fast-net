/**
    Defines the Layer class
    @file Layer.cpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/


#include "../include/Layer.hpp"
#include "../src/Tensor.cpp"
using std::cout;
using std::endl;
using std::vector;

using namespace FastNet::Layer;
using namespace FastNet::Tensor;
using Tensorf = FastNet::Tensor::Tensor<float>;

class Tensor;
Layer::Layer(int rows, int cols) : _Rows{rows}, _Cols{cols}
{
    for (int i = 0; i < this->_Rows; i++)
    {
        vector<float> tmp;
        for (int j = 0; j < this->_Cols; j++)
        {
            tmp.push_back((float)rand() / (RAND_MAX));
        }
        FastNet::Tensor::Tensor<float> _T{tmp}; // constructs a Tensor
        _Weights.push_back(_T);                 // Pushes back the Tensor to make a single row
    }

}; // random number between 0 and 1
const void Layer::Print() const
{
    cout << "Layer( " << endl;

    for (int i = 0; i < this->_Rows; i++)
    {
        _Weights[i].Print();
    }
    cout << ")" << endl;
}

const Tensorf &Layer::operator()(const int &position) const
{
    return this->_Weights.at(position);
}

size_t Layer::GetRows() const
{
    return this->_Rows;
}

constexpr size_t Layer::GetCols() const
{
    return this->_Cols;
}

std::vector<Tensorf> Layer::GetWeights() const
{
    return this->_Weights;
}
Tensorf Layer::dot(std::vector<Tensorf> t1) // // returns a tensor

{
    int sz = t1[0].GetTensor().size();
    //cout<<sz<<endl;
    assert(sz == this->_Rows); // chech if colum of 1st matrix = rows of second matrix
    std::vector<float> tensor;
    for (int i = this->_Rows - 1; i >= 0; i--)
    {

        auto row = t1.back();
        t1.pop_back();
        row.Print();
        this->_Weights[i].Print();
        auto tmp = (this->_Weights[i]).dot(row);
        tensor.push_back(tmp);

    }
    std::reverse(tensor.begin(), tensor.end());
    tensor.resize(_Rows);
    Tensorf t{tensor};
    return t;
}
