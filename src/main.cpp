/**
    Main file to execute the library
    @file main.cpp
    @author Gautam Sharma
    @version 1.1 01/31/2021
*/


#include "../include/Tensor.hpp"
#include "../src/Tensor.cpp"
#include "../include/Functions.hpp"
#include "../include/Layer.hpp"

using Vectorf = std::vector<float>;
using Vectori = std::vector<int>;
using Tensorf = Tensor<float>;
using Tensori = Tensor<int>;
using namespace FastNet::Tensor;
using namespace FastNet::Layer;

int main()
{
    Vectorf input = {1.5, 2, 3, 4};
    Tensorf t{input};
    t.Print();

    Vectorf weights = {1, 2, 3, 4};
    Tensorf t2{weights};
    t.Print();

    auto dot = t.dot(t2); //dot product
    cout << dot << endl;
    t.Print();

    auto z = sigmoid(t);
    z.Print();

    auto x = ForInRange(100);
    x.Print();

    Tensor<float> y = sigmoid(x);
    auto y2 = sigmoid(y);
    auto y3 = sigmoid(y2);
    auto y4 = sigmoid(y3);
    y4.Print();
    cout << y4.Sum() << endl;

    Vectorf input1 = {1, 2, 3, 4};
    Vectorf input2 = {1, 2, 3, 4};
    Vectorf input3 = {1, 2, 3, 4};
    Vectorf input4 = {0, 0, 0, 0};
    vector<Tensorf> t1{input1, input2, input3, input4};
    Layer l2{4, 4};

    l2.Print();
    auto tt = l2.dot(t1);
    cout << "Final answer is:" << endl;
    tt.Print();

}