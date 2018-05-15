#include "linearregression.h"
#include "testdata.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Create test train set by normal distribution
    auto test = TestData::generateLinearTestSet(20, 0, 1, 2, 0.2);

    // Init linear regression model
    LinearRegression<1, 20, double> reg(test.first.data(), test.second.data());

    // Using gradient descent start training
    qDebug()<<reg.gradientDescent(1000, 0.1);

    return a.exec();
}
