#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <QObject>
#include <QGenericMatrix>
#include <QDebug>

template <int N, int M, typename T>
class LinearRegression : public QObject
{
public:

    enum Type{
        StochasticGradientDescent,
        NormalEquations
    };

    LinearRegression(double x[],
                     double y[]) : QObject(),
        mX(x),
        mY(y)
    {
        //https://github.com/adamtahoun/NormalEquationCuda/blob/master/matrix.c
        //https://en.wikipedia.org/wiki/Invertible_matrix
    }

    virtual ~LinearRegression() {}


    QGenericMatrix<1, N, T> gradientDescent(const int& iteration, const double& learningRate){

        auto hypothesis = [](QGenericMatrix<N, M, T> x,
                             QGenericMatrix<1, N, T> w)->QGenericMatrix<1, M, T>{
            return x * w;
        };

        auto gradient = [&hypothesis](QGenericMatrix<N, M, T> x,
                                      QGenericMatrix<1, M, T> y,
                                      QGenericMatrix<1, N, T>& w,
                                      const double& a){

            for(int col = 0; col < N; col++) {
                T sum = 0;
                for(int row = 0; row < M; row++) {
                    auto delta = (hypothesis(x, w) - y) * x(row, col);
                    sum += delta(row, 0);
                }
                w(col, 0) = w(col, 0) - sum * a / M;
            }
        };

        for(int i = 0; i < iteration; i++) {
            gradient(mX, mY, mW, learningRate);
        }

        return mW;
    }

private:
    QGenericMatrix<N, M, T> mX;
    QGenericMatrix<1, M, T> mY;
    QGenericMatrix<1, N, T> mW;

    double getDeterminate(QGenericMatrix<N, M, T> m) {
        int i, j, k;
        double ratio;
        int rows = N, cols = M;
        //We don't want to change the given matrix, just
        //find it's determinate so use a temporary matrix
        //instead for the calculations
        QGenericMatrix<N, M, T> temp = m;
        //If the matrix is not square then the derminate does not exist
        if(rows == cols) {
            //If the matrix is 2x2 matrix then the
            //determinate is ([0,0]x[1,1]) - ([0,1]x[1,0])
             if(rows == 2 && cols == 2) {
                 return (temp(0, 0) * temp(1, 1)) - (temp(0, 1) * temp(1, 0));
             }
            //Otherwise if it is n*n...we do things the long way
            //we will be using a method where we convert the
            //matrix into an upper triangle, and then the det
            //will simply be the multiplication of all diagonal
            //indexes ---Idea from Khan Academy
            for(i = 0; i < rows; i++){
            for(j = 0; j < cols; j++){
                if(j>i){
                    ratio = temp(j, i)/temp(i, i);
                    for(k = 0; k < rows; k++) {
                        temp(j, k) -= ratio * temp(i, k);
                    }
                }
            }
        }
        double det = 1; //storage for determinant
        for(i = 0; i < rows; i++) {
            det *= temp(i, i);
        }
            return det;
        }
        return 0;
    }

    QGenericMatrix<N, M, T> minorOf(QGenericMatrix<N, M, T> temp) {
        QGenericMatrix<N-1, M-1, T> detMatrix;
        QGenericMatrix<N, M, T> minor;
        int rw = 0, cl = 0, numOfIncludes = 0;

        for(int c = 0; c < N; c++) {
            for(int d = 0; d < M; d++) {
                rw = 0;
                for(int i = 0; i < N; i++) {
                    cl = 0;
                    for(int j = 0; j < M; j++) {
                        if(i == c || j == d) {
                            if(numOfIncludes >= N-1) {
                                cl++;
                            }
                        } else {
                            cl++;
                            numOfIncludes++;
                            detMatrix(rw, cl - 1) = temp(i, j);
                        }
                    }
                    if(numOfIncludes >= N-1) {
                        rw++;
                        numOfIncludes = 0;
                    }
                }
                minor(c, d) = getDeterminate(detMatrix);
            }
            rw = 0;
            cl = 0;
        }
        return minor;
    }
};


#endif // LINEARREGRESSION_H
