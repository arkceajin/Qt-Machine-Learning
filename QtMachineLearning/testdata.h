#ifndef TESTDATA_H
#define TESTDATA_H

#include <QDebug>
#include <utility>

/**
 * Generate standard Normal distribution test data
 */
namespace TestData {
    using namespace std;

    /**
     * @brief generateLinearTestSet
     * @param num
     * @param minX
     * @param maxX
     * @param w         Real weight
     * @param variance
     * @param mean
     * @return
     */
    pair<vector<double>, vector<double>>
    generateLinearTestSet(const int& num,
                          const double& minX,
                          const double& maxX,
                          const double& w,
                          const double& variance,
                          const double& mean = 0){
        vector<double> vecX;
        vector<double> vecY;

        mt19937 engine;
        uniform_real_distribution<double> dist(minX, maxX);
        normal_distribution<double> normalDist(mean, variance);

        auto hypothesis = [w](double x)->double{
            return x * w;
        };

        for(int i = 0; i < num; i++) {
            double x = dist(engine);
            vecX.push_back(x);
            double y = hypothesis(x) + normalDist(engine);
            vecY.push_back(y);
        }

        return make_pair(vecX, vecY);
    }
}
#endif // TESTDATA_H
