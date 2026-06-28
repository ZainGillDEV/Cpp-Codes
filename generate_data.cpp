#include <iostream>
#include <fstream>
#include <random>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <num_points> <num_true_clusters> <output_file>\n";
        return 1;
    }

    int numPoints = atoi(argv[1]);
    int numClusters = atoi(argv[2]);
    string outFile = argv[3];

    mt19937 rng(42);
    uniform_real_distribution<double> centerDist(0.0, 1000.0);
    normal_distribution<double> spread(0.0, 25.0);
    uniform_int_distribution<int> clusterPicker(0, numClusters - 1);

    vector<pair<double,double>> centers(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        centers[i] = { centerDist(rng), centerDist(rng) };
    }

    ofstream out(outFile);
    if (!out.is_open()) {
        cerr << "Error: could not open output file " << outFile << "\n";
        return 1;
    }

    out << "x,y\n";
    for (int i = 0; i < numPoints; ++i) {
        int c = clusterPicker(rng);
        double x = centers[c].first + spread(rng);
        double y = centers[c].second + spread(rng);
        out << x << "," << y << "\n";
    }

    out.close();
    cout << "Generated " << numPoints << " points around " << numClusters
         << " true clusters -> " << outFile << "\n";
    return 0;
}
