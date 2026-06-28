#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
using namespace std;

struct Point {
    double x, y;
    int cluster = -1;
};

vector<Point> loadDataset(const string& filename) {
    vector<Point> points;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open dataset file " << filename << "\n";
        exit(1);
    }
    string line;
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        string xs, ys;
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        if (xs.empty() || ys.empty()) continue;
        Point p;
        p.x = stod(xs);
        p.y = stod(ys);
        points.push_back(p);
    }
    return points;
}

vector<Point> initializeCentroids(const vector<Point>& data, int k, unsigned seed = 123) {
    vector<Point> centroids;
    mt19937 rng(seed);
    uniform_int_distribution<size_t> dist(0, data.size() - 1);
    vector<size_t> chosenIdx;
    while (centroids.size() < static_cast<size_t>(k)) {
        size_t idx = dist(rng);
        bool dup = false;
        for (size_t c : chosenIdx) if (c == idx) { dup = true; break; }
        if (dup) continue;
        chosenIdx.push_back(idx);
        Point c = data[idx];
        c.cluster = static_cast<int>(centroids.size());
        centroids.push_back(c);
    }
    return centroids;
}

inline double squaredDistance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

bool assignClustersParallel(vector<Point>& data, const vector<Point>& centroids, int numThreads) {
    int changedCount = 0;
    int k = static_cast<int>(centroids.size());
    long long n = static_cast<long long>(data.size());

    #pragma omp parallel for num_threads(numThreads) schedule(static) reduction(+:changedCount)
    for (long long i = 0; i < n; ++i) {
        double bestDist = numeric_limits<double>::max();
        int bestCluster = -1;
        for (int c = 0; c < k; ++c) {
            double d = squaredDistance(data[i], centroids[c]);
            if (d < bestDist) {
                bestDist = d;
                bestCluster = c;
            }
        }
        if (data[i].cluster != bestCluster) {
            data[i].cluster = bestCluster;
            changedCount += 1;
        }
    }
    return changedCount > 0;
}

void updateCentroidsParallel(const vector<Point>& data, vector<Point>& centroids, int numThreads) {
    int k = static_cast<int>(centroids.size());
    long long n = static_cast<long long>(data.size());

    vector<vector<double>> localSumX(numThreads, vector<double>(k, 0.0));
    vector<vector<double>> localSumY(numThreads, vector<double>(k, 0.0));
    vector<vector<long long>> localCount(numThreads, vector<long long>(k, 0));

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            int c = data[i].cluster;
            localSumX[tid][c] += data[i].x;
            localSumY[tid][c] += data[i].y;
            localCount[tid][c] += 1;
        }
    }

    vector<double> sumX(k, 0.0), sumY(k, 0.0);
    vector<long long> count(k, 0);
    for (int t = 0; t < numThreads; ++t) {
        for (int c = 0; c < k; ++c) {
            sumX[c] += localSumX[t][c];
            sumY[c] += localSumY[t][c];
            count[c] += localCount[t][c];
        }
    }

    for (int c = 0; c < k; ++c) {
        if (count[c] > 0) {
            centroids[c].x = sumX[c] / count[c];
            centroids[c].y = sumY[c] / count[c];
        }
    }
}

double computeInertia(const vector<Point>& data, const vector<Point>& centroids) {
    double total = 0.0;
    for (const auto& p : data) total += squaredDistance(p, centroids[p.cluster]);
    return total;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0]
             << " <input_csv> <k> <max_iters> <num_threads> <output_prefix>\n";
        return 1;
    }

    string inputFile = argv[1];
    int k = atoi(argv[2]);
    int maxIters = atoi(argv[3]);
    int numThreads = atoi(argv[4]);
    string outPrefix = argv[5];

    vector<Point> data = loadDataset(inputFile);
    if (data.empty()) {
        cerr << "Error: dataset is empty or could not be parsed.\n";
        return 1;
    }
    vector<Point> centroids = initializeCentroids(data, k);

    cout << "[Parallel] Loaded " << data.size() << " points, K=" << k
         << ", threads=" << numThreads << "\n";

    auto start = chrono::high_resolution_clock::now();

    int iter = 0;
    bool changed = true;
    while (changed && iter < maxIters) {
        changed = assignClustersParallel(data, centroids, numThreads);
        updateCentroidsParallel(data, centroids, numThreads);
        iter++;
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsedMs = chrono::duration<double, milli>(end - start).count();

    double inertia = computeInertia(data, centroids);

    cout << "[Parallel] Converged after " << iter << " iterations\n";
    cout << "[Parallel] Execution time: " << elapsedMs << " ms\n";
    cout << "[Parallel] Final inertia (SSE): " << inertia << "\n";

    ofstream assignOut(outPrefix + "_assignments.csv");
    assignOut << "x,y,cluster\n";
    for (const auto& p : data) assignOut << p.x << "," << p.y << "," << p.cluster << "\n";
    assignOut.close();

    ofstream centroidOut(outPrefix + "_centroids.csv");
    centroidOut << "x,y\n";
    for (const auto& c : centroids) centroidOut << c.x << "," << c.y << "\n";
    centroidOut.close();

    ofstream timingOut(outPrefix + "_timing.csv", ios::app);
    timingOut << "parallel," << numThreads << "," << data.size() << "," << k << ","
              << iter << "," << elapsedMs << "," << inertia << "\n";
    timingOut.close();

    return 0;
}
