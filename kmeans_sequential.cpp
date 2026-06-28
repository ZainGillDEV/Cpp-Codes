#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
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

bool assignClusters(vector<Point>& data, const vector<Point>& centroids) {
    bool changed = false;
    for (auto& p : data) {
        double bestDist = numeric_limits<double>::max();
        int bestCluster = -1;
        for (size_t c = 0; c < centroids.size(); ++c) {
            double d = squaredDistance(p, centroids[c]);
            if (d < bestDist) {
                bestDist = d;
                bestCluster = static_cast<int>(c);
            }
        }
        if (p.cluster != bestCluster) {
            p.cluster = bestCluster;
            changed = true;
        }
    }
    return changed;
}

void updateCentroids(const vector<Point>& data, vector<Point>& centroids) {
    int k = static_cast<int>(centroids.size());
    vector<double> sumX(k, 0.0), sumY(k, 0.0);
    vector<long long> count(k, 0);

    for (const auto& p : data) {
        sumX[p.cluster] += p.x;
        sumY[p.cluster] += p.y;
        count[p.cluster]++;
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
    for (const auto& p : data) {
        total += squaredDistance(p, centroids[p.cluster]);
    }
    return total;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_csv> <k> <max_iters> <output_prefix>\n";
        return 1;
    }

    string inputFile = argv[1];
    int k = atoi(argv[2]);
    int maxIters = atoi(argv[3]);
    string outPrefix = argv[4];

    vector<Point> data = loadDataset(inputFile);
    if (data.empty()) {
        cerr << "Error: dataset is empty or could not be parsed.\n";
        return 1;
    }
    vector<Point> centroids = initializeCentroids(data, k);

    cout << "[Sequential] Loaded " << data.size() << " points, K=" << k << "\n";

    auto start = chrono::high_resolution_clock::now();

    int iter = 0;
    bool changed = true;
    while (changed && iter < maxIters) {
        changed = assignClusters(data, centroids);
        updateCentroids(data, centroids);
        iter++;
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsedMs = chrono::duration<double, milli>(end - start).count();

    double inertia = computeInertia(data, centroids);

    cout << "[Sequential] Converged after " << iter << " iterations\n";
    cout << "[Sequential] Execution time: " << elapsedMs << " ms\n";
    cout << "[Sequential] Final inertia (SSE): " << inertia << "\n";

    ofstream assignOut(outPrefix + "_assignments.csv");
    assignOut << "x,y,cluster\n";
    for (const auto& p : data) assignOut << p.x << "," << p.y << "," << p.cluster << "\n";
    assignOut.close();

    ofstream centroidOut(outPrefix + "_centroids.csv");
    centroidOut << "x,y\n";
    for (const auto& c : centroids) centroidOut << c.x << "," << c.y << "\n";
    centroidOut.close();

    ofstream timingOut(outPrefix + "_timing.csv", ios::app);
    timingOut << "sequential,1," << data.size() << "," << k << "," << iter << "," << elapsedMs << "," << inertia << "\n";
    timingOut.close();

    return 0;
}
