#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Генерация матрицы со случайными значениями в диапазоне [-1, 1]
vector<vector<double>> generateMatrix(int m) {
    vector<vector<double>> matrix(m, vector<double>(m));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// Вывод матрицы в консоль
void printMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    cout << "+";
    for (int j = 0; j < cols; ++j) {
        cout << "---------+";
    }
    cout << endl;

    for (int i = 0; i < rows; ++i) {
        cout << "| ";
        for (int j = 0; j < cols; ++j) {
            cout << setw(7) << fixed << setprecision(3) << matrix[i][j] << " | ";
        }
        cout << endl;
        cout << "+";
        for (int j = 0; j < cols; ++j) {
            cout << "---------+";
        }
        cout << endl;
    }
}

// Параллельное вычисление матрицы C по формуле
vector<vector<double>> computeMatrixC(const vector<vector<double>>& A, const vector<vector<double>>& B,
                                      const vector<vector<double>>& D, const vector<vector<double>>& E, bool parallel) {
    int m = A.size();
    vector<vector<double>> C(m, vector<double>(m));

    #pragma omp parallel for if(parallel)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = (A[i][j] + 2 * B[i][j]) * D[i][j] - (E[i][j] * D[i][j]);
        }
    }
    return C;
}

bool areMatricesEqual(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
    for (int i = 0; i < mat1.size(); ++i) {
        for (int j = 0; j < mat1[i].size(); ++j) {
            if (abs(mat1[i][j] - mat2[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int m = 5; // Размер матрицы для демонстрации
    vector<vector<double>> A = generateMatrix(m);
    vector<vector<double>> B = generateMatrix(m);
    vector<vector<double>> D = generateMatrix(m);
    vector<vector<double>> E = generateMatrix(m);

    cout << "Матрица A:" << endl;
    printMatrix(A);
    cout << "\nМатрица B:" << endl;
    printMatrix(B);
    cout << "\nМатрица D:" << endl;
    printMatrix(D);
    cout << "\nМатрица E:" << endl;
    printMatrix(E);
    cout << endl;

    // Последовательное вычисление
    auto startSequential = high_resolution_clock::now();
    vector<vector<double>> CSequential = computeMatrixC(A, B, D, E, false);
    auto stopSequential = high_resolution_clock::now();
    auto durationSequential = duration_cast<microseconds>(stopSequential - startSequential);
    cout << "Последовательное вычисление: " << durationSequential.count() << " микросекунд\n";
    printMatrix(CSequential);
    cout << endl;

    // Параллельное вычисление
    auto startParallel = high_resolution_clock::now();
    vector<vector<double>> CParallel = computeMatrixC(A, B, D, E, true);
    auto stopParallel = high_resolution_clock::now();
    auto durationParallel = duration_cast<microseconds>(stopParallel - startParallel);
    cout << "Параллельное вычисление: " << durationParallel.count() << " микросекунд\n";
    printMatrix(CParallel);
    cout << endl;

    // Проверка на равенство матриц
    if (areMatricesEqual(CSequential, CParallel)) {
        cout << "Матрицы C последовательной и параллельной обработки совпадают." << endl;
    } else {
        cout << "Ошибка: Матрицы C не совпадают!" << endl;
    }

    return 0;
}
