#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cfloat>

#define DATA_LENGTH 1981
#define PRECISION 5
#define K 7

void read_file_to_array(long double (&arr)[DATA_LENGTH][2]) {
    std::ifstream inputFile("data.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return;
    }

    long double latitude, longitude;
    char openParenthesis, comma, closeParenthesis;

    // read line by line and fill array
    int curr_index = 0;
    while (inputFile >> openParenthesis >> latitude >> comma >> longitude >> closeParenthesis >> comma) {
        arr[curr_index][0] = latitude;
        arr[curr_index][1] = longitude;
        curr_index++;
    }

    // last line
    inputFile >> openParenthesis >> latitude >> comma >> longitude >> closeParenthesis;
    arr[curr_index][0] = latitude;
    arr[curr_index][1] = longitude;

    inputFile.close();
}

// read file to 2D array
void read_file_to_arrays(double (&arr_x)[DATA_LENGTH], double (&arr_y)[DATA_LENGTH]) {
    std::ifstream inputFile("data.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return;
    }

    double latitude, longitude;
    char openParenthesis, comma, closeParenthesis;

    // read line by line and fill array
    int curr_index = 0;
    while (inputFile >> openParenthesis >> latitude >> comma >> longitude >> closeParenthesis >> comma) {
        arr_x[curr_index] = latitude;
        arr_y[curr_index] = longitude;
        curr_index++;
    }

    // last line
    inputFile >> openParenthesis >> latitude >> comma >> longitude >> closeParenthesis;
    arr_x[curr_index] = latitude;
    arr_y[curr_index] = longitude;

    inputFile.close();
}

struct Point {
    // MEMBER VARIABLES:
    //  x:          latitude
    //  y:          longitude
    //  minDist:    minimum distance from point to cluster
    //  clusterID:  cluster ID that this point belongs to
    double x;
    double y;
    double minDist = DBL_MAX;
    int clusterID = -1;
    
    // CONSTRUCTORS
    Point() : x(0), y(0), minDist(DBL_MAX), clusterID(-1) {};
    Point(const double x, const double y) : x(x), y(y), minDist(DBL_MAX), clusterID(-1) {};
};

struct Centroid {
    // MEMBER VARIABLES:
    //  x:      latitude
    //  y:      longitude
    //  id:     cluster's ID
    double x;
    double y;
    int id;
    int nPoints;
    
    // CONSTRUCTORS
    Centroid() : x(0), y(0), id(-1), nPoints(0) {};
    Centroid(const double x, const double y) : x(x), y(y), id(-1), nPoints(0) {};

    // MEMBER FUNCTIONS:
    // distance:    calculates distance from given centroid to this point
    double distance(const Point& p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

// read file to Points array
void read_file_to_points_arr(Point * &arr) {
    FILE* inputFile = fopen("data.txt", "r");
    if (inputFile == NULL) {
        perror("Error opening the file");
        return;
    }

    double latitude, longitude;
    int curr_index = 0;

    while (fscanf(inputFile, "(%lf, %lf),\n", &latitude, &longitude) == 2 && curr_index < DATA_LENGTH) {
        printf("(%.5lf, %.5lf)\n", latitude, longitude);
        Point *p = new Point();
        p->x = latitude;
        p->y = longitude;
        arr[curr_index] = *p;
        curr_index++;
    }

    // one more iteration for the last line
    if (fscanf(inputFile, "(%lf, %lf)", &latitude, &longitude) == 2) {
        printf("(%.5lf, %.5lf)\n", latitude, longitude);
        Point *p = new Point();
        p->x = latitude;
        p->y = longitude;
        arr[curr_index] = *p;
        curr_index++;
    }

    fclose(inputFile);
}

void write_array_to_file(Point * &arr) {
    FILE* outputFile = fopen("output.csv", "w");
    if (outputFile == NULL) {
        perror("Error opening the file");
        return;
    }

    fprintf(outputFile, "x, y, c\n");

    for (int i = 0; i < DATA_LENGTH; ++i) {
        fprintf(outputFile, "%.5lf, %.5lf, %d\n", arr[i].x, arr[i].y, arr[i].clusterID);
    }

    fclose(outputFile);
}

int main(int argc, char**argv) {
    Point * h_data = (Point *) malloc(sizeof(Point) * DATA_LENGTH);
    Centroid * h_centroids = (Centroid *) malloc(sizeof(Point) * K);
    Point * d_data = (Point*) malloc(sizeof(Point) * DATA_LENGTH);
    Centroid * d_centroids = (Centroid*) malloc(sizeof(Point) * K);;

    // read file data points to h_data
    read_file_to_points_arr(h_data);

    // print data points for debugging
    // for (int i = 0; i < DATA_LENGTH; i++) {
    //     printf("(%f, %f),   minDist: %f,    clusterID: %d\n", h_data[i].x, h_data[i].y, h_data[i].minDist, h_data[i].clusterID);
    // }

    //naive initial selection, picks a random data point as centroids
    for (int i = 0; i < K; i++){
        int rand_ind = rand() % DATA_LENGTH;
        h_centroids[i].x = h_data[rand_ind].x;
        h_centroids[i].y = h_data[rand_ind].y;
        h_centroids[i].id = i;
    }

    // test print centroids data for debugging
    // for (int i = 0; i < K; i++) {
    //     printf("(%f, %f),   clusterID: %d\n", h_centroids[i].x, h_centroids[i].y, h_centroids[i].id);
    // }

    int iterations = 0;
    while (iterations < 20) {
        // assign points to clusters
        for (int i = 0; i < K; i++) {
            int clusterID = h_centroids[i].id;

            for (int j = 0; j < DATA_LENGTH; j++) {
                double distance = h_centroids[i].distance(h_data[j]);
                // printf("distance: %f,     currMinDist: %f\n", distance, h_data[j].minDist);
                if (distance < h_data[j].minDist) {
                    h_data[j].minDist = distance;
                    h_data[j].clusterID = clusterID;
                }
            }
        }

        // print data points for debugging
        // for (int i = 0; i < DATA_LENGTH; i++) {
        //     printf("(%f, %f),   minDist: %f,    clusterID: %d\n", h_data[i].x, h_data[i].y, h_data[i].minDist, h_data[i].clusterID);
        // }

        // iterate over points to sum and calculate new centroid
        double sums[K][2];
        for (int i = 0; i < DATA_LENGTH; i++) {
            int clusterID = h_data[i].clusterID;
            h_centroids[clusterID].nPoints++;
            sums[clusterID][0] += h_data[i].x;
            sums[clusterID][1] += h_data[i].y;

            // reset min distance
            h_data[i].minDist = DBL_MAX;
        }

        // compute new centroids
        for (int i = 0; i < K; i++) {
            h_centroids[i].x = sums[i][0] / h_centroids[i].nPoints;
            h_centroids[i].y = sums[i][1] / h_centroids[i].nPoints;
        }


        // printf("\n");

        // test print centroids data for debugging
        // for (int i = 0; i < K; i++) {
        //     printf("(%f, %f),   clusterID: %d,    nPoints: %d\n", h_centroids[i].x, h_centroids[i].y, h_centroids[i].id, h_centroids[i].nPoints);
        // }

        for (int i = 0; i < K; i++) {
            sums[i][0] = 0;
            sums[i][1] = 0;
            h_centroids[i].nPoints = 0;
        }

        iterations++;
    }

    write_array_to_file(h_data);

    // std::ofstream myfile;
    // myfile.open("output.csv");
    // myfile << "x,y,c" << std::endl;

    // for (int i = 0; i < DATA_LENGTH; i++) {
    //     myfile << h_data[i].x << "," << h_data[i].y << "," << h_data[i].clusterID << std::endl;
    // }
    // myfile.close();


    return 0;
}