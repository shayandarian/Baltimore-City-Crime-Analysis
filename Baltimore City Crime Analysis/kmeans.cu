
// #include <cuda.h>
#include <stdio.h>
#include <cuda.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cfloat>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#define MAX_ITERATIONS 50
#define DATA_LENGTH 1981
#define MAX_THREADS_PER_BLOCK 512
#define K 7                         // K # of clusters

const double epsilon = 1e-6;

struct Point {
    // MEMBER VARIABLES:
    //  x:          latitude
    //  y:          longitude
    //  minDist:    minimum distance from point to cluster
    //  clusterID:  cluster ID that this point belongs to
    double x;
    double y;
    double minDist;
    int clusterID;
    
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
};

__global__ void calculate_min_distance(Point * datapoints, Centroid * centroids, int clusterID){
    int index = threadIdx.x + blockIdx.x * MAX_THREADS_PER_BLOCK;
    if (index < DATA_LENGTH) {
        double x_sqr = pow((datapoints[index].x - centroids[clusterID].x), 2.0);
        double y_sqr = pow((datapoints[index].y - centroids[clusterID].y), 2.0);

        // printf("datapoint_x: %f,    datapoint_y: %f\n", datapoints[index].x, datapoints[index].y);
        // printf("clusterID:  %d,     centroids_x: %f,    centroids_y: %f\n", clusterID, centroids[clusterID].x, centroids[clusterID].y);
        // printf("x_sqr: %f,     y_sqr: %f\n", x_sqr, y_sqr);

        double distance = x_sqr + y_sqr;
        if (distance < datapoints[index].minDist) {
            datapoints[index].minDist = distance;
            datapoints[index].clusterID = clusterID;
        }
    }
}

// compute cluster sums
__global__ void compute_cluster_sums(Point * datapoints, Centroid * centroids, double * xsums, double * ysums) {
    int index = threadIdx.x + blockIdx.x * MAX_THREADS_PER_BLOCK;
    if (index < DATA_LENGTH) {
        int clusterID = datapoints[index].clusterID;

        atomicAdd(&centroids[clusterID].nPoints, 1);
        atomicAdd(&xsums[clusterID], datapoints[index].x);
        atomicAdd(&ysums[clusterID], datapoints[index].y);

        // reset min distance
        datapoints[index].minDist = DBL_MAX;
    }
}

// read file to array
void read_file_to_arr(Point * &arr) {
    FILE* inputFile = fopen("data.txt", "r");
    if (inputFile == NULL) {
        perror("Error opening the file");
        return;
    }

    double latitude, longitude;
    int curr_index = 0;

    while (fscanf(inputFile, "(%lf, %lf),\n", &latitude, &longitude) == 2 && curr_index < DATA_LENGTH) {
        Point *p = new Point();
        p->x = latitude;
        p->y = longitude;
        arr[curr_index] = *p;
        curr_index++;
    }

    // one more iteration for the last line
    if (fscanf(inputFile, "(%lf, %lf)", &latitude, &longitude) == 2) {
        Point *p = new Point();
        p->x = latitude;
        p->y = longitude;
        arr[curr_index] = *p;
        curr_index++;
    }

    fclose(inputFile);
}

// write array to file
void write_array_to_file(Point * &arr) {
    FILE* outputFile = fopen("output.csv", "w");
    if (outputFile == NULL) {
        perror("Error opening the file");
        return;
    }

    fprintf(outputFile, "x,y,c\n");

    for (int i = 0; i < DATA_LENGTH; i++) {
        fprintf(outputFile, "%.5lf,%.5lf,%d\n", arr[i].x, arr[i].y, arr[i].clusterID);
    }

    fclose(outputFile);
}

int main(int argc, char**argv) {
    //init x, y data into separate arrays
    //init k from command line
    int k = K;
    if (argc > 1){
        k = atoi(argv[1]);
    }
    printf("K is: %d\n", k);
    
    //error checking
    if(k > DATA_LENGTH){
        printf("K must be less than the number of data points");
        return 1;
    }

    // allocate host memory
    Point * h_data = (Point*) malloc(sizeof(Point) * DATA_LENGTH); 
    Centroid * h_centroids = (Centroid*) malloc(sizeof(Centroid) * k);
    double * h_xsums = (double*) malloc(sizeof(double) * k);
    double * h_ysums = (double*) malloc(sizeof(double) * k);

    // allocate memory for GPU objects
    Point * d_data = (Point*) malloc(sizeof(Point) * DATA_LENGTH);
    Centroid * d_centroids = (Centroid*) malloc(sizeof(Centroid) * k);
    double * d_xsums = (double*) malloc(sizeof(double) * k);
    double * d_ysums = (double*) malloc(sizeof(double) * k);

    // read file data points to h_data
    read_file_to_arr(h_data);

    // print data points for debugging
    // for (int i = 0; i < DATA_LENGTH; i++) {
    //     printf("(%f, %f),   minDist: %f,    clusterID: %d\n", h_data[i].x, h_data[i].y, h_data[i].minDist, h_data[i].clusterID);
    // }

    //naive initial selection, picks a random data point as centroids
    for (int i = 0; i < k; i++){
        int rand_ind = rand() % DATA_LENGTH;
        h_centroids[i].x = h_data[rand_ind].x;
        h_centroids[i].y = h_data[rand_ind].y;
        h_centroids[i].id = i;
    }

    // test print centroids data for debugging
    // for (int i = 0; i < k; i++) {
    //     printf("clusterID: %d,  (%f, %f)\n", h_centroids[i].x, h_centroids[i].y, h_centroids[i].id);
    // }

    // 512 threads and 4 blocks for current data
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks(static_cast<int>(ceil(static_cast<double>(DATA_LENGTH)/static_cast<double>(MAX_THREADS_PER_BLOCK))));

    // allocate memory and copy data and centroids to GPU
    cudaMalloc((void**) &d_data, sizeof(Point) * DATA_LENGTH);
    cudaMalloc((void**) &d_centroids, sizeof(Centroid) * k);
    cudaMemcpy(d_data, h_data, sizeof(Point) * DATA_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(Centroid) * k, cudaMemcpyHostToDevice);

    //main loop
    int counter = 0;
    while(counter < MAX_ITERATIONS) {
        // for each point, calculate the minimum distance to centroids
        for (int i = 0; i < k; i++) {
            calculate_min_distance<<<blocks, threads>>>(d_data, d_centroids, i);

            // make sure sums and centroids nPoints are reset
            h_xsums[i] = 0;
            h_ysums[i] = 0;
            h_centroids[i].nPoints = 0;
        }

        // allocate memory on GPU for sums and copy sums from host to device
        cudaMalloc((void**) &d_xsums, sizeof(double) * k);
        cudaMalloc((void**) &d_ysums, sizeof(double) * k);
        cudaMemcpy(d_xsums, h_xsums, sizeof(double) * k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ysums, h_ysums, sizeof(double) * k, cudaMemcpyHostToDevice);

        // for each cluster, compute the xsums and the ysums
        compute_cluster_sums<<<blocks, threads>>>(d_data, d_centroids, d_xsums, d_ysums);

        // copy centroids data and sums data from device to host
        cudaMemcpy(h_centroids, d_centroids, sizeof(Centroid) * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xsums, d_xsums, sizeof(double) * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ysums, d_ysums, sizeof(double) * k, cudaMemcpyDeviceToHost); 

        // create a flag for convergence
        bool converges = true;

        // compute new centroids
        for (int i = 0; i < k; i++) {
            // copy old centroid coordinates
            double old_centroid_x = h_centroids[i].x;
            double old_centroid_y = h_centroids[i].y;

            // compute new centroids
            h_centroids[i].x = h_centroids[i].nPoints != 0 ? (h_xsums[i] / h_centroids[i].nPoints) : old_centroid_x;
            h_centroids[i].y = h_centroids[i].nPoints != 0 ? (h_ysums[i] / h_centroids[i].nPoints) : old_centroid_y;

            printf("nPoints: %d\n", h_centroids[i].nPoints);
            printf("old_x:   %f,    old_y:   %f\n", old_centroid_x, old_centroid_y);
            printf("new_x:   %f,    new_y:   %f\n", h_centroids[i].x, h_centroids[i].y);

            // reset cluster points and sums
            h_xsums[i] = 0;
            h_ysums[i] = 0;
            h_centroids[i].nPoints = 0;

            // compare new centroids with old_centroids
            // if not the same, set the flag to false
            if (std::abs(h_centroids[i].x - old_centroid_x) > epsilon ||
                std::abs(h_centroids[i].y - old_centroid_y) > epsilon) {
                converges = false;
            }
        }

        // copy newly computed centroids and resetted sums from host to device
        cudaMemcpy(d_centroids, h_centroids, sizeof(Centroid) * k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_xsums, h_xsums, sizeof(double) * k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ysums, h_ysums, sizeof(double) * k, cudaMemcpyHostToDevice);

        // deallocate and reset sums
        cudaFree(d_xsums);
        cudaFree(d_ysums);

        printf("convergence: %s\n", (converges ? "true" : "false"));
        if (converges) {
            printf("converges at iteration: %d\n", counter+1);
            break;
        }

        counter++;
    }

    // copy datapoints with clusterID data from device to host
    cudaMemcpy(h_data, d_data, sizeof(Point) * DATA_LENGTH, cudaMemcpyDeviceToHost);

    // deallocate GPU memory
    cudaFree(d_data);
    cudaFree(d_centroids);

    write_array_to_file(h_data);

    return 0;
}