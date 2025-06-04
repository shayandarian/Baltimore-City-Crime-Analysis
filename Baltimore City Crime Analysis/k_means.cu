#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <fstream>
#include <iostream>

#define MAX_ITERATIONS 50
#define DATA_LENGTH 1981
#define MAX_THREADS_PER_BLOCK 512
#define K 7


//input: data, centroids, other info. out: centroids of each datapoint in the corresponding index
//arr_x_vals: array of the x coordinates
//arr_y_vals: array of the y coordinates. Note: arr_x_vals[i], arr_y_vals[i] are the locations pairs
//centroid_x_vals: array of x values for each centroid
//centroid_y_vals: array of y values for each centroid. Note: centroid_x_vals[i], centroid_y_vals[i] are the location pairs
//max_int: MAX_INT value, passed for calculating min value
__global__ void find_nearest_centroid(float * arr_x_vals, float * arr_y_vals, float * centroid_x_vals, float * centroid_y_vals, int data_length, int k, float max_float, int * out_array){
    int index = threadIdx.x + blockIdx.x * MAX_THREADS_PER_BLOCK;
    //bounds checking
    if (index < data_length){
        //squared euclidian distance
        float min = max_float;
        int min_cluster = -1;
        if (index == 1980){ printf("Hello from index 1980, I have %f, %f", arr_x_vals[1980], arr_y_vals[1980]); }
        for(int i = 0; i < k; i++){
            float dist = pow((arr_x_vals[index] - centroid_x_vals[i]), 2.0) + pow((arr_y_vals[index] - centroid_y_vals[i]), 2.0);
            /*if (index == 1){ 
                printf("a_x: %f, c_x: %f, a_y: %f, c_y: %f\n", arr_x_vals[index], centroid_x_vals[i], arr_y_vals[index], centroid_y_vals[i]);
                printf("Dist for index %d, iter %d: %f\n", index, i, dist); 
            }*/
            if (dist < min){
                min = dist;
                min_cluster = i;
            }
        }
        out_array[index] = min_cluster;
    }
}

// read file to 2D array
void read_file_to_array_2d(float (&arr)[DATA_LENGTH][2]) {
    std::ifstream inputFile("data.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return;
    }

    float latitude, longitude;
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

// read file to 2 arrays
void read_file_to_arrays(float arr_x[], float arr_y[]) {
    std::ifstream inputFile("data.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return;
    }

    float latitude, longitude;
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

void write_array_to_file(float arr_x[], float arr_y[], int cluster_group[]) {
    FILE* outputFile = fopen("output.csv", "w");
    if (outputFile == NULL) {
        perror("Error opening the file");
        return;
    }

    //header
    fprintf(outputFile, "x,y,c\n");

    for (int i = 0; i < DATA_LENGTH; ++i) {
        fprintf(outputFile, "%.6lf, %.6lf, %d\n", arr_x[i], arr_y[i], cluster_group[i]);
    }

    fclose(outputFile);
}


//will accept k as an input from command line
int main(int argc, char ** argv){
    //init x, y data into separate arrays
    //init k from command line
    int k = K;
    if (argc > 0){
        k = atoi(argv[1]);
    }
    printf("K is: %d\n", k);
    
    //error checking
    if(k > DATA_LENGTH){
        printf("K must be less than the number of data points");
        return 1;
    }
    
    //init x and y arrays
    float * x_vals = (float *) malloc(sizeof(float) * DATA_LENGTH);
    float * y_vals = (float *) malloc(sizeof(float) * DATA_LENGTH);

    read_file_to_arrays(x_vals, y_vals);

    float * centroid_x = (float *) malloc(sizeof(float) * k);
    float * centroid_y = (float *) malloc(sizeof(float) * k);
    int * cluster_center_avgs = (int *) calloc(k, sizeof(int));    //needs to be zeroed so used calloc, used to find total occurrences in each cluster

    //naiive initial selection, picks a random data point as centroids
    time_t t;
    srand((unsigned) time(&t));
    printf("Initial centroids are: \n");
    for (int i = 0; i < k; i++){
        int rand_ind = rand() % DATA_LENGTH;
        centroid_x[i] = x_vals[rand_ind];
        centroid_y[i] = y_vals[rand_ind];
        printf("%d %f, %f\n", i, centroid_x[i], centroid_y[i]);
    }
    
    // define number of threads and blocks
    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks(ceil(float(DATA_LENGTH)/float(MAX_THREADS_PER_BLOCK)));
    printf("# of threads: %d\n# of blocks: %d\n", MAX_THREADS_PER_BLOCK, int(ceil(float(DATA_LENGTH)/float(MAX_THREADS_PER_BLOCK))));

    //the data arrays are not modified, so they only need to be declared once (save resources)
    float * device_in_x_vals;
    cudaMalloc((void **) &device_in_x_vals, sizeof(float) * DATA_LENGTH);
    cudaMemcpy(device_in_x_vals, x_vals, sizeof(float) * DATA_LENGTH, cudaMemcpyHostToDevice);
    float * device_in_y_vals;
    cudaMalloc((void **) &device_in_y_vals, sizeof(float) * DATA_LENGTH);
    cudaMemcpy(device_in_y_vals, y_vals, sizeof(float) * DATA_LENGTH, cudaMemcpyHostToDevice);

    //array data modified, but there is only assignment, so it can be declared once
    int * device_out_array;
    cudaMalloc((void **) &device_out_array, sizeof(int) * DATA_LENGTH);

    int * host_in_array = (int *) malloc(sizeof(int) * DATA_LENGTH);

    float * device_in_centroid_x;
    cudaMalloc((void **) &device_in_centroid_x, sizeof(float) * DATA_LENGTH);
    float * device_in_centroid_y;
    cudaMalloc((void **) &device_in_centroid_y, sizeof(float) * DATA_LENGTH);
    
    //main loop
    int counter = 0;
    while (counter < MAX_ITERATIONS){
        /*printf("Testing centroids before function call: ");
        for (int i = 0; i < k; i++){
            printf("(%f, %f) ", centroid_x[i], centroid_y[i]);
        }
        printf("\n");*/
        
        //load arrays into GPU memory
        cudaMemcpy(device_in_centroid_x, centroid_x, sizeof(float) * k, cudaMemcpyHostToDevice);
        cudaMemcpy(device_in_centroid_y, centroid_y, sizeof(float) * k, cudaMemcpyHostToDevice);
        //call find_nearest_centroid
        find_nearest_centroid<<<blocks, threads>>>(device_in_x_vals, device_in_y_vals, device_in_centroid_x, device_in_centroid_y, DATA_LENGTH, k, FLT_MAX, device_out_array);
        //copy result into array
        cudaMemcpy(host_in_array, device_out_array, sizeof(int) * DATA_LENGTH, cudaMemcpyDeviceToHost);
        
        printf("Testing host_in_array: ");
        for (int i = 0; i < 10; i++){
            printf("%d ", host_in_array[i]);
        }
        printf("\n");

        //determine average x and y values within each centroid using result arr. these are the new 
        float * new_centroid_x = (float *) calloc(k, sizeof(float));
        float * new_centroid_y = (float *) calloc(k, sizeof(float));
        for (int i = 0; i < DATA_LENGTH; i++){
            cluster_center_avgs[host_in_array[i]] += 1;
            new_centroid_x[host_in_array[i]] += x_vals[i];
            new_centroid_y[host_in_array[i]] += y_vals[i];
        }

        printf("Total occurrences in each cluster: \n");
        for (int i = 0; i < k; i++){
            printf("Cluster %d: %d\n", i, cluster_center_avgs[i]);
        }

        //calculate average
        //check for changes, if not data has converged
        bool centroids_same = true;
        for (int i = 0; i < k; i++){
            float divide_by = float(cluster_center_avgs[i]);
            if(divide_by == 0){
                divide_by = 1.0;
            }
            new_centroid_x[i] = new_centroid_x[i] / divide_by;
            new_centroid_y[i] = new_centroid_y[i] / divide_by;
            //reset cluster_center_avgs
            cluster_center_avgs[i] = 0;
            if (new_centroid_x[i] != centroid_x[i] || new_centroid_y[i] != centroid_y[i]){
                centroids_same = false;
            }
        }
        if (centroids_same){
            printf("Converged on iteration %d\n", counter);
            write_array_to_file(x_vals, y_vals, host_in_array);
            //print final clusters
            for (int i = 0; i < k; i++){
                int rand_ind = rand() % DATA_LENGTH;
                centroid_x[i] = x_vals[rand_ind];
                centroid_y[i] = y_vals[rand_ind];
                printf("%d %f, %f\n", i, centroid_x[i], centroid_y[i]);
            }
            break;
        }else{
            //update centroid location
            for (int i = 0; i < k; i++){
                centroid_x[i] = new_centroid_x[i];
                centroid_y[i] = new_centroid_y[i];
            }
        }
        free(new_centroid_x);
        free(new_centroid_y);
        
        
        
        //display k for debugging
        for (int i = 0; i < k; i++){
            printf("End of iteration %d. Centroid %d is %f, %f\n", counter, i, centroid_x[i], centroid_y[i]);
        }
        
        counter++;
    }

    cudaFree(device_in_centroid_x);
    cudaFree(device_in_centroid_y);
    cudaFree(device_in_x_vals);
    cudaFree(device_in_y_vals);
    cudaFree(device_out_array);
    free(host_in_array);
    free(centroid_x);
    free(centroid_y);
    free(x_vals);
    free(y_vals);
}
