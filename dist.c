#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

MPI_Comm MPI_COMM_COMPUTE;

int* rowsPerProc(int numProcs, int dim) {
    int computeDim = dim-2;
    int rowsPerProc = computeDim / numProcs;
    int leftoverRows = computeDim % numProcs;
    int* rows = malloc(sizeof(int)*numProcs);
    for(int i = 0; i < numProcs; i++) {
        rows[i] = rowsPerProc;
        if (leftoverRows != 0) {
            rows[i]++;
            leftoverRows--;
        }
    }	
    return rows;	
}

void printInfo(double* workingArr, int size, int dim, int rank) {
    char strArr[size*dim];
    char *s = malloc(sizeof(char) * 5);
    sprintf(s, "%.2f ", workingArr[0]);
    strcpy(strArr, s);

    for (int i = 1; i < size; i++) {
        sprintf(s, "%.2f ", workingArr[i]);
        if (i % dim == 0) strcat(strArr, "\n");
        strcat(strArr, s);
    }
    printf("Processor with rank: %d has a working array of: \n%s\n", rank, strArr);
}

double* initMasterProc(double* arr, int dim, double precision, int *size) {
    int numProcs;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);

    double *workingArr;

    int* rows = rowsPerProc(numProcs, dim);
    int elemsToCompute[numProcs];
    for (int i = 0; i < numProcs; i++) {
        elemsToCompute[i] = rows[i] * dim;
    }

    int indices[numProcs];
    int nextIndex = 0;
    for (int i = 0; i < numProcs; i++) {
        printf("index for proc %d is: %d\n", i, nextIndex);
        indices[i] = nextIndex;
        int prevIndex = nextIndex;
        nextIndex = elemsToCompute[i] + prevIndex;
    }

    rows = &rows[0];
    *size = elemsToCompute[0] + (2*dim);
    workingArr = malloc(sizeof(double) * *size);
    memcpy(workingArr, &arr[0], sizeof(double) * *size);
    // send arrays to other processes
    MPI_Request req[numProcs-1];
    MPI_Status statuses[numProcs-1];
    for (int i = 1; i < numProcs; i++) {
        int index = indices[i];
        int procArrSize = elemsToCompute[i] + (2*dim);
        MPI_Send(&procArrSize, 1, MPI_INT, i, 1, MPI_COMM_COMPUTE);
        MPI_Isend(&arr[index], procArrSize, MPI_DOUBLE, i, 0, 
                MPI_COMM_COMPUTE, &req[i-1]);
    }
    MPI_Waitall(numProcs-1, req, statuses);

    return workingArr;
}

double* initSlaveProc(int dim, double precision, int size, int rank) {
    double *workingArr;
    int numProcs;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);

    workingArr = malloc(sizeof(double) * size);
    MPI_Recv(workingArr, size, MPI_DOUBLE, 0, 0, MPI_COMM_COMPUTE, MPI_STATUS_IGNORE);

    return workingArr;
}

int main() {
    int dim = 5;
    int precision = 0;

    MPI_Init(NULL, NULL);
    int rank;    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int inUse = rank < dim-2;
    MPI_Comm_split(MPI_COMM_WORLD, inUse, rank, &MPI_COMM_COMPUTE);
    if (!inUse) {
        MPI_Finalize();
        exit(0);
    }

    double *workingArr;
    int size;
    if (rank == 0) {
        // read array from file
        //double arr[36] = {1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1};
        double arr[25] = {1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1};
        workingArr = initMasterProc(&arr[0], dim, precision, &size);
    } else {
        MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_COMPUTE, MPI_STATUS_IGNORE);
        workingArr = initSlaveProc(dim, precision, size, rank);
    }
    printInfo(workingArr, size, dim, rank);
    
    //relax(workingArr, size, dim, precision, rank);

    // relax working array, sending top and bottom rows to top and bottom ranks
    // send out settled flag to everyone async

    MPI_Finalize();
}
