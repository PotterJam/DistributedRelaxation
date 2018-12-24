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
    printf("Processor with rank %d has a working array of: \n%s\n", rank, strArr);
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

double* initSlaveProc(int dim, double precision, int size) {
    double *workingArr;
    int numProcs;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);

    workingArr = malloc(sizeof(double) * size);
    MPI_Recv(workingArr, size, MPI_DOUBLE, 0, 0, MPI_COMM_COMPUTE, MPI_STATUS_IGNORE);

    return workingArr;
}

double avgElem(double* arr, int index, int dim) {
    double up = arr[index-dim];
    double down = arr[index+dim];
    double left = arr[index-1];
    double right = arr[index+1];
    return (up+down+left+right)/4;
}

void averageRow(double* workingArr, double* avgArr, int dim, int row) {
    int startIndex = row * dim;

    // fixed elements at edge of matrix
    avgArr[startIndex] = workingArr[startIndex];
    avgArr[startIndex+dim-1] = workingArr[startIndex+dim-1];

    // average other elements in row
    for (int i = 1; i < dim-1; i++) {
        avgArr[startIndex+i] = avgElem(workingArr, startIndex+i, dim); 
    }
}

double* relax(double* workingArr, int size, int dim, double precision) {
    int numProcs;
    int rank;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numRows = size/dim;
    int *rowIndices = malloc(sizeof(int) * numRows);
    double* avgArr = malloc(sizeof(double) * size);

    int nextRowIndex = 0;
    for (int i = 0; i < numRows; i++) {
        rowIndices[i] = nextRowIndex;
        nextRowIndex += dim;
    }
    
    int topEdge = rank == 0;
    int bottomEdge = rank == numProcs-1;

    MPI_Request topRowReq, bottomRowReq;
    MPI_Request topRowSend, bottomRowSend;
    // async receive top and bottom rows
    averageRow(workingArr, avgArr, dim, 1);
    if (topEdge) {
        memcpy(&avgArr[0], &workingArr[0], sizeof(double) * dim);
    } else {
        MPI_Isend(&avgArr[dim], dim, MPI_DOUBLE, rank-1, 3, MPI_COMM_COMPUTE, &topRowSend);
        MPI_Irecv(&avgArr[0], dim, MPI_DOUBLE, rank-1, 4, MPI_COMM_COMPUTE, &topRowReq);
    }

    averageRow(workingArr, avgArr, dim, numRows-2);
    int lastI = rowIndices[numRows-1];
    int secondLastI = rowIndices[numRows-2];
    if (bottomEdge) {
        memcpy(&avgArr[lastI], &workingArr[lastI], sizeof(double) * dim);
    } else {
        MPI_Isend(&avgArr[secondLastI], dim, MPI_DOUBLE, rank+1, 4, MPI_COMM_COMPUTE, &bottomRowSend);
        MPI_Irecv(&avgArr[lastI], dim, MPI_DOUBLE, rank+1, 3, MPI_COMM_COMPUTE, &bottomRowReq);
    }

    for (int i = 2; i < numRows-2; i++) {
        averageRow(workingArr, avgArr, dim, i);            
    }

    if (!topEdge) {
        MPI_Wait(&topRowReq, MPI_STATUS_IGNORE);
        MPI_Wait(&topRowSend, MPI_STATUS_IGNORE);
    }
    
    if (!bottomEdge) {
        MPI_Wait(&bottomRowReq, MPI_STATUS_IGNORE);
        MPI_Wait(&bottomRowSend, MPI_STATUS_IGNORE);
    }

    return avgArr;
}

// TAG 0 is the array
// TAG 1 is the size
// TAG 2 is the settle flag
// TAG 3 is updated rows
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
        workingArr = initSlaveProc(dim, precision, size);
    }

    double *newArr = relax(workingArr, size, dim, precision);
    printInfo(newArr, size, dim, rank);

    // relax working array, sending top and bottom rows to top and bottom ranks
    // send out settled flag to everyone async

    MPI_Finalize();
}
