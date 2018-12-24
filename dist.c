#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    free(s);
    printf("Processor with rank %d has a working array of: \n%s\n", rank, strArr);
}

double* initMasterProc(double* arr, int* elemsToCompute, int* indices, 
        int dim, double precision, int *size) {
    int numProcs;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);

    double *workingArr;

    int* rows = rowsPerProc(numProcs, dim);
    for (int i = 0; i < numProcs; i++) {
        elemsToCompute[i] = rows[i] * dim;
    }
    free(rows);

    int nextIndex = 0;
    for (int i = 0; i < numProcs; i++) {
        indices[i] = nextIndex;
        int prevIndex = nextIndex;
        nextIndex = elemsToCompute[i] + prevIndex;
    }

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
    MPI_Recv(workingArr, size, MPI_DOUBLE, 0, 0, 
            MPI_COMM_COMPUTE, MPI_STATUS_IGNORE);

    return workingArr;
}

double avgElem(double* arr, int index, int dim) {
    double up = arr[index-dim];
    double down = arr[index+dim];
    double left = arr[index-1];
    double right = arr[index+1];
    return (up+down+left+right)/4;
}

int fequal(double a, double b, double precision) {
    return fabs(a-b) < precision;
}

void averageRow(double* workingArr, double* avgArr, int dim, int row,
         int *settled, double precision) {
    int startIndex = row * dim;

    // fixed elements at edge of matrix
    avgArr[startIndex] = workingArr[startIndex];
    avgArr[startIndex+dim-1] = workingArr[startIndex+dim-1];

    // average other elements in row
    for (int i = 1; i < dim-1; i++) {
        double avg = avgElem(workingArr, startIndex+i, dim);

        // update settled flag if within precision
        if (*settled == 1 && !fequal(avg, workingArr[startIndex+i], precision)) {
            *settled = 0;
        }

        avgArr[startIndex+i] = avg;
    }
}

void relax(double* workingArr, int size, int dim, double precision, int numProcs, int rank) {
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

    MPI_Request topRowReq, topRowSend;
    MPI_Request bottomRowReq, bottomRowSend;
    MPI_Request settledReq[numProcs], settledSend[numProcs];

    int procSettled[numProcs];
    int settled = 0;

    // start loop here
    while (!settled) {

        // reset settled
        settled = 1;

        averageRow(workingArr, avgArr, dim, 1, &settled, precision);
        if (topEdge) {
            memcpy(&avgArr[0], &workingArr[0], sizeof(double) * dim);
        } else {
            MPI_Isend(&avgArr[dim], dim, MPI_DOUBLE, rank-1, 3,
                    MPI_COMM_COMPUTE, &topRowSend);
            MPI_Irecv(&avgArr[0], dim, MPI_DOUBLE, rank-1, 4,
                    MPI_COMM_COMPUTE, &topRowReq);
        }

        averageRow(workingArr, avgArr, dim, numRows-2, &settled, precision);
        int lastI = rowIndices[numRows-1];
        int secondLastI = rowIndices[numRows-2];
        if (bottomEdge) {
            memcpy(&avgArr[lastI], &workingArr[lastI], sizeof(double) * dim);
        } else {
            MPI_Isend(&avgArr[secondLastI], dim, MPI_DOUBLE, rank+1, 4, 
                    MPI_COMM_COMPUTE, &bottomRowSend);
            MPI_Irecv(&avgArr[lastI], dim, MPI_DOUBLE, rank+1, 3, 
                    MPI_COMM_COMPUTE, &bottomRowReq);
        }

        // bulk of processing here, averages all the other rows
        for (int i = 2; i < numRows-2; i++) {
            averageRow(workingArr, avgArr, dim, i, &settled, precision);            
        }

        // send settled value to other processes
        procSettled[rank] = settled;
        for (int i = 0; i < numProcs; i++) {
            if (i != rank) {
                MPI_Isend(&settled, 1, MPI_INT, i, 2, 
                        MPI_COMM_COMPUTE, &settledSend[i]);
                MPI_Irecv(&procSettled[i], 1, MPI_INT, i, 2, 
                        MPI_COMM_COMPUTE, &settledReq[i]); 
            }
        }

        if (!topEdge) {
            MPI_Wait(&topRowReq, MPI_STATUS_IGNORE);
            MPI_Wait(&topRowSend, MPI_STATUS_IGNORE);
        }

        if (!bottomEdge) {
            MPI_Wait(&bottomRowReq, MPI_STATUS_IGNORE);
            MPI_Wait(&bottomRowSend, MPI_STATUS_IGNORE);
        }

        // settled send and recv waits
        for (int i = 0; i < numProcs; i++) {
            if (i != rank) {
                MPI_Wait(&settledSend[i], MPI_STATUS_IGNORE);
                MPI_Wait(&settledReq[i], MPI_STATUS_IGNORE);
            }
        }

        for (int i = 0; i < numProcs; i++) {
            if (procSettled[i] == 0) {
                settled = 0;
                break;
            }
        }

        // swap avgArr and workingArr pointers if not settled
        if (!settled) {
            double* temp = avgArr;
            avgArr = workingArr;
            workingArr = temp;
        }
    }  
    free(rowIndices);
    free(avgArr);
}

// TAG 0 is the initial array
// TAG 1 is the size
// TAG 2 is the settle flag
// TAG 3 is updated rows (top)
// TAG 4 is updated rows (bottom)
int main() {
    int dim = 6;
    double precision = 0.01;

    MPI_Init(NULL, NULL);
    int rank;    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int inUse = rank < dim-2;
    MPI_Comm_split(MPI_COMM_WORLD, inUse, rank, &MPI_COMM_COMPUTE);

    int numProcs;
    MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);
    if (!inUse) {
        MPI_Finalize();
        exit(0);
    }
    
    int *elemsToCompute, *indices;
    double *workingArr;
    int size;
    if (rank == 0) {
        elemsToCompute = malloc(sizeof(int) * numProcs);
        indices = malloc(sizeof(int) * numProcs);
        
        // read array from file
        double arr[36] = {1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1};
        workingArr = initMasterProc(&arr[0], elemsToCompute, indices, dim, precision, &size);
    } else {
        MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_COMPUTE, MPI_STATUS_IGNORE);
        workingArr = initSlaveProc(dim, precision, size);
    }

    relax(workingArr, size, dim, precision, numProcs, rank);
    printInfo(workingArr, size, dim, rank);

    double *finalArr; 
    if (rank == 0) {
        finalArr = malloc(sizeof(double) * dim * dim);
    }
    
    int computeSize = size - (dim*2);
    MPI_Gatherv(&workingArr[dim], computeSize, MPI_DOUBLE, &finalArr[dim],
            elemsToCompute, indices, MPI_DOUBLE, 0, MPI_COMM_COMPUTE);
    
    free(workingArr);
    if (rank == 0) {
        printInfo(finalArr, dim*dim, dim, rank);
        free(finalArr);
    }

    MPI_Finalize();
}
