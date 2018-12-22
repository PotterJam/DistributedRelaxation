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

void relaxation(double* arr, int dim, double precision) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int inUse = rank < dim-2;

	MPI_Comm_split(MPI_COMM_WORLD, inUse, rank, &MPI_COMM_COMPUTE);

	int numProcs;
	MPI_Comm_size(MPI_COMM_COMPUTE, &numProcs);

	if (inUse) {
		// PLAN:
		// Implement basic mpi, where rank 0 sends array to other
		// ranks while they wait for it, once everyone has their
		// arrays, they start processing. When settled, gather.
        int rows;
        int size;
        double *workingArr;
        
		if (rank == 0) {
    		int* rows = rowsPerProc(numProcs, dim);
    		int myRows = rows[rank]+2;
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

            // set rank 0 rows and arr
            rows = &rows[0];
            size = elemsToCompute[0] + (2*dim);
            workingArr = malloc(sizeof(double) * size);
            memcpy(workingArr, &arr[0], sizeof(double) * size);

            // send arrays to other processes
            MPI_Request req[numProcs-1];
            MPI_Status statuses[numProcs-1];
            for (int i = 1; i < numProcs; i++) {
                int index = indices[i];
                int procArrSize = elemsToCompute[i] + (2*dim);
                MPI_Send(&procArrSize, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Isend(&arr[index], procArrSize, MPI_DOUBLE, i, 0, 
                    MPI_COMM_WORLD, &req[i-1]);
            }
            MPI_Waitall(numProcs-1, req, statuses);
        } else 
        {
            MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            workingArr = malloc(sizeof(double) * size);
            MPI_Recv(workingArr, size, MPI_DOUBLE, 0, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // test everything gets their elements
        char strArr[size*5];
        char *s = malloc(sizeof(char) * 5);
        sprintf(s, "%.2f ", workingArr[0]);
        strcpy(strArr, s);
    
        for (int i = 1; i < size; i++) {
            sprintf(s, "%.2f ", workingArr[i]);
            strcat(strArr, s);
        }
        printf("Processor with rank: %d has a working array of: %s\n", rank, strArr);

	}
}

int main() {
    double arr[25] = {1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1};
    MPI_Init(NULL, NULL);
    relaxation(&arr[0], 5, 0);
    MPI_Finalize();
}
