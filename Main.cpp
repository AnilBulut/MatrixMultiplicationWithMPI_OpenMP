
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>


double* fillTwoDimensionalArray(double* matrix_A, int numberOfRowsPerTask,int totalSize, int upper, int lower, int taskid);
double* fillOneDimensionalArray(double* matrix_B, int numberOfRowsPerTask, int upper, int lower, int taskid);
void rowMatrixVectorMultiply(int n, double* a, double* b, double* x, MPI_Comm comm,int taskID);

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("You didn't enter the matrix size, please restart the program!\n");
        exit(0);
    }
    else if (argc == 2)
    {
        int	numberOfTasks, taskID; //this variables will be used in MPI.                              
        int matrixSize = atoi(argv[1]); //square matrix size which will come as input
       
        int rowsPerTask = 0; //initiliazer for rows per task

        /* this part is generated random elements for matrix A and matrix B
        each task generates random number in a range between ((1,2)* taskID)
        */
        int lower = 1, upper = 2;
        srand(time(0)); //seed for random number generator

        //initiliazing mpi library
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
        MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);

        double t1 = MPI_Wtime();
        //checking number of arguments to ensure program works correctly
        if (matrixSize < numberOfTasks)
        {
            printf("Matrix size cannot be lower than number of processors!");
            MPI_Finalize();
            exit(0);
        }
        else if (matrixSize % numberOfTasks != 0)
        {
            printf("Matrix size must be divisible without remainder with number of processors!");
            MPI_Finalize();
            exit(0);
        }



        rowsPerTask = matrixSize / numberOfTasks; //finding number of rows which are handled by each task
        //creating matrix A and B for each task
        double* matrix_A = (double*)malloc(rowsPerTask * matrixSize * sizeof(double)); //creating matrix a in the size of N*N
        double* matrix_B = (double*)malloc(rowsPerTask * sizeof(double)); //creating matrix in the size of N*1
        double* matrix_X = (double*)malloc(rowsPerTask * sizeof(double)); //creating matrix in the size of N*1
        matrix_A = fillTwoDimensionalArray(matrix_A, rowsPerTask, matrixSize, upper, lower, taskID); //filling matrix A with random numbers
        matrix_B = fillOneDimensionalArray(matrix_B, rowsPerTask, upper, lower, taskID); //filling matrix B with random numbers
        //MPI_Barrier(MPI_COMM_WORLD); // I've used MPI_Barriare here to ensure all tasks created their random rows before multiply them

       
        rowMatrixVectorMultiply(matrixSize, matrix_A, matrix_B, matrix_X, MPI_COMM_WORLD, taskID); //multiplying created matrixes

        double t2 = MPI_Wtime();
        printf("Elapsed time is %f\n", t2 - t1);
        MPI_Finalize();
    }
    else
    {
        printf("Wrong number of arguments, you should just enter the matrix size!\n");
        exit(0);
    }
    return 0;
   
}

double* fillTwoDimensionalArray(double* matrix_A, int numberOfRowsPerTask, int totalSize,int upper, int lower, int taskid)
{
    size_t i;
    size_t j;

    
    for (i = 0; i < numberOfRowsPerTask; i++)
    {
        for (j = 0; j < totalSize; j++)
        {
            
            double rand_num = (rand() % (upper - lower + 1)) + upper;                  
            rand_num = rand_num*(taskid+1);          
            *(matrix_A + i * totalSize + j) = rand_num;         
        }

    }
    
    return matrix_A;
}
double* fillOneDimensionalArray(double* matrix_B, int numberOfRowsPerTask, int upper, int lower, int taskid)
{
    int i;
    for (i = 0; i < numberOfRowsPerTask; i++)
    {
        double rand_num = (rand() % (upper - lower + 1)) + upper;
        rand_num = rand_num * (taskid + 1);
         *(matrix_B + i) = rand_num;
    }
    return matrix_B;
}

void rowMatrixVectorMultiply(int n, double* a, double* b, double* x,MPI_Comm comm, int taskID)
{
    size_t i, j;
    int nlocal; /* Number of locally stored rows of A */
    double* fb; /* Will point to a buffer that stores the entire vector b */
    int npes, myrank;
    MPI_Status status;

    /* Get information about the communicator */
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    /* Allocate the memory that will store the entire vector b */
    fb = (double*)malloc(n * sizeof(double));

    nlocal = n / npes;

    /* Gather the entire vector b on each processor using MPI's ALLGATHER op. */
    MPI_Allgather(b, nlocal, MPI_DOUBLE, fb, nlocal, MPI_DOUBLE, comm);

    /* Perform the matrix-vector mult. involving the locally stored submatrix */
    for (i = 0; i < nlocal; i++) {
        x[i] = 0.0;
        for (j = 0; j < n; j++)
            x[i] += a[i * n + j] * fb[j];
    }
 
    free(fb);
}