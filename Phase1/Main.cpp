#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double* fillTwoDimensionalArray(double* matrix_A, int numberOfRowsPerTask,int totalSize,int taskID); //filling matrix A
double* fillOneDimensionalArray(double* matrix_B, int numberOfRowsPerTask,int taskid); //filling matrix B
void rowMatrixVectorMultiply(int n, double* a, double* b, double* x, MPI_Comm comm,int taskID); //matrix multiplication
double calculateMemoryUsage(int numberOfRowsPerTask, int matrixSize); //calculating total used memory
void writeFile(char resultString[],char fileName[]); //writing results to txt file

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("You didn't enter the matrix size, please restart the program!\n");
        exit(0);
    }
    else if (argc == 2)
    {
        printf("Matrix Size:%d\n", atoi(argv[1]));
        int	numberOfTasks, taskID; //this variables will be used in MPI.                              
        int matrixSize = atoi(argv[1]); //square matrix size which will come as input
        double totalTime = 0 , totalMemory = 0;  //total elapsed time and total used memory
        int rowsPerTask = 0; //initiliazer for rows per task

        //initiliazing mpi library
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
        MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);

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

        matrix_A = fillTwoDimensionalArray(matrix_A, rowsPerTask, matrixSize, taskID); //filling matrix A with random numbers
        matrix_B = fillOneDimensionalArray(matrix_B, rowsPerTask, taskID); //filling matrix B with random numbers

        MPI_Barrier(MPI_COMM_WORLD);  //using barrier before multiplication
        double t1 = MPI_Wtime();
        rowMatrixVectorMultiply(matrixSize, matrix_A, matrix_B, matrix_X, MPI_COMM_WORLD, taskID); //multiplying created matrixes
        MPI_Barrier(MPI_COMM_WORLD); //using barrier after multiplication
        double t2 = MPI_Wtime();
        
        double result = t2 - t1;


        //since we've used barrier, only checking one task to find elapsed time
        //used memory calculation is same for all tasks
        if (taskID == 0)
        {
            totalTime = result;
            totalMemory = calculateMemoryUsage(rowsPerTask, matrixSize);
            totalMemory = totalMemory * numberOfTasks; //calculating total memory usage from all processor

            //preparing the result string which is going to be in txt file
            char resultString[100];
            snprintf(resultString, 100, "%d\t%d\t%.3f\t%.3f\n", matrixSize, numberOfTasks, totalTime, totalMemory);

            char fileName[] = "res.txt";
            writeFile(resultString, fileName);
        }

        MPI_Finalize();
        free(matrix_A);
        free(matrix_B);
        free(matrix_X);
    }
    else
    {
        printf("Wrong number of arguments, you should just enter the matrix size!\n");
        exit(0);
    }

    
    return 0;
   
}

//matrix A is filling with taskID+1 
double* fillTwoDimensionalArray(double* matrix_A, int numberOfRowsPerTask, int matrixSize, int taskID)
{
    size_t i;
    size_t j;

    for (i = 0; i < numberOfRowsPerTask; i++)
    {
        for (j = 0; j < matrixSize; j++)
        {
            *(matrix_A + i * matrixSize + j) = (taskID+1);
        }

    }
    
    return matrix_A;
}
//matrix B is filling with taskID+1 
double* fillOneDimensionalArray(double* matrix_B, int numberOfRowsPerTask,int taskID)
{
    int i;
    for (i = 0; i < numberOfRowsPerTask; i++)
    {
        *(matrix_B + i) = (taskID + 1);
    }
    return matrix_B;
}

void rowMatrixVectorMultiply(int n, double* a, double* b, double* x,MPI_Comm comm, int taskID)
{
    size_t i, j;
    int nlocal; /* Number of locally stored rows of A */
    double* fb; /* Will point to a buffer that stores the entire vector b */
    int npes, myrank;

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

double calculateMemoryUsage(int numberOfRowsPerTask, int matrixSize)
{
    unsigned long resultAsByte = 0;
    resultAsByte = numberOfRowsPerTask * matrixSize * sizeof(double); //matrix A
    resultAsByte = resultAsByte + numberOfRowsPerTask * sizeof(double); //matrix B
    resultAsByte = resultAsByte + numberOfRowsPerTask * sizeof(double); //matrix X
    double resultAsGB = (double)resultAsByte / (1024 * 1024 * 1024); //converting to GB
 
    return resultAsGB;
}

void writeFile(char resultString[], char fileName[])
{
    FILE* filePointer;
    filePointer = fopen(fileName, "a");
    fputs(resultString, filePointer);
    fclose(filePointer);
}