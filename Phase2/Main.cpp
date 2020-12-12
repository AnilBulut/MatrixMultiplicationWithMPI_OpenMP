#define _CRT_SECURE_NO_DEPRECATE

#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double* fillTwoDimensionalArray(double* matrix_A,int matrixSize); //filling matrix A
double* fillOneDimensionalArray(double* matrix_B, int matrixSize); //filling matrix B
double* ParallelRowMatrixVectorMultiply(int matrixSize,double* matrixA, double* matrixB, double* matrix_X_Buffer, MPI_Comm comm);
double* SequentialMatrixVectorMultiply(int matrixSize, double* matrixA, double* matrixB, double* matrixResult);
double calculateL2Norm(int matrixSize,double* parallelResult,double* sequentialResult);
void writeFile(char resultString[], char fileName[]);

int main(int argc, char* argv[])
{
	int numOfThreads = omp_get_max_threads();
	printf("Threat Number:%d\n", numOfThreads);
	if (argc != 2)
	{
		printf("You didn't enter the matrix size, please restart the program!\n");
		exit(0);
	}
	else
	{
		double* matrix_A = NULL; //input matrix A
		double* matrix_B = NULL; //input matrix B
		int rank, size; //mpi parameters
		double startTimeMPI, endTimeMPI,timeDifferenceMPI; //measuring time for MPI
		double startTimeOpenMP, endTimeOpenMP, timeDifferenceOpenMP; //measuring time for MPI
		double averageTimeMPI = 0, maxTimeMPI = 0, minTimeMPI = 0;
		int matrixSize = atoi(argv[1]); //matrix size which will come as input
		int numberOfRowsPerProcessor = 0;


		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank); //ID of processor
		MPI_Comm_size(MPI_COMM_WORLD, &size); //amount of processor


		numberOfRowsPerProcessor = matrixSize / size; //calculating number of rows per each processor
		double* rowsOfMatrix_A = (double*)malloc(numberOfRowsPerProcessor * matrixSize * sizeof(double)); 		//allocate memory for each processor for scattering rows of matrix A
		double* matrix_X_Parallel = (double*)malloc(matrixSize * sizeof(double)); //Tthe result matrix will be used in parallel computing
		
		
		if (rank == 0) {
		    matrix_A = (double*)malloc(matrixSize * matrixSize * sizeof(double)); //creating matrix a in the size of N*N
			matrix_B = (double*)malloc(matrixSize * sizeof(double)); //creating matrix in the size of N*1
					
			matrix_A = fillTwoDimensionalArray(matrix_A, matrixSize); //filling matrix A with 1s
			matrix_B = fillOneDimensionalArray(matrix_B, matrixSize); //filling matrix B with 1s			
		}
		if (rank != 0)//allocationg space for matrix B which will be broadcasted by processor 0
		{
			matrix_B = (double*)malloc(matrixSize * sizeof(double));
		}
		MPI_Barrier(MPI_COMM_WORLD); 
		startTimeMPI = MPI_Wtime();
		// Scatter rows of  matrix A to different processes
		MPI_Scatter(matrix_A, numberOfRowsPerProcessor * matrixSize, MPI_DOUBLE, rowsOfMatrix_A, numberOfRowsPerProcessor * matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

		//each process make multiplication operation with the rows of A and entire B matrix
		//ParallelRowMatrixVectorMultiply() function returns a part of result matrix for each processor
		matrix_X_Parallel = ParallelRowMatrixVectorMultiply(matrixSize, rowsOfMatrix_A, matrix_B, matrix_X_Parallel, MPI_COMM_WORLD);
		
		endTimeMPI = MPI_Wtime();
		timeDifferenceMPI = endTimeMPI - startTimeMPI;

		MPI_Reduce(&timeDifferenceMPI, &maxTimeMPI, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&timeDifferenceMPI, &minTimeMPI, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&timeDifferenceMPI, &averageTimeMPI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			averageTimeMPI /= size;
			char resultStringMPITimes[100];
			snprintf(resultStringMPITimes, 100, "%d\t%d\t%.5f\t%.5f\t%.5f\n", matrixSize, size, minTimeMPI, maxTimeMPI, averageTimeMPI);
			char fileName1[] = "mpiTimes.txt";
			writeFile(resultStringMPITimes, fileName1);
		}
				
		if (rank == 0)
		{
			double* matrix_X_Sequential = (double*)malloc(matrixSize * sizeof(double)); //The result matrix will be used in sequential computing
			matrix_X_Sequential = SequentialMatrixVectorMultiply(matrixSize, matrix_A, matrix_B, matrix_X_Sequential);		

			MPI_Barrier(MPI_COMM_WORLD);
			startTimeOpenMP = omp_get_wtime();
			double L2Norm;
			for (int i = 0; i < 100.000; i++)
			{
				L2Norm = calculateL2Norm(matrixSize, matrix_X_Parallel, matrix_X_Sequential);
			}
				
				
			endTimeOpenMP = omp_get_wtime();
			timeDifferenceOpenMP = endTimeOpenMP - startTimeOpenMP;
			
		

			char resultStringGeneral[100];
			snprintf(resultStringGeneral, 100, "%d\t%d\t%d\t%.5f\t%.5f\t%.12e\n", matrixSize, size, numOfThreads, timeDifferenceMPI, timeDifferenceOpenMP, L2Norm);
			char fileName2[] = "generalResult.txt";
			writeFile(resultStringGeneral, fileName2);
		}
		
		
		MPI_Finalize();
	}
	
	return 0;
}


//matrix A is filling with 1s
double* fillTwoDimensionalArray(double* matrix_A, int matrixSize)
{
	size_t i;
	size_t j;

	for (i = 0; i < matrixSize; i++)
	{
		for (j = 0; j < matrixSize; j++)
		{
			*(matrix_A + i * matrixSize + j) = 1;
		}

	}

	return matrix_A;
}
//matrix B is filling with 1s 
double* fillOneDimensionalArray(double* matrix_B, int matrixSize)
{
	size_t i;
	for (i = 0; i < matrixSize; i++)
	{
		*(matrix_B + i) = 1;
	}
	return matrix_B;
}

double* ParallelRowMatrixVectorMultiply(int matrixSize, double* matrixA, double* matrixB, double* matrix_X_Parallel, MPI_Comm comm)
{
	size_t i, j;
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size); //amount of processor
	int numberOfRowPerProcessors = matrixSize / size;

	//broadcast b to all other processors
	MPI_Bcast(matrixB, matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD); //to make sure all processes got  entire matrix B

	double* matrix_X_Buffer_Parallel = (double*)malloc(numberOfRowPerProcessors * sizeof(double)); //the result matrix part in each processor which will be gathered in result matrix
	
	
	for (i = 0; i < numberOfRowPerProcessors; i++) {
		matrix_X_Buffer_Parallel[i] = 0.0;
		for (j = 0; j < matrixSize; j++)
			matrix_X_Buffer_Parallel[i] += matrixA[i * matrixSize + j] * matrixB[j];
	}

	//gathering part of result matrix to 0.processor to build entire result matrix
	MPI_Gather(matrix_X_Buffer_Parallel, numberOfRowPerProcessors, MPI_DOUBLE, matrix_X_Parallel, numberOfRowPerProcessors, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	return matrix_X_Parallel;
}

double* SequentialMatrixVectorMultiply(int matrixSize, double* matrixA, double* matrixB, double* matrixResult)
{
	size_t i, j;
	for (i = 0; i < matrixSize; i++) {
		matrixResult[i] = 0.0;
		for (j = 0; j < matrixSize; j++) {
			size_t index = i * matrixSize + j;
			matrixResult[i] += matrixA[index] * matrixB[j];
		}
	}
	return matrixResult;
}

double calculateL2Norm(int matrixSize, double* parallelResult, double* sequentialResult)
{
	int i;
	double* differenceMatrix = (double*)malloc(matrixSize * sizeof(double));
	double sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < matrixSize; i++)
	{
		differenceMatrix[i] = parallelResult[i] - sequentialResult[i];
		sum += differenceMatrix[i] * differenceMatrix[i];
	}
	return sqrt(sum);
}

void writeFile(char resultString[], char fileName[])
{
	FILE* filePointer;
	filePointer = fopen(fileName, "a");
	fputs(resultString, filePointer);
	fclose(filePointer);
}