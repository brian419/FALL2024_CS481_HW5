// added this for reference for working code for hw4.
/* File:     hw4_kamal.c
 *Name: Md Kamal Hossain Chowdhury
 *
 * Homework #:4
 *
 *
 * Compile:  mpicc -g -Wall -o hw4_kamal hw4_kamal.c
 * Run:      mpiexec -n <comm_sz> ./hw4_kamal 4
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define DIES 0
#define ALIVE 1

void Check_for_error(int local_ok, char fname[], char message[],
                     MPI_Comm comm);

void Allocate_vectors(int **local_x_pp, int **local_y_pp,
                      int **local_z_pp, int local_n, MPI_Comm comm);
void Allocate_vectors_for_life(int **local_x_pp, int local_n, int n, MPI_Comm comm);
void Read_vector(int local_a[], int *counts, int *displs, int n, char vec_name[],
                 int my_rank, MPI_Comm comm);
void Print_vector(int local_b[], int *counts, int *displs, int n, char title[],
                  int my_rank, MPI_Comm comm);
void Parallel_vector_sum(int local_x[], int local_y[],
                         int local_z[], int local_n);
void Print_local_vector(int local_b[], int *counts, char title[], int my_rank, MPI_Comm comm);
void compute_local(int local_x[], int n, int counts[], int my_rank, int comm_sz, MPI_Comm comm);

double gettime(void)
{
    struct timeval tval;

    gettimeofday(&tval, NULL);

    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}
void printarray(int **a, int row, int col, int k)
{
    int i, j;
    printf("Life after %d iterations:\n", k);
    for (i = 1; i < row + 1; i++)
    {
        for (j = 1; j < col + 1; j++)
            printf("%d ", a[i][j]);
        printf("\n");
    }
    printf("\n");
}
void printarray_1d(int *a, int N, int k)
{
    int i, j;
    printf("Life after %d iterations:\n", k);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", a[i * N + j]);
        printf("\n");
    }
    printf("\n");
}
int **allocarray(int P, int Q)
{
    int i, *p, **a;

    p = (int *)malloc(P * Q * sizeof(int));
    a = (int **)malloc(P * sizeof(int *));
    for (i = 0; i < P; i++)
        a[i] = &p[i * Q];

    return a;
}

void freearray(int **a)
{
    free(&a[0][0]);
    free(a);
}

void print_life(int **life, int nRowsGhost, int nColsGhost, int my_rank, int local_n)
{
    int i, j;
    printf("\n----life print----\n");

    printf("my_rank=%d local_n=%d \n", my_rank, local_n);
    for (i = 0; i < nRowsGhost; i++)
    {
        for (j = 0; j < nColsGhost; j++)
        {

            printf(" %3d ", life[i][j]);
        }

        printf("\n");
    }
}

int compute(int **life, int **temp, int nRows, int nCols)
{
    int i, j, value, flag = 0;

    for (i = 1; i < nRows + 1; i++)
    {
        for (j = 1; j < nCols + 1; j++)
        {
            /* find out the value of the current cell */
            value = life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] + life[i][j - 1] + life[i][j + 1] + life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

            /* check if the cell dies or life is born */
            if (life[i][j])
            { // cell was alive in the earlier iteration
                if (value < 2 || value > 3)
                {
                    temp[i][j] = DIES;
                    flag++; // value changed
                }
                else                    // value must be 2 or 3, so no need to check explicitly
                    temp[i][j] = ALIVE; // no change
            }
            else
            { // cell was dead in the earlier iteration
                if (value == 3)
                {
                    temp[i][j] = ALIVE;
                    flag++; // value changed
                }
                else
                    temp[i][j] = DIES; // no change
            }
        }
    }

    return flag;
}

/*-------------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int n, local_n, i, remain;
    int comm_sz, my_rank, *counts;
    int *local_x;

    MPI_Comm comm;
    int *displs;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    n = atoi(argv[1]);
    // NTIMES = atoi(argv[2]);
    // n=4;

    /* compute counts and displacements */
    counts = (int *)malloc(comm_sz * sizeof(int));
    displs = (int *)malloc(comm_sz * sizeof(int));
    remain = n % comm_sz;
    for (i = 0; i < comm_sz; i++)
    {
        counts[i] = n / comm_sz + ((i < remain) ? 1 : 0);
        counts[i] = counts[i] * n;
#ifdef DEBUG3
        printf("counts[%d]=%d\n", i, counts[i]);
#endif
    }
    displs[0] = 0;
    for (i = 1; i < comm_sz; i++)
    {
        displs[i] = displs[i - 1] + counts[i - 1];
#ifdef DEBUG3
        printf("displs[%d]=%d\n", i, displs[i]);
#endif
    }
    local_n = counts[my_rank];

#ifdef DEBUG
    printf("Proc %d > n = %d, local_n = %d\n", my_rank, n, local_n);
#endif

    Allocate_vectors_for_life(&local_x, local_n, n, comm);

    Read_vector(local_x, counts, displs, n, "x", my_rank, comm);

    compute_local(local_x, n, counts, my_rank, comm_sz, comm);

    free(local_x);

    MPI_Finalize();

    return 0;
} /* main */

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so,
 *            print message and terminate all processes.  Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0
 *               otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error:  should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(
    int local_ok /* in */,
    char fname[] /* in */,
    char message[] /* in */,
    MPI_Comm comm /* in */)
{
    int ok;

    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0)
    {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0)
        {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname,
                    message);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
} /* Check_for_error */

/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x
 * In args:   local_n:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors_for_life(
    int **local_x_pp /* out */,
    int local_n /* in  */,
    int n /* in*/,
    MPI_Comm comm /* in  */)
{
    int local_ok = 1;
    char *fname = "Allocate_vectors";

    *local_x_pp = malloc(local_n * sizeof(int));

    if (*local_x_pp == NULL)
        local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)",
                    comm);
}
/* Allocate_vectors for x */

/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x, y, and z
 * In args:   local_n:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp, local_z_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(
    int **local_x_pp /* out */,
    int **local_y_pp /* out */,
    int **local_z_pp /* out */,
    int local_n /* in  */,
    MPI_Comm comm /* in  */)
{
    int local_ok = 1;
    char *fname = "Allocate_vectors";

    *local_x_pp = malloc(local_n * sizeof(int));
    *local_y_pp = malloc(local_n * sizeof(int));
    *local_z_pp = malloc(local_n * sizeof(int));

    if (*local_x_pp == NULL || *local_y_pp == NULL ||
        *local_z_pp == NULL)
        local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)",
                    comm);
} /* Allocate_vectors */

/*-------------------------------------------------------------------
 * Function:   Read_vector
 * Purpose:    Read a vector from stdin on process 0 and distribute
 *             among the processes using a block distribution.
 * In args:    local_n:  size of local vectors
 *             n:        size of global vector
 *             vec_name: name of vector being read (e.g., "x")
 *             my_rank:  calling process' rank in comm
 *             comm:     communicator containing calling processes
 * Out arg:    local_a:  local vector read
 *
 * Errors:     if the malloc on process 0 for temporary storage
 *             fails the program terminates
 *
 * Note:
 *    This function assumes a block distribution and the order
 *   of the vector evenly divisible by comm_sz.
 */
void Read_vector(
    int local_a[] /* out */,
    int counts[] /* in  */,
    int displs[] /* in  */,
    int n /* in  */,
    char vec_name[] /* in  */,
    int my_rank /* in  */,
    MPI_Comm comm /* in  */)
{

    int *a = NULL;
    int i, local_n;
    int local_ok = 1;
    char *fname = "Read_vector";

    local_n = counts[my_rank];
    if (my_rank == 0)
    {
        a = malloc(n * n * sizeof(int));
        if (a == NULL)
            local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector",
                        comm);
        // printf("Enter the vector %s\n", vec_name);
        // for (i = 0; i < n*n; i++)
        //   a[i]=i+1;
        /* Initialize the life matrix */
        for (i = 0; i < n * n; i++)
        {
            srand(54321 | i);
            if (drand48() < 0.5)
                a[i] = ALIVE;
            else
                a[i] = DIES;
        }
#ifdef DEBUG2
        printf("rank=%d\n", my_rank);
        /* Display the life matrix */
        printarray_1d(a, n, 0);
#endif
        // scanf("%lf", &a[i]);
        MPI_Scatterv(a, counts, displs, MPI_INT, local_a, local_n, MPI_INT, 0,
                     comm);
        free(a);
    }
    else
    {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector",
                        comm);
        MPI_Scatterv(a, counts, displs, MPI_INT, local_a, local_n, MPI_INT, 0, comm);
    }
} /* Read_vector */

void compute_local(
    int local_x[] /* in */,
    int n,
    int counts[] /* in  */,
    int my_rank /* in */,
    int comm_sz /* in */,
    MPI_Comm comm /* in */)
{

    // int* b = NULL;
    int i, j, local_n;
    // int nCols=5;
    // int local_ok = 1;
    // int N, NTIMES;
    int **life = NULL, **temp = NULL, **ptr;
    local_n = counts[my_rank];
    int nCols = n;
    int nRows = local_n / n;
    int upper_rank, down_rank;
    double t1, t2;
    int flag = 1, k;

    MPI_Status status;

    int nRowsGhost = nRows + 2;
    int nColsGhost = nCols + 2;
    life = allocarray(nRowsGhost, nColsGhost);
    temp = allocarray(nRowsGhost, nColsGhost);
    // printf("\n ---This is here n=%d rank=%d, local_n=%d nRowsGhost=%d nColsGhost=%d local_x[0]=%d\n",n,my_rank,local_n,nColsGhost,nColsGhost,local_x[0]);
    // printf(" \ndddddd  rank=%d local_x[0]=%d\n",my_rank,local_x[0]);

    int row = 0;
    int col = 0;
    // life[0][nRows+1]=5;
    // printf("\n--life[0][nRows+1]=%d\n",life[0][nRows+1]);
    /* Initialize the boundaries of the life matrix */
    for (i = 0; i < nRowsGhost; i++)
    {
        for (j = 0; j < nColsGhost; j++)
        {
            if (i == 0 || j == 0 || i == nRowsGhost - 1 || j == nColsGhost - 1)
            {
                life[i][j] = DIES;
                temp[i][j] = DIES;
            }
        }
    }
    // printf("\n--life[0][6]=%d\n",life[0][6]);

    for (i = 0; i < local_n; i++)
    {
        row = i / n;
        col = i % n;
        // give space for ghost cell
        row = row + 1;
        col = col + 1;
#ifdef DEBUG3
        printf("\n----local_x[%d]=%d-----", i, local_x[i]);
#endif
        life[row][col] = local_x[i];
#ifdef DEBUG3
        printf(" life[%d][%d]= %d \n", row, col, life[row][col]);
#endif
    }
#ifdef DEBUG2
    printf("rank=%d\n", my_rank);
    /* Display the life matrix */
    printarray(life, row, col, 0);
#endif

    upper_rank = my_rank + 1;
    if (upper_rank >= comm_sz)
        upper_rank = MPI_PROC_NULL;
    down_rank = my_rank - 1;
    if (down_rank < 0)
        down_rank = MPI_PROC_NULL;
    int NTIMES = 1000;
    if (my_rank == 0)
    {
        t1 = gettime();
    }
    /* Play the game of life for given number of iterations */
    for (k = 0; k < NTIMES; k++)
    {
        flag = 0;

        if ((my_rank % 2) == 0)
        {
            /* exchange up */
            MPI_Sendrecv(&(life[nRows][0]), nColsGhost, MPI_INT, upper_rank, 0,
                         &(life[nRows + 1][0]), nColsGhost, MPI_INT, upper_rank, 0,
                         comm, &status);
        }
        else
        {
            /* exchange down */
            MPI_Sendrecv(&(life[1][0]), nColsGhost, MPI_INT, down_rank, 0,
                         &(life[0][0]), nColsGhost, MPI_INT, down_rank, 0,
                         comm, &status);
        }

        /* Do the second set of exchanges */
        if ((my_rank % 2) == 1)
        {
            /* exchange up */
            MPI_Sendrecv(&(life[nRows][0]), nColsGhost, MPI_INT, upper_rank, 1,
                         &(life[nRows + 1][0]), nColsGhost, MPI_INT, upper_rank, 1,
                         comm, &status);
        }
        else
        {
            /* exchange down */
            MPI_Sendrecv(&(life[1][0]), nColsGhost, MPI_INT, down_rank, 1,
                         &(life[0][0]), nColsGhost, MPI_INT, down_rank, 1,
                         comm, &status);
        }

        // print_life(life,nRowsGhost,nColsGhost,my_rank,local_n);
        //   if(my_rank==0){

        // 	 //int** serial_life=life;

        //   }
        flag = compute(life, temp, nRows, nCols);

        // Each MPI process sends its rank to reduction, root MPI process collects the result
        int reduction_flag = 0;
        MPI_Allreduce(&flag, &reduction_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            if (!reduction_flag)
                printf("The sum of all flag is %d after k=%d.\n", reduction_flag, k);
        }
        if (!reduction_flag)
        {
            break;
        }

        MPI_Barrier(comm);
        /* copy the new values to the old array */
        ptr = life;
        life = temp;
        temp = ptr;

        // printf("\n----After change-----\n");
        // print_life(life,nRowsGhost,nColsGhost,my_rank,local_n);

        /* copy the new values to the old array */
        /*ptr = life;
        life = temp;
        temp = ptr;
        */
#ifdef DEBUG2
        /* Print no. of cells alive after the current iteration */
        printf("No. of cells whose value changed in iteration %d = %d\n", k + 1, flag);
        printf("rank=%d\n", my_rank);
        /* Display the life matrix */
        printarray(life, row, col, k + 1);
#endif
    }
    if (my_rank == 0)
    {
        t2 = gettime();

        printf("Time taken %f seconds for %d iterations\n", t2 - t1, k);
    }

    freearray(life);
    freearray(temp);
    // freearray(ptr);
} /* Print_life */
