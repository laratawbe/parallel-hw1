#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255
#define ROWS_PER_BLOCK 20

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;

    double z_real2, z_imag2, lengthsq;

    int iter = 0;
    while (iter < MAX_ITER) {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq =  z_real2 + z_imag2;
        iter++;

        if (lengthsq >= 4.0) {
            break;
        }
    }

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int trials = 10;
    double total_elapsed = 0;
    int rank, size;
    int flag = 0;
    double total_comm = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int trial = 0; trial < trials; trial++) {
        double start_time = MPI_Wtime();
        double comm_time = 0;

        if (rank == 0) {
            int buffer[ROWS_PER_BLOCK][WIDTH];
            MPI_Status status;
            int rows = 0;
            int count = 0;
            struct complex c;
            int image[HEIGHT][WIDTH];
            MPI_Request request;

            // Computation phase
            for (int i = 1; i < size; i++, rows += ROWS_PER_BLOCK) {
                MPI_Isend(&rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                count += ROWS_PER_BLOCK;
            }

            while (count > 0) {
                MPI_Irecv(image + rows, ROWS_PER_BLOCK * WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request); // Receive 2
                double recv_start = MPI_Wtime();
                MPI_Wait(&request, &status);
                double recv_end = MPI_Wtime();
                comm_time += recv_end - recv_start;
                count -= ROWS_PER_BLOCK;
                if (rows < HEIGHT - ROWS_PER_BLOCK) {
                    MPI_Isend(&rows, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
                    double send_start = MPI_Wtime();
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    double send_end = MPI_Wtime();
                    comm_time += send_end - send_start;
                    count += ROWS_PER_BLOCK;
                    rows += ROWS_PER_BLOCK;
                } else {
                    MPI_Isend(&rows, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &request);
                    double send_start = MPI_Wtime();
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    double send_end = MPI_Wtime();
                    comm_time += send_end - send_start;
                    count -= ROWS_PER_BLOCK;
                }
            }

            save_pgm("dynamic.pgm", image);
        } else {
            // Communication phase
            int r;
            MPI_Request request;
            MPI_Status status;
            while (1) {
                MPI_Irecv(&r, 1, MPI_INT, 0, MPI_ANY_TAG , MPI_COMM_WORLD, &request);
                double recv_start = MPI_Wtime();
                MPI_Wait(&request, &status);
                double recv_end = MPI_Wtime();
                comm_time += recv_end - recv_start;
                if (status.MPI_TAG == 1 || flag == 1) {
                    break;
                }
                int buffer[ROWS_PER_BLOCK][WIDTH];
                struct complex c;
                for (int x = 0; x < WIDTH; x++) {
                    for (int y = r; y < r + ROWS_PER_BLOCK && y < HEIGHT; y++) {
                        c.real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
                        c.imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
                        buffer[y - r][x] = cal_pixel(c);
                    }
                }
                MPI_Isend(buffer, ROWS_PER_BLOCK * WIDTH, MPI_INT, 0, rank, MPI_COMM_WORLD, &request);
                double send_start = MPI_Wtime();
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                double send_end = MPI_Wtime();
                comm_time += send_end - send_start;
            }
        }

        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        total_comm += comm_time;

        if (rank == 0) {
            printf("Trial %d: Elapsed time: %f seconds\n", trial + 1, elapsed_time);
            printf("Trial %d: Communication time: %f seconds\n", trial + 1, comm_time);
        }

        total_elapsed += elapsed_time;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double avg_elapsed = (total_elapsed) / trials * 1000;
    double avg_comm = (total_comm) / trials * 1000;

    if (rank == 0) {
        printf("Average Elapsed Time for %d trials: %f ms\n", trials, avg_elapsed);
        printf("Average Communication Time for %d trials: %f ms\n", trials, avg_comm);
    }

    MPI_Finalize();
    return 0;
}
