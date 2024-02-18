#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int *image, int width, int height) {
    FILE *pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", width, height);
    fprintf(pgmimg, "255\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            temp = image[i * width + j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *image = NULL;
    double AVG = 0;
    int N = 10;
    double total_time[N], total_communication_time[N];
    struct complex c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width_per_proc = WIDTH / size;
    int start_col = rank * width_per_proc;
    int *col = (int *)malloc(sizeof(int) * HEIGHT * width_per_proc);
    int *recv_buf = NULL;

    if (rank == 0) {
        image = (int *)malloc(sizeof(int) * HEIGHT * WIDTH);
        recv_buf = (int *)malloc(sizeof(int) * HEIGHT * WIDTH);
    }

    for (int k = 0; k < N; k++) {
        double start_time = MPI_Wtime();
        double start_communication_time = 0, end_communication_time = 0;

        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < width_per_proc; j++) {
                c.real = (start_col + j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                col[i * width_per_proc + j] = cal_pixel(c);
            }
        }

        MPI_Gather(col, HEIGHT * width_per_proc, MPI_INT, recv_buf, HEIGHT * width_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

        double end_time = MPI_Wtime();
        total_time[k] = end_time - start_time;
        total_communication_time[k] = end_communication_time - start_communication_time;
        printf("Execution time of trial [%d]: %f seconds\n", k, total_time[k]);
        printf("Communication time of trial [%d]: %f seconds\n", k, total_communication_time[k]);

        AVG += total_time[k];
    }

    if (rank == 0) {
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                image[i * WIDTH + j] = recv_buf[i * WIDTH + j];
            }
        }

        save_pgm("mandelbrot.pgm", image, WIDTH, HEIGHT);
        printf("The average execution time of %d trials is: %f seconds\n", N, AVG / N);
        free(image);
        free(recv_buf);
    }

    free(col);
    MPI_Finalize();

    return 0;
}
