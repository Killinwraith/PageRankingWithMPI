/*
 * ECE 420 Lab 4 – Distributed PageRank with MPI
 *
 * === Design ===
 *
 * Partitioning
 *   Each process owns a contiguous stripe of nodes [local_start, local_end).
 *   node_init() is called with those bounds so each process only allocates
 *   memory for its own stripe.
 *
 * Global out-degree array
 *   inlinks[k] for a node in our stripe can point to ANY node in the graph,
 *   so we need num_out_links for every node, not just our stripe.
 *   We build a small int array `out_deg[nodecount]` on every process by
 *   reading the meta file — this is cheap (one integer per node) and avoids
 *   the need to load the full graph on every process.
 *
 * Per-iteration communication
 *   Each process computes local_r[0..local_count) then one MPI_Allgatherv
 *   assembles the full r vector on every process.  No other communication.
 *
 * Convergence
 *   Every process has the full r and r_pre vectors after the gather, so
 *   rel_error() is computed locally — no extra reduction needed.
 *
 * Timing
 *   GET_TIME() brackets only the PageRank loop (not I/O).
 *   Only rank 0 saves the output.
 */

#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "Lab4_IO.h"
#include "timer.h"

#define EPSILON 1e-5
#define DAMPING_FACTOR 0.85

int main(void)
{
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ----------------------------------------------------------------
     * 1. Read node count from meta file
     * -------------------------------------------------------------- */
    int nodecount;
    {
        FILE *fp = fopen("data_input_meta", "r");
        if (!fp)
        {
            fprintf(stderr, "Rank %d: cannot open data_input_meta\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 253);
        }
        fscanf(fp, "%d\n", &nodecount);
        fclose(fp);
    }

    /* ----------------------------------------------------------------
     * 2. Build global out-degree array (all ranks, cheap)
     *    Reads the meta file once; stores one int per node.
     * -------------------------------------------------------------- */
    int *out_deg = malloc(nodecount * sizeof(int));
    if (!out_deg)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    {
        FILE *fp = fopen("data_input_meta", "r");
        int dummy_n;
        fscanf(fp, "%d\n", &dummy_n);
        for (int i = 0; i < nodecount; i++)
        {
            int id, nin, nout;
            fscanf(fp, "%d\t%d\t%d\n", &id, &nin, &nout);
            out_deg[i] = nout;
        }
        fclose(fp);
    }

    /* ----------------------------------------------------------------
     * 3. Stripe decomposition
     * -------------------------------------------------------------- */
    int *counts = malloc(nprocs * sizeof(int));
    int *displs = malloc(nprocs * sizeof(int));

    {
        int base = nodecount / nprocs;
        int rem = nodecount % nprocs;
        for (int p = 0; p < nprocs; p++)
        {
            counts[p] = base + (p < rem ? 1 : 0);
            displs[p] = (p == 0) ? 0 : displs[p - 1] + counts[p - 1];
        }
    }

    int local_start = displs[rank];
    int local_count = counts[rank];
    int local_end = local_start + local_count; /* exclusive */

    /* ----------------------------------------------------------------
     * 4. Load graph stripe for this process
     *    node_init(nodehead, start, end) loads nodes [start, end)
     * -------------------------------------------------------------- */
    struct node *nodehead;
    if (node_init(&nodehead, local_start, local_end))
    {
        fprintf(stderr, "Rank %d: node_init failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 254);
    }

    /* ----------------------------------------------------------------
     * 5. Allocate rank vectors
     *    r, r_pre  – full length (needed after Allgatherv and for rel_error)
     *    local_r   – only this process's slice (written into r via gather)
     * -------------------------------------------------------------- */
    double *r = malloc(nodecount * sizeof(double));
    double *r_pre = malloc(nodecount * sizeof(double));
    double *local_r = malloc(local_count * sizeof(double));
    if (!r || !r_pre || !local_r)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialise r(0) = 1/N */
    for (int i = 0; i < nodecount; i++)
        r[i] = 1.0 / nodecount;

    double const_term = (1.0 - DAMPING_FACTOR) / nodecount;

    /* ----------------------------------------------------------------
     * 6. PageRank iteration — timed
     * -------------------------------------------------------------- */
    double start, end;
    GET_TIME(start);

    int itercount = 0;
    do
    {
        ++itercount;

        /* Copy current r -> r_pre for convergence check */
        vec_cp(r, r_pre, nodecount);

        /* Compute updated ranks for our stripe */
        for (int li = 0; li < local_count; li++)
        {
            double sum = 0.0;
            for (int k = 0; k < nodehead[li].num_in_links; k++)
            {
                int src = nodehead[li].inlinks[k];
                /* r[src] is available — every process holds the full r.
                   out_deg[src] gives the out-degree of any node globally. */
                sum += r[src] / (double)out_deg[src];
            }
            local_r[li] = const_term + DAMPING_FACTOR * sum;
        }

        /* Assemble the full r on every process */
        MPI_Allgatherv(local_r, local_count, MPI_DOUBLE, r, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        /* Convergence check is local — no extra MPI call needed */
    } while (rel_error(r, r_pre, nodecount) >= EPSILON);

    GET_TIME(end);

    /* ----------------------------------------------------------------
     * 7. Save output (rank 0 only)
     * -------------------------------------------------------------- */
    if (rank == 0)
    {
        printf("Converged in %d iterations, time = %.6f s\n", itercount, end - start);
        Lab4_saveoutput(r, nodecount, end - start);
    }

    /* ----------------------------------------------------------------
     * 8. Clean up
     * -------------------------------------------------------------- */
    node_destroy(nodehead, local_count);
    free(r);
    free(r_pre);
    free(local_r);
    free(out_deg);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}