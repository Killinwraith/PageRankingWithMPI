# ECE 420 Lab 4 – MPI PageRank
# No optimisation flags per submission requirements.

CC      = mpicc
CFLAGS  = -Wall -std=c99
LDFLAGS = -lm

all: main

main: main.o Lab4_IO.o
	$(CC) $(CFLAGS) -o main main.o Lab4_IO.o $(LDFLAGS)

main.o: main.c Lab4_IO.h timer.h
	$(CC) $(CFLAGS) -c main.c

Lab4_IO.o: Lab4_IO.c Lab4_IO.h
	$(CC) $(CFLAGS) -c Lab4_IO.c

clean:
	rm -f *.o main