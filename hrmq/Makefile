CC=g++-12
#CFLAGS=-std=c++11 -O3 -DNDEBUG -march=native
CFLAGS=-std=c++11 -O3 -DNDEBUG -Ofast -march=znver3 -mtune=znver3 -ftree-vectorize

all: index

Basic_rmq.o: includes/Basic_rmq.cpp
	$(CC) $(CFLAGS) -c includes/Basic_rmq.cpp

RMQRMM64.o: RMQRMM64.cpp
	$(CC) $(CFLAGS) -c RMQRMM64.cpp

index: Basic_rmq.o RMQRMM64.o
	ar rc rmqrmmBP.a Basic_rmq.o RMQRMM64.o

test: 
	@$(CC) $(CFLAGS) rmqrmmBP.cpp -o rmqrmmBP rmqrmmBP.a 

clean:
	-rm *~ *.o *.bak 
cleanall:
	-rm *~ *.o *.bak *.a
