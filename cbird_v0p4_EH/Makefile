CXX = g++
CXXFlags = -Wall -Wextra -pedantic -g -O4 -DNDEBUG

EXEC =	cbird
SRC = $(wildcard *.cpp)
OBJ = $(SRC: .cpp=.o) 
STD = -std=c++14
GSL = -lm -lgsl -lgslcblas
CUBA = -L/Users/xflorian/Downloads/Cuba-4.2 -lcuba -lm
FFTW = -lfftw3


all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $^ -o $@ $(STD) $(GSL) $(FFTW) $(CUBA)
	
%.o: %.cpp
	$(CXX) $(CXXFlags) -o $@ -c $^ 

.PHONY: clean

clean:
	rm $(EXEC) *~

