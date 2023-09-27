# Compiler
GCC = g++

# Include and Library paths
INCLUDE_PATH = -I/usr/local/include
LIBRARY_PATH = -L/usr/local/lib

# Libraries to link against
LIBRARIES = -lginac -lcln

# Source and Object files
SRC = kkt_solver.cpp
OBJ = main.o
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(GCC) $(LIBRARY_PATH) -o $@ $^ $(LIBRARIES)

$(OBJ): $(SRC)
	$(GCC) $(INCLUDE_PATH) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
