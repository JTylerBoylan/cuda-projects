# Makefile

# Compiler
NVCC = nvcc

# Target executable name
TARGET = output

# Source file
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
