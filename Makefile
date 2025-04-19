# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )
CC := gcc
NVCC := nvcc

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)
OPT := -g -std=c11 -O3 -Wall -Wextra -lm -march=native -funroll-loops
NV_OPT := -O3 --gpu-architecture=sm_89 -m64

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
LIB_FOLDER := lib
INCLUDE_FOLDER := include
BATCH_OUT_FOLDER := outputs

# Find all .c files in the lib directory
LIB_SOURCES := $(wildcard $(LIB_FOLDER)/*.c)
# Create corresponding object file paths
LIB_OBJECTS := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SOURCES))


# Get all of the main c files
MAIN_SOURCES := $(wildcard $(SRC_FOLDER)/*.c)
# Get all of the main cu files
MAIN_CUDA_SOURCES := $(wildcard $(SRC_FOLDER)/*.cu)
# Create corresponding binary file paths
EXECUTABLES := $(patsubst $(SRC_FOLDER)/%.c, $(BIN_FOLDER)/%, $(MAIN_SOURCES))
# Create it also for cuda
CUDA_EXECUTABLES := $(patsubst $(SRC_FOLDER)/%.cu, $(BIN_FOLDER)/%.exec, $(MAIN_CUDA_SOURCES))

# Default target
all: $(EXECUTABLES) $(CUDA_EXECUTABLES)

# Rule for creating executables
$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.c $(LIB_OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	$(CC) $< $(LIB_OBJECTS) -o $@ $(OPT) -fopenmp

# Rule for creating object files
$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) -c $< -o $@ $(OPT)

# Rule for compiling cuda
$(BIN_FOLDER)/%.exec: $(SRC_FOLDER)/%.cu $(LIB_OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@bash -c "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && $(NVCC) $< -o $@ $(NV_OPT)"

# Create necessary directories
directories:
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)

# Clean target
clean:
	rm -rf $(BIN_FOLDER)
	rm -rf $(OBJ_FOLDER)

clean_batch_outputs:
	rm -f $(BATCH_OUT_FOLDER)/*