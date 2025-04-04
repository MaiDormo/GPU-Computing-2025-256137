# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )
CC := gcc

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)
OPT := -g -std=c11 -O3 -Wall -Wextra -lm -march=native -funroll-loops

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
LIB_FOLDER := lib
BATCH_OUT_FOLDER := outputs

# Find all .c files in the lib directory
LIB_SOURCES := $(wildcard $(LIB_FOLDER)/*.c)
# Create corresponding object file paths
LIB_OBJECTS := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SOURCES))


# Get all of the main files
MAIN_SOURCES := $(wildcard $(SRC_FOLDER)/*.c)
# Create corresponding binary file paths
EXECUTABLES := $(patsubst $(SRC_FOLDER)/%.c, $(BIN_FOLDER)/%, $(MAIN_SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule for creating executables
$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.c $(LIB_OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	$(CC) $< $(LIB_OBJECTS) -o $@ $(OPT) -fopenmp

# Rule for creating object files
$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) -c $< -o $@ $(OPT)

# Create necessary directories
directories:
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)

# Clean target
clean:
	rm -rf $(BIN_FOLDER)
	rm -rf $(OBJ_FOLDER)

clean_batch_outputs:
	rm -f $(BATCH_OUT_FOLDER)/*
