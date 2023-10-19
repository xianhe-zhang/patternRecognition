CC = g++
CXXFLAGS=-std=c++17
LIBS = `pkg-config --cflags --libs opencv4`

main: main.cpp functions.h utils/csv_util.h
	mkdir -p database
	${CC} main.cpp functions.cpp utils/csv_util.cpp $(CXXFLAGS) -o ./main $(LIBS)

