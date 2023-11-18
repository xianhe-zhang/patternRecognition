CC = g++
CXXFLAGS=-std=c++11 
SRC = main.cpp functions.cpp
LIBS = `pkg-config --cflags --libs opencv4`

main: main.cpp functions.h
	${CC} ${SRC} $(CXXFLAGS) -o ./main $(LIBS)

