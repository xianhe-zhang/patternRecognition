CC = g++
CXXFLAGS=-std=c++11 
LDFLAGS=`pkg-config --cflags --libs opencv4`
SRC = vidDisplay.cpp filter.cpp

# g++ tutorial.cpp -std=c++11 -o output_name `pkg-config --cflags --libs opencv4`

all: output/imgDisplay output/vidDisplay

output/imgDisplay: imgDisplay.cpp
	mkdir -p output
	g++ imgDisplay.cpp $(CXXFLAGS) -o output/imgDisplay $(LDFLAGS)

output/vidDisplay: ${SRC}
	mkdir -p output
	g++ ${SRC} $(CXXFLAGS) -o output/vidDisplay $(LDFLAGS)