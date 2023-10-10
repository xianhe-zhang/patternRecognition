CC = g++
CXXFLAGS=-std=c++11 
# SRC = ../util/csv_util.cpp ../util/histogram_util.cpp match_functions.cpp matchimages.cpp
SRC = build_db/build.cpp build_db/build_helper.cpp utils/utils.cpp utils/csv_util.cpp
M_SRC = match_pics/matching.cpp match_pics/match_helper.cpp build_db/build.cpp build_db/build_helper.cpp utils/utils.cpp utils/csv_util.cpp 
LIBS = `pkg-config --cflags --libs opencv4`
DEPS = utils/utils.h utils/csv_util.h build_db/build_helper.h
m_DEPS = match_pics/match_helper.h utils/utils.h utils/csv_util.h build_db/build_helper.h


all: build match
build: build_db/build.cpp ${DEPS}
	rm -rf csv_files
	mkdir -p csv_files
	touch csv_files/baseline.csv
	touch csv_files/histogram.csv
	touch csv_files/top_histogram.csv
	touch csv_files/down_histogram.csv
	touch csv_files/color_histogram.csv
	touch csv_files/magnitude_histogram.csv
	touch csv_files/hsv.csv
	mkdir -p output
	${CC} ${SRC} $(CXXFLAGS) -o output/build $(LIBS)


match: match_pics/matching.cpp ${m_DEPS}
	${CC} ${M_SRC} $(CXXFLAGS) -o output/match $(LIBS)

