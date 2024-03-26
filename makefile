# output: main.o linkedlist.o polycalculator.o
# 	g++ main.o linkedlist.o polycalculator.o -o output

# main.o: main.cpp
# 	g++ -c main.cpp

# linkedlist.o: linkedlist.cpp linkedlist.h
# 	g++ -c linkedlist.cpp

# polycalculator.o: polycalculator.cpp polycalculator.h
# 	g++ -c polycalculator.cpp

# clean:
# 	rm polycalculator.o linkedlist.o main.o output



hotelFinder: main.o hashtable.o hotel.o 
	g++ main.o hashtable.o hotel.o -o hotelFinder

main.o: main.cpp
	g++ -c main.cpp

hashtable.o: hashtable.cpp hashtable.hpp bst.hpp
	g++ -c hashtable.cpp

# bst.o: bst.hpp
# 	g++ -c bst.hpp

hotel.o: hotel.cpp hotel.hpp
	g++ -c hotel.cpp

clean:
	rm bst.o hashtable.o hotel.o main.o

