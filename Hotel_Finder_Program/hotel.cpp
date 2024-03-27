#include "hotel.hpp"
#include <iostream>
#include <iomanip>
#include <string>

//======================================================
//constructor

Hotel::Hotel(string name,string city, string stars,string price,string country,string address) 
{
	this->name = name;
	this->city = city;
	this->stars = stars;
	this->price = price;
	this->country = country;
	this->address = address;
}

//======================================================
//getname method

string Hotel::getName()
{
	return name;
}

//======================================================
//getCity method

string Hotel::getCity()
{
	return city;
}

//======================================================
//getStars method

string Hotel::getStars()
{
	return stars;
}

//======================================================
//getPrice method

string Hotel::getPrice()
{
	return price;
}

//======================================================
//getCountry method

string Hotel::getCountry()
{
	return country;
}

//======================================================
//getAddress method

string Hotel::getAddress()
{
	return address;
}

//======================================================
//toString method

string Hotel::toString()
{
	return name + "," + city + "," + stars + "," + price + "," + country + "," + address;
	
}

//======================================================
//print method

void Hotel::print()
{
	
    cout << "Name     : " << name << endl;
    cout << "City     : " << city << endl;
    cout << "Stars    : " << stars << endl;
    cout << "Price    : " << price << " $/nigh" <<endl;
    cout << "Country  : " << country << endl;
    cout << "Address  : " << address << endl;
}
//======================================================
//Print2 method

void Hotel::print2()
{
	cout << std::left << setw(50) << name << setw(22) << city << setw(20) << stars << setw(20) << price << setw(30) << country << address << endl; 
}

//======================================================
