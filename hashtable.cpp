#include "hashtable.hpp"
#include "iostream"

//======================================================
//constructor

HashTable::HashTable(int capacity)
{
	buckets = new list<Entry>[capacity]; //Array of lists of type Entries for Chaining

	this->capacity = capacity;
	this->size = 0; //assigning size as zero
	this->collisions = 0; //assigning collisions as zero
}

//======================================================
//Destructor

HashTable::~HashTable () 
{
	delete[] buckets;
}

//======================================================
//Hashcode method

unsigned long HashTable::hashCode(string key) 
{
	unsigned long h = 0;
	const int leftShift = 5;
	const int rightShift = 32-leftShift;
	int keyLength = key.length();

	for (int i=0; i<keyLength; i++) {
		h += (unsigned int) key[i];
		h = (h<<leftShift)|(h>>rightShift);
	}

	return ((h*7)+11) % capacity;
}

//======================================================
//insert method

void HashTable::insert(string key, Hotel* value)
{
	unsigned long keyIndex = hashCode(key);

	//if an empty slot with empty list is found, then input the value
	if (buckets[keyIndex].empty()) {
		buckets[keyIndex].push_back(Entry(key, value));
		size++;			
	}

	//if a list already exists in that index
	else {
		//if the existing key is the same as inserted, then update the value
		for (auto& entry : buckets[keyIndex]) {
			if (entry.key == key) {
				entry.value = value;
				//connecing with the BST
   				Node<string, Hotel*>* root = cityRecords.getRoot();
    			cityRecords.insert(root, value->getCity(), value);
				cout << "Existing record has been updated." <<endl;
				return;
			}
		}

		//if the key does not match to any of the existing keys in the list
		buckets[keyIndex].push_back(Entry(key, value));
		size++;
		collisions++;

	}
	//connecing with the BST
    Node<string, Hotel*>* root = cityRecords.getRoot();
    cityRecords.insert(root, value->getCity(), value);

    cout << "New record has been successfully added to the Database."<<endl;
}

//======================================================
//Find method

Hotel* HashTable::find(string key)
{
	unsigned long keyIndex = hashCode(key);
	//if the index is empty
	if (buckets[keyIndex].empty()) {
		cout << "Record not found..!" <<endl;			
	}
	//iterating through the list of the specific index
	else {
		int cnt = 0;
		for (auto& entry : buckets[keyIndex]) {
			cnt++;
			if (entry.key == key) {
				cout << "Record found after " << cnt<<" comparision(s)" << endl;
				return entry.value;
			}
			else {
				cout <<"Record not found..!" <<endl;
			}
		}
	}
}

//======================================================
//FindAll method

void HashTable::findAll(string city,string stars)
{
	cityRecords.find(cityRecords.getRoot(), city, stars); //calling find method from bst


	// unsigned long keyIndex = hashCode(city);
	// //if the index is empty
	// if (buckets[keyIndex].empty()) {
	// 	cout<< "Record not found..!" <<endl;			
	// }
	// //iterating through the list of the specific index
	// else {
	// 	for (auto& entry : buckets[keyIndex]) {
	// 		if (entry.key == city) {
	// 			entry.value; //->print(stars);
	// 			return;
	// 		}
	// 	}
	// }
	// cout<< "Record not found..!"<<endl;	
}

//======================================================
//erase method
void HashTable::erase(string key)
{

	unsigned long keyIndex = hashCode(key);
	for (auto it = buckets[keyIndex].begin(); it != buckets[keyIndex].end(); it++) { //iterating through the list to match key
		if (it->key == key) {
		    Node<string, Hotel*> *root = cityRecords.getRoot();
    		cityRecords.remove(root, it->value->getCity(), it->value);
    		it = buckets[keyIndex].erase(it); //delete if the key is same;
			size --; //decrease the size
			cout << key <<" has been successfully deleted." <<endl;
			break;
		}
		else {
			cout << "Record not found...!" <<endl;
		}
	}

}

//======================================================
//getSize method
unsigned int HashTable::getSize()
{
	return size;
}
//======================================================
//getCollisions method
unsigned int HashTable::getCollisions()
{
	return collisions;
}

//======================================================
//dump method
void HashTable::dump(string path) 
{
	ofstream fout(path); //opening the file for writing
	fout <<"hotelName,cityName,stars,price,countryName,address" << endl;
	if (fout.is_open()) {
		for (int i=0; i<capacity; i++) {
			for(auto &entry : buckets[i]) {
				fout << entry.value->toString() << endl;
			}
		}
	}
	else {
		cout << "Sorry! Could not open the file.";
	}
	fout.close(); //closes the file stream
	cout << getSize() << " records has been successfully exported to " << path <<endl;
}




