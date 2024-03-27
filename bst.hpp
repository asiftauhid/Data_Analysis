#pragma once
#include<iostream>
#include<cstdlib>
#include<sstream>
#include<iomanip>
#include<math.h>
#include<queue>
#include<vector>

using namespace std;
template <typename T1, typename T2>
class Node
{
	private:
		T1 key;					//city name
		vector<T2> value;		//list of hotels in the city
		Node<T1,T2>* left;		//left child
		Node<T1,T2>* right;		//right child
		Node<T1,T2>* parent;	//pointer to the parent node
	public:
		Node(T1 key): key(key),left(nullptr),right(nullptr), parent(nullptr) 
		{}
		void print(string stars="")	//print all or only those hotels with specific stars.
		{
			int counter=0;
			cout<<std::left<<setw(50)<<"Name"<<" "<<setw(20)<<"City"<<" "<<setw(20)<<"Stars"<<" "<<setw(20)<<"Price"<<" "<<setw(25)<<"Country"<<" "<<"Address"<<endl;
			cout<<"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"<<endl;

			for(auto it:value) {
				if(stars=="" or stars==it->getStars())
				{
					it->print2();
					counter++;
				}
			}
			
			cout<<"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"<<endl;
			cout<<counter<<" records found."<<endl;
		}

		template<typename,typename> friend class BST;
};
//=============================================================================
template <typename T1, typename T2>
class BST
{
	private:
		Node<T1,T2> *root;
	public:
		BST();									//constructor
		~BST();									//destructor
		Node<T1,T2>* getRoot();						//returns the root of the Tree
		void insert(Node<T1,T2>* ptr,T1 key, T2 value);			//Insert key into tree/subtree with root ptr
		int height(Node<T1,T2> *ptr);				    //Find the height of a tree/subtree with root ptr
		Node<T1,T2>* find(Node<T1,T2> *ptr,T1 key,string stars);		//Find and returns the node with key
		Node<T1,T2>* findMin(Node<T1,T2> *ptr);				//Find and return the Node<T1,T2> with minimum key value from a tree/subtree with root ptr
		void remove(Node<T1,T2> *ptr,T1 key, T2 value);			//Remove a node with key from the tree/subtree with root

		//helper methods
		void clearTree(Node<T1,T2> *ptr);
		void insertAtExternal(Node<T1, T2>* ptr, T1 key, T2 value);
		void removeAboveExternal(Node<T1,T2> *external);

};
//=====================================================================
//write implementation of all methods below

//======================================================
//constructor

template <typename T1, typename T2>
BST<T1, T2>::BST() 
{
	this->root = nullptr;
}

//======================================================
//desdtructor

template <typename T1, typename T2>
BST<T1, T2>::~BST() 
{
	clearTree(root);
}

///////////////////////
//helper function clearTree
template <typename T1, typename T2>
void BST<T1, T2>::clearTree(Node<T1, T2> *ptr) {
	if (ptr != nullptr) { 
		clearTree(ptr->left); //recursively delete the left tree of ptr
		clearTree(ptr->right); //recursively delete the right tree of ptr
		delete ptr; //deleting the root
	}
}


//======================================================
//getRoot method

template <typename T1, typename T2>
Node<T1,T2>* BST<T1, T2>::getRoot() 
{
	return root;
}

//======================================================
//insert method

template <typename T1, typename T2>
void BST<T1, T2>::insert(Node<T1,T2>* ptr,T1 key, T2 value) 
{
	//When there is no element in the tree
	if (ptr == nullptr) {
		root = new Node<T1, T2>(key);
		root->value.push_back(value);
		return;
	}

	//When the key is less the the root
	if (key < ptr->key) {
		//when there is no left children of ptr
		if (ptr->left == nullptr) {
			insertAtExternal(ptr, key, value);
		}
		//when there is a left child of ptr
		else {
			insert(ptr->left, key, value);
		}
	}

	//when the key is bigger than the root
	else if (key > ptr->key) {
		//when there is no right children of ptr
		if (ptr->right == nullptr) {
			insertAtExternal(ptr,key,value);
		}
		//when there is a right child of ptr
		else {
			insert(ptr->right, key, value);
		}
	}

	//key is equal to the root, key already exists
	else if (key == ptr->key){
		for (int i=0; i< ptr->value.size(); i++) {
			if(ptr->value[i]->getName() == value->getName()){
				ptr->value[i] = value;
				return;
			}
		}
		ptr->value.push_back(value);
	}
}

//////////////////////////////////
//helper function insertAtExternal
template <typename T1, typename T2>
void BST<T1, T2>::insertAtExternal(Node<T1, T2>* ptr, T1 key, T2 value)
{
	Node<T1, T2> *newNode = new Node<T1, T2>(key); //creating a new Node for the external node
	newNode->parent = ptr; //assigning ptr as the parent node of the newNode
	newNode->value.push_back(value); //storing the value in the newNode

	//when the key is smaller then the ptr
	if (key < ptr->key) {
		ptr->left = newNode;
	}
	//when the key is bigger than the ptr
	else {
		ptr->right = newNode;
	}
}

//======================================================
//height method
template <typename T1, typename T2>
int BST<T1, T2>::height(Node<T1, T2>* ptr)
{	
	//when there is no root
	if (ptr == nullptr) {
		return 0;
	}

	//when there exist a root
	//iterate through the left trees
	int leftTreeHeight = height(ptr->left); //recursing through the left subtree

	//interate through the right trees
	int rightTreHeight = height(ptr->right); //recusring through the right subtree

	return 1 + max(leftTreeHeight, rightTreHeight);
}

//======================================================
//find method
template <typename T1, typename T2>
Node<T1,T2>* BST<T1, T2>::find(Node<T1,T2> *ptr,T1 key,string stars)
{
	//when there is no root
	if (ptr == nullptr) {
		return nullptr;
	}

	//when key is less than ptr
	if (key < ptr->key) {
		find(ptr->left, key, stars); //recursively applying find method in the left subtree
	}

	//when key bigger than ptr
	if (key > ptr->key) {
		find(ptr->right, key, stars); //recursively applying find method in the right subtree
	}

	//when the key is equal to the ptr
	else if(key == ptr->key){
		ptr->print(stars);
		return ptr;
	}

}

//======================================================
//findMin method
template <typename T1, typename T2>
Node<T1,T2>* BST<T1, T2>::findMin(Node<T1,T2> *ptr)
{
	//when there is no root
	if (ptr == nullptr) {
		return nullptr;
	}

	while (ptr->left != nullptr) { //iterating through the left subtree
		ptr = ptr->left; //whenever the last element is found
	}
	return ptr; //it's returned
}

//======================================================
//remove method
template <typename T1, typename T2>
void BST<T1, T2>::remove(Node<T1,T2>* ptr,T1 key, T2 value) 
{
	if (ptr == nullptr) {
		return;
	}

	//key is smaller than the ptrs
	if (key < ptr->key) {
		remove(ptr->left, key, value); //recursively finding the key on left subtree to remove
	}
	else if (key > ptr->key) {
		remove (ptr->right, key, value); //recursively finding the key on right subtree to remove
	}
	//else must found the key
	else {
        // auto it = find(ptr->value.begin(), ptr->value.end(), value);
        // if (it != ptr->value.end()) {
        //     ptr->value.erase(it);  // Remove the value from the vector
        //     break;
        // }
        auto it = std::find(ptr->value.begin(), ptr->value.end(), value);
        if (it != ptr->value.end()) {
            ptr->value.erase(it);
        }

		//if no value left in the list of hotels in the same city
		if (ptr->value.empty()) {
			removeAboveExternal(ptr);
		}

	}
}



////////////////////////////////////////////
//helper function removeAboveExternal method
template <typename T1, typename T2>
void BST<T1, T2>::removeAboveExternal(Node<T1,T2>* external) 
{
	//Assigning parent Node
	Node<T1, T2> *parentNode = external->parent;
	//assigning grandparent Node
	Node<T1, T2> *grandParent = parentNode->parent;
	//assigning singling node
	Node<T1, T2> *sinbling;
	if (external == parentNode->left) {
		sinbling = parentNode->right;
	}
	else {
		sinbling = parentNode->left;
	}

	//if parentNode is the root
	if (parentNode == root) {
		root = sinbling;
		if (sinbling != nullptr) {
			sinbling->parent = nullptr;
		}
	}

	//when parentNode has a parent
	else {
		//if the parentNode is the left child of the grandparent
		if(parentNode == grandParent->left) {
			grandParent->left = sinbling;
		}
		//if the parentNode is the right child of the grandparent
		else {
			grandParent->right = sinbling;
		}
		//when sibling is not nullptr
		if (sinbling != nullptr) {
			sinbling->parent = grandParent; //assigning grandparent as the parent
		}
	}
	//deleting both nodes
	delete parentNode;
	delete external;

}


