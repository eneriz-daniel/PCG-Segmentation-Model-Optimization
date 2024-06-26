// Copyright (C) 2023 Daniel Enériz and Antonio Rodriguez-Almeida
// 
// This file is part of PCG Segmentation Model Optimization.
// 
// PCG Segmentation Model Optimization is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// PCG Segmentation Model Optimization is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with PCG Segmentation Model Optimization.  If not, see <http://www.gnu.org/licenses/>.

#include <bitset>
#include <stdexcept>
#include <typeinfo>
#include <cstring>
#include <string>
#include <fstream>

//Max array sizes
#define MAX_NDARRAY_SIZE 100000 //Maximum number of elements to read in a NDARRAY
#define MAX_NDARRAY_DIM 4 //Maximum dimensions in a NDARRAY


float TextStringToDouble(string words, int nbytes) {
  /* Takes a string containing binary data and returns the value
  associated to its representation, that could be either Float32
  (float) or Float64 (double).
  Args:
    words - String containing binary data to use as the
    representation of the floating point number
    nbytes - Number of bytes of the representation. Must be either
    4 (Float32) or 8 (Float64). Otherwise, an exception is thrown.
  Returns:
    The floating point number represented in the nbytes data.
  */

  reverse(words.begin(), words.end());
  string binaryString = "";

  union {
      double f;  // assuming 32-bit IEEE 754 single-precision
      uint64_t  i;    // assuming 32-bit 2's complement int
  } u64;

  union {
      float f;  // assuming 32-bit IEEE 754 single-precision
      int  i;    // assuming 32-bit 2's complement int
  } u32;
      
  for (char& _char : words) {
      binaryString +=bitset<8>(_char).to_string();
  }

  switch(nbytes){
    case 8:
      u64.i  = bitset<64>(binaryString).to_ullong();
      return u64.f;
    case 4:
      u32.i  = bitset<32>(binaryString).to_ulong();
      return u32.f;
    default:
      throw invalid_argument( "nbytes argument must be 4 or 8" );
  }
}

void GetFlatArrFromNpy(string npypath, float ndarray[MAX_NDARRAY_SIZE], int shape[MAX_NDARRAY_DIM]){
  /* Takes the path to a npy file containing a Numpy's n dimensional
  array of with np.single or np.double datatypes and fills ndarray with
  the flattened version the array and shape with the original shape.
  Args:
    npypath - String containing the .npy file path
    ndarray - Array to save the flattened version of the n dimensional
    array saved in the .npy. Its maximum size is MAX_NDARRAY_SIZE.
    shape - Array to save the shape of the n dimensional array. Its
    maximum size is MAX_NDARRAY_DIM.npy
  */   
  size_t loc1, loc2, loc3; //Three helpful variables when parsing the .npy header
  
  ifstream infile(npypath, ifstream::binary); //Opening the .npy file
  
  // Read the size
  int size;        
  infile.read(reinterpret_cast<char *>(&size), sizeof(size));
  
  // Allocate a string, make it large enough to hold the input
  string buffer;
  buffer.resize(size);
  
  // Read the text into the string
  infile.read(&buffer[0],  buffer.size() );
  infile.close();

  //PARSING THE HEADER
  //First, the data format is determined. Now only float is supported.
  loc1 = buffer.find("descr");
  loc2 = buffer.find(",", loc1+1);

  string descr = buffer.substr(loc1+1+6, loc2-loc1-1-6);

  int nbytes = (int)(descr[descr.find("<f")+2] - '0');
  
  //Second, the ndarray shape
  loc1 = buffer.find("shape");

  loc1 = buffer.find("(", loc1+1);
  loc2 = buffer.find(")", loc1+1);
  loc3 = buffer.find(",", loc1+1);

  string shape_str = buffer.substr(loc1+1, loc2-loc1-1);

  int ndim;
  if(loc2-loc3 == 1) ndim = 1; // One dimension array is shaped as (N,)
  else ndim = count(shape_str.begin(), shape_str.end(), ',') + 1;

  loc1 = -1;
  int nelements = 1;
  for(int i=0; i<ndim; i++){
    loc2 = shape_str.find(",", loc1+1);
    shape[i] = stoi(shape_str.substr(loc1+1, loc2-loc1-1), nullptr);
    nelements *= shape[i];
    loc1 = loc2;
  }
  
  //READING THE NDARRAY DATA
  string element_str;
  int elt_idx;
  int data_loc = buffer.find('\n', loc1+1)+1;

  for(elt_idx=0; elt_idx<nelements; elt_idx++){
    element_str = buffer.substr(data_loc+elt_idx*nbytes, nbytes);
    ndarray[elt_idx] = TextStringToDouble(element_str, nbytes);
  }
}