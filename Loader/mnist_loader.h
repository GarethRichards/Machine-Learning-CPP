#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "boost\numeric\ublas\vector.hpp"

using namespace boost::numeric;

// Loads the MNIST data files
template<typename T>
class mnist_loader
{
public:
	mnist_loader(const std::string &FileData,
		const std::string &FileLabels,
		std::vector<std::pair<ublas::vector<T>, ublas::vector<T>>> &mnist_data)
	{
		{
			std::ifstream myFile(FileData, std::wifstream::in | std::wifstream::binary);
			if (!myFile) throw "File does not exist";
			int MagicNumber(0); int nItems(0); int nRows(0); int nCol(0);
			myFile.read((char *)&MagicNumber, 4);
			MagicNumber=_byteswap_ulong(MagicNumber);
			if (MagicNumber != 2051) throw "Magic number for training data incorrect";
			myFile.read((char *)&nItems, 4);
			nItems = _byteswap_ulong(nItems);
			myFile.read((char *)&nRows, 4);
			nRows = _byteswap_ulong(nRows);
			myFile.read((char *)&nCol, 4);
			nCol = _byteswap_ulong(nCol);
			std::unique_ptr<unsigned char[]> buf(new unsigned char[nRows*nCol]);
			for (auto i = 0; i < nItems; ++i)
			{
				myFile.read((char *)buf.get(), nRows*nCol);
				ublas::vector<T> data(nRows*nCol);
				for (auto j = 0; j < nRows*nCol; ++j)
				{
					data[j] = static_cast<T>(buf[j]) / static_cast<T>(255.0);
				}
				mnist_data.push_back(make_pair(data, ublas::zero_vector<T>(10)));
			}
		}
		{
			std::ifstream myFile(FileLabels, std::wifstream::in | std::wifstream::binary);
			if (!myFile) throw "File does not exist";
			int MagicNumber(0); int nItems(0);
			myFile.read((char *)&MagicNumber, 4);
			MagicNumber = _byteswap_ulong(MagicNumber);
			if (MagicNumber != 2049) throw "Magic number for label file incorrect";
			myFile.read((char *)&nItems, 4);
			nItems = _byteswap_ulong(nItems);
			for (int i = 0; i < nItems; ++i)
			{
				char data;
				myFile.read(&data, 1);
				mnist_data[i].second[data] = 1.0;
			}
		}
	}
};
