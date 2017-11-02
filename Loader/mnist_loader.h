#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "boost\numeric\ublas\vector.hpp"

using namespace boost::numeric;


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
			if (!myFile) throw("File does not exist");
			int MagicNumber; int nItems; int nRows; int nCol;
			myFile.read((char *)&MagicNumber, 4);
			MagicNumber=_byteswap_ulong(MagicNumber);
			myFile.read((char *)&nItems, 4);
			nItems = _byteswap_ulong(nItems);
			myFile.read((char *)&nRows, 4);
			nRows = _byteswap_ulong(nRows);
			myFile.read((char *)&nCol, 4);
			nCol = _byteswap_ulong(nCol);
			unsigned char *buf = new unsigned char[nRows*nCol];
			for (int i = 0; i < nItems; ++i)
			{
				myFile.read((char *)buf, nRows*nCol);
				ublas::vector<T> data(nRows*nCol);
				for (int j = 0; j < nRows*nCol; ++j)
				{
					data[j] = static_cast<T>(buf[j]) / 255.0;
				}
				mnist_data.push_back(make_pair(data, ublas::zero_vector<T>(10)));
			}
		}
		{
			std::ifstream myFile(FileLabels, std::wifstream::in | std::wifstream::binary);
			if (!myFile) throw("File does not exist");
			int MagicNumber; int nItems;
			myFile.read((char *)&MagicNumber, 4);
			MagicNumber = _byteswap_ulong(MagicNumber);
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