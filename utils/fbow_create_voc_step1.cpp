// Second step,creates the vocabulary from the set of features. It can be slow
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
//
#include "vocabulary_creator.h"
// OpenCV
#include <opencv2/core/core.hpp>

// command line parser
class CmdLineParser
{
	int argc;
	char **argv;

public:
	CmdLineParser(int _argc, char **_argv) : argc(_argc), argv(_argv) {}
	bool operator[](std::string param)
	{
		int idx = -1;
		for (int i = 0; i < argc && idx == -1; i++)
			if (std::string(argv[i]) == param)
				idx = i;
		return (idx != -1);
	}
	std::string operator()(std::string param, std::string defvalue = "-1")
	{
		int idx = -1;
		for (int i = 0; i < argc && idx == -1; i++)
			if (std::string(argv[i]) == param)
				idx = i;
		if (idx == -1)
			return defvalue;
		else
			return (argv[idx + 1]);
	}
};

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
std::vector<cv::Mat> readFeaturesFromFile(std::string filename, std::string &desc_name)
{
	std::vector<cv::Mat> features;
	// test it is not created
	std::ifstream ifile(filename, std::ios::binary);
	if (!ifile.is_open())
	{
		std::cerr << "could not open input file" << std::endl;
		exit(0);
	}

	char _desc_name[20];
	ifile.read(_desc_name, 20);
	desc_name = _desc_name;

	uint32_t size;
	ifile.read((char *)&size, sizeof(size));
	features.resize(size);
	for (size_t i = 0; i < size; i++)
	{

		uint32_t cols, rows, type;
		ifile.read((char *)&cols, sizeof(cols));
		ifile.read((char *)&rows, sizeof(rows));
		ifile.read((char *)&type, sizeof(type));
		features[i].create(rows, cols, type);
		ifile.read((char *)features[i].ptr<uchar>(0), features[i].total() * features[i].elemSize());
	}
	return features;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
	try
	{
		CmdLineParser cml(argc, argv);
		if (cml["-h"] || argc < 3)
		{
			std::cerr << "Usage:  features output.fbow [-k k] [-l L] [-t nthreads] [-maxIters <int>:0 default] [-v verbose on]. " << std::endl;
			std::cerr << "Creates the vocabylary of k^L" << std::endl;
			std::cerr << "By default, we employ a random selection center without runnning a single iteration of the k means.\n"
						 "As indicated by the authors of the flann library in their paper, the result is not very different from using k-means, but speed is much better\n";
			return -1;
		}

		std::string desc_name;
		auto features = readFeaturesFromFile(argv[1], desc_name);

		std::cout << "DescName=" << desc_name << std::endl;
		fbow::VocabularyCreator::Params params;
		params.k = stoi(cml("-k", "10"));
		params.L = stoi(cml("-l", "6"));
		params.nthreads = stoi(cml("-t", "1"));
		params.maxIters = std::stoi(cml("-maxIters", "0"));
		params.verbose = cml["-v"];
		srand(0);
		fbow::VocabularyCreator voc_creator;
		fbow::Vocabulary voc;
		std::cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << std::endl;
		auto t_start = std::chrono::high_resolution_clock::now();
		voc_creator.create(voc, features, desc_name, params);
		auto t_end = std::chrono::high_resolution_clock::now();
		std::cout << "time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << " msecs" << std::endl;
		std::cout << "nblocks=" << voc.size() << std::endl;
		std::cerr << "Saving " << argv[2] << std::endl;
		voc.saveToFile(argv[2]);
	}
	catch (std::exception &ex)
	{
		std::cerr << ex.what() << std::endl;
	}

	return 0;
}
