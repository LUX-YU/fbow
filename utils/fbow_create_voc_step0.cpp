
// First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
#include <fstream>
#include <vector>
#include "fbow.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "dirreader.h"

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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
	std::cout << std::endl << "Press enter to continue" << std::endl;
	getchar();
}

std::vector<cv::Mat> loadFeatures(std::vector<std::string> path_to_images, std::string descriptor = "")
{
	// select detector
	cv::Ptr<cv::Feature2D> fdetector;
	if (descriptor == "orb")
		fdetector = cv::ORB::create(2000);
	else if (descriptor == "brisk")
		fdetector = cv::BRISK::create();
#ifdef OPENCV_VERSION_3
	else if (descriptor == "akaze")
		fdetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4f);
#endif
#ifdef USE_CONTRIB
	else if (descriptor == "surf")
		fdetector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

	else
		throw std::runtime_error("Invalid descriptor");
	assert(!descriptor.empty());
	std::vector<cv::Mat> features;

	std::cout << "Extracting   features..." << std::endl;
	for (size_t i = 0; i < path_to_images.size(); ++i)
	{
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		std::cout << "reading image: " << path_to_images[i] << std::endl;
		cv::Mat image = cv::imread(path_to_images[i], 0);
		if (image.empty())
		{
			std::cerr << "Could not open image:" << path_to_images[i] << std::endl;
			continue;
		}
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		std::cout << "extracting features: total= " << keypoints.size() << std::endl;
		features.push_back(descriptors);
		std::cout << "done detecting features" << std::endl;
	}
	return features;
}

// ----------------------------------------------------------------------------
void saveToFile(std::string filename, const std::vector<cv::Mat> &features, std::string desc_name, bool rewrite = true)
{
	// test it is not created
	if (!rewrite)
	{
		std::fstream ifile(filename, std::ios::binary);
		if (ifile.is_open()) // read size and rewrite
			std::runtime_error("ERROR::: Output File " + filename + " already exists!!!!!");
	}
	std::ofstream ofile(filename, std::ios::binary);
	if (!ofile.is_open())
	{
		std::cerr << "could not open output file" << std::endl;
		exit(0);
	}

	desc_name.resize(std::min(size_t(19), desc_name.size()));
	ofile << desc_name;

	size_t size = features.size();
	ofile.write((char *)&size, sizeof(size));
	for (auto &f : features)
	{
		if (!f.isContinuous())
		{
			std::cerr << "Matrices should be continuous" << std::endl;
			exit(0);
		}
		uint32_t aux = f.cols;
		ofile.write((char *)&aux, sizeof(aux));
		aux = f.rows;
		ofile.write((char *)&aux, sizeof(aux));
		aux = f.type();
		ofile.write((char *)&aux, sizeof(aux));
		ofile.write((char *)f.ptr<uchar>(0), f.total() * f.elemSize());
	}
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{

	try
	{
		CmdLineParser cml(argc, argv);
		if (cml["-h"] || argc < 4)
		{
			std::cerr << "Usage:  descriptor_name output dir_with_images \n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)" << std::endl;
			return -1;
		}

		std::string descriptor = argv[1];
		std::string output = argv[2];

		auto images = DirReader::read(argv[3]);
		std::vector<cv::Mat> features = loadFeatures(images, descriptor);

		// save features to file
		std::cerr << "saving to " << argv[2] << std::endl;
		saveToFile(argv[2], features, descriptor);
	}
	catch (std::exception &ex)
	{
		std::cerr << ex.what() << std::endl;
	}

	return 0;
}
