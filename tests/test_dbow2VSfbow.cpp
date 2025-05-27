#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"

#include "fbow.h"
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

using ORBVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

class CmdLineParser
{
    int argc;
    char **argv;

public:
    CmdLineParser(int _argc, char **_argv)
        : argc(_argc), argv(_argv) {}

    bool operator[](std::string param)
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
        {
            if (std::string(argv[i]) == param)
                idx = i;
        }
        return (idx != -1);
    }

    std::string operator()(std::string param, std::string defvalue = "-1")
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
        {
            if (std::string(argv[i]) == param)
                idx = i;
            if (idx == -1)
                return defvalue;
            else
                return (argv[idx + 1]);
        }
    };
};

std::vector<cv::Mat> toDescriptorvector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
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

    std::cout << "Extracting features..." << std::endl;
    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::cout << "reading image: " << path_to_images[i] << std::endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty())
            throw std::runtime_error("Could not open image" + path_to_images[i]);
        std::cout << "extracting features" << std::endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        std::cout << "done detecting features" << std::endl;
    }
    return features;
}

int main(int argc, char **argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 4 || cml["-h"])
            throw std::runtime_error("Usage: dbowfile.txt   image  fbowfile.fbow ");
        std::cout << "extracting features" << std::endl;
        std::vector<cv::Mat> features = loadFeatures({argv[2]}, "orb");

        double dbow2_load, dbow2_transform;
        double fbow_load, fbow_transform;

        {
            ORBVocabulary voc;
            std::cout << "loading dbow2 voc...." << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();
            voc.loadFromTextFile(argv[1]);
            auto t_end = std::chrono::high_resolution_clock::now();
            auto desc_vector = toDescriptorvector(features[0]); // transform into the mode required by dbow2
            dbow2_load = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
            std::cout << "load time=" << dbow2_load << " ms" << std::endl;
            std::cout << "processing image 1000 times" << std::endl;
            DBoW2::BowVector vv;
            t_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 1000; i++)
                voc.transform(desc_vector, vv);
            t_end = std::chrono::high_resolution_clock::now();

            std::cout << vv.begin()->first << " " << vv.begin()->second << std::endl;
            std::cout << vv.rbegin()->first << " " << vv.rbegin()->second << std::endl;
            dbow2_transform = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
            std::cout << "DBOW2 time=" << dbow2_transform << " ms" << std::endl;
        }
        // repeat with fbow

        fbow::Vocabulary fvoc;
        std::cout << "loading fbow voc...." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        fvoc.readFromFile(argv[3]);
        auto t_end = std::chrono::high_resolution_clock::now();
        fbow_load = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        std::cout << "load time=" << fbow_load << " ms" << std::endl;

        {
            std::cout << "processing image 1000 times" << std::endl;
            fbow::fBow vv;
            t_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 1000; i++)
            {
                vv = fvoc.transform(features[0]);
            }
            t_end = std::chrono::high_resolution_clock::now();

            std::cout << vv.begin()->first << " " << vv.begin()->second << std::endl;
            std::cout << vv.rbegin()->first << " " << vv.rbegin()->second << std::endl;
            fbow_transform = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
            std::cout << "FBOW time=" << fbow_transform << " ms" << std::endl;
        }

        std::cout << "Fbow load speed up=" << dbow2_load / fbow_load << " transform Speed up=" << dbow2_transform / fbow_transform << std::endl;
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
}
