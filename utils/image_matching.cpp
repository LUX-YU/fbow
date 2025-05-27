// loads a vocabulary, and a image. Extracts image feaures and then  compute the bow of the image
#include "fbow.h"
#include <iostream>
#include <map>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

#include <chrono>
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
    CmdLineParser cml(argc, argv);
    try
    {
        if (argc < 3 || cml["-h"])
            throw std::runtime_error("Usage: fbow   image [descriptor]");
        fbow::Vocabulary voc;
        voc.readFromFile(argv[1]);

        std::string desc_name = voc.getDescName();
        std::cout << "voc desc name=" << desc_name << std::endl;
        std::vector<std::vector<cv::Mat>> features(argc - 3);
        std::vector<std::map<double, int>> scores;
        std::vector<std::string> filenames(argc - 3);
        std::string outDir = argv[2];
        for (int i = 3; i < argc; ++i)
        {
            filenames[i - 3] = {argv[i]};
        }
        for (size_t i = 0; i < filenames.size(); ++i)
            features[i] = loadFeatures({filenames[i]}, desc_name);

        fbow::fBow vv, vv2;
        double avgScore = 0;
        int counter = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < features.size(); ++i)
        {
            vv = voc.transform(features[i][0]);
            std::map<double, int> score;
            for (size_t j = 0; j < features.size(); ++j)
            {

                vv2 = voc.transform(features[j][0]);
                double score1 = vv.score(vv, vv2);
                counter++;
                //		if(score1 > 0.01f)
                {
                    score.insert(std::pair<double, int>(score1, (int)j));
                }
                printf("%f, ", score1);
            }
            printf("\n");
            scores.push_back(score);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        avgScore += double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

        std::string command;
        int j = 0;
        for (size_t i = 0; i < scores.size(); i++)
        {
            std::stringstream str;

            command = "mkdir ";
            str << i;
            command += str.str();
            command += "/";
            system(command.c_str());

            command = "cp ";
            command += filenames[i];
            command += " ";
            command += str.str();
            command += "/source.JPG";

            system((std::string("cd ") + outDir).c_str());
            system(command.c_str());
            j = 0;
            for (auto it = scores[i].begin(); it != scores[i].end(); it++)
            {
                ++j;
                std::stringstream str2;
                command = "cp ";
                command += filenames[it->second];
                command += " ";
                command += str.str();
                command += "/";
                str2 << j << "-";
                str2 << it->first;
                command += str2.str();
                command += ".JPG";
                system(command.c_str());
            }
        }
        /*
        {
        std::cout<<vv.begin()->first<<" "<<vv.begin()->second<<std::endl;
        std::cout<<vv.rbegin()->first<<" "<<vv.rbegin()->second<<std::endl;
        }
        */
        std::cout << "avg score: " << avgScore << " # of features: " << features.size() << std::endl;
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
}
