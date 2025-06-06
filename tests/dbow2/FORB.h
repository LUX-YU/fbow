/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include "FClass.h"
namespace DBoW2 {

/// Functions to manipulate ORB descriptors
class FORB: protected FClass
{
public:

    /// Descriptor type
    typedef cv::Mat TDescriptor; // CV_8U
    /// Pointer to a single descriptor
    typedef const TDescriptor *pDescriptor;
    /// Descriptor length (in bytes)

    static inline int L() {return 32;}
    /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
    static inline void meanValue(const std::vector<pDescriptor> &descriptors,
                                 TDescriptor &mean);

    /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
    static inline int distance(const TDescriptor &a, const TDescriptor &b);

    /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
    static inline std::string toString(const TDescriptor &a);

    /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
    static inline void fromString(TDescriptor &a, const std::string &s);

    /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
    static inline void toMat32F(const std::vector<TDescriptor> &descriptors,
                                cv::Mat &mat);

    static inline void toMat8U(const std::vector<TDescriptor> &descriptors,
                               cv::Mat &mat);

};


// --------------------------------------------------------------------------


void FORB::meanValue(const std::vector<FORB::pDescriptor> &descriptors,
                     FORB::TDescriptor &mean)
{
    if(descriptors.empty())
    {
        mean.release();
        return;
    }
    else if(descriptors.size() == 1)
    {
        mean = descriptors[0]->clone();
    }
    else
    {
        std::vector<int> sum(FORB::L() * 8, 0);

        for(size_t i = 0; i < descriptors.size(); ++i)
        {
            const cv::Mat &d = *descriptors[i];
            const unsigned char *p = d.ptr<unsigned char>();

            for(int j = 0; j < d.cols; ++j, ++p)
            {
                if(*p & (1 << 7)) ++sum[ j*8     ];
                if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
                if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
                if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
                if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
                if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
                if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
                if(*p & (1))      ++sum[ j*8 + 7 ];
            }
        }

        mean = cv::Mat::zeros(1, FORB::L(), CV_8U);
        unsigned char *p = mean.ptr<unsigned char>();

        const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2;
        for(size_t i = 0; i < sum.size(); ++i)
        {
            if(sum[i] >= N2)
            {
                // set bit
                *p |= 1 << (7 - (i % 8));
            }

            if(i % 8 == 7) ++p;
        }
    }
}

// --------------------------------------------------------------------------

int FORB::distance(const FORB::TDescriptor &a,
                   const FORB::TDescriptor &b)
{
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

// --------------------------------------------------------------------------

std::string FORB::toString(const FORB::TDescriptor &a)
{
    std::stringstream ss;
    const unsigned char *p = a.ptr<unsigned char>();

    for(int i = 0; i < a.cols; ++i, ++p)
    {
        ss << (int)*p << " ";
    }

    return ss.str();
}

// --------------------------------------------------------------------------

void FORB::fromString(FORB::TDescriptor &a, const std::string &s)
{
    a.create(1, FORB::L(), CV_8U);
    unsigned char *p = a.ptr<unsigned char>();

    std::stringstream ss(s);
    for(int i = 0; i < FORB::L(); ++i, ++p)
    {
        int n;
        ss >> n;

        if(!ss.fail())
            *p = (unsigned char)n;
    }

}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor> &descriptors,
                    cv::Mat &mat)
{
    if(descriptors.empty())
    {
        mat.release();
        return;
    }

    const size_t N = descriptors.size();

    mat.create((int)N, FORB::L()*8, CV_32F);
    float *p = mat.ptr<float>();

    for(size_t i = 0; i < N; ++i)
    {
        const int C = descriptors[i].cols;
        const unsigned char *desc = descriptors[i].ptr<unsigned char>();

        for(int j = 0; j < C; ++j, p += 8)
        {
            p[0] = static_cast<float>((desc[j] & (1 << 7) ? 1 : 0));
            p[1] = static_cast<float>((desc[j] & (1 << 6) ? 1 : 0));
            p[2] = static_cast<float>((desc[j] & (1 << 5) ? 1 : 0));
            p[3] = static_cast<float>((desc[j] & (1 << 4) ? 1 : 0));
            p[4] = static_cast<float>((desc[j] & (1 << 3) ? 1 : 0));
            p[5] = static_cast<float>((desc[j] & (1 << 2) ? 1 : 0));
            p[6] = static_cast<float>((desc[j] & (1 << 1) ? 1 : 0));
            p[7] = static_cast<float>(desc[j] & (1));
        }
    }
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor> &descriptors,
                   cv::Mat &mat)
{
    mat.create((int)descriptors.size(), 32, CV_8U);

    unsigned char *p = mat.ptr<unsigned char>();

    for(size_t i = 0; i < descriptors.size(); ++i, p += 32)
    {
        const unsigned char *d = descriptors[i].ptr<unsigned char>();
        std::copy(d, d+32, p);
    }

}

} // namespace DBoW2

#endif

