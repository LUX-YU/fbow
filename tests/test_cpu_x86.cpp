
#include <iostream>
#include "cpu.h"

using namespace fbow;
void print(const char* label, bool yes){
    std::cout << label;
    std::cout << (yes ? "Yes" : "No") << std::endl;
}
void print(cpu  host_info)  {
    std::cout << "CPU Vendor:" << std::endl;
    print("    AMD         = ", host_info.Vendor_AMD);
    print("    Intel       = ", host_info.Vendor_Intel);
    std::cout << std::endl;

    std::cout << "OS Features:" << std::endl;
#ifdef _WIN32
    print("    64-bit      = ", host_info.OS_x64);
#endif
    print("    OS AVX      = ", host_info.OS_AVX);
    print("    OS AVX512   = ", host_info.OS_AVX512);
    std::cout << std::endl;

    std::cout << "Hardware Features:" << std::endl;
    print("    MMX         = ", host_info.HW_MMX);
    print("    x64         = ", host_info.HW_x64);
    print("    ABM         = ", host_info.HW_ABM);
    print("    RDRAND      = ", host_info.HW_RDRAND);
    print("    BMI1        = ", host_info.HW_BMI1);
    print("    BMI2        = ", host_info.HW_BMI2);
    print("    ADX         = ", host_info.HW_ADX);
    print("    MPX         = ", host_info.HW_MPX);
    print("    PREFETCHWT1 = ", host_info.HW_PREFETCHWT1);
    std::cout << std::endl;

    std::cout << "SIMD: 128-bit" << std::endl;
    print("    SSE         = ", host_info.HW_SSE);
    print("    SSE2        = ", host_info.HW_SSE2);
    print("    SSE3        = ", host_info.HW_SSE3);
    print("    SSSE3       = ", host_info.HW_SSSE3);
    print("    SSE4a       = ", host_info.HW_SSE4a);
    print("    SSE4.1      = ", host_info.HW_SSE41);
    print("    SSE4.2      = ", host_info.HW_SSE42);
    print("    AES-NI      = ", host_info.HW_AES);
    print("    SHA         = ", host_info.HW_SHA);
    std::cout << std::endl;

    std::cout << "SIMD: 256-bit" << std::endl;
    print("    AVX         = ", host_info.HW_AVX);
    print("    XOP         = ", host_info.HW_XOP);
    print("    FMA3        = ", host_info.HW_FMA3);
    print("    FMA4        = ", host_info.HW_FMA4);
    print("    AVX2        = ", host_info.HW_AVX2);
    std::cout << std::endl;

    std::cout << "SIMD: 512-bit" << std::endl;
    print("    AVX512-F    = ", host_info.HW_AVX512_F);
    print("    AVX512-CD   = ", host_info.HW_AVX512_CD);
    print("    AVX512-PF   = ", host_info.HW_AVX512_PF);
    print("    AVX512-ER   = ", host_info.HW_AVX512_ER);
    print("    AVX512-VL   = ", host_info.HW_AVX512_VL);
    print("    AVX512-BW   = ", host_info.HW_AVX512_BW);
    print("    AVX512-DQ   = ", host_info.HW_AVX512_DQ);
    print("    AVX512-IFMA = ", host_info.HW_AVX512_IFMA);
    print("    AVX512-VBMI = ", host_info.HW_AVX512_VBMI);
    std::cout << std::endl;

    std::cout << "Summary:" << std::endl;
    print("    Safe to use AVX:     ", host_info.HW_AVX && host_info.OS_AVX);
    print("    Safe to use AVX512:  ",host_info. HW_AVX512_F && host_info.OS_AVX512);
    std::cout << std::endl;
}

int main(){

    std::cout << "CPU Vendor String: " ;
    std::cout<< cpu::get_vendor_string() ;
    cpu features;
    features.detect_host();
    print(features);

#if _WIN32
    system("pause");
#endif
}
