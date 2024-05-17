#pragma once
#ifndef PY_C_TEST_EXCUTE_H
#define PY_C_TEST_EXCUTE_H

#define USE_RGB_FILE 0
#define USE_TCP_LINK 1

#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>  
#include <mutex>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/core/core.hpp"
#include <cstring>
#include <thread>
#include <iomanip>
#include <sstream>  

std::string date_time(std::time_t posix);
std::string stamp();
void writeBinaryDataToFile(const std::string& filename, const uint8_t* data, int len_of_data);

typedef struct Result {
    int a;
    const char* b;
} result;

#define SERVER_PORT 1200

typedef struct VideoFrame
{
    int dts;
    int width, height, channel;

    uint8_t buffer[1920][1080][3];
}VideoFrame;

class Execute {
public:
    // 传递int数组到c++
    Execute();
    void show_matrix(int* matrix, int rows, int columns);

    void show_matrix_uint8(uint8_t* matrix, int rows, int columns, int channel);

    // 传递uchar数组到c++
    void show_uchar_matrix(char* matrix, int rows, int columns);

    int TestReadFromFile(char* filename);

    void TransformRGBtoMatrix(uint8_t* rgb, uint8_t ***rgbMatrix, int width, int height, int channel);
    void TransformYUVtoMatrix(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel);

    void TransformYUVtoMatrixByTable(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel);

    void DumpVideoFrame(VideoFrame* ptr);

    int  GetFrameMatrix(char* matrix, int pts);
    int FinishFrameGenerate(char* matrix, int pts);
    int  GetQueueSize() { return inputQueue.size(); }
    bool isRecvSocketOpen();

#if USE_TCP_LINK
    void startTcpServerThread();
    void TcpServerThread();
    void stopTcpServerThread();

#endif
private:
    std::vector<VideoFrame*> inputQueue;
    std::mutex               m_videoFrameQueueMutex;

    std::unordered_map<int, bool> finishMap;
#if USE_RGB_FILE
    static const int itemsize = 1920 * 1080 * 3;
#elif USE_TCP_LINK
    static const int itemsize = sizeof(VideoFrame);
#else
    static const int itemsize = 1920 * 1080 * 2;
#endif
    // 分配内存以存储文件内容  
    uint8_t buff[itemsize];

    short m_rv[256] = { 0 };
    short m_gv[256] = { 0 };
    short m_gu[256] = { 0 };
    short m_bu[256] = { 0 };
#if USE_TCP_LINK
    std::thread		m_tcpServerThread;	//tcp接收线程
    int             sock;	// 通信套接字
    bool            sockOpen = false;
    int             client_sock;
    bool            client_sockOpen = false;
    int             continueToRecv = 1;
#endif
};


#endif //PY_C_TEST_EXCUTE_H