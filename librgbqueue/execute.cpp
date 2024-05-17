#include "execute.h"
#include <algorithm>
#include <thread>
#if USE_TCP_LINK
extern "C"
{
    #include <stdio.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <string.h>        // strerror
    #include <ctype.h>
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <stdlib.h>
    #include <errno.h>
}
    
#endif



Execute::Execute()
{
    //��ʼ��ϵ��
    int rv = 0;     // ���� R ֵ V ϵ��
    int gu = 0;     // ���� G ֵ U ϵ��
    int gv = 0;     // ���� G ֵ V ϵ��
    int bu = 0;     // ���� B ֵ U ϵ��

    rv = 256 * 1.28033 + 0.5 ;
    gv = 256 * 0.38059 + 0.5;
    gu = 256 * 0.21482 + 0.5;
    bu = 256 * 2.12798 + 0.5;

    for (int i = 0; i < 256; i++)
    {
        m_rv[i] = ((i - 128) * rv) >> 8;
        m_gu[i] = ((i - 128) * gu) >> 8;
        m_gv[i] = ((i - 128) * gv) >> 8;
        m_bu[i] = ((i - 128) * bu) >> 8;
    }
}

// python �������鵽 c++ ��ʱ, ��Ҫ������ƽΪ 1 ά.
void Execute::show_matrix(int* matrix, int rows, int columns) {
    int row, col;
    printf("the matrix point %p ==>%lld\n", matrix, (long long)matrix);

    for (row = 0; row < rows; row++) {
        for (col = 0; col < columns; col++) {
            printf("matrix[%d][%d] = %d\n", row, col, matrix[row * columns + col]);
        }
    }
}


void Execute::show_matrix_uint8(uint8_t* matrix, int rows, int columns,int channel) {
    int row, col,cha;
    printf("the matrix point %p ==>%lld\n", matrix, (long long)matrix);

    for (row = 0; row < rows; row++) {
        for (col = 0; col < columns; col++)
            for (cha = 0;  cha < channel; cha++)
            {
                printf("matrix[%d][%d][%d] = %d\n", row, col, cha, matrix[row * columns*3 + col*3 + cha]);
            }
    }
}
void Execute::show_uchar_matrix(char* matrix, int rows, int columns) {
    int row, col;
    printf("the matrix point %p ==>%lld\n", matrix, (long long)matrix);
    for (row = 0; row < rows; row++) {
        for (col = 0; col < columns; col++) {
            printf("matrix[%d][%d] = %d\n", row, col, int(matrix[row * columns + col]));
        }
    }
}
/**
 * ���Դ��ļ���ȡRGB����.
 * 
 */
int Execute::TestReadFromFile(char * filename)
{
    std::ifstream file(filename, std::ios::binary);

    std::cout << stamp() << "Execute::TestReadFromFile begin,filename:" << filename << std::endl;

    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return -1;
    }

    // ��ȡ�ļ���С  
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    int Cnt = size / itemsize;

    // ��ȡ�ļ����ݵ�������  
    for (size_t i = 0; i < Cnt; i++)
    {
        VideoFrame* ptr = new VideoFrame();
        if (!ptr)
        {
            std::cerr << "malloc mem failed!" << std::endl;
            file.close();
            return -1;
        }
        if (!file.read((char *)buff, itemsize)) {
            std::cerr << "Failed to read file!" << std::endl;
            file.close();
            return -1;
        }
        ptr->dts = i;
        ptr->width = 1920;
        ptr->height = 1080;
        ptr->channel = 3;
#if USE_RGB_FILE
        std::cout << stamp() << "Execute::TransformRGBtoMatrix:pts "<<i<< std::endl;
        TransformRGBtoMatrix(buff, (uint8_t ***)ptr->buffer, 1920, 1080, 3);
#else
        TransformYUVtoMatrixByTable(buff, (uint8_t ***)ptr->buffer, 1920, 1080, 3);
#endif

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        inputQueue.push_back(ptr);
    }
    // �ر��ļ�  
    file.close();

    std::cout << stamp() << "Execute::TestReadFromFile end cnt("<< Cnt<<")"<< std::endl;
    return 0;
}
/**
 * ����ͨRGBת��Ϊ����ʽ.
 * 
 */
void Execute::TransformRGBtoMatrix(uint8_t *rgb, uint8_t *** rgbMatrix, int width, int height, int channel)
{
    std::cout << stamp() << "Execute::TransformRGBtoMatrix begin" << std::endl;

    uint8_t* rgbPtr = rgb;
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;

    uint8_t* rgbMatrixPtr = (uint8_t*)rgbMatrix;
    uint8_t* tmpPtr;
    memcpy(rgbMatrixPtr, rgb, itemsize);

  /* for (size_t k = 0; k < channel; k++)
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                tmpPtr = rgbMatrixPtr + i * width * channel + j * channel + k;
                *tmpPtr = *rgbPtr;
                rgbPtr++;
            }
        }
    }
    */
    for (size_t i = 0; i < 6; i++)
    {
        printf("%x ", rgbMatrixPtr[i]);
    }
    printf("\n");
    std::cout << stamp() << "Execute::TransformRGBtoMatrix end" << std::endl;

}

/**
 * ��YUV422����ת��ΪRGB����.
 * 
 * \param yuv422
 * \param rgbMatrix
 * \param width
 * \param height
 * \param channel
 */
void Execute::TransformYUVtoMatrix(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel)
{
    std::cout << stamp() << "Execute::TransformYUVtoMatrix begin" << std::endl;

    uint8_t* yPtr, *uPtr, *vPtr;
    uint8_t* rgbMatrixPtr = (uint8_t*)rgbMatrix;

    int ylen = width * height;
    int ulen = ylen >> 1;
    int vlen = ulen;

    yPtr = yuv422;
    uPtr = yPtr + ylen;
    vPtr = uPtr + ulen;
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint8_t* rgb = rgbMatrixPtr + i * width * channel + j*channel;
            //rgb[0] = (*yPtr)-128 + 1.28033 * ((*vPtr)-128) + 128;
            //rgb[1] = (*yPtr)-128 - 0.21482 * ((*uPtr)-128) - 0.38059*((*vPtr)-128)+128;
            //rgb[2] = (*yPtr) - 128 - 2.12798 * ((*uPtr)-128) + 128;
            double y = *yPtr - 16;
            double u = *uPtr - 128;
            double v = *vPtr - 128;
            //
            //double r = 1.164 * y + 1.596 * v;
            //double g = 1.164 * y - 0.392 * u - 0.813 * v;
            //double b = 1.164 * y + 2.017 * u;

            double r =  y + 1.28033 * v;
            double g =  y - 0.21482 * u - 0.38059 * v;
            double b =  y + 2.12798 * u;
            
            if (r > 0.0)
            {
                rgb[0] = r <= 255.0 ? r : 255;
            }
            else
            {
                rgb[0] = 0;
            }

            if (g > 0.0)
            {
                rgb[1] = g <= 255.0 ? g : 255;
            }
            else
            {
                rgb[1] = 0;
            }

            if (b > 0.0)
            {
                rgb[2] = b <= 255.0 ? b : 255;
            }
            else
            {
                rgb[2] = 0;
            }

            //int y = *yPtr;
            //int u = *uPtr;
            //int v = *vPtr;

            //rgb[0] = ((298 * y + 408 * v) >> 8) - 223;
            //rgb[1] = ((298 * y - 100 * u - 208 * (v)) >> 8) + 136;
            //rgb[2] = ((298 * y + 516 * u) >> 8) - 277;

            yPtr++;
            uPtr += j % 2;
            vPtr += j % 2;
        }
    }
    std::cout << stamp() << "Execute::TransformYUVtoMatrix end" << std::endl;
}
inline uint8_t clamp_uint8(short value) {
    return (value > 255) ? 255 : (value < 0) ? 0 : (uint8_t)value;
}
/**
 * ���ʽת��YUV.
 * 
 * \param yuv422
 * \param rgbMatrix
 * \param width
 * \param height
 * \param channel
 */
void Execute::TransformYUVtoMatrixByTable(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel)
{
    std::cout << stamp() << "Execute::TransformYUVtoMatrixByTable begin" << std::endl;

    uint8_t* yPtr, * uPtr, * vPtr;
    uint8_t* rgbMatrixPtr = (uint8_t*)rgbMatrix;

    int ylen = width * height;
    int ulen = ylen >> 1;
    int vlen = ulen;

    short tmpR, tmpG, tmpB;
    short y, u, v;

    yPtr = yuv422;
    uPtr = yPtr + ylen;
    vPtr = uPtr + ulen;
    /*
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint8_t* rgb = rgbMatrixPtr + i * width * channel + j * channel;
            //rgb[0] = (*yPtr)-128 + 1.28033 * ((*vPtr)-128) + 128;
            //rgb[1] = (*yPtr)-128 - 0.21482 * ((*uPtr)-128) - 0.38059*((*vPtr)-128)+128;
            //rgb[2] = (*yPtr) - 128 - 2.12798 * ((*uPtr)-128) + 128;
            y = *yPtr;
            u = *uPtr;
            v = *vPtr;
            tmpR = y + m_rv[v] - 16;
            tmpG = y - m_gu[u] - m_gv[v] - 16;
            tmpB = y + m_bu[u] - 16;

            //rgb[0] = tmpR < 0 ? 0 : (tmpR >>8 ? 255 : tmpR);
            //rgb[1] = tmpG < 0 ? 0 : (tmpG >>8 ? 255 : tmpG);
            //rgb[2] = tmpB < 0 ? 0 : (tmpB >>8 ? 255 : tmpB);

            rgb[0] = (tmpR > 255) ? 255 : (tmpR < 0) ? 0 : (uint8_t)tmpR;
            rgb[1] = (tmpG > 255) ? 255 : (tmpG < 0) ? 0 : (uint8_t)tmpG;
            rgb[2] = (tmpB > 255) ? 255 : (tmpB < 0) ? 0 : (uint8_t)tmpB;

            yPtr++;
            uPtr += j % 2;
            vPtr += j % 2;
        }
    }
    */
    uint8_t* rowPtr = rgbMatrixPtr;
    for (size_t i = 0; i < height; i++) {
        uint8_t* rgb = rowPtr;
        for (size_t j = 0; j < width; j++) {
            y = (*yPtr++) - 16;
            u = (j & 0x1) == 0 ? *uPtr++ : *(uPtr - 1); // ����YUV�Ĳ����ʸ���uPtr  
            v = (j & 0x1) == 0 ? *vPtr++ : *(vPtr - 1); // ����YUV�Ĳ����ʸ���vPtr  

            //tmpR = y + m_rv[v] - 16;
            //tmpG = y - m_gu[u] - m_gv[v] - 16;
            //tmpB = y + m_bu[u] - 16;

            rgb[0] = clamp_uint8(y + m_rv[v]);
            rgb[1] = clamp_uint8(y - m_gu[u] - m_gv[v]);
            rgb[2] = clamp_uint8(y + m_bu[u]);

            rgb += channel; // ������һ�����ص�RGBֵ  
        }
        rowPtr += width * channel; // ������һ�е���ʼλ��  
    }
    std::cout << stamp() << "Execute::TransformYUVtoMatrixByTable end" << std::endl;
}
/**
 * ��ӡ��Ƶ����.
 * 
 * \param ptr
 */
void Execute::DumpVideoFrame(VideoFrame* ptr)
{
    printf("%s video dts(%d) w(%d)h(%d) Rgb:", stamp().c_str(), ptr->dts, ptr->width, ptr->height);
    for (size_t i = 0; i < 10; i++)
    {
        printf("%x %x %x,", ptr->buffer[0][i][0], ptr->buffer[0][i][1], ptr->buffer[0][i][2]);
    }
    printf("\n");
}

int Execute::GetFrameMatrix(char* matrix, int pts)
{
    size_t i;
    std::cout << stamp() << "Execute::GetFrameMatrix begin pts(" << pts << ")" << std::endl;

    std::unique_lock<std::mutex> lock(m_videoFrameQueueMutex);
    std::cout << stamp() << "Execute::GetFrameMatrix getLock pts(" << pts << ")" << std::endl;

    for (i= 0; i < inputQueue.size(); i++)
    {
        if (inputQueue[i]->dts == pts)
        {
            memcpy(matrix, inputQueue[i]->buffer, sizeof(inputQueue[i]->buffer));
            break;
        }

        if (inputQueue[i]->dts > pts)
        {
            std::cout << stamp() << "Execute::GetFrameMatrix end pts inputQueue[i]->dts > pts" << std::endl;
            return -1;
        }
    }

    if (i >= inputQueue.size())
    {
        std::cout << stamp() << "Execute::GetFrameMatrix end pts(" << pts << "): i > size(" <<i <<" >,"<< inputQueue.size()<<")" << std::endl;
        return -11;
    }
    std::cout << stamp() << "Execute::GetFrameMatrix end pts(" << pts << ")" << std::endl;
    return 0;
}

bool Execute::isRecvSocketOpen()
{
    return client_sockOpen;
}


/**
 * ��֡���ɻص������ڽ���֡���͸��ͻ���.
 * 
 * \param matrix �³ɵ���֡RGB����
 * \param pts    ʱ������Ϣ
 * \return       ������0--OK����ֵ����
 */
int Execute::FinishFrameGenerate(char* matrix, int pts)
{
    std::cout << stamp() << "Execute::FinishFrameGenerate begin pts("<<pts<<")" << std::endl;

    //1.������pts��֡����������֡��������֡
    finishMap[pts] = true;
    //writeBinaryDataToFile("outputRGB.rgb", (const uint8_t*)matrix, 1920 * 1080 * 3);

    VideoFrame* ptr = nullptr;

    {
        //2.��ȡ����֡��preFrame����preFrame��ԭʼ�����Ƴ�
        std::unique_lock<std::mutex> lock(m_videoFrameQueueMutex);
        std::cout << stamp() << "Execute::FinishFrameGenerate getLock pts(" << pts << ")" << std::endl;

        for (auto it = inputQueue.begin(); it != inputQueue.end() && (*it)->dts <= pts; it++)
        {
            if ((*it)->dts == pts)
            {
                ptr = *it;
                std::cout << stamp() << "Execute::FinishFrameGenerate get frame inqueue pts(" << pts << ")" << std::endl;
                
                it = inputQueue.erase(it);
                break;
            }
        }
    }

    char* buffer = (char*)ptr;
    ssize_t size_sent = 0;
    if (client_sockOpen && ptr)
    {
        memcpy(ptr->buffer, matrix, sizeof(ptr->buffer));

        do
        {
            ssize_t size = write(client_sock, buffer + size_sent, itemsize - size_sent);

            if (-1 == size) {
                fprintf(stdout, "%s Execute::FinishFrameGenerate  write error, reason: %d %s\n", stamp().c_str(), errno, strerror(errno));
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            size_sent += size;
            if (size_sent < itemsize)
            {
                fprintf(stdout, "%s Execute::FinishFrameGenerate  write new gen size [%ld] for dts[%d]\n", stamp().c_str(), size, ((VideoFrame*)buffer)->dts);
            }
            else
            {
                fprintf(stdout, "%s Execute::FinishFrameGenerate  write sendfullframe size [%ld] for dts[%d]\n", stamp().c_str(), size_sent, ((VideoFrame*)buffer)->dts);
                DumpVideoFrame((VideoFrame*)buffer);
                break;
            }

        } while (true);

        delete ptr;
        ptr = nullptr;
    }
    else //�쳣���
    {
        std::cout << stamp() << "Execute::FinishFrameGenerate  break now socket unopen break " << std::endl;
    }

    std::cout << stamp() << "Execute::FinishFrameGenerate end pts(" << pts << ")" << std::endl;
    return 0;
}


/**
 * �߳� ���ڽ���RGB���ݣ���RGB֡���뵽������.
 * 
 */
void Execute::TcpServerThread()
{

    int ret = 0;
    struct sockaddr_in server_addr;
    // 1.����ͨ���׽���
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (-1 == sock) {
        fprintf(stderr, "create socket error, reason: %s\n", strerror(errno));
        exit(-1);
    }

    // 2.��ձ�ǩ��д�ϵ�ַ�Ͷ˿ں�
    bzero(&server_addr, sizeof(server_addr));

    uint16_t hostshort = SERVER_PORT;

    server_addr.sin_family = AF_INET;	// ѡ��Э����ipv4
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);	// ������������IP��ַ
    server_addr.sin_port = htons(SERVER_PORT);			// �󶨶˿ں�

    // 3.��
    ret = bind(sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (-1 == ret) {
        fprintf(stderr, "socket bind error, reason: %s\n", strerror(errno));
        close(sock);
        return;
    }

    // 4.������ͬʱ����128������
    ret = listen(sock, 128);
    if (-1 == ret) {
        fprintf(stderr, "listen error, reason: %s\n", strerror(errno));
        close(sock);
        return;
    }

    sockOpen = true;

    printf("�ȴ��ͻ��˵�����\n");

    continueToRecv = 1;

    struct sockaddr_in client;
    char client_ip[64];
    int len = 0;
    //char buf[256];

    socklen_t client_addr_len;
    client_addr_len = sizeof(client);
    // 5.����
    //int clientfd = accept4(listenfd, &clientaddr, &addrlen, SOCK_NONBLOCK);

    client_sock = accept4(sock, (struct sockaddr*)&client, &client_addr_len, SOCK_NONBLOCK);
    if (-1 == client_sock) {
        perror("accept error");
        close(sock);
        return;
    }
    client_sockOpen = true;
    // ��ӡ�ͻ���IP��ַ�Ͷ˿ں�
    printf("client ip: %s\t port: %d\n", inet_ntop(AF_INET, &client.sin_addr.s_addr, client_ip, sizeof(client_ip)),ntohs(client.sin_port));

    do {

        ssize_t total_bytes_read = 0;
        ssize_t n = 0;

        VideoFrame* ptr = new VideoFrame();

        while (total_bytes_read < itemsize) {
            // ��ȡ���ݵ�������  
            if (client_sockOpen)
            {
                n = read(client_sock, (char*)ptr + total_bytes_read, (itemsize - total_bytes_read));
            }
            if (n < 0) {
                // ��ȡ����  
                int error = errno;
                fprintf(stdout, "%s Execute::TcpServerThread err errmsg, reason: %d %s\n", stamp().c_str(), errno, strerror(errno));
                if (EAGAIN == error)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                continueToRecv = 0;
                break;
            }
            else if (n == 0) {
                // ��ȡ��EOF������TCP�׽�����˵����ͨ�����ᷢ�����������ӱ��رգ�  
                fprintf(stdout, "%s Execute::TcpServerThread EOF reached\n", stamp().c_str());
                continueToRecv = 0;
                break;
            }
            else {
                // �����ȡ�������ݣ�����ֻ�Ǽ򵥼�����  
                total_bytes_read += n;
            }
        }

        if (!continueToRecv)
        {
            // 8.�رտͻ����׽���
            if (client_sockOpen)
            {
                fprintf(stdout, "%s Execute::TcpServerThread !continueToRecv\n", stamp().c_str());
                close(client_sock);
                client_sockOpen = false;
            }
            break;
        }

        // ����Ƿ��ȡ�����������ֽ���  
        if (total_bytes_read == itemsize) {
            fprintf(stdout, "%s Execute::TcpServerThread Successfully read %d bytes.\n", stamp().c_str(), itemsize);
            if (ptr->channel == -1) //��-1��ʾ���Զ�Ҫ�����
            {
                std::cout << stamp() << "Execute::TcpServerThread ptr->channel == -1 "  << std::endl;
                close(client_sock);
                client_sockOpen = false;
                break;
            }
            std::cout << stamp() << " Execute::TcpServerThread videodump:" << std::endl;
            DumpVideoFrame((VideoFrame *)ptr);
            {
                std::unique_lock<std::mutex> lock(m_videoFrameQueueMutex);
                inputQueue.push_back(ptr);
            }
        }
        else {
            printf("Read %zd bytes, expected %d bytes.\n", total_bytes_read, itemsize);
        }
    } while (continueToRecv);

    // 9.�رշ������׽���
    close(sock);
    sockOpen = false;

    std::cout << stamp() << "Execute::TcpServerThread End!"  << std::endl;
}

/**
 * ����TCP�����߳�.
 *
 */
void Execute::startTcpServerThread()
{
    m_tcpServerThread = std::thread(&Execute::TcpServerThread, this);
}

/**
 * ֹͣ�߳�.
 *
 */
void Execute::stopTcpServerThread()
{
    std::cout << stamp() << "Execute::stopTcpServerThread end " << std::endl;

    if (m_tcpServerThread.joinable())
    {
        if (client_sockOpen)
        {
            continueToRecv = 0;
            client_sockOpen = false;
            close(client_sock);
        }
        m_tcpServerThread.join();
    }
    std::cout << stamp() << "Execute::stopTcpServerThread " << std::endl;

}
std::string date_time(std::time_t posix)
{
    char buf[20]; // big enough for 2015-07-08 10:06:51\0
    std::tm tp = *std::localtime(&posix);
    return { buf, std::strftime(buf, sizeof(buf), "%F %T", &tp) };
}

std::string stamp()
{
    using namespace std;
    using namespace std::chrono;

    auto now = system_clock::now();

    // use microseconds % 1000000 now
    auto us = duration_cast<microseconds>(now.time_since_epoch()) % 1000000;

    std::ostringstream oss;
    oss.fill('0');

    oss << date_time(system_clock::to_time_t(now));
    oss << '.' << setw(6) << us.count();
    oss << "[" << std::this_thread::get_id() << "]";

    return oss.str();

}

void writeBinaryDataToFile(const std::string& filename, const uint8_t* data, int len_of_data) {
    // ��һ���ļ�����д�����������  
    std::cout << stamp() << "Execute::writeBinaryDataToFile begin " << std::endl;

    std::ofstream file(filename, std::ios::binary | std::ios::app); // ʹ�� std::ios::app ����׷��д��  

    // ����ļ��Ƿ�ɹ���  
    if (!file.is_open()) {
        std::cerr << "�޷����ļ� " << filename << " ����д��" << std::endl;
        return;
    }

    // �� uint8_t ����д���ļ�  
    file.write(reinterpret_cast<const char*>(data), len_of_data);

    // �ر��ļ�  
    file.close();
    std::cout << stamp() << "Execute::writeBinaryDataToFile begin " << std::endl;
}
