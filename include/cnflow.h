#ifndef CNFLOW_CNFLOW_H_
#define CNFLOW_CNFLOW_H_

#include <memory>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include "tsque.h"
#include "cnmodel.h"

namespace cnflow {

typedef struct HostDeviceInputArray {
    std::vector<cv::Mat> hosts;
    void **in_mlu_ptr;
    void **out_mlu_ptr;

    std::vector<std::string> imagenames;
    std::vector<float> ratios;

    HostDeviceInputArray() {}
    HostDeviceInputArray(std::vector<cv::Mat> hosts, void **in_mlu_ptr, void **out_mlu_ptr): 
        hosts(hosts), in_mlu_ptr(in_mlu_ptr), out_mlu_ptr(out_mlu_ptr) {}
} Host_DeviceInputArray;

typedef struct HostDeviceInput {
    cv::Mat host;
    void **in_mlu_ptr;
    void **out_mlu_ptr;

    std::string imagename;
    float ratio;

    HostDeviceInput() {}
    HostDeviceInput(cv::Mat host, void **in_mlu_ptr, void **out_mlu_ptr): 
        host(host), in_mlu_ptr(in_mlu_ptr), out_mlu_ptr(out_mlu_ptr) {}
} Host_DeviceInput;

class CnFlow {
  public:
    CnFlow();
    ~CnFlow();

    void showQueueSize();

    void join();
    void detach();

    void putImageList(const std::vector<std::string> &imagePath, int epoch);

    void addReadImage(int parallelism);
    void runReadImage();
    void runReadImage_ex(const std::vector<std::string> &imagePath);
    // void runReadImage(const std::vector<std::string> &imagePath);

    void addFaceBoxesPreProcess(int parallelism);
    void runFaceBoxesPreProcess();

    void addFaceBoxesForBatch(int parallelism);
    void runFaceBoxesForBatch();

    void addFaceBoxesPreprocessEx(int parallelism);
    void runFaceBoxesPreprocessEx();

    void addFaceBoxesInfer(int parallelism, int dp=1);
    void runFaceBoxesInfer(int dp, bool need_buffer);
    
    void addFaceBoxesPostProcess(int parallelism);
    void runFaceBoxesPostProcess();

    std::vector<std::thread *> threads;

    std::vector<cnmodel::CnModel *> faceboxesModels;

    tsque::TsQueue<std::string> imagePathQueue;
    tsque::TsQueue<cv::Mat> faceboxesRawImageQueue;
    tsque::TsQueue<Host_DeviceInput> faceboxesInputQueue;
    tsque::TsQueue<Host_DeviceInputArray> faceBoxesBatchInputQueue;
    tsque::TsQueue<Host_DeviceInputArray> faceboxesOutputQueue;
    tsque::TsQueue<Host_DeviceInput> faceboxesPostQueue;

    tsque::TsQueue<uint64_t> timeQueue;
    tsque::TsQueue<uint64_t> faceBoxesPreprocessTimeQueue;
    tsque::TsQueue<uint64_t> FaceBoxesInferTimeQueue;
    tsque::TsQueue<uint64_t> FaceBoxesPostProcessTimeQueue;

    int epoch = 1;
    std::vector<std::string> imagePath;
    std::string faceboxes_model_path;
    std::string faceboxes_func_name = "fusion_0";
    int faceboxes_height = 0;
    int faceboxes_width = 0;
    uint64_t time_start;
    uint64_t one_thrid_time;
    uint64_t two_thrid_time;
    int num_input = 0;
    int device = 0;
    bool fake_input = false;
};

}  // namespace mlu

#endif  // CNFLOW_CNFLOW_H_
