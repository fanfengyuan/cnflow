#include <memory>

#include "cnflow.h"
#include "cnmodel.h"
#include "faceboxes_preprocess.h"

namespace cnflow {

#define DEVICE 0

void setdevice(int device) {
    cnrtDev_t dev;
    CNRT_CHECK_V2(cnrtGetDeviceHandle(&dev, device));
    CNRT_CHECK_V2(cnrtSetCurrentDevice(dev));
}

CnFlow::CnFlow() {
    CNRT_CHECK_V2(cnrtInit(0));

    faceBoxesBatchInputQueue.resize(320);
    faceboxesOutputQueue.resize(320);
}

CnFlow::~CnFlow() {}

void CnFlow::join() {
    for (auto thread : threads) {
        thread->join();
    }
}

void CnFlow::detach() {
    for (auto thread : threads) {
        thread->detach();
    }
}

void CnFlow::putImageList(const std::vector<std::string> &imagePath) {
    num_input += imagePath.size();
    for (auto path : imagePath) {
        imagePathQueue.push(path);
    }
    time_start = cnmodel::time();
}

void CnFlow::addFaceBoxesPreprocessEx(int parallelism) {
    for (int i = 0; i < parallelism; ++i) {
        threads.push_back(new std::thread(&CnFlow::runFaceBoxesPreprocessEx, this));
    }
}

void CnFlow::runFaceBoxesPreprocessEx() {
    setdevice(0);
    do {
        USLEEP(100);
    } while (faceboxesModels.size() <= 0);

    while (true) {
        auto images = imagePathQueue.pop_n(faceboxesModels[0]->dp);

        uint64_t t1 = cnmodel::time();

        std::vector<float> ratios;
        std::vector<cv::Mat> faceboxes_imgs;
        for (int i = 0; i < images.size(); ++i) {
            float ratio = 1.f;
            cv::Mat rawimg = cv::imread(images[i].c_str());
            std::cout << "Image: " << images[i] << " shape: [" << rawimg.rows << ", " << rawimg.cols << ", " << rawimg.channels() << "]" << std::endl;
            cv::Mat rszd_img = faceboxes_preprocess(rawimg, faceboxes_height, faceboxes_width, ratio);
            faceboxes_imgs.emplace_back(rszd_img);
            ratios.push_back(ratio);
        }

        std::shared_ptr<uint8_t> imgsptr = copyto<uint8_t>(faceboxes_imgs, faceboxesModels[0]->dp);
        uint8_t *p_imgsptr = imgsptr.get();

        void **in_mlu = faceboxesModels[0]->deviceAllocInput();
        void **out_mlu = faceboxesModels[0]->deviceAllocOutput();
        faceboxesModels[0]->copyin(in_mlu, (void **)&p_imgsptr);

        uint64_t t2 = cnmodel::time();
        faceBoxesPreprocessTimeQueue.push(t2 - t1);

        HostDeviceInputArray faceboxesBatchInput(std::move(faceboxes_imgs), in_mlu, out_mlu);
        faceboxesBatchInput.imagenames = std::move(images);
        faceboxesBatchInput.ratios = std::move(ratios);
        faceBoxesBatchInputQueue.push(std::move(faceboxesBatchInput));
    }
}

void CnFlow::addFaceBoxesInfer(int parallelism, int dp) {
    bool need_buffer = true;
    for (int i = 0; i < parallelism; ++i) {
        threads.push_back(new std::thread(&CnFlow::runFaceBoxesInfer, this, dp, need_buffer));
        need_buffer = false;
    }
} 

void CnFlow::runFaceBoxesInfer(int dp, bool need_buffer) {
    size_t align_input_size = ALIGN_UP(sizeof(uint8_t) * faceboxes_height * faceboxes_width * 4, 64 * 1024);
    std::vector<size_t> input_bytes = {align_input_size};
    cnmodel::CnModel *moder = new cnmodel::CnModel("offline_models/faceboxes-500x500.cambricon", "fusion_0", 0, dp, input_bytes, need_buffer, CNRT_UINT8, CNRT_NHWC);
    if (need_buffer)
        faceboxesModels.push_back(moder);

    while (true) {
        auto faceboxesinput = faceBoxesBatchInputQueue.pop();

        uint64_t t1 = cnmodel::time();

        auto img = faceboxesinput.hosts;
        void **_input_mlu_ptrS = faceboxesinput.in_mlu_ptr;
        void **_output_mlu_ptrS = faceboxesinput.out_mlu_ptr;
        moder->invoke_ex(_input_mlu_ptrS, _output_mlu_ptrS);

        uint64_t t2 = cnmodel::time();
        FaceBoxesInferTimeQueue.push(t2 - t1);

        Host_DeviceInputArray faceboxesoutput(std::move(img), _input_mlu_ptrS, _output_mlu_ptrS);
        faceboxesoutput.imagenames = std::move(faceboxesinput.imagenames);
        faceboxesoutput.ratios = std::move(faceboxesinput.ratios);
        faceboxesOutputQueue.push(std::move(faceboxesoutput));
    }
}

void CnFlow::addFaceBoxesPostProcess(int parallelism) {
    for (int i = 0; i < parallelism; ++i) {
        threads.push_back(new std::thread(&CnFlow::runFaceBoxesPostProcess, this));
    }
}

void CnFlow::runFaceBoxesPostProcess() {
    setdevice(0);

    do {
        USLEEP(100);
    } while (faceboxesModels.size() <= 0);

    while (true) {
        auto faceboxesoutput = faceboxesOutputQueue.pop();

        uint64_t t1 = cnmodel::time();

        std::shared_ptr<std::shared_ptr<float>> faceboxes = faceboxesModels[0]->copyout(faceboxesoutput.out_mlu_ptr);
        faceboxesModels[0]->freeInput(faceboxesoutput.in_mlu_ptr);
        faceboxesModels[0]->freeOutput(faceboxesoutput.out_mlu_ptr);

        auto images = faceboxesoutput.hosts;
        auto imagenames = faceboxesoutput.imagenames;
        auto ratios = faceboxesoutput.ratios;
        for (int i = 0; i < images.size(); i++) {
            float *location = faceboxes.get()[0].get() + i * faceboxesModels[0]->output_data_counts[0];
            float *confidence = faceboxes.get()[1].get() + i * faceboxesModels[0]->output_data_counts[1];

            /* auto boxes = postprocess(boxes) */
            
            Host_DeviceInput empty_data;
            faceboxesPostQueue.push(std::move(empty_data));
        }

        uint64_t t2 = cnmodel::time();
        FaceBoxesPostProcessTimeQueue.push(t2 - t1);

        if (num_input == faceboxesPostQueue.size()) {
            auto current_time = cnmodel::time();

            double ptv = static_cast<double>(current_time - time_start) / static_cast<double>(num_input);
            printf("sec: %ld us\n", current_time - time_start);
            printf("qps: %lf\n", 1000000. / ptv);
        }
    }    
}

}  // namespace mlu
