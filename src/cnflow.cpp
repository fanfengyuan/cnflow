#include <memory>

#include "cnflow.h"
#include "cnmodel.h"
#include "faceboxes_preprocess.h"

namespace cnflow {

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
    setdevice(device);
    do {
        USLEEP(100);
    } while (faceboxesModels.size() <= 0);

    while (true) {
        // TODO: maybe not input_shapes[0]
        auto images = imagePathQueue.pop_n(faceboxesModels[0]->dp * faceboxesModels[0]->input_shapes[0].n);
        if (images.size() != faceboxesModels[0]->dp * faceboxesModels[0]->input_shapes[0].n) {
            LOG(WARNING) << "need " << faceboxesModels[0]->dp * faceboxesModels[0]->input_shapes[0].n << " pop " << images.size();
        }

        uint64_t t1 = cnmodel::time();

        std::vector<float> ratios;
        std::vector<cv::Mat> faceboxes_imgs;
        for (int i = 0; i < images.size(); ++i) {
            float ratio = 1.f;
            if (fake_input) {
                // Maybe not CV_8UC3
                static std::vector<uint8_t> ones(faceboxes_height * faceboxes_width * 3, 1);
                cv::Mat fake_img(faceboxes_height, faceboxes_width, CV_8UC3, ones.data());
                faceboxes_imgs.emplace_back(fake_img);
            }
            else {
                cv::Mat rawimg = cv::imread(images[i].c_str());
                cv::Mat rszd_img = faceboxes_preprocess(rawimg, faceboxes_height, faceboxes_width, ratio);
                faceboxes_imgs.emplace_back(rszd_img);
            }
            ratios.push_back(ratio);
        }

        std::shared_ptr<uint8_t> imgsptr = copyto<uint8_t>(faceboxes_imgs, faceboxesModels[0]->dp * faceboxesModels[0]->input_shapes[0].n);
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
    cnmodel::CnModel *moder = new cnmodel::CnModel(faceboxes_model_path.c_str(), faceboxes_func_name.c_str(), device, dp, need_buffer, CNRT_UINT8, CNRT_NHWC);

    // TODO: maybe not input_shapes[0]
    this->faceboxes_height = moder->input_shapes[0].h;
    this->faceboxes_width = moder->input_shapes[0].w;
    LOG(INFO) << " shape: [" << moder->input_shapes[0].n << ", " << moder->input_shapes[0].c 
              << ", " << moder->input_shapes[0].h << ", " << moder->input_shapes[0].w << "]" << std::endl;
    if (need_buffer)
        faceboxesModels.push_back(moder);

    while (true) {
        if (faceBoxesBatchInputQueue.empty()) {
            LOG(WARNING) << "faceBoxesBatchInputQueue size == 0";
        }

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

        if (faceboxesOutputQueue.full()) {
            LOG(WARNING) << "faceboxesOutputQueue is full";
        }

        faceboxesOutputQueue.push(std::move(faceboxesoutput));
    }
}

void CnFlow::addFaceBoxesPostProcess(int parallelism) {
    for (int i = 0; i < parallelism; ++i) {
        threads.push_back(new std::thread(&CnFlow::runFaceBoxesPostProcess, this));
    }
}

void CnFlow::runFaceBoxesPostProcess() {
    setdevice(device);

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
            timeQueue.push(cnmodel::time());
        }

        uint64_t t2 = cnmodel::time();
        FaceBoxesPostProcessTimeQueue.push(t2 - t1);

        if (num_input == faceboxesPostQueue.size()) {
            auto current_time = cnmodel::time();

            double ptv = static_cast<double>(current_time - time_start) / static_cast<double>(num_input);
            printf("sec: %ld us\n", current_time - time_start);
            printf("qps: %lf\n", 1000000. / ptv);

            auto _start_cnt = num_input / 3;
            auto _end_cnt = num_input / 3 * 2;
            auto _tstart = timeQueue[_start_cnt];
            auto _tend = timeQueue[_end_cnt];

            double full_ptv = static_cast<double>(_tend - _tstart) / static_cast<double>(_end_cnt - _start_cnt);
            printf("full-utili qps: %lf\n", 1000000. / full_ptv);
        }
    }    
}

}  // namespace mlu
