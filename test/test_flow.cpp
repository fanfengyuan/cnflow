#include "cnflow.h"

#include <string>
#include <vector>

#include "cnrt.h"

int main(int argc, char* argv[]) {
    std::string model_path("/share/projects/mxnet-helper/model_fusion_1.cambricon");

    if (argc != 2) {
        LOG(ERROR) << "Usage: ./test_flow model_path";
        exit(-1);
    }
    model_path = argv[1];
    
    const int device = 0;

    std::vector<std::string> imagepaths(10000, "datas/face.jpg");
    bool fake_input = false;

    LOG(INFO) << "data path: datas/face.jpg";
    LOG(INFO) << "fake input: " << fake_input;

    // Set to 1 for MLU270
    const int dp_faceboxes = 1;

    // The cycle time for imagepaths. If -1, it will not stop.
    int epoch = 1;

    LOG(INFO) << "model path: " << model_path;

    cnflow::CnFlow flower;
    // Set the cambricon offline model path.
    flower.faceboxes_model_path = model_path;
    // Set the cambricon offline model func name.
    flower.faceboxes_func_name = "fusion_0";
    // Set the device id.
    flower.device = device;
    // If true, use the fake image (all 1) for input.
    flower.fake_input = fake_input;
    // // The number of parallelism models.
    // int num_models = batch_size + 1;
    // // The buffer size for model input and output deviceMemory.
    // int buffer_size = 2 * num_models - 2;

    flower.putImageList(imagepaths, epoch);
    flower.addFaceBoxesPreprocessEx(32);
    flower.addFaceBoxesInfer(dp_faceboxes);
    flower.addFaceBoxesPostProcess(32);
    flower.join();

    return 0;
}
