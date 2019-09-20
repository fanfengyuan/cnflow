#ifndef CNFLOW_CNMODEL_H_
#define CNFLOW_CNMODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "cnrt.h"
#include "tsque.h"

namespace cnmodel {

#define CNRT_CHECK_V2(condition) \
do { \
    cnrtRet_t status = condition; \
    CHECK_EQ(status, CNRT_RET_SUCCESS) << " " << cnrtGetErrorStr(status); \
} while (0)

#define ALIGN_UP(x, a) (((x) + (a) - 1) / (a) * (a))

uint64_t time();

class CnMemManager {
public:
    CnMemManager() {}
    CnMemManager(size_t num_buffers, std::vector<size_t> size_in_bytes);
    ~CnMemManager();

    void **pop();
    void push(void **ptrs);

private:
    tsque::TsQueue<void **> buffer;
    int n_size;
};

struct Shape {
    int n;
    int c;
    int h;
    int w;
};

class CnModel {
public:
    std::string modelpath;
    std::string funcname;
    int device; 
    int dp;

    int input_num, output_num;
    std::vector<size_t> input_data_bytes;
    std::vector<size_t> output_data_bytes;
    std::vector<int> output_data_counts;
    cnrtStream_t stream;
    cnrtEvent_t event_start, event_end;
    cnrtModel_t model;
    cnrtFunction_t function;
    cnrtDataDescArray_t input_descS, output_descS;
    cnrtInvokeFuncParam_t invoke_func_param;

    CnModel(const char *_modelpath, 
             const char *_funcname, 
             int _device, 
             int _dp,
             bool need_buffer=false,
             cnrtDataType_t _input_dtype=CNRT_FLOAT32, 
             cnrtDimOrder_t _input_order=CNRT_NCHW, 
             cnrtDataType_t _output_dtype=CNRT_FLOAT32, 
             cnrtDimOrder_t _output_order=CNRT_NCHW);
    void invoke_ex(void **_input_mlu_ptrS, void **_output_mlu_ptrS);
    std::shared_ptr<std::shared_ptr<float>> invoke(void **ptr);
    void **deviceAllocInput();
    void **deviceAllocOutput();
    void copyin(void **mlu_ptr, void **cpu_ptr);
    std::shared_ptr<std::shared_ptr<float>> copyout(void **mlu_ptr);
    void freeInput(void **input_mlu);
    void freeOutput(void **output_mlu);
    ~CnModel();

    std::vector<Shape> input_shapes;

private:
    float ptv = 0;
    bool muta = false;
    u32_t affinity = 0x01;
    cnrtDim3_t dim = {1, 1, 1};
    cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;

    CnMemManager *input_buffer;
    CnMemManager *output_buffer;
};

}  // namespace cnmodel

#endif  // CNFLOW_CNMODEL_H_
