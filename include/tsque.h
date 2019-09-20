#ifndef CNFLOW_TSQUE_H_
#define CNFLOW_TSQUE_H_

#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace tsque {

#define USLEEP(t) \
{ \
    std::this_thread::sleep_for(std::chrono::microseconds(t)); \
}

/* By default, push data to tail, pop data from head.
 * 
 */
typedef enum TsQueuePosition {
    TSQUE_HEAD,
    TSQUE_TAIL
} TsQueuePosition_t;

template <typename T>
class TsQueue {
public:
    TsQueue();
    explicit TsQueue(int capacity);
    ~TsQueue();
    void reset();

    T operator[](int i);

    void resize(int capacity);
    int size();
    bool empty();
    bool full();

    int push(const T &data, TsQueuePosition_t pos=TSQUE_TAIL);
    void push_n(const std::vector<T> &datas, TsQueuePosition_t pos=TSQUE_TAIL);

    T pop(TsQueuePosition_t pos=TSQUE_HEAD);
    /* pop_ex: pop data, if fail (e.g. no data in queue), the ret will be set fo false. */
    T pop_ex(bool &ret, TsQueuePosition_t pos=TSQUE_HEAD);
    std::vector<T> force_pop_n(int n, TsQueuePosition_t pos=TSQUE_HEAD);
    std::vector<T> pop_n(int n, TsQueuePosition_t pos=TSQUE_HEAD);

private:
    int force_push(const T &data, TsQueuePosition_t pos);
    T force_pop(TsQueuePosition_t pos);

    std::deque<T> datas;
    int _capacity = 0x7fffffff;
    int _size = 0;
    std::mutex locker;
};

template <typename T>
TsQueue<T>::TsQueue() {}

template <typename T>
TsQueue<T>::TsQueue(int capacity): _capacity(capacity) {}

template <typename T>
TsQueue<T>::~TsQueue() {}

template <typename T>
void TsQueue<T>::reset() {
    locker.lock();
    datas = std::deque<T>();
    _size = 0;
    locker.unlock();
}

template <typename T>
void TsQueue<T>::resize(int capacity) {
    locker.lock();
    _capacity = capacity;
    locker.unlock();
}

template <typename T>
T TsQueue<T>::operator[](int i) {
    locker.lock();
    if (i < 0) {
        i += _size;
    }
    T &data = datas[i];
    locker.unlock();
    return data;
}

template <typename T>
bool TsQueue<T>::empty() {
    return _size <= 0;
}

template <typename T>
bool TsQueue<T>::full() {
    return _size >= _capacity;
}


template <typename T>
int TsQueue<T>::force_push(const T &data, TsQueuePosition_t pos) {
    int ret = 0;
    switch (pos) {
        case TSQUE_TAIL:
            datas.emplace_back(data);
            ++_size;
            break;
        case TSQUE_HEAD:
            datas.emplace_front(data);
            ++_size;
            break;
        default:
            ret = -1;
            break;
    }
    return ret;
}

/* Push data to deque. It will block if the size equal to capacity.
 * 
 */
template <typename T>
int TsQueue<T>::push(const T &data, TsQueuePosition_t pos) {
    int ret = 0;
    locker.lock();
    while (_size >= _capacity) {
        locker.unlock();
        USLEEP(100);
        locker.lock();
    }

    force_push(data, pos);
    locker.unlock();
    return ret;
}

template <typename T>
T TsQueue<T>::force_pop(TsQueuePosition_t pos) {
    T data;
    switch (pos) {
        case TSQUE_TAIL:
            data = datas.back();
            datas.pop_back();
            --_size;
            break;
        case TSQUE_HEAD:
            data = datas.front();
            datas.pop_front();
            --_size;
            break;
        default:
            break;
    }
    return std::move(data);
}

/* Pop data from deque. It will block if the size is 0. 
 *
 */
template <typename T>
T TsQueue<T>::pop(TsQueuePosition_t pos) {
    T data;
    locker.lock();
    while (_size <= 0) {
        locker.unlock();
        USLEEP(100);
        locker.lock();
    }

    data = force_pop(pos);
    locker.unlock();
    return std::move(data);
}

/* pop_ex will not block but may false when queue is empty.
 * 
 */
template <typename T>
T TsQueue<T>::pop_ex(bool &ret, TsQueuePosition_t pos) {
    T data;
    locker.lock();
    if (_size <= 0) {
        ret = false;
        locker.unlock();
        return std::move(data);
    }
    else {
        ret = true;
        data = force_pop(pos);
        locker.unlock();
        return std::move(data);
    }
}

template <typename T>
std::vector<T> TsQueue<T>::force_pop_n(int n, TsQueuePosition_t pos) {
    std::vector<T> list;
    for (int i = 0; i < n; ++i) {
        list.emplace_back(pop(pos));
    }
    return std::move(list);
}

template <typename T> 
std::vector<T> TsQueue<T>::pop_n(int n, TsQueuePosition_t pos) {
    std::vector<T> list;
    bool ret;
    list.emplace_back(pop(pos));
    for (int i = 1; i < n; ++i) {
        T data = pop_ex(ret, pos);
        if (ret == false) {
            break;
        }
        list.emplace_back(data);
    }
    return std::move(list);
}

template <typename T>
void TsQueue<T>::push_n(const std::vector<T> &datas, TsQueuePosition_t pos) {
    for (auto data : datas) {
        push(data, pos);
    }
}

template <typename T>
int TsQueue<T>::size() {
    int qsize;
    locker.lock();
    qsize = _size;
    locker.unlock();
    return qsize;
}

}  // namespace tsque

#endif  // CNFLOW_TSQUE_H_
