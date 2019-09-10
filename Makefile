all:
	CNRT_HOME=/usr/local/neuware/
	GLOG_HOME=/share/projects/glog/prefix/install/
	OPENCV_HOME=/share/projects/opencv-2.4/install/

	g++ -std=c++11 -O3 src/*.cpp -shared -fPIC -g -o lib/libcnflow.so \
		-I include \
		-I /usr/local/neuware/include -L /usr/local/neuware/lib64 -lcnrt \
		-I /share/projects/glog/prefix/install/include -L /share/projects/glog/prefix/install/lib -lglog \
		-I /share/projects/opencv-2.4/install/include -L /share/projects/opencv-2.4/install/lib `pkg-config opencv --libs --cflags` \

	g++ -std=c++11 -O3 test/test_flow.cpp -g -o bin/test_flow \
		-I include -L lib -lcnflow \
		-I /usr/local/neuware/include -L /usr/local/neuware/lib64 -lcnrt \
		-I /share/projects/glog/prefix/install/include -L /share/projects/glog/prefix/install/lib -lglog \
		-I /share/projects/opencv-2.4/install/include -L /share/projects/opencv-2.4/install/lib `pkg-config opencv --libs --cflags` \
