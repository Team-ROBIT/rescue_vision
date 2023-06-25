#include <ros/ros.h>
#include <string>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "std_msgs/String.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

namespace vision_rescue
{

    class Hazmat
    {
    public:
        Hazmat(int argc, char **argv);
        ~Hazmat();
        Mat *original;
        Mat clone_mat;
        Mat frame;
        Mat blob;
        void run();
        void update();
        void all_clear();
        cv::dnn::Net net;
        std::vector<std::string> class_names;

        bool isRecv;
        bool isOverlapping;
        bool init();
        bool isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2);

        void set_yolo();

        ros::Publisher img_result;

        std::string param;

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
    };

}