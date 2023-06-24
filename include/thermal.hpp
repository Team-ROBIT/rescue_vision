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

    class Thermal
    {
    public:
        Thermal(int argc, char **argv);
        ~Thermal();
        Mat *original;
        Mat clone_mat;
        Mat gray_clone;
        Mat output_thermal;
        Mat thermal_8bit;
        Mat thermal_binary;

        void run();
        void update();
        void all_clear();

        double minVal, maxVal;
        double minTemp, maxTemp;

        Point minLoc, maxLoc;

        bool isRecv;
        bool init();

        String info;

        ros::Publisher img_thermal;
        ros::Publisher img_thermal_gray;

        std::string param;

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
    };
}