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
using namespace std;

namespace vision_rescue
{

    class Victimboard
    {
    public:
        Victimboard(int argc, char **argv);
        ~Victimboard();
        Mat *original;
        Mat clone_mat;
        Mat gray_clone;
        Mat output_qr;
        Mat Image_to_Binary_OTSU;
        Mat Image_to_Binary_adaptive;
        Mat mask = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

        void run();
        void update();
        void all_clear();
        vector<Point> points;

        bool isRecv;
        bool init();

        String info;

        ros::Publisher img_ad;

        std::string param;

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
    };

}