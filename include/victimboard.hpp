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
        Mat Captured_Image_RGB;
        Mat Captured_Image_to_Binary;
        Mat img_labeling; // connectedComponentsWithStats_output
        Mat stats;        // connectedComponentsWithStats_stats
        Mat centroids;
        Mat roi;
        Mat adaptive_capture;
        Mat result;

        typedef struct
        {
            Mat Image;
            Rect position;
            vector<KeyPoint> Keypoints;
        } Data;

        Data divided_Image_data[8];

        int image_x;
        int image_y;
        int image_width_divided3;
        int image_height_divided3;
        // connectedComponentsWithStats_center coordinates

        void run();
        void update();
        void all_clear();
        void labeling(const Mat &input_img, int &_x, int &_y, int &_width, int &_height);
        void divide_box();

        vector<Point> points;

        bool isRecv;
        bool ifCaptured = false;
        bool init();

        String info;

        ros::Publisher img_ad;
        ros::Publisher img_divide;

        std::string param;

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
    };

}