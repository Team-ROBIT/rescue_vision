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
#include <opencv2/xfeatures2d.hpp>

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
        Mat *original_thermal;
        Mat clone_mat;
        Mat clone_thermal_mat;
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
        Mat frame;
        Mat blob;

        typedef struct
        {
            Mat Image;
            Rect position;
            vector<KeyPoint> Keypoints;
        } Data;

        Data divided_Image_data[8];

        int save_image_position__Rotation_Direction[2] = {
            9,
        }; // save Rotation_Direction image Index

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
        void detect_location();
        void set_yolo();

        std::vector<std::string> class_names;
        std::vector<Point2i> hazmat_loc;

        Ptr<ORB> orb = ORB::create();

        cv::dnn::Net net;

        vector<Point> points;

        bool isRecv;
        // bool isRecv_thermal;
        bool ifCaptured = false;
        bool isOverlapping;
        bool init();
        bool isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2);

        String info;

        ros::Publisher img_ad;
        ros::Publisher img_divide;
        // ros::Publisher img_thermal_vt;

        std::string param;

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        // void imageCallBack_thermal(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
        // image_transport::Subscriber img_sub2;
    };

}