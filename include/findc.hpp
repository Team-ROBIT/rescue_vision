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

    class Findc
    {

    public:
        Findc(int argc, char **argv);
        ~Findc();
        Mat *original;
        Mat clone_mat;
        Mat clone_binary;
        Mat in_cup_mat;
        Mat contour_lane;
        Mat roi;
        Mat gray_clone;
        Mat corner_img;
        Mat in_cup_gray;
        Mat in_cup_canny;
        Mat in_cup_binary;
        Mat cup_temp_mat;
        Mat ok_cup_mat;
        Mat expand_cup_mat;
        Mat expand_cup_gray;
        Mat expand_cup_binary;
        Mat expand_cup_mat2;
        Mat expand_cup_gray2;
        Mat expand_cup_binary2;
        Mat last_binary;


        Vec3i c;
        vector<Vec3f> circles; //(중심좌표x, 중심좌표y, 반지름r)
        vector<Point2f> corners;
        vector<Point2f> corners2;
        vector<Point2f> real_corners;
        vector<vector<Point>> contours;
        vector<Vec4i> lines;

        int radius;
        int range_radius_small = 10; // 50
        int range_radius_big = 120;
        int distance;
        
        bool check_black(const Mat & binary_mat);
        void find_contour();
        void random_image_save(const Mat &input_img, const char *format);
        void catch_c(Point center, int radius);
        void run();
        void update();
        void all_clear();
        void choose_circle(Point center, int radius);
        void detect_way();
        void remove_text();

        double slope;
        double radians;
        double degrees;
        double temp_corner_left;
        double temp_corner_right;
        double averageAngle_;

        bool isRecv;
        bool init();
        bool find_ok=false;
        bool second_ring=false;
        bool first_ring=false;
    
        ros::Publisher img_tr;
        ros::Publisher img_cup;
        ros::Publisher img_cup_binary;
        ros::Publisher img_expand_binary;
        ros::Publisher img_expand_binary2;


    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
        image_transport::Subscriber img_sub2;
    };

}