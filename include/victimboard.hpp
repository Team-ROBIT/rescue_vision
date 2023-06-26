#include <ros/ros.h>
#include <string>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "robit_master_vision.hpp"
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
#include <std_msgs/Float64MultiArray.h>

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
        Mat thermal_mat;
        Mat gray_clone;
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
        Mat motion1;
        Mat motion2;
        Mat movement_Find_center_img;
        Mat Divided_Image__Rotation_Direction[2];
        Mat C_up;
        Mat C_down;

        vector<Vec3f> circles;

        Point Center;
        Point Vec1, Vec2;

        double Rotation_Direction[2] = {
            0,
        };

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
        int count__Rotation_Direction = 0;
        int count_Movement_find_circle = 0;
        int movement_count = 0;
        int nROI_radius = 110;
        // connectedComponentsWithStats_center coordinates

        void
        run();
        void update();
        void all_clear();
        void labeling(const Mat &input_img, int &_x, int &_y, int &_width, int &_height);
        void divide_box();
        void detect_location();
        void set_yolo();
        void img_Detect_movement(Mat &input_img);
        void CLAHE(Mat &image);
        void img_cvtcolor_gray(Mat &input, Mat &output);
        void set_thermal();
        void thermal_location(int loc);
        void detect_C(Mat &C);

        std::vector<std::string> class_names;
        std::vector<Point2i> hazmat_loc;

        Ptr<ORB> orb = ORB::create();

        cv::dnn::Net net;

        vector<Point> points;

        Rect first_thermal;
        Rect second_thermal;

        bool isRecv;
        bool isRecv_thermal;
        bool ifCaptured = false;
        bool isOverlapping;
        bool start_motion = false;
        bool exist_c = false;
        bool init();
        bool isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2);
        bool motion_exit = false;

        Mat region_of_interest(Mat input_img);

        String info;

        ros::Publisher img_ad;
        ros::Publisher img_result;
        ros::Publisher img_result_thermal;
        ros::Publisher img_binary_vt;
        ros::Publisher findc_pub;
        ros::Publisher up_c;
        ros::Publisher down_c;
        ros::Publisher img_binary_thermal;

        ros::Publisher bingle_data_;

        std::string param;

        std_msgs::Float64MultiArray bingle_msgs;

        float bingle_save[4] = {
            0,
        };

        Rect motion_loc_save[2];

    private:
        int init_argc;
        char **init_argv;
        void imageCallBack(const sensor_msgs::ImageConstPtr &msg_img);
        void imageCallBack_thermal(const sensor_msgs::ImageConstPtr &msg_img);
        image_transport::Subscriber img_sub;
        image_transport::Subscriber img_sub_thermal;
    };

}