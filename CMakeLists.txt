cmake_minimum_required(VERSION 3.0.2)
project(rescue_vision)

find_package(catkin REQUIRED COMPONENTS qt_build
roscpp
rospy
cv_bridge
message_runtime
message_generation
image_transport
std_msgs
roslib
sensor_msgs
image_transport
)
find_package(OpenCV 4.5.2 REQUIRED)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_executable(findc src/findc.cpp)
add_executable(hazmat src/hazmat.cpp)
add_executable(qr_detect src/qr_detect.cpp)
add_executable(thermal src/thermal.cpp)

target_link_libraries(findc 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
)

target_link_libraries(hazmat
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)

target_link_libraries(qr_detect
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
<<<<<<< Updated upstream
)
=======
)

target_link_libraries(thermal
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)

##install(TARGETS rescue_vision RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/rescue_vision_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_rescue_vision.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
>>>>>>> Stashed changes
