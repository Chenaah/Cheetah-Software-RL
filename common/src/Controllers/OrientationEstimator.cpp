/*! @file OrientationEstimator.cpp
 *  @brief All Orientation Estimation Algorithms
 *
 *  This file will contain all orientation algorithms.
 *  Orientation estimators should compute:
 *  - orientation: a quaternion representing orientation
 *  - rBody: coordinate transformation matrix (satisfies vBody = Rbody * vWorld)
 *  - omegaBody: angular velocity in body frame
 *  - omegaWorld: angular velocity in world frame
 *  - rpy: roll pitch yaw
 */

#include "Controllers/OrientationEstimator.h"

/*!
 * Get quaternion, rotation matrix, angular velocity (body and world),
 * rpy, acceleration (world, body) by copying from cheater state data
 */
template <typename T>
void CheaterOrientationEstimator<T>::run() {
  this->_stateEstimatorData.result->orientation =
      this->_stateEstimatorData.cheaterState->orientation.template cast<T>();
  this->_stateEstimatorData.result->rBody = ori::quaternionToRotationMatrix(
      this->_stateEstimatorData.result->orientation);
  this->_stateEstimatorData.result->omegaBody =
      this->_stateEstimatorData.cheaterState->omegaBody.template cast<T>();
  this->_stateEstimatorData.result->omegaWorld =
      this->_stateEstimatorData.result->rBody.transpose() *
      this->_stateEstimatorData.result->omegaBody;
  this->_stateEstimatorData.result->rpy =
      ori::quatToRPY(this->_stateEstimatorData.result->orientation);
  this->_stateEstimatorData.result->aBody =
      this->_stateEstimatorData.cheaterState->acceleration.template cast<T>();
  this->_stateEstimatorData.result->aWorld =
      this->_stateEstimatorData.result->rBody.transpose() *
      this->_stateEstimatorData.result->aBody;
}


template <typename T>
void VectorNavOrientationEstimator<T>::handleLCM()
{
    while ( !_interfaceLcmQuit ) {
        myOritationLCM.handle();
    }
}

template <typename T>
void VectorNavOrientationEstimator<T>::handleT265LCM ( const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                                                           const T265position_t* msg ){
    ( void ) rbuf;
    ( void ) chan;
    t265_roll = msg->rpyBOdy[0];
    t265_pitch = msg->rpyBOdy[1];
    t265_yaw = msg->rpyBOdy[2];
    t265_omega_x = msg->omegaBody[0];
    t265_omega_y = msg->omegaBody[1];
    t265_omega_z = msg->omegaBody[2];

    printf("received t265: %.2f\t%.2f\n",t265_roll,t265_pitch);
}
template <typename T>
VectorNavOrientationEstimator<T>::VectorNavOrientationEstimator()//:myLCM ( getLcmUrl ( 255 ) )
{
    _t265OritationLcmThread = std::thread ( &VectorNavOrientationEstimator<T>::handleLCM, this );
    if ( !myOritationLCM.good() ) {
        printf ( "my lcm _interfaceLCM failed to initialize\n");
    }

    myOritationLCM.subscribe ( "t265_position_msg", &VectorNavOrientationEstimator<T>::handleT265LCM , this );
}
/*!
 * Get quaternion, rotation matrix, angular velocity (body and world),
 * rpy, acceleration (world, body) from vector nav IMU
 */
template <typename T>
void VectorNavOrientationEstimator<T>::run() {
  this->_stateEstimatorData.result->orientation[0] =
      this->_stateEstimatorData.vectorNavData->quat[3];
  this->_stateEstimatorData.result->orientation[1] =
      this->_stateEstimatorData.vectorNavData->quat[0];
  this->_stateEstimatorData.result->orientation[2] =
      this->_stateEstimatorData.vectorNavData->quat[1];
  this->_stateEstimatorData.result->orientation[3] =
      this->_stateEstimatorData.vectorNavData->quat[2];

  // std::cout << "[DEBUG] QUAT FROM vectorNavData: [" << this->_stateEstimatorData.vectorNavData->quat[0] << ", " << this->_stateEstimatorData.vectorNavData->quat[1] << ", " << this->_stateEstimatorData.vectorNavData->quat[2] << ", " << this->_stateEstimatorData.vectorNavData->quat[3] << "] " << std::endl;

  if(_b_first_visit){
    Vec3<T> rpy_ini = ori::quatToRPY(this->_stateEstimatorData.result->orientation);
    rpy_ini[0] = 0;
    rpy_ini[1] = 0;
    _ori_ini_inv = rpyToQuat(-rpy_ini);
    _b_first_visit = false;

//      cfg.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
//      // Start pipeline with chosen configuration
//      pipe.start(cfg);
  }

/*    pipe_cout++;
//    if(pipe_cout%4==0){
//        auto frames = pipe.wait_for_frames();
//        // Get a frame from the pose stream
//        auto f = frames.first_or_default(RS2_STREAM_POSE);
//        // Cast the frame to pose_frame and get its data
//        auto pose_data = f.as<rs2::pose_frame>().get_pose_data();
//        if(pipe_cout%40==0)
//        {
//            //四元数到真实对应
////            printf("T265: %.3f\t%.3f\t%.3f\t%.3f\n",pose_data.rotation.w,-pose_data.rotation.z,-pose_data.rotation.x,pose_data.rotation.y);
////            printf("IMU: %.3f\t%.3f\t%.3f\t%.3f\n",this->_stateEstimatorData.result->orientation[0],this->_stateEstimatorData.result->orientation[1],
////                   this->_stateEstimatorData.result->orientation[2],this->_stateEstimatorData.result->orientation[3]);
//            //角速度到真实对应
////            printf("T265: %.3f\t%.3f\t%.3f\n",-pose_data.angular_velocity.z,-pose_data.angular_velocity.x,pose_data.angular_velocity.y);
////            printf("IMU: %.3f\t%.3f\t%.3f\n",this->_stateEstimatorData.result->omegaBody[0],this->_stateEstimatorData.result->omegaBody[1],
////                   this->_stateEstimatorData.result->omegaBody[2]);
//           //加速度不对，没对应起来
//            printf("T265: %.3f\t%.3f\t%.3f\n",-pose_data.acceleration.z,-pose_data.acceleration.x,pose_data.acceleration.y+9.8);
//            printf("IMU: %.3f\t%.3f\t%.3f\n",this->_stateEstimatorData.result->aBody[0],this->_stateEstimatorData.result->aBody[1],
////                   this->_stateEstimatorData.result->aBody[2]);
//        }
//   }*/

  // this->_stateEstimatorData.result->orientation =
  //   ori::quatProduct(_ori_ini_inv, this->_stateEstimatorData.result->orientation);
  this->_stateEstimatorData.result->orientation = this->_stateEstimatorData.result->orientation;

  this->_stateEstimatorData.result->rpy =
      ori::quatToRPY(this->_stateEstimatorData.result->orientation);

  this->_stateEstimatorData.result->rBody = ori::quaternionToRotationMatrix(
      this->_stateEstimatorData.result->orientation);

  this->_stateEstimatorData.result->omegaBody =
      this->_stateEstimatorData.vectorNavData->gyro.template cast<T>();

  this->_stateEstimatorData.result->omegaWorld =
      this->_stateEstimatorData.result->rBody.transpose() *
      this->_stateEstimatorData.result->omegaBody;

  this->_stateEstimatorData.result->aBody =
      this->_stateEstimatorData.vectorNavData->accelerometer.template cast<T>(); // 0 0 9.8

  this->_stateEstimatorData.result->aWorld =
      this->_stateEstimatorData.result->rBody.transpose() *
      this->_stateEstimatorData.result->aBody;
   static int times_show(0);
   times_show++;
//   if(times_show%10==1)
//   {
//      printf("rpy:%.2f\t%.2f\t%.2f\n",this->_stateEstimatorData.result->rpy(0),this->_stateEstimatorData.result->rpy(1),
//               this->_stateEstimatorData.result->rpy(2));
//       printf("body rpy rate:%.2f\t%.2f\t%.2f\n",this->_stateEstimatorData.result->omegaBody(0),this->_stateEstimatorData.result->omegaBody(1),
//              this->_stateEstimatorData.result->omegaBody(2));
//       printf("body acc:%.2f\t%.2f\t%.2f\n",this->_stateEstimatorData.result->aBody(0),this->_stateEstimatorData.result->aBody(1),
//              this->_stateEstimatorData.result->aBody(2));
//       printf("-------------------------------------------------\n");
//   }
}

template class CheaterOrientationEstimator<float>;
template class CheaterOrientationEstimator<double>;

template class VectorNavOrientationEstimator<float>;
template class VectorNavOrientationEstimator<double>;
