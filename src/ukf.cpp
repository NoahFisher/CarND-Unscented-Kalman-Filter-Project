#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  NIS_radar_ = 0;
  NIS_laser_ = 0;

  // State dimension
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  num_sigma_points = 2 * n_aug_ + 1;

  weights_ = VectorXd(num_sigma_points);
  // sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, num_sigma_points);
  Xsig_pred_.fill(0.0);

  // spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
        Convert radar from polar to cartesian coordinates and initialize state.
        */
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhodot = meas_package.raw_measurements_(2);

      x_ << rho * cos(phi), rho * sin(phi), rhodot * cos(phi), rhodot * sin(phi);
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //set the state with the initial location and zero velocity
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  long dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);


  // measurement update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    cout << "UpdateRadar\n";
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    cout << "UpdateLidar\n";
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**********************************
   * 1. Create Augmented Sigma Points
   *********************************
   */
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, num_sigma_points);

  // augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L_ = P_aug.llt().matrixL();

  //create augmented sigma points
  // Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++) {
    if(i==0){
      Xsig_aug.col(0) = x_aug;
    }
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L_.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L_.col(i);
  }

  /**********************************
   * 2. Predict Sigma Points
   **********************************
   */
  //predict sigma points
  for(int i=0; i < num_sigma_points; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if(yawd > 0.001) {
        px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p += 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p += 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p += nu_a*delta_t;

    yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p += nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  /**********************************
   * 3. Predict Mean and Covariance
   **********************************
   */
  // set weights and predicted state mean
  VectorXd x_pred = VectorXd(n_x_);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  x_pred.fill(0);
  P_pred.fill(0);

  for (int i=0; i < num_sigma_points; i++) {  //2n+1 weights
    if(i == 0) {
      weights_(i) = lambda_/(lambda_+n_aug_);
    } else {
      weights_(i) = 0.5/(n_aug_+lambda_);
    }
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  for (int i = 0; i < num_sigma_points; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));
    P_pred += weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x_pred;
  P_ = P_pred;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //set measurement dimension, lidar can measure px and py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, num_sigma_points);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  for (int i = 0; i < num_sigma_points; i++) {
    //transform sigma points into measurement space
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);

    Zsig.col(i) << px,
                   py;

    //calculate mean predicted measurement
    z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  for (int i = 0; i < num_sigma_points; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add R to S
  MatrixXd R = MatrixXd(2,2);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S += R;

  //create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  //calculate cross correlation matrix
  for (int i = 0; i < num_sigma_points; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = NormalizeAngle(x_diff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc += weights_(i) * x_diff * z_diff.transpose();

  }

  // residual
  VectorXd z_diff = z - z_pred;

  //calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**********************************
   * Predict Radar Sigma Points
   **********************************
   rho - radius of object from origin
   phi - angle of object wrt x axis
   rhod - radial velocity
   */

  //create matrix for sigma points in measurement space
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, num_sigma_points);
  Zsig.fill(0.0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  for (int i = 0; i < num_sigma_points; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    double rho = sqrt(p_x*p_x + p_y*p_y);
    Zsig(0,i) = rho;
    Zsig(1,i) = atan2(p_y,p_x);
    // avoid division by zero (eps = 0.001)
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / max(0.001, rho);

    // mean predicted measurement
    z_pred += weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < num_sigma_points; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S += R;

  /**********************************
   * Predict Radar Sigma Points
   **********************************
   */

  //incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);

  //cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  //calculate cross correlation matrix
  for (int i = 0; i < num_sigma_points; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //normalize angles
    z_diff(1) = NormalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();
  // residual
  VectorXd z_diff = z - z_pred;
  //normalize angles
  z_diff(1) = NormalizeAngle(z_diff(1));

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();

}

double UKF::NormalizeAngle(double angle) {
  const double Max = M_PI;
  const double Min = -M_PI;

  return angle < Min
    ? Max + std::fmod(angle - Min, Max - Min)
    : std::fmod(angle - Min, Max - Min) + Min;
}
