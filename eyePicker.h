#ifndef __COMMON_INCLUDE__
#define __COMMON_INCLUDE__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#endif

#ifndef __EYE_PICKER__
#define __EYE_PICKER__

/* Includes */

/* Constants */
#define CHERRY 3
#define EP__K_EYE_PERCENT_WIDTH		35
#define EP__K_EYE_PERCENT_HEIGHT	27
#define EP__K_EYE_PERCENT_TOP		27
#define EP__K_EYE_PERCENT_SIDE		13
#define EP__K_WEIGHT_BLUR_SIZE		5
#define EP__K_FAST_EYE_WIDTH		50
#define EP__GRADIENT_THRESHOLD		30.0
#define EP__KALMAN_DIFF_THRESHOLD	25

#define EP__EYE_STATE_CLOSE			0
#define EP__EYE_STATE_OPEN			1
#define EP__EYE_STATE_LEFT_CLOSED 	2
#define EP__EYE_STATE_RIGHT_CLOSED 	3

/* Types */
struct detectionData{
	cv::Rect faceRegion;
	cv::Rect leftEyeRegion;
	cv::Rect rightEyeRegion;
	cv::Point leftEyePosition;
	cv::Point rightEyePosition;
	cv::Point resultleftEyePosition;
	cv::Point resultrightEyePosition;

	int eyeWidth;
	int eyeHeight;
	int eyeState;
};

struct kalmanEyePositionData{
	double KPLX_xhat;
	double KPLX_xhatMinus;
	double KPLX_p;
	double KPLX_pMinus;

	double* pKPLX_xhat;
	double* pKPLX_xhatMinus;
	double* pKPLX_p;
	double* pKPLX_pMinus;

	double KPLY_xhat;
	double KPLY_xhatMinus;
	double KPLY_p;
	double KPLY_pMinus;

	double* pKPLY_xhat;
	double* pKPLY_xhatMinus;
	double* pKPLY_p;
	double* pKPLY_pMinus;

	double KPRX_xhat;
	double KPRX_xhatMinus;
	double KPRX_p;
	double KPRX_pMinus;

	double* pKPRX_xhat;
	double* pKPRX_xhatMinus;
	double* pKPRX_p;
	double* pKPRX_pMinus;

	double KPRY_xhat;
	double KPRY_xhatMinus;
	double KPRY_p;
	double KPRY_pMinus;

	double* pKPRY_xhat;
	double* pKPRY_xhatMinus;
	double* pKPRY_p;
	double* pKPRY_pMinus;
};

struct kalmanEyeOpenAndCloseData
{
	double KLIP_xhat;
	double KLIP_xhatMinus;
	double KLIP_p;
	double KLIP_pMinus;

	double* pKLIP_xhat;
	double* pKLIP_xhatMinus;
	double* pKLIP_p;
	double* pKLIP_pMinus;

	double KRIP_xhat;
	double KRIP_xhatMinus;
	double KRIP_p;
	double KRIP_pMinus;

	double* pKRIP_xhat;
	double* pKRIP_xhatMinus;
	double* pKRIP_p;
	double* pKRIP_pMinus;
};

class EyePicker{
	protected:
		detectionData outputData;
		kalmanEyePositionData eyePositionData;
		kalmanEyeOpenAndCloseData eyeInnerProductData;

		int counta = 0;
		int gaussian_leftx[100];
		int gaussian_lefty[100];
		int gaussian_rightx[100];
		int gaussian_righty[100];
		
		double totalsum_mean = 0;
		double sumtoralmean_test = 0;
		double left_right_total = 0;

		double mean_odd[100];
		double mean_even[100];

		double edgecountsum = 0;

		double mean_update = 1;
		double mean_update_th = 1;

		int maxPointX = 0;
		int maxPointY = 0;

		int skipnum = 1;
		int nump = 0;
		int LockEscape = 0;

		double lastLeftKalmanResult = 0;
		double lastRightKalmanResult = 0;
		
    public:
		EyePicker(){
			//kalman of eye position
			this->eyePositionData.pKPLX_xhat = &this->eyePositionData.KPLX_xhat;
			this->eyePositionData.pKPLX_xhatMinus = &this->eyePositionData.KPLX_xhatMinus;
			this->eyePositionData.pKPLX_p = &this->eyePositionData.KPLX_p;
			this->eyePositionData.pKPLX_pMinus = &this->eyePositionData.KPLX_pMinus;

			this->eyePositionData.pKPLY_xhat = &this->eyePositionData.KPLY_xhat;
			this->eyePositionData.pKPLY_xhatMinus = &this->eyePositionData.KPLY_xhatMinus;
			this->eyePositionData.pKPLY_p = &this->eyePositionData.KPLY_p;
			this->eyePositionData.pKPLY_pMinus = &this->eyePositionData.KPLY_pMinus;

			this->eyePositionData.pKPRX_xhat = &this->eyePositionData.KPRX_xhat;
			this->eyePositionData.pKPRX_xhatMinus = &this->eyePositionData.KPRX_xhatMinus;

			this->eyePositionData.pKPRX_p = &this->eyePositionData.KPRX_p;
			this->eyePositionData.pKPRX_pMinus = &this->eyePositionData.KPRX_pMinus;

			this->eyePositionData.pKPRY_xhat = &this->eyePositionData.KPRY_xhat;
			this->eyePositionData.pKPRY_xhatMinus = &this->eyePositionData.KPRY_xhatMinus;
			this->eyePositionData.pKPRY_p = &this->eyePositionData.KPRY_p;
			this->eyePositionData.pKPRY_pMinus = &this->eyePositionData.KPRY_pMinus;

			//kalman of eye inner product
			this->eyeInnerProductData.pKLIP_xhat = &this->eyeInnerProductData.KLIP_xhat;
			this->eyeInnerProductData.pKLIP_xhatMinus = &this->eyeInnerProductData.KLIP_xhatMinus;
			this->eyeInnerProductData.pKLIP_p = &this->eyeInnerProductData.KLIP_p;
			this->eyeInnerProductData.pKLIP_pMinus = &this->eyeInnerProductData.KLIP_pMinus;

			this->eyeInnerProductData.pKRIP_xhat = &this->eyeInnerProductData.KRIP_xhat;
			this->eyeInnerProductData.pKRIP_xhatMinus = &this->eyeInnerProductData.KRIP_xhatMinus;
			this->eyeInnerProductData.pKRIP_p = &this->eyeInnerProductData.KRIP_p;
			this->eyeInnerProductData.pKRIP_pMinus = &this->eyeInnerProductData.KRIP_pMinus;

			this->eyeInnerProductData.KLIP_xhat = 0.0;
			this->eyeInnerProductData.KRIP_xhat = 0.0;

			this->eyeInnerProductData.KLIP_p = 1.0;
			this->eyeInnerProductData.KRIP_p = 1.0;
		}
        void selectEyeArea(cv::Mat frame_gray, cv::Rect face, detectionData &outputData);
        
		cv::Point detectEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow, int& num, detectionData& outputData);
        double kalmanEyePosition(double measurement, double* xhat, double* xhatMinus, double* p, double* pMinus, double Q, double R, int nump);
        
		void scaleToFastSize(const cv::Mat &src, cv::Mat &dst);
        cv::Mat computeMatXGradient(const cv::Mat& mat);
        cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
        double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
        void FindComponents(cv::Mat &out, int label, int &count, int cutLine, int num, int maxX, int maxY);
        void calcAccuInnerProduct(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out, cv::Mat &open);
        double kalmanEyeInnerProduct(double measurement, double* xhat, double* xhatMinus, double* p, double* pMinus, double Q, double R, int num);
        cv::Point unscalePoint(cv::Point p, cv::Rect origSize);
		
        void Search(cv::Mat &out, int label, int y, int x, int &count, int cutLine);
};

#endif